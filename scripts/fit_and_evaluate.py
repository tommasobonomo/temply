import logging
import pprint
from pathlib import Path

import hydra
import lightning as pl
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from tqdm import tqdm

from settings import Config, SplitType
from src.dataset import get_dataloader
from src.evaluation import (
    Evaluator,
    evaluate_arguments,
    evaluate_trigger,
    reduce_metrics,
)
from src.logger import logger_factory
from src.model import DEE, predict
from src.tokenizer import tokenizer_factory
from src.typing_classes import EvaluationMetrics, Event

config_store = hydra.core.config_store.ConfigStore.instance()
config_store.store(name="config", node=Config)


def get_model_checkpoint(config: Config, console_logger: logging.Logger):
    final_model_dir = config.model.model_dir / config.model.model_name
    if config.model.model_checkpoint_file is None:
        # There should be a model prefixed with [best]. If no model found, raise error
        final_model_checkpoint = next(
            (
                possible_checkpoint
                for possible_checkpoint in final_model_dir.iterdir()
                if possible_checkpoint.stem.startswith("[best]")
                and possible_checkpoint.suffix == config.model.checkpoint_extension
            ),
            None,
        )
    else:
        final_model_checkpoint = final_model_dir / config.model.model_checkpoint_file

    if final_model_checkpoint is None or not final_model_checkpoint.exists():
        raise RuntimeError(
            f"Model was not found in {(config.model.model_dir / config.model.model_name).as_posix()}.\n"
            + "Either specify a checkpoint path in `config.model.model_checkpoint_file` or make sure that "
            + "one of the checkpoints in the folder is prefixed with `[best]`."
        )

    console_logger.info(
        f"Loading model from checkpoint path {final_model_checkpoint.as_posix()}..."
    )

    return final_model_checkpoint


@hydra.main(version_base=None, config_name="config")
def main(config: Config):
    """
    Main function of the script. Parameters can be passed through modification of the default Config object
    using the Hydra library.
    """

    console_logger = logger_factory("fit_and_evaluate", level=logging.INFO)
    console_logger.info("Starting fit and evaluate script...")

    console_logger.info("Getting tokenizer...")
    tokenizer = tokenizer_factory(
        config.model.transformer_model, config.model.additional_special_tokens
    )

    if not config.fit:
        console_logger.info("Not fitting model, loading with checkpoints...")
        # Instantiating trainer for evaluation
        if config.model.model_name == "":
            raise ValueError(
                f"Must specify a valid name of a model in directory {config.model.model_dir}"
            )
        model_name = config.model.model_name

        final_model_checkpoint = get_model_checkpoint(config, console_logger)
        model = DEE.load_from_checkpoint(final_model_checkpoint, config=config)
    else:
        console_logger.info("Started fitting of model...")

        # Load datasets and dataloaders
        train_dataloader = get_dataloader(
            config=config, split=SplitType.train, tokenizer=tokenizer, shuffled=True
        )
        val_dataloader = get_dataloader(
            config=config, split=SplitType.dev, tokenizer=tokenizer, shuffled=False
        )

        # Load model from checkpoint if `fit=True` and a `model.model_checkpoint_file` is specified
        if config.model.model_name == "":
            model = DEE(config)
        else:
            # config.model.model_name != ""
            init_checkpoint = get_model_checkpoint(config, console_logger)
            model = DEE.load_from_checkpoint(init_checkpoint, config=config)

        # W&B management
        if config.enable_wandb:
            wandb_logger = WandbLogger(
                project="dee", log_model="best", save_dir=str(config.wandb_dir)
            )
            experiment_name = wandb_logger.experiment.name
        else:
            experiment_name = None
            wandb_logger = None
        model_name = experiment_name if experiment_name else "local_run"

        # Base logger and all loggers
        base_logger = TensorBoardLogger(
            save_dir=config.model.model_dir, name=model_name
        )
        loggers = (
            [base_logger, wandb_logger] if wandb_logger is not None else [base_logger]
        )

        learning_rate_callback = LearningRateMonitor(logging_interval="step")

        # Checkpoint saving
        checkpoint_callback = ModelCheckpoint(
            dirpath=config.model.model_dir / model_name,
            monitor="val/<ace2005>-argument-classify-f1",
            mode="max",
            save_weights_only=False,
            save_top_k=5,
        )
        checkpoint_callback.FILE_EXTENSION = config.model.checkpoint_extension

        trainer = pl.Trainer(  # type: ignore
            **config.trainer,
            callbacks=[checkpoint_callback, learning_rate_callback],
            logger=loggers,
        )

        # Start fitting
        trainer.fit(
            model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
        )

        final_model_checkpoint = Path(checkpoint_callback.best_model_path)
        final_model_checkpoint.rename(
            final_model_checkpoint.parent / f"[best]{final_model_checkpoint.name}"
        )

    if not config.evaluate:
        console_logger.info("Not evaluating model, exiting...")
        return
    else:
        console_logger.info("Evaluating model...")
        val_dataloader = get_dataloader(
            config=config, split=SplitType.dev, tokenizer=tokenizer, shuffled=False
        )

        decoded_sources, decoded_targets, decoded_predictions = predict(
            model=model,
            dataloader=val_dataloader,
            tokenizer=tokenizer,
            ckpt_paht=final_model_checkpoint,
            use_gpu=config.trainer.accelerator != "cpu" and torch.cuda.is_available(),
        )

        console_logger.info("Saving predictions...")
        with open(
            config.model.model_dir / model_name / config.model.predictions_file_name,
            "w+",
        ) as f:
            f.writelines(
                f"{idx}:\n{source}\n{target}\n{pred}\n\n"
                for idx, (source, target, pred) in enumerate(
                    zip(decoded_sources, decoded_targets, decoded_predictions)
                )
            )

        console_logger.info("Instantiating evaluator...")
        evaluator = Evaluator(config)

        predicted_events: dict[str, list[Event | None]] = {
            key: [] for key in evaluator.definitions.keys()
        }
        for prediction in tqdm(
            decoded_predictions,
            desc="Discretizing predictions",
        ):
            compositional_token, event = evaluator(prediction)
            predicted_events[compositional_token].append(event)

        golden_events: dict[str, list[Event | None]] = {
            key: [] for key in evaluator.definitions.keys()
        }
        for target in tqdm(decoded_targets, desc="Discretizing target labels"):
            compositional_token, event = evaluator(target)
            golden_events[compositional_token].append(event)

        all_datasets = list(evaluator.definitions.keys())
        trigger_scores: dict[str, EvaluationMetrics] = {}
        argument_identification_scores: dict[str, EvaluationMetrics] = {}
        argument_classification_scores: dict[str, EvaluationMetrics] = {}
        for key in all_datasets:
            ds_golden_events = golden_events[key]
            ds_predicted_events = predicted_events[key]

            trigger_scores[key] = evaluate_trigger(
                ds_golden_events, ds_predicted_events
            )
            (
                argument_identification_scores[key],
                argument_classification_scores[key],
            ) = evaluate_arguments(ds_golden_events, ds_predicted_events)

        # Calculate overall score as harmonic mean of score on all datasets
        trigger_scores["overall"] = reduce_metrics(trigger_scores, "harmonic_mean")
        argument_identification_scores["overall"] = reduce_metrics(
            argument_identification_scores, "harmonic_mean"
        )
        argument_classification_scores["overall"] = reduce_metrics(
            argument_classification_scores, "harmonic_mean"
        )

        for key in all_datasets + ["overall"]:
            if config.enable_wandb and config.fit:
                keys = [
                    f"val/{key}-trigger-classify-precision",
                    f"val/{key}-trigger-classify-recall",
                    f"val/{key}-trigger-classify-f1",
                    f"val/{key}-argument-identify-precision",
                    f"val/{key}-argument-identify-recall",
                    f"val/{key}-argument-identify-f1",
                    f"val/{key}-argument-classify-precision",
                    f"val/{key}-argument-classify-recall",
                    f"val/{key}-argument-classify-f1",
                ]
                values = [
                    trigger_scores[key].precision,
                    trigger_scores[key].recall,
                    trigger_scores[key].f1,
                    argument_identification_scores[key].precision,
                    argument_identification_scores[key].recall,
                    argument_identification_scores[key].f1,
                    argument_classification_scores[key].precision,
                    argument_classification_scores[key].recall,
                    argument_classification_scores[key].f1,
                ]
                wandb_logger.log_metrics(dict(zip(keys, values)))  # type: ignore

            console_logger.info(
                f"[{key}] Trigger classification scores: {pprint.pformat(trigger_scores[key].to_dict())}"
            )
            console_logger.info(
                f"[{key}] Argument identification scores: {pprint.pformat(argument_identification_scores[key].to_dict())}"
            )
            console_logger.info(
                f"[{key}] Argument classification scores: {pprint.pformat(argument_classification_scores[key].to_dict())}"
            )

        console_logger.info("Evaluation done.")


if __name__ == "__main__":
    seed_everything(78)
    torch.set_float32_matmul_precision("medium")
    main()
