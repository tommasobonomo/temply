import math
from functools import partial
from pathlib import Path
from typing import Callable

import lightning as pl
import torch
from torch.optim import RAdam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BartForConditionalGeneration,
    GenerationConfig,
    LogitsProcessor,
    PreTrainedTokenizerFast,
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.optimization import get_linear_schedule_with_warmup  # noqa: F401

import wandb
from settings import Config
from src.dataset import Batch
from src.evaluation import (
    EvaluationMetrics,
    Evaluator,
    evaluate_arguments,
    evaluate_trigger,
    reduce_metrics,
)
from src.tokenizer import remove_token, tokenizer_factory
from src.typing_classes import Event


class PrefixAllowedTokens:
    def __init__(self, compositional_tokens: torch.Tensor) -> None:
        # compositional_tokens should be a tensor of size [batch_size, compositional_rank]
        # Basically the prefix tokens that were passed in the input batch.
        self.compositional_tokens = compositional_tokens

    def __call__(self, batch_id: int, input_ids: torch.Tensor) -> list[int] | None:
        # Here we want to return the correct compositional token based on the input ones
        if input_ids.size(0) == 2:
            return self.compositional_tokens[batch_id].tolist()
        else:
            return None


class CustomPrefixConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], list[int] | None],
        num_beams: int,
    ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.Tensor:
        mask = torch.full_like(scores, -math.inf)

        for batch_id, beam_sent in enumerate(
            input_ids.view(-1, self._num_beams, input_ids.shape[-1])
        ):
            for beam_id, sent in enumerate(beam_sent):
                prefix_constrained_token_ids = self._prefix_allowed_tokens_fn(
                    batch_id, sent
                )
                if prefix_constrained_token_ids is None:
                    mask[
                        batch_id * self._num_beams + beam_id,
                        :,
                    ] = 0
                else:
                    mask[
                        batch_id * self._num_beams + beam_id,
                        prefix_constrained_token_ids,
                    ] = 0

        return scores + mask


class DEE(pl.LightningModule):
    def __init__(
        self, config: Config, tokenizer: PreTrainedTokenizerFast | None = None
    ) -> None:
        super().__init__()

        # seq2seq model
        self.seq2seq = BartForConditionalGeneration.from_pretrained(
            config.model.transformer_model
        )

        self.learning_rate = config.model.learning_rate
        self.num_warmup_steps = config.model.num_warmup_steps
        self.num_training_steps = config.model.num_training_steps
        self.wandb_logging = config.enable_wandb

        # Resize token embeddings if there are additional special tokens
        if len(config.model.additional_special_tokens) > 0:
            initial_token_embeddings_size = (
                self.seq2seq.get_input_embeddings().weight.size(0)
            )
            self.seq2seq.resize_token_embeddings(
                initial_token_embeddings_size
                + len(config.model.additional_special_tokens)
            )

        self.generation_config = GenerationConfig(
            max_new_tokens=300, num_beams=5, early_stopping=False
        )
        self.compositional_rank = config.compositional_rank

        self.complete_eval_in_dev = config.complete_eval_in_dev
        if self.complete_eval_in_dev:
            # We want to run complete evaluation in validation step
            self.evaluator = Evaluator(config)
            # Instantiate tokenizer if not passed
            if tokenizer is None:
                self.tokenizer = tokenizer_factory(
                    config.model.transformer_model,
                    config.model.additional_special_tokens,
                )

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = RAdam(self.parameters(), lr=self.learning_rate)
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, self.num_warmup_steps, self.num_training_steps
        # )
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "step",
        #         "name": "hf_slanted_triangle_scheduler",
        #     },
        # }
        return optimizer

    def forward(self, batch: Batch) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        if self.compositional_rank > 0:
            input_compositional_tokens = input_ids[:, 1 : 1 + self.compositional_rank]
            prefix_allowed_tokens_fn = PrefixAllowedTokens(input_compositional_tokens)
            logits_processor = [
                CustomPrefixConstrainedLogitsProcessor(
                    prefix_allowed_tokens_fn,
                    self.generation_config.num_beams
                    // self.generation_config.num_beam_groups,
                )
            ]
        else:
            logits_processor = []

        with torch.no_grad():
            out = self.seq2seq.generate(
                input_ids,
                attention_mask=attention_mask,
                generation_config=self.generation_config,
                logits_processor=logits_processor,
            )
        return out

    def step(self, batch: Batch) -> Seq2SeqLMOutput:
        out = self.seq2seq(**batch)
        return out

    def training_step(self, batch: Batch, *args, **kwargs) -> torch.Tensor:  # type: ignore
        out = self.step(batch)
        self.log("train/loss", out.loss)
        return out.loss

    def on_fit_start(self) -> None:
        if self.wandb_logging:
            all_datasets = list(self.evaluator.definitions.keys()) + ["overall"]
            for key in all_datasets:
                # Trigger classification
                wandb.define_metric(
                    f"val/{key}-trigger-classify-precision", summary="max"
                )
                wandb.define_metric(f"val/{key}-trigger-classify-recall", summary="max")
                wandb.define_metric(f"val/{key}-trigger-classify-f1", summary="max")
                # Argument identification
                wandb.define_metric(
                    f"val/{key}-argument-identify-precision", summary="max"
                )
                wandb.define_metric(
                    f"val/{key}-argument-identify-recall", summary="max"
                )
                wandb.define_metric(f"val/{key}-argument-identify-f1", summary="max")
                # Argument classification
                wandb.define_metric(
                    f"val/{key}-argument-classify-precision", summary="max"
                )
                wandb.define_metric(
                    f"val/{key}-argument-classify-recall", summary="max"
                )
                wandb.define_metric(f"val/{key}-argument-classify-f1", summary="max")

    def on_validation_start(self) -> None:
        if self.complete_eval_in_dev:
            self.predicted_events: dict[str, list[Event | None]] = {
                key: [] for key in self.evaluator.definitions.keys()
            }
            self.golden_events: dict[str, list[Event | None]] = {
                key: [] for key in self.evaluator.definitions.keys()
            }

    def validation_step(self, batch: Batch, *args, **kwargs) -> torch.Tensor:  # type: ignore
        out = self.step(batch)
        self.log("val/loss", out.loss, prog_bar=True)

        if self.complete_eval_in_dev:
            # Run complete evaluation, i.e. decoding predictions and comparing to gold labels
            decoding_fn = partial(
                self.tokenizer.batch_decode,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            generated_pred = self.forward(batch)
            string_targets = decoding_fn(remove_token(batch["labels"], -100))
            string_preds = decoding_fn(
                remove_token(generated_pred, self.tokenizer.pad_token_id)
            )
            for string_target in string_targets:
                compositional_token, gold_event = self.evaluator(string_target)
                self.golden_events[compositional_token].append(gold_event)

            for string_pred in string_preds:
                compositional_token, gold_event = self.evaluator(string_pred)
                self.predicted_events[compositional_token].append(gold_event)

        return out.loss

    def on_validation_epoch_end(self) -> None:
        if self.complete_eval_in_dev:
            all_datasets = list(self.evaluator.definitions.keys())
            trigger_scores: dict[str, EvaluationMetrics] = {}
            arg_id_scores: dict[str, EvaluationMetrics] = {}
            arg_cls_scores: dict[str, EvaluationMetrics] = {}
            for key in all_datasets:
                golden_events = self.golden_events[key]
                predicted_events = self.predicted_events[key]

                trigger_scores[key] = evaluate_trigger(golden_events, predicted_events)
                (
                    arg_id_scores[key],
                    arg_cls_scores[key],
                ) = evaluate_arguments(golden_events, predicted_events)

            # Calculate overall score as harmonic mean of score on all datasets
            trigger_scores["overall"] = reduce_metrics(trigger_scores, "harmonic_mean")
            arg_id_scores["overall"] = reduce_metrics(arg_id_scores, "harmonic_mean")
            arg_cls_scores["overall"] = reduce_metrics(arg_cls_scores, "harmonic_mean")
            for key in all_datasets + ["overall"]:
                metrics_values = {
                    # Trigger classify
                    f"val/{key}-trigger-classify-precision": trigger_scores[
                        key
                    ].precision,
                    f"val/{key}-trigger-classify-recall": trigger_scores[key].recall,
                    f"val/{key}-trigger-classify-f1": trigger_scores[key].f1,
                    # Argument identify
                    f"val/{key}-argument-identify-precision": arg_id_scores[
                        key
                    ].precision,
                    f"val/{key}-argument-identify-recall": arg_id_scores[key].recall,
                    f"val/{key}-argument-identify-f1": arg_id_scores[key].f1,
                    # Argument classify
                    f"val/{key}-argument-classify-precision": arg_cls_scores[
                        key
                    ].precision,
                    f"val/{key}-argument-classify-recall": arg_cls_scores[key].recall,
                    f"val/{key}-argument-classify-f1": arg_cls_scores[key].f1,
                }
                self.log_dict(metrics_values)


def predict(
    model: DEE,
    dataloader: DataLoader,
    tokenizer: PreTrainedTokenizerFast,
    ckpt_paht: Path | None,
    use_gpu: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    if ckpt_paht is not None and ckpt_paht.exists() and ckpt_paht.is_file():
        model.load_from_checkpoint(ckpt_paht)

    if use_gpu:
        model = model.cuda()

    decoding_fn = partial(
        tokenizer.batch_decode,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=True,
    )
    sources = []
    targets = []
    predictions = []
    for batch in tqdm(dataloader, desc="Predicting"):
        # Save sources, replacing pad token with space to keep all special tokens except the pad
        sources += decoding_fn(remove_token(batch["input_ids"], tokenizer.pad_token_id))

        # Replace stray -100 in labels and decode them
        targets += decoding_fn(remove_token(batch["labels"], -100))

        # Pass through the model and save the decoded predictions
        with torch.no_grad():
            if use_gpu:
                batch["input_ids"] = batch["input_ids"].cuda()
                batch["attention_mask"] = batch["attention_mask"].cuda()

            raw_prediction = model.forward(batch)
            predictions += decoding_fn(
                remove_token(raw_prediction, tokenizer.pad_token_id)
            )

    return sources, targets, predictions
