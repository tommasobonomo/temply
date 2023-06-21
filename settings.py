import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List

import hydra

data_dir = Path("data/geneva")
extension = ".jsonl"

# Disable tokenizers parallelism -- impacts torch dataloaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SplitType(str, Enum):
    train = "train"
    validation = "validation"
    test = "test"


@dataclass
class ModelConfig:
    """
    Model config.
    Has fields that are custom and used to configure the model and its trianing,
    hopefully they should be self-explanatory.
    """

    transformer_model: str = "bert-base-uncased"
    learning_rate: float = 5e-5
    batch_size: int = 64
    use_bias: bool = False
    model_dir: Path = Path("models")
    model_name: str = ""
    num_warmup_steps: int = 0
    num_training_steps: int = 0
    additional_special_tokens: List[str] = field(default_factory=lambda: [])
    checkpoint_extension: str = ".ckpt"


@dataclass
class TrainerConfig:
    """
    Trainer config. Will be deconstructed as arguments to initialise a `pytorch_lightning.Trainer`,
    so one should take care not to add fields that are not compatible with the `Trainer`'s initalisation arguments.
    """

    accelerator: str = "auto"
    fast_dev_run: bool = False
    max_epochs: int = 50
    log_every_n_steps: int = 5


@dataclass
class Config:
    """
    Root config class. Holds both `ModelConfig` and `TrainerConfig` as fields.
    The rest of the fields concern how the script should behave (fit and train, only fit, etc.)
    and where the data is saved and should be loaded from (`split_paths`).
    """

    split_paths: Dict[SplitType, Path] = field(
        default_factory=lambda: {
            SplitType.train: data_dir / f"{SplitType.train}{extension}",
            SplitType.validation: data_dir / f"{SplitType.validation}{extension}",
            SplitType.test: data_dir / f"{SplitType.test}{extension}",
        }
    )
    model: ModelConfig = ModelConfig()
    trainer: TrainerConfig = TrainerConfig()
    # Script configs
    fit: bool = True
    evaluate: bool = True
    num_workers: int = 4
    wandb_dir: Path = Path("wandb")
    enable_wandb: bool = True
    class2id_path: Path = data_dir / "class2id.json"
    eval_split: SplitType = SplitType.validation
    data_pct: float = 1.0


config_store = hydra.core.config_store.ConfigStore.instance()
config_store.store(name="config", node=Config)
config_store.store(name="model", node=ModelConfig)
config_store.store(name="trainer", node=TrainerConfig)
