from typing import Optional, List, Union
import os
import time
import json
from dataclasses import dataclass, field, asdict

from transformers import TrainingArguments


def save_arguments(args_list, json_path):
    all_args = {}
    for tmp_args in args_list:
        all_args.update(asdict(tmp_args))
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as json_file:
            json.dump(all_args, json_file, indent=4)
    print(f"Arguments are saved to {json_path}.")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    adapter_architecture: Optional[str] = field(
        default="resnet18",
        metadata={
            "help": "The architecture of the adapter to use."
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    local_files_only: bool = field(
        default=True,
        metadata={
            "help": "Whether to only use local files and not download from the internet."
        },
    )
    use_orders: List[int] = field(
        default_factory=lambda: [1, 2, 3],
        metadata={"help": "The orders of attention maps to use."},
    )
    remove_outliers: bool = field(
        default=True,
        metadata={"help": "Whether to remove outliers from attention maps."},
    )
    outliers_sigma_multiplier: float = field(
        default=3.0,
        metadata={"help": "The sigma multiplier to determine outliers."},
    )
    eval_models_name_list: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The list of models to evaluate."},
    )
    eval_adapter_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the adapter checkpoint to evaluate."},
    )
    visual_models_name_list: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The list of models to visualize."},
    )

    def __post_init__(self):
        if len(self.eval_models_name_list) == 0:
            self.eval_models_name_list.append(self.model_name_or_path)


@dataclass
class DataArguments:
    dataset_name: str = field(
        metadata={"help": "The name of the dataset to use."},
    )
    num_classes: int = field(
        default=None,
        metadata={"help": "The number of classes in the dataset"}
    )
    random_order: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle the order of tasks randomly."},
    )
    cutoff_len: int = field(
        default=320,
        metadata={"help": "The maximum length of the input sequence."},
    )
    visual_text_file: Optional[str] = field(
        default="datasets/visualize/all_text.txt",
        metadata={"help": "The file containing texts to visual attention maps."},
    )

    def __post_init__(self):
        if self.num_classes is None:
            if self.dataset_name == "imdb":
                self.num_classes = 2
            else:
                raise ValueError(f"Dataset {self.dataset_name} is not supported.")


@dataclass
class OLALMTrainingArguments(TrainingArguments):
    # general settings
    seed: int = field(
        default=2025,
        metadata={
            "help": "Random seed that will be set at the beginning of training."
        },
    )
    experiment_name: Optional[str] = field(
        default=None, metadata={"help": "Name of the experiment"}
    )
    output_dir: str = field(
        default="outputs",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    do_visualize: bool = field(
        default=False,
        metadata={"help": "Whether to visualize the attention maps."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.experiment_name is None:
            self.experiment_name = time.strftime(
                "%Y%m%d_%H%M%S", time.localtime())
        self.output_dir = os.path.join(self.output_dir, self.experiment_name)
