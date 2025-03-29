from typing import Optional, List
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
    train_models_name_list: List[str] = field(
        metadata={
            "help": "The list of models to train."
        }
    )
    adapter_architecture: Optional[str] = field(
        default="textcls_resnet18",
        metadata={
            "help": "The architecture of the adapter to use."
        },
    )
    unet_init_features: int = field(
        default=64,
        metadata={
            "help": "The initial number of features in the UNet."
        },
    )
    axial_tf_layers: int = field(
        default=5,
        metadata={
            "help": "The number of layers in the axial transformer."
        },
    )
    rnn_layers: int = field(
        default=1,
        metadata={
            "help": "The number of layers in the rnn."
        },
    )
    adapter_hidden_size: int = field(
        default=768,
        metadata={
            "help": "The hidden size of the transformer."
        },
    )
    adapter_params: Optional[dict] = field(
        default=None,
        metadata={
            "help": "The parameters for the adapter."
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
    ola_augments: Optional[List[dict]] = field(
        default=None,
        metadata={"help": "The augmentations to use for OLA."},
    )
    eval_models_name_list: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The list of models to evaluate."},
    )
    eval_adapter_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the adapter checkpoint to evaluate."},
    )
    sp_aug: str = field(
        default='none',
        metadata={"help": "same augments."},
    )

    def __post_init__(self):
        # init adapter parameters
        self.adapter_params = {}
        if self.unet_init_features is not None:
            self.adapter_params["unet_init_features"] = self.unet_init_features
        if self.axial_tf_layers is not None:
            self.adapter_params["axial_tf_layers"] = self.axial_tf_layers
        if self.rnn_layers is not None:
            self.adapter_params["rnn_layers"] = self.rnn_layers
        if self.adapter_hidden_size is not None:
            self.adapter_params["hidden_size"] = self.adapter_hidden_size
        if len(self.eval_models_name_list) == 0:
            self.eval_models_name_list += self.train_models_name_list
        if self.ola_augments is None:
            assert self.sp_aug in ['none', 'gemma', 'qwen', 'llama']
            if self.sp_aug == 'none':
                self.ola_augments = [
                    # {
                    #     "class_name": "RandomTemperatureScaling",
                    #     "params": {
                    #         "p": 0.2,
                    #         "min_temp": 0.6,
                    #         "max_temp": 3
                    #     }
                    # },
                    {
                        "class_name": "RandomHightlightColumns",
                        "params": {
                            "p": 0.3,
                            "min_columns": 1,
                            "max_columns": 3,
                            "ref_rank1": 0,
                            "ref_rank2": 1
                        }
                    },
                    # {
                    #     "class_name": "RandomHightlightColumns",
                    #     "params": {
                    #         "p": 0.2,
                    #         "min_columns": 1,
                    #         "max_columns": 6,
                    #         "ref_rank1": 3,
                    #         "ref_rank2": 4
                    #     }
                    # },
                    {
                        "class_name": "AddGuassianNoise",
                        "params": {
                            "p": 0.3,
                            "std_ratio": 0.2
                        }
                    },
                ]
            elif self.sp_aug == 'gemma':
                self.ola_augments = [
                    {
                        "class_name": "RandomHightlightColumns",
                        "params": {
                            "p": 0.3,
                            "min_columns": 1,
                            "max_columns": 3,
                            "ref_rank1": 0,
                            "ref_rank2": 1
                        }
                    },
                    {
                        "class_name": "AddGuassianNoise",
                        "params": {
                            "p": 0.15,
                            "std_ratio": 0.13
                        }
                    }
                ]
            elif self.sp_aug == 'qwen':
                self.ola_augments = [
                    {
                        "class_name": "RandomHightlightColumns",
                        "params": {
                            "p": 0.3,
                            "min_columns": 1,
                            "max_columns": 3,
                            "ref_rank1": 0,
                            "ref_rank2": 1
                        }
                    },
                    {
                        "class_name": "AddGuassianNoise",
                        "params": {
                            "p": 0.3,
                            "std_ratio": 0.2
                        }
                    }
                ]
            elif self.sp_aug == 'llama':
                self.ola_augments = [
                    {
                        "class_name": "RandomTemperatureScaling",
                        "params": {
                            "p": 0.2,
                            "min_temp": 0.6,
                            "max_temp": 3
                        }
                    },
                    {
                        "class_name": "RandomHightlightColumns",
                        "params": {
                            "p": 0.3,
                            "min_columns": 1,
                            "max_columns": 3,
                            "ref_rank1": 0,
                            "ref_rank2": 1
                        }
                    },
                    {
                        "class_name": "AddGuassianNoise",
                        "params": {
                            "p": 0.3,
                            "std_ratio": 0.2
                        }
                    }
                ]


@dataclass
class DataArguments:
    dataset_name: str = field(
        default="conll2000_pos",
        metadata={"help": "The name of the dataset to use."},
    )
    othertest_dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the other dataset to use(cross language)."},
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
    pad_to_multiple_of: int = field(
        default=16,
        metadata={"help": "The multiple to pad the sequence length to."},
    )
    visual_text_file: Optional[str] = field(
        default="datasets/visualize/all_text.txt",
        metadata={"help": "The file containing texts to visual attention maps."},
    )
    visual_annot_size: int = field(
        default=1,
        metadata={"help": "The size of each annotation in the visualization."},
    )
    visual_label_size: int = field(
        default=3,
        metadata={"help": "The size of the label in the visualization."},
    )
    use_generated_oladata: bool = field(
        default=False,
        metadata={"help": "Whether to use the generated OLA data."},
    )
    attn_type: str = field(
        default="ola",
        metadata={"help": "The attention map type(ola, tandem, first, last)."},
    )
    do_classify_data_generate: bool = field(
        default=False,
        metadata={"help": "Whether to do classify data generate."},
    )

    def __post_init__(self):
        if self.num_classes is None:
            if self.dataset_name == "imdb":
                self.num_classes = 2
            elif self.dataset_name == "conll2000_pos":
                # 1 for padding or special tokens such as [CLS], [SEP], [MASK], etc.
                self.num_classes = 44 + 1
            elif self.dataset_name == "conll2000_chunk":
                self.num_classes = 23
            elif self.dataset_name == "conll2012cn_pos":
                # 1 for padding or special tokens such as [CLS], [SEP], [MASK], etc.
                self.num_classes = 36 + 1
            elif self.dataset_name == "conll2012en_pos":
                # 1 for padding or special tokens such as [CLS], [SEP], [MASK], etc.
                self.num_classes = 49 + 1
            elif self.dataset_name == "conll2012cn_entity":
                self.num_classes = 37
            elif self.dataset_name == "conll2012en_entity":
                self.num_classes = 37
            elif self.dataset_name == "semeval_re":
                self.num_classes = 19
            elif self.dataset_name == "ud_english_gum":
                self.num_classes = 55
            elif self.dataset_name == "ud_english_ewt":
                self.num_classes = 55
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
    eval_during_checkpointing: bool = field(
        default=True,
        metadata={"help": "Whether to run evaluation during checkpointing."},
    )
    do_visualize: bool = field(
        default=False,
        metadata={"help": "Whether to visualize the attention maps."},
    )
    do_gen_save_ola: bool = field(
        default=False,
        metadata={"help": "Whether to generate and save the OLA data."},
    )
    task: str = field(
        default="pos",
        metadata={"help": "Task name."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.experiment_name is None:
            self.experiment_name = time.strftime(
                "%Y%m%d_%H%M%S", time.localtime())
        self.output_dir = os.path.join(self.output_dir, self.experiment_name)
