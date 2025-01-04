from typing import Optional, List
from dataclasses import asdict
import os
import json

import torch
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer import WEIGHTS_NAME, TRAINING_ARGS_NAME
from transformers.utils import logging, SAFE_WEIGHTS_NAME
import safetensors

from models import OLAModel
from data_utils import DataManager
from utils.arguments import OLALMTrainingArguments
from utils.evaluate import evaluate_ola_adapter_with_multi_llms


logger = logging.get_logger(__name__)
ADAPTERS_CKPT_NAME = "ola_adapter_weight.bin"
TRAINING_ARGS_JSON = "training_args.json"


class OLALMTrainer(Trainer):
    def __init__(
        self, 
        eval_models_name_list: List[str],
        data_manager: DataManager,
        task: str,
        attn_type: str,
        use_generated_oladata: bool,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.eval_models_name_list = eval_models_name_list
        self.data_manager = data_manager
        self.task = task
        self.attn_type = attn_type
        self.use_generated_oladata = use_generated_oladata
        self.args: OLALMTrainingArguments

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if isinstance(self.model, OLAModel):
            self.model.save_adapter(
                os.path.join(output_dir, ADAPTERS_CKPT_NAME)
            )
        elif not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                self.accelerator.unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if hasattr(self, "processing_class"):
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        with open(os.path.join(output_dir, TRAINING_ARGS_JSON), "w") as json_file:
            json.dump(asdict(self.args), json_file, indent=4)

        # evaluate during checkpointing
        if self.args.eval_during_checkpointing and self.use_generated_oladata:
            # set model to eval mode
            self.model.eval()
            # load arguments
            eval_args = os.path.join(output_dir, "..", "args.json")
            with open(eval_args, "r") as f:
                eval_args = json.load(f)
            # evaluate
            evaluate_ola_adapter_with_multi_llms(
                self.eval_models_name_list,
                eval_args,
                output_dir,
                self.data_manager,
                self.task,
                self.args.per_device_eval_batch_size,
                self.attn_type,
                self.use_generated_oladata,
                model=self.model
            )
            # set model to train mode
            self.model.train()
