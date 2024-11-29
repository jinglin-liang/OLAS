from typing import Tuple
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json

from torch.utils.data import DataLoader
from transformers import (
    set_seed,
    HfArgumentParser,
)

from utils import (
    ADAPTERS_CKPT_NAME,
    save_arguments,
    evaluate_ola_adapter,
    visualize_attn_map,
    TextClsMetric,
    TokenClsMetric,
    ModelArguments, 
    DataArguments, 
    OLALMTrainingArguments as TrainingArguments,
    OLALMTrainer,
)
from data_utils import DataManager
from models.ola_model import OLAModel


def args_parser() -> Tuple[ModelArguments, DataArguments, TrainingArguments]:
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    json_args_dict = {}
    other_args = []
    # if the first argument is a JSON file, parse it
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        with open(os.path.abspath(sys.argv[1]), 'r') as f:
            json_args_dict = json.load(f)
        other_args = sys.argv[2:]
    else:
        other_args = sys.argv[1:]
    # convert json args to list
    json_args_list = []
    for key, value in json_args_dict.items():
        if value is None:
            continue
        json_args_list.append(f"--{key}")
        if isinstance(value, list):
            json_args_list.extend(map(str, value))
        else:
            json_args_list.append(str(value))
    # combine json args and other args, other args have higher priority
    combined_args = json_args_list + other_args
    # parse arguments
    model_args, data_args, training_args = (
        parser.parse_args_into_dataclasses(combined_args)
    )
    return model_args, data_args, training_args


def main():
    # parser arguments
    model_args, data_args, training_args = args_parser()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    # set random seed
    set_seed(training_args.seed)

    # create data manager
    data_manager = DataManager(
        dataset_name=data_args.dataset_name,
        cutoff_len=data_args.cutoff_len,
        train_model_name_or_paths=[model_args.model_name_or_path],
        test_model_name_or_paths=model_args.eval_models_name_list
    )

    # do train
    if training_args.do_train:
        # save arguments
        save_arguments([model_args, data_args, training_args], 
                    os.path.join(training_args.output_dir, "args.json"))

        # create OLAModel
        model = OLAModel(
            base_model_name_or_path=model_args.model_name_or_path,
            adapter_architecture=model_args.adapter_architecture,
            num_classes=data_args.num_classes,
            use_orders=model_args.use_orders,
            remove_outliers=model_args.remove_outliers,
            outliers_sigma_multiplier=model_args.outliers_sigma_multiplier,
            local_files_only=model_args.local_files_only
        )
        model.print_trainable_parameters()
        model = model.train().cuda()
        # load train dataset
        train_dataset, data_collator = data_manager.get_dataset_collator(
            model_args.model_name_or_path, "train"
        )
        trainer = OLALMTrainer(
            model=model,
            train_dataset=train_dataset,
            data_collator=data_collator,
            args=training_args,
        )
        trainer.train()
        # save last checkpoint
        last_ckpt_path = os.path.join(
            training_args.output_dir, "checkpoint-last", ADAPTERS_CKPT_NAME
        )
        os.makedirs(os.path.dirname(last_ckpt_path), exist_ok=True)
        model.save_adapter(last_ckpt_path)

    # do eval
    if training_args.do_eval:
        # load eval adapter checkpoint
        if model_args.eval_adapter_checkpoint is None:
            eval_adapter_checkpoint = os.path.join(
                training_args.output_dir, "checkpoint-last", ADAPTERS_CKPT_NAME
            )
        else:
            eval_adapter_checkpoint = model_args.eval_adapter_checkpoint
        # evaluate each model
        for eval_model_name in model_args.eval_models_name_list:
            print(f"Evaluating model {eval_model_name}")
            # load eval dataset
            eval_dataset, data_collator = data_manager.get_dataset_collator(
                eval_model_name, "test"
            )
            eval_dataloader = DataLoader(
                eval_dataset,
                collate_fn=data_collator,
                batch_size=training_args.per_device_eval_batch_size,
                shuffle=False,
            )
            # load eval metric
            if data_args.dataset_name.lower() == "imdb":
                eval_metric = TextClsMetric()
            elif data_args.dataset_name.lower() == "conll2000":
                eval_metric = TokenClsMetric(
                    label_names=eval_dataset.features["pos_tags"].feature.names,
                    tokenizer=data_manager.tokenizer_dict[eval_model_name],
                )
            else:
                raise NotImplemented
            # create OLAModel
            model = OLAModel(
                base_model_name_or_path=eval_model_name,
                adapter_architecture=model_args.adapter_architecture,
                num_classes=data_args.num_classes,
                use_orders=model_args.use_orders,
                remove_outliers=model_args.remove_outliers,
                outliers_sigma_multiplier=model_args.outliers_sigma_multiplier,
            )
            output_dir = os.path.join(
                os.path.dirname(eval_adapter_checkpoint),
                f"eval_{os.path.basename(eval_model_name)}"
            )
            # evaluate
            evaluate_ola_adapter(
                eval_dataloader=eval_dataloader,
                eval_metric=eval_metric,
                eval_ola_model=model,
                eval_adapter_ckpt=eval_adapter_checkpoint,
                output_dir=output_dir,
            )

    # do visualize
    if training_args.do_visualize:
        # save arguments
        save_arguments([model_args, data_args, training_args], 
                    os.path.join(training_args.output_dir, "args.json"))
        with open(data_args.visual_text_file, "r") as f:
            text_list = f.readlines()
        text_list = [text.rstrip('\n') for text in text_list]
        visualize_attn_map(
            model_args.visual_models_name_list,
            model_args.use_orders,
            text_list,
            training_args.output_dir,
            data_args.cutoff_len,
            model_args.outliers_sigma_multiplier,
            data_args.visual_annot_size,
            data_args.visual_label_size,
        )


if __name__ == "__main__":
    main()
