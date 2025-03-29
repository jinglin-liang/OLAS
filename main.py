from typing import Tuple
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json

import torch
from transformers import (
    set_seed,
    HfArgumentParser,
)

from utils import (
    ADAPTERS_CKPT_NAME,
    save_arguments,
    evaluate_ola_adapter_with_multi_llms,
    visualize_attn_map,
    visualize_layer_attn_map,
    ModelArguments, 
    DataArguments, 
    OLALMTrainingArguments as TrainingArguments,
    OLALMTrainer,
)
from data_utils import (
    generate_save_ola_data,
    calc_flop,
    get_oladata_dir_path,
    DataManager,
)
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

    if not training_args.do_visualize:
        # create data manager
        pad_to_multiple_of = data_args.pad_to_multiple_of if model_args.adapter_architecture != "tokencls_unet" else 32
        data_manager = DataManager(
            dataset_name=data_args.dataset_name,
            cutoff_len=data_args.cutoff_len,
            train_model_name_or_paths=model_args.train_models_name_list,
            test_model_name_or_paths=model_args.eval_models_name_list,
            use_generated_oladata=data_args.use_generated_oladata,
            attn_type=data_args.attn_type,
            pad_to_multiple_of=pad_to_multiple_of,
            do_classify_data_generate=data_args.do_classify_data_generate,
            classify_sentence_len=data_args.classify_sentence_len,
            classify_sentence_num=data_args.classify_sentence_num
        )

        if data_args.othertest_dataset_name != None:
            _data_manager = DataManager(
                dataset_name=data_args.othertest_dataset_name,
                cutoff_len=data_args.cutoff_len,
                train_model_name_or_paths=model_args.train_models_name_list,
                test_model_name_or_paths=model_args.eval_models_name_list,
                use_generated_oladata=data_args.use_generated_oladata,
                attn_type=data_args.attn_type
            )

    # do train
    if training_args.do_train:
        # save arguments
        save_arguments([model_args, data_args, training_args], 
                    os.path.join(training_args.output_dir, "args.json"))

        # create OLAModel
        model = OLAModel(
            base_model_name_list=model_args.train_models_name_list,
            adapter_architecture=model_args.adapter_architecture,
            num_classes=data_args.num_classes,
            use_orders=model_args.use_orders,
            remove_outliers=model_args.remove_outliers,
            outliers_sigma_multiplier=model_args.outliers_sigma_multiplier,
            local_files_only=model_args.local_files_only,
            abandom_base_lm=data_args.use_generated_oladata,
            ola_augments=model_args.ola_augments,
            attn_type=data_args.attn_type,
            **model_args.adapter_params
        )
        model.print_trainable_parameters()
        model = model.train().cuda()
        # load train dataset
        train_dataset, data_collator = data_manager.get_dataset_collator(
            model_args.train_models_name_list, "train", training_args.task
        )
        # prepare args for eval during checkpointing
        args_for_eval = [
            model_args.eval_models_name_list,
            data_manager,
            training_args.task,
            data_args.attn_type,
            data_args.use_generated_oladata
        ]
        # create trainer
        trainer = OLALMTrainer(
            *args_for_eval,
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
        # Explicitly delete the model and clear cache
        del model, trainer
        torch.cuda.empty_cache()

    # do eval
    if training_args.do_eval:
        # load eval adapter checkpoint
        if model_args.eval_adapter_checkpoint is None:
            eval_adapter_checkpoint = os.path.join(
                training_args.output_dir, "checkpoint-last", ADAPTERS_CKPT_NAME
            )
        else:
            eval_adapter_checkpoint = model_args.eval_adapter_checkpoint
        # load arguments
        eval_args = os.path.join(
            os.path.dirname(eval_adapter_checkpoint), 
            "..", "args.json"
        )
        with open(eval_args, "r") as f:
            eval_args = json.load(f)
        # set output dir
        output_dir = os.path.dirname(eval_adapter_checkpoint)
        # evaluate
        evaluate_ola_adapter_with_multi_llms(
            model_args.eval_models_name_list,
            eval_args,
            output_dir,
            data_manager,
            training_args.task,
            training_args.per_device_eval_batch_size,
            data_args.attn_type,
            data_args.use_generated_oladata,
            eval_adapter_checkpoint
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
            model_args.train_models_name_list,
            model_args.use_orders,
            text_list,
            training_args.output_dir,
            None,
            data_args.attn_type,
            data_args.cutoff_len,
            model_args.outliers_sigma_multiplier,
            data_args.visual_annot_size,
            data_args.visual_label_size,
            model_args.load_method
        )
        # visualize_layer_attn_map(
        #     model_args.train_models_name_list,
        #     model_args.use_orders,
        #     text_list,
        #     training_args.output_dir,
        #     model_args.ola_augments,
        #     model_args.adapter_hidden_size,
        #     model_args.num_layers,
        #     data_args.cutoff_len,
        #     model_args.outliers_sigma_multiplier,
        #     data_args.visual_annot_size,
        #     data_args.visual_label_size
        # )

    # do ola data generation
    if training_args.do_gen_save_ola:
        for model_name_or_path in model_args.eval_models_name_list:
            model = OLAModel(
                base_model_name_list=[model_name_or_path,],
                adapter_architecture="tokencls_axialtranformer",
                num_classes=data_args.num_classes,
                use_orders=model_args.use_orders,
                remove_outliers=True,
                outliers_sigma_multiplier=3,
                attn_type=data_args.attn_type,
                load_method=model_args.load_method,
            )
            splits = ["train", "test"] if not data_args.do_classify_data_generate else ["train"]
            for split in splits:
                gen_dataset, gen_data_collator = data_manager.get_dataset_collator(
                    [model_name_or_path], split
                )
                gen_data_collator.data_collator.pad_to_multiple_of = None
                save_dir = get_oladata_dir_path(data_args.dataset_name, model_name_or_path, split, data_args.attn_type, data_args.do_classify_data_generate, model_args.load_method, data_args.classify_sentence_len, data_args.classify_sentence_num)
                save_arguments([model_args, data_args, training_args], 
                                os.path.join(save_dir, "args.json"))
                generate_save_ola_data(
                    model,
                    gen_dataset,
                    gen_data_collator,
                    save_dir,
                    data_args.attn_type
                )
                # calc_flop(
                #     model,
                #     gen_dataset,
                #     gen_data_collator,
                #     data_args.attn_type
                # )
            # Explicitly delete the model and clear cache
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
