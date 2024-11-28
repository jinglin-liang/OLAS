import functools
from typing import List, Dict, Any

import datasets
from transformers import AutoTokenizer
from transformers.data.data_collator import (
    DataCollatorWithPadding, 
    DataCollatorForTokenClassification
)

from data_utils.data import load_raw_data


class PartPaddingDataCollator:
    def __init__(
        self,
        data_collator,
        keys_to_ignore: List[str] = None,
    ) -> None:
        self.data_collator = data_collator
        if keys_to_ignore is None:
            self.keys_to_ignore = [
                "text", "token_pos_tags", "token_chunk_tags", 
                "tokens", "pos_tags", "chunk_tags", "id"
            ]
        else:
            self.keys_to_ignore = keys_to_ignore

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_need_padding = [
            {k: v for k, v in feature.items() if k not in self.keys_to_ignore}
            for feature in features
        ]
        batch = self.data_collator(features_need_padding)
        for key in self.keys_to_ignore:
            if key in features[0].keys():
                batch[key] = [feature[key] for feature in features]
        return batch


class DataManager:
    def __init__(
        self, 
        dataset_name: str,
        cutoff_len: int,
        train_model_name_or_paths: List[str],
        test_model_name_or_paths: List[str],
    ) -> None:
        self.dataset_name = dataset_name
        self.cutoff_len = cutoff_len
        self.train_model_name_or_paths = train_model_name_or_paths
        self.test_model_name_or_paths = test_model_name_or_paths
        # init tokenizer
        self._init_tokenizers(
            train_model_name_or_paths + test_model_name_or_paths
        )
        # load data
        (train_data, test_data), standardize_function = \
            load_raw_data(dataset_name)
        train_data: datasets.Dataset
        test_data: datasets.Dataset
        self.data = {
            "train": {},
            "test": {},
        }
        kwargs = {}
        kwargs["cutoff_len"] = self.cutoff_len
        if dataset_name == "conll2000":
            kwargs["pos_tags_names"] = train_data.features["pos_tags"].feature.names
            kwargs["chunk_tags_names"] = train_data.features["chunk_tags"].feature.names
        for tmp_train_model in train_model_name_or_paths:
            kwargs["tokenizer"] = self.tokenizer_dict[tmp_train_model]
            preprocess_func = functools.partial(
                standardize_function,
                **kwargs
            )
            # self.data["train"][tmp_train_model] = \
            #     train_data.map(preprocess_func, load_from_cache_file=False)
            self.data["train"][tmp_train_model] = train_data.map(preprocess_func)
        
        for tmp_test_model in test_model_name_or_paths:
            kwargs["tokenizer"] = self.tokenizer_dict[tmp_test_model]
            preprocess_func = functools.partial(
                standardize_function,
                **kwargs
            )
            # self.data["test"][tmp_test_model] = \
            #     test_data.map(preprocess_func, load_from_cache_file=False)
            self.data["test"][tmp_test_model] = test_data.map(preprocess_func)

    def _init_tokenizers(self, all_model_name_or_paths: List[str]):
        self.tokenizer_dict = {}
        for tmp_model in set(all_model_name_or_paths):
            tokenizer = AutoTokenizer.from_pretrained(
                tmp_model, local_files_only=True)
            tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
            tokenizer.padding_side = "left"  # Allow batched inference
            self.tokenizer_dict[tmp_model] = tokenizer

    def get_dataset_collator(self, model_name_or_path: str, split: str):
        assert split in ["train", "test"]
        assert model_name_or_path in self.data[split].keys()

        if self.dataset_name == "conll2000":
            base_data_collator = DataCollatorForTokenClassification(
                tokenizer=self.tokenizer_dict[model_name_or_path],
                padding="longest",
                max_length=self.cutoff_len,
                pad_to_multiple_of=8
            )
        elif self.dataset_name == "imdb":
            base_data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer_dict[model_name_or_path],
                padding="longest",
                max_length=self.cutoff_len,
                pad_to_multiple_of=8
            )

        return self.data[split][model_name_or_path], PartPaddingDataCollator(base_data_collator)
