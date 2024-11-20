import functools
from typing import List, Dict, Any
from dataclasses import dataclass, field

import datasets
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorWithPadding

from data_utils.data import load_raw_data


@dataclass
class DataCollatorWithPartPadding(DataCollatorWithPadding):
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
        keys_to_ignore (`List[str]`, *optional*):
            List of keys in the input dictionary to ignore when padding. This is useful to avoid padding the text.
    """

    keys_to_ignore: List[str] = field(default_factory=lambda: ["text"])

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_need_padding = [
            {k: v for k, v in feature.items() if k not in self.keys_to_ignore}
            for feature in features
        ]
        batch = super().__call__(features_need_padding)
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
        (train_data, test_data), standardize_function = \
            load_raw_data(dataset_name)
        train_data: datasets.Dataset
        test_data: datasets.Dataset
        self.data = {
            "train": {},
            "test": {},
        }
        for tmp_train_model in train_model_name_or_paths:
            preprocess_func = functools.partial(
                self._tokenize_data_func,
                tokenizer=self.tokenizer_dict[tmp_train_model],
                standardize_function=standardize_function,
            )
            self.data["train"][tmp_train_model] = \
                train_data.map(preprocess_func)
        
        for tmp_test_model in test_model_name_or_paths:
            preprocess_func = functools.partial(
                self._tokenize_data_func,
                tokenizer=self.tokenizer_dict[tmp_test_model],
                standardize_function=standardize_function,
            )
            self.data["test"][tmp_test_model] = \
                test_data.map(preprocess_func)

    def _init_tokenizers(self, all_model_name_or_paths: List[str]):
        self.tokenizer_dict = {}
        for tmp_model in set(all_model_name_or_paths):
            tokenizer = AutoTokenizer.from_pretrained(
                tmp_model, local_files_only=True)
            tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
            tokenizer.padding_side = "left"  # Allow batched inference
            self.tokenizer_dict[tmp_model] = tokenizer

    def _tokenize_data_func(self, data_point, tokenizer, standardize_function):
        data_point = standardize_function(data_point)
        text = data_point["text"]
        tokenized_text = tokenizer(
            text,
            truncation=True,
            max_length=self.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": tokenized_text["input_ids"],
            "attention_mask": tokenized_text["attention_mask"],
            "label": data_point["label"],
        }

    def get_dataset_collator(self, model_name_or_path: str, split: str):
        assert split in ["train", "test"]
        assert model_name_or_path in self.data[split].keys()

        data_collator = DataCollatorWithPartPadding(
            tokenizer=self.tokenizer_dict[model_name_or_path],
            padding="longest",
            max_length=self.cutoff_len,
            pad_to_multiple_of=8
        )

        return self.data[split][model_name_or_path], data_collator
