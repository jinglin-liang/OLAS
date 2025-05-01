import functools
from typing import List, Dict, Any
import json

import torch
from torch.utils.data import ConcatDataset
import datasets
from transformers import AutoTokenizer
from transformers.data.data_collator import (
    DataCollatorWithPadding, 
    DataCollatorForTokenClassification
)

from data_utils.data import (
    load_raw_data
)
from data_utils.ola_dataset import (
    get_oladata_dir_path,
    OLADataset,
    OLADataset_conll2012,
    ClassifyDataset,
    OLADataset_SemEvalRe,
)
from data_utils.dp_utils import OLADataset_UDDP, UddpPreProcessor


class PartPaddingDataCollator:
    def __init__(
        self,
        data_collator,
        keys_to_ignore: List[str] = None,
        task: str = "pos"
    ) -> None:
        self.data_collator = data_collator
        self.task = task
        if keys_to_ignore is None:
            self.keys_to_ignore = [
                "text", "token_pos_tags", "token_chunk_tags", "token_named_entities_tags",
                "tokens", "pos_tags", "chunk_tags", "named_entities_tags", "id", "ola", "task",
                "e1_s", "e1_e", "e2_s", "e2_e", "relation", "begin_mask", "heads", "dp_rels"
            ]
        else:
            self.keys_to_ignore = keys_to_ignore

    def _padding_ola_tensor(self, ola_tensors: List[Dict[str, Any]], target_len: int):
        all_orders = list(ola_tensors[0].keys())
        ret_ola = {}
        for tmp_order in all_orders:
            ret_ola[tmp_order] = []
            for ola_tensor in ola_tensors:
                tmp_ola = torch.zeros(size=(1, 1, target_len, target_len))
                tmp_size = ola_tensor[tmp_order].shape[-1]
                if self.data_collator.tokenizer.padding_side == "left":
                    tmp_ola[:, :, -tmp_size:, -tmp_size:] = ola_tensor[tmp_order]
                else:
                    tmp_ola[:, :, :tmp_size, :tmp_size] = ola_tensor[tmp_order]
                ret_ola[tmp_order].append(tmp_ola)
            ret_ola[tmp_order] = torch.cat(ret_ola[tmp_order], dim=0)
        return ret_ola

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_need_padding = [
            {k: v for k, v in feature.items() if k not in self.keys_to_ignore}
            for feature in features
        ]
        batch = self.data_collator(features_need_padding)
        for key in self.keys_to_ignore:
            if key in features[0].keys() and key != "ola":
                batch[key] = [feature[key] for feature in features]
        # padding ola tensors
        if "ola" in features[0].keys():
            batch["ola"] = self._padding_ola_tensor(
                [feature["ola"] for feature in features], 
                batch["input_ids"].shape[1]
            )
        if "e1_s" in batch.keys():
            batch["e1_s"] = torch.LongTensor(batch["e1_s"])
            batch["e1_e"] = torch.LongTensor(batch["e1_e"])
            batch["e2_s"] = torch.LongTensor(batch["e2_s"])
            batch["e2_e"] = torch.LongTensor(batch["e2_e"])
        if "begin_mask" in batch.keys():
            tgt_len = batch["input_ids"].shape[1]
            begin_mask = torch.zeros((len(batch["begin_mask"]), tgt_len))
            for i, tmp_begin_mask in enumerate(batch["begin_mask"]):
                begin_mask[i, -len(tmp_begin_mask):] = torch.tensor(tmp_begin_mask)
            batch["begin_mask"] = begin_mask.bool()
        if "heads" in batch.keys():
            tgt_len = batch["input_ids"].shape[1]
            heads = torch.zeros((len(batch["heads"]), tgt_len)).fill_(-200)
            for i, tmp_heads in enumerate(batch["heads"]):
                tmp_heads = torch.tensor(tmp_heads)
                assert tmp_heads.max() < batch["attention_mask"][i].sum().item()
                adjust_heads = tmp_heads.clone()
                for j in range(len(adjust_heads)):
                    if adjust_heads[j] != -100 and adjust_heads[j] != -200:
                        adjust_heads[j]  += (1 - batch["attention_mask"][i]).sum().item()
                # adjust_heads[adjust_heads != -100] += (1 - batch["attention_mask"][i]).sum().item()
                heads[i, -len(adjust_heads):] = adjust_heads
            assert heads.max() < tgt_len
            batch["heads"] = heads.long()
        if "dp_rels" in batch.keys():
            tgt_len = batch["input_ids"].shape[1]
            dp_rels = torch.zeros((len(batch["dp_rels"]), tgt_len)).fill_(-200)
            for i, tmp_dp_rels in enumerate(batch["dp_rels"]):
                tmp_dp_rels = torch.tensor(tmp_dp_rels)
                dp_rels[i, -len(tmp_dp_rels):] = tmp_dp_rels
            batch["dp_rels"] = dp_rels.long()
        batch["task"] = self.task
        if "begin_mask" in batch.keys():
            assert (batch["begin_mask"].sum(-1) == heads.ne(-200).sum(-1)).all()
        return dict(batch)


class DataManager:
    def __init__(
        self, 
        dataset_name: str,
        cutoff_len: int,
        train_model_name_or_paths: List[str],
        test_model_name_or_paths: List[str],
        use_generated_oladata: bool = False,
        attn_type: str = "ola",
        pad_to_multiple_of: int = 8,
        do_classify_data_generate: bool = False,
        classify_sentence_len: int = 50,
        classify_sentence_num: int = 2000
    ) -> None:
        self.dataset_name = dataset_name
        self.cutoff_len = cutoff_len
        self.train_model_name_or_paths = train_model_name_or_paths
        self.test_model_name_or_paths = test_model_name_or_paths
        self.use_generated_oladata = use_generated_oladata
        self.pad_to_multiple_of = pad_to_multiple_of
        # init tokenizer
        self._init_tokenizers(
            train_model_name_or_paths + test_model_name_or_paths
        )
        # load data
        (raw_train_data, raw_test_data), standardize_function = \
            load_raw_data(dataset_name)
        raw_train_data: datasets.Dataset
        raw_test_data: datasets.Dataset
        self.data = {
            "train": {},
            "test": {},
        }

        do_cnt = False
        if do_cnt:
            cnt = 0
            for i in range(len(raw_train_data)):
                for j in range(len(raw_train_data[i]['sentences'])):
                    cnt += len(raw_train_data[i]['sentences'][j]['words'])
            for i in range(len(raw_test_data)):
                for j in range(len(raw_test_data[i]['sentences'])):
                    cnt += len(raw_test_data[i]['sentences'][j]['words'])
            cnt = cnt/84666


        kwargs = {}
        kwargs["cutoff_len"] = self.cutoff_len
        if dataset_name in ["conll2000_pos", "conll2000_chunk"]:
            kwargs["pos_tags_names"] = raw_train_data.features["pos_tags"].feature.names
            kwargs["chunk_tags_names"] = raw_train_data.features["chunk_tags"].feature.names
        if dataset_name in ["conll2012cn_pos", "conll2012cn_entity", "conll2012en_pos", "conll2012en_entity"]:
            kwargs["pos_tags_names"] = raw_train_data.features['sentences'][0]['pos_tags'].feature.names
            kwargs["named_entities_names"] = raw_train_data.features['sentences'][0]['named_entities'].feature.names
        if dataset_name == "semeval_re":
            kwargs["relation_names"] = raw_train_data.features["relation"].names
        
        if do_classify_data_generate:
            final_sentence_len = classify_sentence_len
            sentence_num = classify_sentence_num
            print(f"per sentence len={final_sentence_len}, sentence_num={sentence_num}")
            use_generated_classify_data = True
            assert dataset_name in ["conll2012en_entity", "imdb"]
            if use_generated_classify_data:
                if dataset_name == "conll2012en_entity":
                    file_name = "conll2012_classify_sentences"
                elif dataset_name == "imdb":
                    file_name = "imdb_classify_sentences"
                else:
                    raise NotImplementedError
                with open(f'datasets/{file_name}/classify_sentences_len{final_sentence_len}_num{sentence_num}.json', 'r') as f:
                    final_sentences = json.load(f)
                final_sentences = final_sentences[:sentence_num]
            else:
                final_sentences = []
                if dataset_name == "conll2012en_entity":
                    for document_id in range(raw_train_data.__len__()):
                        tmp_doc_words = []
                        for sentence_id in range(len(raw_train_data[document_id]['sentences'])):
                            tmp_doc_words.extend(raw_train_data[document_id]['sentences'][sentence_id]['words'])
                            if len(tmp_doc_words) >= final_sentence_len:
                                final_sentences.append(tmp_doc_words[:final_sentence_len])
                                tmp_doc_words = []
                            if len(final_sentences) >= sentence_num:
                                break
                    final_sentences = final_sentences[:sentence_num]
                    print("final sentence num =", len(final_sentences))
                    with open(f'datasets/conll2012_classify_sentences/classify_sentences_len{final_sentence_len}_num{sentence_num}.json', 'w') as f:
                        json.dump(final_sentences, f)
                elif dataset_name == 'imdb':
                    for data in raw_train_data:
                        tmp_doc_words = data['text'].split()
                        if len(tmp_doc_words) >= final_sentence_len and len(tmp_doc_words) < (final_sentence_len + 10):
                            final_sentences.append(tmp_doc_words)
                        elif len(tmp_doc_words) >= (final_sentence_len + 10):
                            final_sentences.append(tmp_doc_words[:final_sentence_len])
                        if len(final_sentences) >= sentence_num:
                            break
                        final_sentences = final_sentences[:sentence_num]
                    print("final sentence num =", len(final_sentences))
                    with open(f'datasets/imdb_classify_sentences/classify_sentences_len{final_sentence_len}_num{sentence_num}.json', 'w') as f:
                        json.dump(final_sentences, f)
                else:
                    raise NotImplementedError
                print("save final sentences done")

        for tmp_train_model in train_model_name_or_paths:
            if self.use_generated_oladata:
                data_dir_path = get_oladata_dir_path(
                    dataset_name, tmp_train_model, "train", attn_type, do_classify_data_generate
                )
                self.data["train"][tmp_train_model] = OLADataset(data_dir_path)
            elif do_classify_data_generate:
                kwargs["tokenizer"] = self.tokenizer_dict[tmp_train_model]
                kwargs["language"] = "en"
                kwargs["task"] = dataset_name.split("_")[-1]
                kwargs["cutoff_len"] = final_sentence_len + 100
                self.data["train"][tmp_train_model] = ClassifyDataset(final_sentences, **kwargs)
            elif dataset_name in ["conll2012cn_pos", "conll2012cn_entity"]:
                kwargs["tokenizer"] = self.tokenizer_dict[tmp_train_model]
                kwargs["language"] = "cn"
                kwargs["task"] = dataset_name.split("_")[-1]
                self.data["train"][tmp_train_model] = OLADataset_conll2012(raw_train_data, **kwargs)
            elif dataset_name in ["conll2012en_pos", "conll2012en_entity"]:
                kwargs["tokenizer"] = self.tokenizer_dict[tmp_train_model]
                kwargs["language"] = "en"
                kwargs["task"] = dataset_name.split("_")[-1]
                self.data["train"][tmp_train_model] = OLADataset_conll2012(raw_train_data, **kwargs)
            elif dataset_name == "semeval_re":
                kwargs["tokenizer"] = self.tokenizer_dict[tmp_train_model]
                self.data["train"][tmp_train_model] = OLADataset_SemEvalRe(raw_train_data, **kwargs)
            elif dataset_name in ["ud_english_gum", "ud_english_ewt"]:
                kwargs["min_freq"] = 2
                kwargs["tokenizer"] = self.tokenizer_dict[tmp_train_model]
                vocab = UddpPreProcessor.from_corpus(raw_train_data, **kwargs)
                self.data["train"][tmp_train_model] = OLADataset_UDDP(vocab.numericalize(raw_train_data, **kwargs))
                # self.data["train"][tmp_train_model] = OLADataset_UDDP(raw_train_data, **kwargs)
            else:
                kwargs["tokenizer"] = self.tokenizer_dict[tmp_train_model]
                preprocess_func = functools.partial(
                    standardize_function,
                    **kwargs
                )
                # self.data["train"][tmp_train_model] = \
                #     raw_train_data.map(preprocess_func, load_from_cache_file=False)
                self.data["train"][tmp_train_model] = raw_train_data.map(preprocess_func, load_from_cache_file=False)
        
        for tmp_test_model in test_model_name_or_paths:
            if self.use_generated_oladata and not do_classify_data_generate:
                data_dir_path = get_oladata_dir_path(
                    dataset_name, tmp_test_model, "test", attn_type
                )
                self.data["test"][tmp_test_model] = OLADataset(data_dir_path)
            elif do_classify_data_generate:
                self.data["test"][tmp_test_model] = None
            elif dataset_name in ["conll2012cn_pos","conll2012cn_entity"]:
                kwargs["tokenizer"] = self.tokenizer_dict[tmp_test_model]
                kwargs["language"] = "cn"
                kwargs["task"] = dataset_name.split("_")[-1]
                self.data["test"][tmp_test_model] = OLADataset_conll2012(raw_test_data, **kwargs)
            elif dataset_name in ["conll2012en_pos","conll2012en_entity"]:
                kwargs["tokenizer"] = self.tokenizer_dict[tmp_test_model]
                kwargs["language"] = "en"
                kwargs["task"] = dataset_name.split("_")[-1]
                self.data["test"][tmp_test_model] = OLADataset_conll2012(raw_test_data, **kwargs)
            elif dataset_name == "semeval_re":
                kwargs["tokenizer"] = self.tokenizer_dict[tmp_test_model]
                self.data["test"][tmp_test_model] = OLADataset_SemEvalRe(raw_test_data, **kwargs)
            elif dataset_name in ["ud_english_gum", "ud_english_ewt"]:
                kwargs["min_freq"] = 2
                kwargs["tokenizer"] = self.tokenizer_dict[tmp_test_model]
                vocab = UddpPreProcessor.from_corpus(raw_train_data, **kwargs)
                self.data["test"][tmp_test_model] = OLADataset_UDDP(vocab.numericalize(raw_test_data, **kwargs))
            else:
                kwargs["tokenizer"] = self.tokenizer_dict[tmp_test_model]
                preprocess_func = functools.partial(
                    standardize_function,
                    **kwargs
                )
                # self.data["test"][tmp_test_model] = \
                #     raw_test_data.map(preprocess_func, load_from_cache_file=False)
                self.data["test"][tmp_test_model] = raw_test_data.map(preprocess_func, load_from_cache_file=False)

    def _init_tokenizers(self, all_model_name_or_paths: List[str]):
        self.tokenizer_dict = {}
        for tmp_model in set(all_model_name_or_paths):
            tokenizer = AutoTokenizer.from_pretrained(
                tmp_model, local_files_only=True)
            tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
            tokenizer.padding_side = "left"  # Allow batched inference
            self.tokenizer_dict[tmp_model] = tokenizer

    def get_dataset_collator(self, model_name_list: List[str], split: str, task: str = "pos"):
        assert split in ["train", "test"]
        for model_name_or_path in model_name_list:
            assert model_name_or_path in self.data[split].keys()

        ret_dataset = ConcatDataset(
            [self.data[split][model_name_or_path] for model_name_or_path in model_name_list]
        )

        if self.dataset_name in ["conll2000_pos", "conll2000_chunk", "conll2012cn_pos", "conll2012en_pos", "conll2012cn_entity", "conll2012en_entity"]:
            base_data_collator = DataCollatorForTokenClassification(
                tokenizer=self.tokenizer_dict[model_name_list[0]],
                padding="longest",
                max_length=self.cutoff_len,
                pad_to_multiple_of=self.pad_to_multiple_of
            )
        elif self.dataset_name in ["imdb", "semeval_re", "ud_english_gum", "ud_english_ewt"]:
            base_data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer_dict[model_name_list[0]],
                padding="longest",
                max_length=self.cutoff_len,
                pad_to_multiple_of=self.pad_to_multiple_of
            )

        return ret_dataset, PartPaddingDataCollator(base_data_collator, task=task)
