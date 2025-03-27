import os
import inspect
import lmdb
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import data_utils
from data_utils.data import DATASET_NAME_TO_PATH


def get_oladata_dir_path(dataset_name, model_name_or_path, split, attn_type, do_classify_data_generate=False, load_method="origin"):
    if os.path.isfile(DATASET_NAME_TO_PATH[dataset_name]):
        data_root_dir = os.path.dirname(DATASET_NAME_TO_PATH[dataset_name])
    elif os.path.isdir(DATASET_NAME_TO_PATH[dataset_name]):
        data_root_dir = DATASET_NAME_TO_PATH[dataset_name]
    else:
        raise ValueError("Invalid dataset path")
    data_root_dir = data_root_dir + "_" + attn_type
    if dataset_name == "conll2000_pos":
        data_root_dir = data_root_dir + "_pos"
    elif dataset_name == "conll2000_chunk":
        data_root_dir = data_root_dir + "_chunk"
    elif dataset_name == "conll2012cn_pos":
        data_root_dir = data_root_dir + "_cn_pos"
    elif dataset_name == "conll2012en_pos":
        data_root_dir = data_root_dir + "_en_pos"
    elif dataset_name == "conll2012cn_entity":
        data_root_dir = data_root_dir + "_cn_entity"
    elif dataset_name == "conll2012en_entity":
        data_root_dir = data_root_dir + "_en_entity"
    if do_classify_data_generate:
        data_root_dir = data_root_dir + "_classify"
    data_root_dir = data_root_dir + '_' + load_method
    save_dir = os.path.join(
        data_root_dir, 
        os.path.basename(model_name_or_path),
        split
    )
    return save_dir


def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def generate_save_ola_data(
    model,
    dataset,
    data_collator,
    save_dir: str,
    attn_type: str = "ola"
):
    # create dataloader
    dataloader = DataLoader(
        dataset,
        collate_fn=data_collator,
        batch_size=1,
        shuffle=False,
    )
    # set model to eval mode
    model = model.eval().cuda()
    # create lmdb
    os.makedirs(save_dir, exist_ok=True)
    env = lmdb.open(save_dir, map_size=1099511627776)
    # generate data
    interested_keys = inspect.signature(model.forward).parameters.keys()
    bar = tqdm(dataloader, desc="Generating OLA data")
    cnt = 0
    cache = {}
    for data in bar:
        if data["input_ids"].shape[-1] == 0:
            continue
        with torch.no_grad():
            input_dict = {k: v for k, v in data.items() if k in interested_keys}
            input_dict["output_attn"] = True
            input_dict["labels"] = None
            input_dict["attn_type"] = attn_type
            output = model(
                **input_dict
            )
        attn = output.order_level_attention
        tmp_data = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                # "input_ids", "attention_mask", "labels"
                tmp_data[k] = v[0].tolist()
            else:
                # "token_pos_tags", "token_chunk_tags"
                tmp_data[k] = v[0]
        tmp_data["ola"] = {k: v.cpu() for k, v in attn.items()}
        data_byte = pickle.dumps(tmp_data)
        data_id = str(cnt).encode("utf-8")
        cache[data_id] = data_byte
        cnt += 1
        if cnt % 400 == 0:
            write_cache(env, cache)
            cache = {}
        # if cnt == 20:
        #     break
    cache["num_samples".encode('utf-8')] = str(cnt).encode("utf-8")
    if isinstance(dataset.datasets[0], OLADataset_conll2012):
        features = {
            "pos_tags_names": dataset.datasets[0].pos_tags_names,
            "named_entities_names": dataset.datasets[0].named_entities_names
        }
        cache["features".encode('utf-8')] = pickle.dumps(features)
    elif hasattr(dataset.datasets[0], "features"):
        features = dataset.datasets[0].features
        cache["features".encode('utf-8')] = pickle.dumps(features)
    write_cache(env, cache)
    print('save {} samples to {}'.format(cnt, save_dir))
    env.close()


class OLADataset:
    def __init__(self, data_dir):
        self.env = lmdb.open(data_dir, max_readers=8, readonly=True, lock=False, readahead=True, meminit=True)
        with self.env.begin(write=False) as txn:
            self.num_samples = int(txn.get('num_samples'.encode('utf-8')).decode("utf-8"))
            self.features = pickle.loads(txn.get('features'.encode('utf-8')))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            data_id = str(idx).encode("utf-8")
            data_byte = txn.get(data_id)
            data = pickle.loads(data_byte)
        return data


class OLADataset_conll2012:
    def __init__(self, raw_train_data, tokenizer, cutoff_len, pos_tags_names, named_entities_names, language, task):
        self.pos_tags_names = pos_tags_names
        self.named_entities_names = named_entities_names
        self.data = []
        id = -1
        for document in raw_train_data:
            for sentence in document['sentences']:
                id += 1
                tokens = sentence['words']
                pos_tags = sentence["pos_tags"]
                named_entities_tags = sentence["named_entities"]
                assert len(tokens) == len(pos_tags) and \
                    len(tokens) ==  len(named_entities_tags), "The length of tokens and pos_tags should be the same."
                test_tokeizer = tokenizer("test")
                all_special_tokens = [v for _, v in tokenizer.special_tokens_map.items()]
                need_bos_token = tokenizer.decode(test_tokeizer["input_ids"][0]) in all_special_tokens
                need_eos_token = tokenizer.decode(test_tokeizer["input_ids"][-1]) in all_special_tokens
                bos_token_id = test_tokeizer["input_ids"][0] if need_bos_token else None
                eos_token_id = test_tokeizer["input_ids"][-1] if need_eos_token else None
                input_ids = []
                attention_mask = []
                ret_pos_tags = []
                ret_named_entities_tags = []
                if need_bos_token:
                    input_ids.append(bos_token_id)
                    attention_mask.append(1)
                    ret_pos_tags.append(len(pos_tags_names))
                    ret_named_entities_tags.append(named_entities_names.index("O"))
                for tmp_idx, (tmp_word, tmp_pos, tmp_named_entities) in enumerate(zip(tokens, pos_tags, named_entities_tags)):
                    # add " " to tmp_word to avoid LM such as GPT2 to ignore the space
                    if language == "en":
                        if tmp_idx > 0:
                            tmp_word = " " + tmp_word
                    if hasattr(tokenizer, "vocab_file") and tokenizer.vocab_file != None:
                        if "Yi-1.5" in tokenizer.vocab_file:
                            tmp_word = tmp_word.lstrip(" ")
                    tokenized_word = tokenizer(
                        tmp_word,
                        truncation=True,
                        max_length=cutoff_len,
                        padding=False,
                        return_tensors=None,
                    )
                    start_idx = int(need_bos_token)
                    end_idx = len(tokenized_word["input_ids"]) - int(need_eos_token)
                    tmp_input_id = tokenized_word["input_ids"][start_idx:end_idx]
                    tmp_attention_mask = tokenized_word["attention_mask"][start_idx:end_idx]
                    tmp_pos_tag = [tmp_pos for _ in range(len(tmp_input_id))]
                    tmp_named_entities_tag = [tmp_named_entities for _ in range(len(tmp_input_id))]
                    input_ids += tmp_input_id
                    attention_mask += tmp_attention_mask
                    ret_pos_tags += tmp_pos_tag
                    ret_named_entities_tags += tmp_named_entities_tag
                    if len(input_ids) + int(need_bos_token) >= cutoff_len:
                        end_idx = cutoff_len - int(need_bos_token)
                        input_ids = input_ids[:end_idx]
                        attention_mask = attention_mask[:end_idx]
                        ret_pos_tags = ret_pos_tags[:end_idx]
                        ret_named_entities_tags = ret_named_entities_tags[:end_idx]
                        break
                if need_eos_token:
                    input_ids.append(eos_token_id)
                    attention_mask.append(1)
                    ret_pos_tags.append(len(pos_tags_names))
                    ret_named_entities_tags.append(named_entities_names.index("O"))
                text = " ".join(tokens) if language == "en" else "".join(tokens)
                labels = ret_pos_tags if task == "pos" else ret_named_entities_tags
                if task == "pos" or (task == "entity" and sum(ret_named_entities_tags) != (named_entities_names.index("O") * len(ret_named_entities_tags))):
                    self.data.append({
                        "id": id,
                        "tokens": sentence['words'],
                        "pos_tags": sentence["pos_tags"],
                        "named_entities_tags": sentence["named_entities"],
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "token_pos_tags": ret_pos_tags,
                        "token_named_entities_tags": ret_named_entities_tags,
                        "text": text,
                        "labels": labels,
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class ClassifyDataset:
    def __init__(self, sentences, tokenizer, cutoff_len, pos_tags_names, named_entities_names, language, task):
        # self.pos_tags_names = pos_tags_names
        # self.named_entities_names = named_entities_names
        self.data = []
        id = -1
        for sentence in sentences:
            id += 1
            tokens = sentence
            # pos_tags = sentence["pos_tags"]
            # named_entities_tags = sentence["named_entities"]
            # assert len(tokens) == len(pos_tags) and \
            #     len(tokens) ==  len(named_entities_tags), "The length of tokens and pos_tags should be the same."
            test_tokeizer = tokenizer("test")
            all_special_tokens = [v for _, v in tokenizer.special_tokens_map.items()]
            need_bos_token = tokenizer.decode(test_tokeizer["input_ids"][0]) in all_special_tokens
            need_eos_token = tokenizer.decode(test_tokeizer["input_ids"][-1]) in all_special_tokens
            bos_token_id = test_tokeizer["input_ids"][0] if need_bos_token else None
            eos_token_id = test_tokeizer["input_ids"][-1] if need_eos_token else None
            input_ids = []
            attention_mask = []
            # ret_pos_tags = []
            # ret_named_entities_tags = []
            if need_bos_token:
                input_ids.append(bos_token_id)
                attention_mask.append(1)
                # ret_pos_tags.append(len(pos_tags_names))
                # ret_named_entities_tags.append(named_entities_names.index("O"))
            for tmp_idx, tmp_word in enumerate(tokens):
                # add " " to tmp_word to avoid LM such as GPT2 to ignore the space
                if language == "en":
                    if tmp_idx > 0:
                        tmp_word = " " + tmp_word
                if hasattr(tokenizer, "vocab_file") and "Yi-1.5" in tokenizer.vocab_file:
                    tmp_word = tmp_word.lstrip(" ")
                tokenized_word = tokenizer(
                    tmp_word,
                    truncation=True,
                    max_length=cutoff_len,
                    padding=False,
                    return_tensors=None,
                )
                start_idx = int(need_bos_token)
                end_idx = len(tokenized_word["input_ids"]) - int(need_eos_token)
                tmp_input_id = tokenized_word["input_ids"][start_idx:end_idx]
                tmp_attention_mask = tokenized_word["attention_mask"][start_idx:end_idx]
                # tmp_pos_tag = [tmp_pos for _ in range(len(tmp_input_id))]
                # tmp_named_entities_tag = [tmp_named_entities for _ in range(len(tmp_input_id))]
                input_ids += tmp_input_id
                attention_mask += tmp_attention_mask
                # ret_pos_tags += tmp_pos_tag
                # ret_named_entities_tags += tmp_named_entities_tag
                if len(input_ids) + int(need_bos_token) >= cutoff_len:
                    end_idx = cutoff_len - int(need_bos_token)
                    input_ids = input_ids[:end_idx]
                    attention_mask = attention_mask[:end_idx]
                    # ret_pos_tags = ret_pos_tags[:end_idx]
                    # ret_named_entities_tags = ret_named_entities_tags[:end_idx]
                    break
            if need_eos_token:
                input_ids.append(eos_token_id)
                attention_mask.append(1)
                # ret_pos_tags.append(len(pos_tags_names))
                # ret_named_entities_tags.append(named_entities_names.index("O"))
            text = " ".join(tokens) if language == "en" else "".join(tokens)
            # labels = ret_pos_tags if task == "pos" else ret_named_entities_tags
            # if task == "pos" or (task == "entity" and sum(ret_named_entities_tags) != (named_entities_names.index("O") * len(ret_named_entities_tags))):
            labels = input_ids
            self.data.append({
                "id": id,
                "tokens": sentence,
                # "pos_tags": sentence["pos_tags"],
                # "named_entities_tags": sentence["named_entities"],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                # "token_pos_tags": ret_pos_tags,
                # "token_named_entities_tags": ret_named_entities_tags,
                "text": text,
                "labels": labels,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]