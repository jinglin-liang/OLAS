from datasets import load_dataset


DATASET_NAME_TO_PATH = {
    "imdb": "datasets/imdb",
    "conll2000_pos": "datasets/conll2000/conll2000.py",
    "conll2000_chunk": "datasets/conll2000/conll2000.py",
}


def imdb_standardize_function(data_point, tokenizer, cutoff_len, **kwargs):
    text = data_point["text"]
    tokenized_text = tokenizer(
        text,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    return {
        "input_ids": tokenized_text["input_ids"],
        "attention_mask": tokenized_text["attention_mask"],
        "label": data_point["label"],
    }


def conll2000_standardize_function(data_point, tokenizer, cutoff_len, pos_tags_names, chunk_tags_names):
    tokens = data_point['tokens']
    pos_tags = data_point["pos_tags"]
    chunk_tags = data_point["chunk_tags"]
    assert len(tokens) == len(pos_tags) and \
        len(tokens) == len(chunk_tags), "The length of tokens, pos_tags, and chunk_tags should be the same."
    test_tokeizer = tokenizer("test")
    all_special_tokens = [v for _, v in tokenizer.special_tokens_map.items()]
    need_bos_token = tokenizer.decode(test_tokeizer["input_ids"][0]) in all_special_tokens
    need_eos_token = tokenizer.decode(test_tokeizer["input_ids"][-1]) in all_special_tokens
    bos_token_id = test_tokeizer["input_ids"][0] if need_bos_token else None
    eos_token_id = test_tokeizer["input_ids"][-1] if need_eos_token else None
    input_ids = []
    attention_mask = []
    ret_pos_tags = []
    ret_chunk_tags = []
    if need_bos_token:
        input_ids.append(bos_token_id)
        attention_mask.append(1)
        ret_pos_tags.append(len(pos_tags_names))
        ret_chunk_tags.append(chunk_tags_names.index("O"))
    for tmp_idx, (tmp_word, tmp_pos, tmp_chunk) in enumerate(zip(tokens, pos_tags, chunk_tags)):
        # add " " to tmp_word to avoid LM such as GPT2 to ignore the space
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
        tmp_pos_tag = [tmp_pos for _ in range(len(tmp_input_id))]
        tmp_chunk_tag = [tmp_chunk for _ in range(len(tmp_input_id))]
        input_ids += tmp_input_id
        attention_mask += tmp_attention_mask
        ret_pos_tags += tmp_pos_tag
        ret_chunk_tags += tmp_chunk_tag
        if len(input_ids) + int(need_bos_token) >= cutoff_len:
            end_idx = cutoff_len - int(need_bos_token)
            input_ids = input_ids[:end_idx]
            attention_mask = attention_mask[:end_idx]
            ret_pos_tags = ret_pos_tags[:end_idx]
            ret_chunk_tags = ret_chunk_tags[:end_idx]
            break
    if need_eos_token:
        input_ids.append(eos_token_id)
        attention_mask.append(1)
        ret_pos_tags.append(len(pos_tags_names))
        ret_chunk_tags.append(chunk_tags_names.index("O"))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_pos_tags": ret_pos_tags,
        "token_chunk_tags": ret_chunk_tags,
        "text": " ".join(tokens)
    }


def conll2000_pos_standardize_function(data_point, tokenizer, cutoff_len, pos_tags_names, chunk_tags_names):
    ret = conll2000_standardize_function(data_point, tokenizer, cutoff_len, pos_tags_names, chunk_tags_names)
    ret["labels"] = ret["token_pos_tags"]
    return ret


def conll2000_chunk_standardize_function(data_point, tokenizer, cutoff_len, pos_tags_names, chunk_tags_names):
    ret = conll2000_standardize_function(data_point, tokenizer, cutoff_len, pos_tags_names, chunk_tags_names)
    ret["labels"] = ret["token_chunk_tags"]
    return ret


def load_raw_data(data_name: str):
    '''
    return (train_data, test_data), standardize_function
    train_data and test_data are datasets.Dataset
    standardize_function is a function that takes a data point and returns a standardized data point {"text": str, "label": int}
    '''
    if data_name.lower() == "imdb":
        data = load_dataset(DATASET_NAME_TO_PATH[data_name])
        train_data = data["train"]
        test_data = data["test"]
        # unsupervised_data = data["unsupervised"]
        return (train_data, test_data), imdb_standardize_function
    elif data_name.lower() == "conll2000_pos":  # conll2000 pos tagging
        data = load_dataset(DATASET_NAME_TO_PATH[data_name], trust_remote_code=True)
        train_data = data["train"]
        test_data = data["test"]
        return (train_data, test_data), conll2000_pos_standardize_function
    elif data_name.lower() == "conll2000_chunk":  # conll2000 text chunking
        data = load_dataset(DATASET_NAME_TO_PATH[data_name], trust_remote_code=True)
        train_data = data["train"]
        test_data = data["test"]
        return (train_data, test_data), conll2000_chunk_standardize_function
    else:
        raise NotImplementedError(f"Dataset {data_name} is not supported.")
