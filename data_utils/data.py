from typing import Dict, Tuple, List, Any, Callable
from collections import namedtuple
from datasets import load_dataset

from conllu import parse_incr




DATASET_NAME_TO_PATH = {
    "imdb": "datasets/imdb",
    "conll2000_pos": "datasets/conll2000/conll2000.py",
    "conll2000_chunk": "datasets/conll2000/conll2000.py",
    "conll2012cn_pos": "datasets/conll2012/conll2012_ontonotesv5.py",
    "conll2012en_pos": "datasets/conll2012/conll2012_ontonotesv5.py",
    "conll2012cn_entity": "datasets/conll2012/conll2012_ontonotesv5.py",
    "conll2012en_entity": "datasets/conll2012/conll2012_ontonotesv5.py",
    "semeval_re": "datasets/sem_eval_2010_task_8/data",
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


def process_multiword_tokens(annotation):
    """
    Processes CoNLLU annotations for multi-word tokens.
    If the token id returned by the conllu library is a tuple object (either a multi-word token or an elided token),
    then the token id is set to None so that the token won't be used later on by the model.
    """
    
    for i in range(len(annotation)):
        conllu_id = annotation[i]["id"]
        if type(conllu_id) == tuple:
            if "-" in conllu_id:
                conllu_id = str(conllu_id[0]) + "-" + str(conllu_id[2])
                annotation[i]["multi_id"] = conllu_id
                annotation[i]["id"] = None
            elif "." in conllu_id:
                annotation[i]["id"] = None
                annotation[i]["multi_id"] = None
        else:
            annotation[i]["multi_id"] = None
    
    return annotation


UD_Sentence = namedtuple(typename='UD_Sentence',
                      field_names=['ids', 'words', 'lemmas', 'upos_tags',
                                   'xpos_tags', 'feats', 'heads', 'rels',
                                   'multiword_ids', 'multiword_forms'],
                      defaults=[None]*10)


class UniversalDependenciesDatasetReader(object):
    def __init__(self):
        self.sentences = []
        self.ids = []
        self.ROOT = "<ROOT>"

    def load(self, file_path):
        counter = 1
        with open(file_path, 'r') as conllu_file:

            for annotation in parse_incr(conllu_file):
                # CoNLLU annotations sometimes add back in words that have been elided
                # in the original sentence; we remove these, as we're just predicting
                # dependencies for the original sentence.
                # We filter by None here as elided words have a non-integer word id,
                # and we replace these word ids with None in process_multiword_tokens.
                annotation = process_multiword_tokens(annotation)               
                multiword_tokens = [x for x in annotation if x["multi_id"] is not None]     
                annotation = [x for x in annotation if x["id"] is not None]

                if len(annotation) == 0:
                    continue

                def get_field(tag: str, map_fn: Callable[[Any], Any] = None) -> List[Any]:
                    map_fn = map_fn if map_fn is not None else lambda x: x
                    return [map_fn(x[tag]) if x[tag] is not None else "_" for x in annotation if tag in x]

                # Extract multiword token rows (not used for prediction, purely for evaluation)
                ids = [x["id"] for x in annotation]
                multiword_ids = [x["multi_id"] for x in multiword_tokens]
                multiword_forms = [x["form"] for x in multiword_tokens]

                words = get_field("form")
                lemmas = get_field("lemma")
                # upos_tags = get_field("upostag")
                # xpos_tags = get_field("xpostag")
                upos_tags = get_field("upos")
                xpos_tags = get_field("xpos")
                feats = get_field("feats", lambda x: "|".join(k + "=" + v for k, v in x.items())
                                                     if hasattr(x, "items") else "_")
                heads = get_field("head")
                dep_rels = get_field("deprel")
                sentence = UD_Sentence(ids,words,lemmas,upos_tags,xpos_tags,feats,heads, dep_rels,multiword_ids,multiword_forms)
                self.sentences.append(sentence)
                self.ids.append(counter)
                counter = counter + 1

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        return self.sentences[item]


    @property
    def words(self):
        return [[self.ROOT] + list(sentence.words) for sentence in self.sentences]

    @property
    def tags(self):
        return [[self.ROOT] + list(sentence.upos_tags) for sentence in self.sentences]

    @property
    def heads(self):
        return [[0] + list(map(int, sentence.heads)) for sentence in self.sentences]

    @property
    def rels(self):
        return [[self.ROOT] + list(sentence.rels) for sentence in self.sentences]

    @heads.setter
    def heads(self, sequences):
        self.sentences = [sentence._replace(heads=sequence)
                          for sentence, sequence in zip(self.sentences, sequences)]

    @rels.setter
    def rels(self, sequences):
        self.sentences = [sentence._replace(rels=sequence)
                          for sentence, sequence in zip(self.sentences, sequences)]

    def save(self, fname):
        with open(fname, 'w') as f:
            for item in self.sentences:
                output = self.write(item)
                f.write(output)

    def write(self, outputs) :
        outputs = dict(outputs._asdict())
        word_count = len([word for word in outputs["words"]])
        lines = zip(*[outputs[k] if k in outputs else ["_"] * word_count
                      for k in ["ids", "words", "lemmas", "upos_tags", "xpos_tags", "feats",
                                "heads", "rels"]])

        multiword_map = None
        if outputs["multiword_ids"]:
            multiword_ids = [[id] + [int(x) for x in id.split("-")] for id in outputs["multiword_ids"]]
            multiword_forms = outputs["multiword_forms"]
            multiword_map = {start: (id_, form) for (id_, start, end), form in zip(multiword_ids, multiword_forms)}

        output_lines = []
        for i, line in enumerate(lines):
            line = [str(l) for l in line]

            # Handle multiword tokens
            if multiword_map and i+1 in multiword_map:
                id_, form = multiword_map[i+1]
                row = f"{id_}\t{form}" + "".join(["\t_"] * 8)
                output_lines.append(row)

            row = "\t".join(line) + "".join(["\t_"] * 2)
            output_lines.append(row)

        return "\n".join(output_lines) + "\n\n"


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
    elif data_name.lower() in ["conll2012cn_pos", "conll2012cn_entity"]:  # conll2012 pos tagging
        data = load_dataset(DATASET_NAME_TO_PATH[data_name], 'chinese_v4', trust_remote_code=True)
        train_data = data["train"]
        test_data = data["test"]
        return (train_data, test_data), None
    elif data_name.lower() in ["conll2012en_pos", "conll2012en_entity"]:  # conll2012 pos tagging
        data = load_dataset(DATASET_NAME_TO_PATH[data_name], 'english_v4', trust_remote_code=True)
        train_data = data["train"]
        test_data = data["test"]
        return (train_data, test_data), None
    elif data_name.lower() == "semeval_re":  # semeval relation extraction
        data = load_dataset(DATASET_NAME_TO_PATH[data_name])
        train_data = data["train"]
        test_data = data["test"]
        return (train_data, test_data), None
    elif data_name.lower() == "ud_english_gum":
        train_data = UniversalDependenciesDatasetReader()
        train_data.load("datasets/UD_English-GUM/en_gum-ud-train.conllu")
        test_data = UniversalDependenciesDatasetReader()
        test_data.load("datasets/UD_English-GUM/en_gum-ud-test.conllu")
        return (train_data, test_data), None
    elif data_name.lower() == "ud_english_ewt":
        train_data = UniversalDependenciesDatasetReader()
        train_data.load("datasets/UD_English-EWT/en_ewt-ud-train.conllu")
        test_data = UniversalDependenciesDatasetReader()
        test_data.load("datasets/UD_English-EWT/en_ewt-ud-test.conllu")
        return (train_data, test_data), None
    else:
        raise NotImplementedError(f"Dataset {data_name} is not supported.")
