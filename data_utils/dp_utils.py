from collections import Counter
import os
from tqdm import tqdm
import regex
import torch
from torch.utils.data import Dataset
# from transformers import *


class OLADataset_UDDP(Dataset):

    def __init__(self, items, n_buckets=1):
        super(OLADataset_UDDP, self).__init__()
        self.keys = ["input_ids", "attention_mask", "begin_mask", "heads", "dp_rels"]
        self.items = items

    def __getitem__(self, index):
        return {key: item[index] for key, item in zip(self.keys, self.items)}

    def __len__(self):
        return len(self.items[0])

    @property
    def lengths(self):
        return [len(i.nonzero()) for i in self.items[0]]


class UddpPreProcessor(object):
    PAD = '[PAD]'
    UNK = '[UNK]'
    BERT = '[BERT]'

    def __init__(self, tokenizer, words, tags, rels):

        # self.config = config
        # self.batchnorm = config.batchnorm_key or config.batchnorm_value
        # self.max_pad_length = config.max_seq_length
        
        self.words = [self.PAD, self.UNK] + sorted(words)
        
        self.tags = sorted(tags)
        self.tags = [self.PAD, self.UNK] + ['<t>:'+tag for tag in self.tags]
        
        self.rels = sorted(rels)
        self.rels = [self.PAD, self.UNK, self.BERT] + self.rels
        self.word_dict = {word: i for i,word in enumerate(self.words)}
        self.punct = [word for word, i in self.word_dict.items() if regex.match(r'\p{P}+$', word)]

        # print("Use Normal Bert model")
        # self.bertmodel = BertModel.from_pretrained(config.bert_path)

        # if config.use_japanese:
        #     self.tokenizer = BertJapaneseTokenizer.from_pretrained(config.bert_path)
        # else:
        #     self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)

        self.tokenizer = tokenizer
        # self.tokenizer.add_tokens(self.tags + ['<ROOT>']+ self.punct)
        
        # self.bertmodel.resize_token_embeddings(len(self.tokenizer))
        # Train our model
        # self.bertmodel.train()

        # if os.path.exists(config.modelpath + "/model_temp") != True:
        #     os.mkdir(config.modelpath + "/model_temp")
            
        # ### Now let's save our model and tokenizer to a directory
        # self.bertmodel.save_pretrained(config.modelpath + "/model_temp")

        self.tag_dict = {tag: i for i,tag in enumerate(self.tags)}
        
        self.rel_dict = {rel: i for i, rel in enumerate(self.rels)}
        
        self.root_label = self.rel_dict['<ROOT>']
        self.sbert_label = self.rel_dict['[BERT]']

        self.puncts = []
        for punct in self.punct:
            self.puncts.append(self.tokenizer.convert_tokens_to_ids(punct))
        
        self.pad_index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.PAD))[0]
        self.unk_index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.UNK))[0]
        
        self.n_words = len(self.words)
        self.n_tags = len(self.tags)
        self.n_rels = len(self.rels)
        self.n_train_words = self.n_words
        self.unk_count = 0
        self.total_count = 0
        self.long_seq = 0

    def __repr__(self):
        info = f"{self.__class__.__name__}: "
        info += f"{self.n_words} words, "
        info += f"{self.n_tags} tags, "
        info += f"{self.n_rels} rels"

        return info

    def map_arcs_bert_pred(self, corpus, predicted_corpus, training=True):

        all_words = []
        all_tags = []
        all_heads = []
        all_masks = []
        all_rels = []
        all_sbert_arc = []
        all_sbert_rel = []
        all_predicted_head = []
        all_predicted_rel = []

        for i, (words, tags, heads, rels, predicted_heads, predicted_rels) in enumerate(zip(corpus.words,
                                                                            corpus.tags, corpus.heads,corpus.rels,
                                                                            predicted_corpus.heads, predicted_corpus.rels)):

            old_to_new_node = {0: 0}
            tokens_org, tokens_length = self.word2id(words)
            tokens = [item for sublist in tokens_org for item in sublist]

            index = 0
            for token_id, token_length in enumerate(tokens_length):
                index += token_length
                old_to_new_node[token_id + 1] = index

            # CLS heads and tags
            new_heads = []
            new_predicted_heads = []
            new_tags = []
            new_subword_head = [0]

            offsets = torch.tensor(list(old_to_new_node.values()))[:-1] + 1

            for token_id, (offset, token_length) in enumerate(zip(offsets, tokens_length)):
                new_predicted_heads.append(old_to_new_node[predicted_heads[token_id]] + 1)
                new_heads.append(old_to_new_node[heads[token_id]] + 1)
                for sub_token in range(token_length):
                    new_tags.append(tags[token_id])

                    new_subword_head.append(int(offset))
            new_subword_head.append(0)

            new_subword_rel = [self.PAD] + len(tokens) * [self.BERT] + [self.PAD]
            ### add one for CLS #igonore CLS, ROOT, SEP

            words_id = torch.tensor(self.tokenizer.build_inputs_with_special_tokens(tokens))

            self.unk_count += len((words_id == 100).nonzero())
            self.total_count += len(words_id)
            
            tags = torch.tensor(self.tokenizer.build_inputs_with_special_tokens(self.tag2id(new_tags)))

            masks = torch.zeros(len(words_id)).long()
            masks[offsets[1:]] = 1
            rels = self.rel2id(rels[1:])
            rels_predicted = self.rel2id(predicted_rels[1:])

            if len(masks) < 512:
                all_words.append(words_id)
                all_tags.append(tags)
                all_masks.append(masks.bool())
                assert masks.sum() == len(new_heads[1:])
                all_heads.append(torch.tensor(new_heads[1:]))
                all_predicted_head.append(torch.tensor(new_predicted_heads[1:]))
                all_rels.append(rels)
                all_predicted_rel.append(rels_predicted)
                all_sbert_arc.append(torch.tensor(new_subword_head))
                all_sbert_rel.append(self.rel2id(new_subword_rel))
            else:
                self.long_seq += 1

        print("Percentage of unknown tokens in BERT:{}".format(self.unk_count * 1.0 / self.total_count * 100))
        self.unk_count = 0
        self.total_count = 0

        return all_words, all_tags, all_heads, all_rels, all_predicted_head, all_predicted_rel,\
                all_masks, all_sbert_arc, all_sbert_rel

    
    def map_arcs_bert(self, corpus, training=True):
        
        all_words = []
        all_tags = []
        all_heads = []
        all_masks = []
        all_rels = []
        all_sbert_arc = []
        all_sbert_rel = []

        if not training:
            all_offsets = []

        for i,(words,tags,heads,rels) in enumerate(zip(corpus.words, corpus.tags, corpus.heads, corpus.rels)):

            old_to_new_node = {0:0}
            tokens_org, tokens_length = self.word2id(words)
            tokens = [item for sublist in tokens_org for item in sublist]
            index = 0
            for token_id, token_length in enumerate(tokens_length):
                index += token_length
                old_to_new_node[token_id+1] = index
            # CLS heads and tags
            new_heads = []
            # new_tags = []
            new_subword_head = [0]
            
            offsets = torch.tensor(list(old_to_new_node.values() ))[:-1] + 1
            for token_id, (offset, token_length) in enumerate(zip(offsets, tokens_length)):
                new_heads.append(old_to_new_node[heads[token_id]]+1)
                for sub_token in range(token_length):
                    # new_tags.append(tags[token_id])
                    
                    new_subword_head.append(int(offset))
            new_subword_head.append(0)
            
            new_subword_rel = [self.PAD] + len(tokens) * [self.BERT] + [self.PAD]

            words_id = torch.tensor(self.tokenizer.build_inputs_with_special_tokens(tokens))

            self.unk_count += len( (words_id==100).nonzero() )
            self.total_count += len(words_id)

            # tags = torch.tensor(self.tokenizer.build_inputs_with_special_tokens(self.tag2id(new_tags) ))
            masks = torch.zeros(len(words_id)).long()
            masks[offsets[1:]] = 1
            rels = self.rel2id(rels[1:])

            if len(masks) < 512:
                all_words.append(words_id)
                # all_tags.append(tags)
                all_masks.append(masks.bool())
                assert masks.sum() == len(new_heads[1:])
                all_heads.append(torch.tensor(new_heads[1:]))
                all_rels.append(rels)
                all_sbert_arc.append(torch.tensor(new_subword_head))
                all_sbert_rel.append(self.rel2id(new_subword_rel))
                if not training:
                    assert len(offsets) == len(rels)+1
                    all_offsets.append(offsets[1:])
            else:
                self.long_seq += 1

        print("Percentage of unknown tokens in BERT:{}".format(self.unk_count * 1.0 / self.total_count * 100))
        self.unk_count = 0
        self.total_count = 0

        if not training:
            return all_words,all_heads,all_rels,all_masks,all_sbert_arc,all_sbert_rel,all_offsets
        else:
            return all_words,all_heads,all_rels,all_masks,all_sbert_arc,all_sbert_rel

    def map_ola_data(self, corpus, tokenizer, cutoff_len):
        all_input_ids = []
        all_attention_masks = []
        all_heads = []
        all_begin_masks = []
        all_rels = []
        # check if the tokenizer has bos_token or eos_token
        test_tokeizer = tokenizer("test")
        all_special_tokens = [v for _, v in tokenizer.special_tokens_map.items()]
        need_bos_token = tokenizer.decode(test_tokeizer["input_ids"][0]) in all_special_tokens
        need_eos_token = tokenizer.decode(test_tokeizer["input_ids"][-1]) in all_special_tokens
        bos_token_id = test_tokeizer["input_ids"][0] if need_bos_token else None
        eos_token_id = test_tokeizer["input_ids"][-1] if need_eos_token else None
        # process each sentence
        pbar = tqdm(enumerate(zip(corpus.words, corpus.heads, corpus.rels)), desc="Tokenizing UD data")
        for sentence_idx, (words, heads, rels) in pbar:
            sentence_input_ids = []
            sentence_attention_mask = []
            sentence_heads = []
            sentence_rels = []
            old_to_new_node = {}
            if need_bos_token:
                sentence_input_ids.append(bos_token_id)
                sentence_attention_mask.append(1)
            for tmp_word_idx, (tmp_word, tmp_head, tmp_rel) in enumerate(zip(words, heads, rels)):
                if tmp_word == "<ROOT>":
                    continue
                old_to_new_node[tmp_word_idx] = len(sentence_input_ids)
                # add " " to tmp_word to avoid LM such as GPT2 to ignore the space
                if tmp_word_idx > 1: # ignore <ROOT>
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
                word_input_id = tokenized_word["input_ids"][start_idx:end_idx]
                word_attention_mask = tokenized_word["attention_mask"][start_idx:end_idx]
                sentence_input_ids += word_input_id
                sentence_attention_mask += word_attention_mask
                if tmp_head == 0:
                    assert tmp_rel == "root"
                    sentence_heads.append(tmp_word_idx)
                else:
                    sentence_heads.append(tmp_head)
                sentence_rels.append(tmp_rel)
                if len(sentence_input_ids) + int(need_bos_token) >= cutoff_len:
                    end_idx = cutoff_len - int(need_bos_token)
                    sentence_input_ids = sentence_input_ids[:end_idx]
                    sentence_attention_mask = sentence_attention_mask[:end_idx]
                    break
            if need_eos_token:
                sentence_input_ids.append(eos_token_id)
                sentence_attention_mask.append(1)
            sentence_heads = torch.tensor([old_to_new_node.get(head, -100) for head in sentence_heads])
            sentence_begin_mask = torch.zeros(len(sentence_input_ids))
            sentence_begin_mask[list(old_to_new_node.values())] = 1
            # sentence_begin_mask[]
            assert torch.Tensor(sentence_heads).max() < len(sentence_input_ids)
            assert sentence_begin_mask.sum() == len(sentence_heads)
            all_input_ids.append(torch.tensor(sentence_input_ids))
            all_attention_masks.append(torch.tensor(sentence_attention_mask))
            all_heads.append(sentence_heads)
            all_begin_masks.append(sentence_begin_mask.bool())
            all_rels.append(self.rel2id(sentence_rels))
        # return all sentences
        return all_input_ids, all_attention_masks, all_begin_masks, all_heads, all_rels
        

    def word2id(self, sequence):
        WORD2ID = []
        lengths = []
        for word in sequence:
            x = self.tokenizer.tokenize(word)
            if len(x) == 0:
                x = ['[UNK]']
            x = self.tokenizer.convert_tokens_to_ids(x)
            lengths.append(len(x))
            WORD2ID.append(x)
        return WORD2ID,lengths
    
    def tag2id(self, sequence):
        
        tags = []
        for tag in sequence:
            tokenized_tag = self.tokenizer.tokenize('<t>:'+tag)
            if len(tokenized_tag) != 1:
                tags.append(self.unk_index)
            else:
                tags.append(self.tokenizer.convert_tokens_to_ids(tokenized_tag)[0])
        return tags

    def rel2id(self, sequence):
        return torch.tensor([self.rel_dict.get(rel, 1)
                             for rel in sequence])

    def id2rel(self, ids):
        return [self.rels[i] for i in ids]

    def extend(self, words):
        self.words.extend(sorted(set(words).difference(self.word_dict)))
        self.word_dict = {word: i for i, word in enumerate(self.words)}
        self.puncts = sorted(i for word, i in self.word_dict.items()
                             if regex.match(r'\p{P}+$', word))
        self.n_words = len(self.words)


    def numericalize(self, corpus, tokenizer, cutoff_len, predicted_corpus = None, training=True, **kwargs):
        return self.map_ola_data(corpus, tokenizer, cutoff_len)
        # if predicted_corpus is None:
        #     return self.map_arcs_bert(corpus,training)
        # else:
        #     return self.map_arcs_bert_pred(corpus,predicted_corpus,training)

    @classmethod
    def from_corpus(cls, corpus, tokenizer, min_freq=1, **kwargs):
        words = Counter(word for seq in corpus.words for word in seq)
        words = list(word for word, freq in words.items() if freq >= min_freq)
        tags = list({tag for seq in corpus.tags for tag in seq})
        rels = list({rel for seq in corpus.rels for rel in seq})
        vocab = cls(tokenizer, words, tags, rels)

        return vocab