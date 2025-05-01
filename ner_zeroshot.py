# from transformers import BertForMaskedLM, BertTokenizer
from transformers import AutoConfig, AutoTokenizer
import torch
import re, ast
from tqdm import tqdm

from data_utils.data import (
    load_raw_data
)

# 预定义的依存关系标签列表
NER_LABELS = ['O', 'B-PERSON', 'I-PERSON', 'B-NORP', 'I-NORP', 'B-FAC', 'I-FAC', 'B-ORG', 'I-ORG', 'B-GPE', 'I-GPE', 'B-LOC', 'I-LOC', 'B-PRODUCT', 'I-PRODUCT', 'B-DATE', 'I-DATE', 'B-TIME', 'I-TIME', 'B-PERCENT', 'I-PERCENT', 'B-MONEY', 'I-MONEY', 'B-QUANTITY', 'I-QUANTITY', 'B-ORDINAL', 'I-ORDINAL', 'B-CARDINAL', 'I-CARDINAL', 'B-EVENT', 'I-EVENT', 'B-WORK_OF_ART', 'I-WORK_OF_ART', 'B-LAW', 'I-LAW', 'B-LANGUAGE', 'I-LANGUAGE']
# NER_LABELS_STR = '0: O\n1: B-PERSON\n2: I-PERSON\n3: B-NORP\n4: I-NORP\n5: B-FAC\n6: I-FAC\n7: B-ORG\n8: I-ORG\n9: B-GPE\n10: I-GPE\n11: B-LOC\n12: I-LOC\n13: B-PRODUCT\n14: I-PRODUCT\n15: B-DATE\n16: I-DATE\n17: B-TIME\n18: I-TIME\n19: B-PERCENT\n20: I-PERCENT\n21: B-MONEY\n22: I-MONEY\n23: B-QUANTITY\n24: I-QUANTITY\n25: B-ORDINAL\n26: I-ORDINAL\n27: B-CARDINAL\n28: I-CARDINAL\n29: B-EVENT\n30: I-EVENT\n31: B-WORK_OF_ART\n32: I-WORK_OF_ART\n33: B-LAW\n34: I-LAW\n35: B-LANGUAGE\n36: I-LANGUAGE\n'
NER_LABELS_STR1 = '0: PERSON\n1: NORP\n2: FACILITY\n3: ORGANIZATION\n4: Geo-Political Entity\n5: LOCATION\n6: PRODUCT\n7: DATE\n8: TIME\n9: PERCENT\n10: MONEY\n11: QUANTITY\n12: ORDINAL\n13: CARDINAL\n14: EVENT\n15: WORK_OF_ART\n16: LAW\n17: LANGUAGE\n'
NER_LABELS_STR2 = '''[PERSON, NORP, FACILITY, ORGANIZATION, Geo-Political Entity, LOCATION, PRODUCT, DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL, EVENT, WORK_OF_ART, LAW, LANGUAGE]'''
NER_LABELS2 = ['PERSON', 'NORP', 'FACILITY', 'ORGANIZATION', 'Geo-Political Entity', 'LOCATION', 'PRODUCT', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']


def is_causal_lm(model_type: str) -> bool:
    causal_models = {"gpt2", "opt", "llama", "qwen2", "gemma2", "bloom", "mistral"}
    non_causal_models = {"bert", "roberta", "albert", "deberta-v2", "electra"}
    if model_type.lower() in causal_models:
        return True
    elif model_type.lower() in non_causal_models:
        return False
    else:
        raise ValueError(f"Model type {model_type} is not tested.")

def build_english_prompt_mlm(sentence, span, mask_str):
    return f"""Act as a named entity recognition tagging tool. Given the sentence: "{sentence}", determine whether the span "{span}" is a named entity. 
    If not a named entity, respond strictly with "none". 
    If it is a named entity, select the correct category from {NER_LABELS_STR1} and respond only with the corresponding number.

    Response: {mask_str}."""

def build_english_prompt_clm(sentence):
    return f"""Act as a named entity recognition tagging tool. Find all entities and their classes in a sentence according to these rules:
    1. Choose the correct named entity class from {NER_LABELS_STR2}.
    2. Do not explain or add extra text.

    Sentence: {sentence}.
    Response as tuples, and each tuple must have exactly two elements: first element is the named entity text (as a string), second element is the named entity class (as a string), e.g. (<entity1>, <class1>), (<entity2>, <class2>), ...
    Response: """

def extract_number(s):
    """从预测文本中提取数字（处理如'4'或'##4'的情况）"""
    numbers = re.findall(r'\d+', s)
    return int(numbers[0]) if numbers else 0

def extract_number2(s):
    """从预测文本中提取数字（处理如'4'或'##4'的情况）"""
    numbers = re.findall(r'\d+', s)
    return int(numbers[0]) if numbers else -1

def predict_dependency_mlm(model, tokenizer, sentence, span, error):
    prompt = build_english_prompt_mlm(sentence, span, tokenizer.mask_token)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    # 定位所有[MASK]位置
    mask_positions = [i for i, token_id in enumerate(inputs['input_ids'][0]) 
                      if token_id == tokenizer.mask_token_id]
    if len(mask_positions) != 1:
        error += 1
        return 0, error
    
    # 预测
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0]
    
    # 解析结果
    ner_token = tokenizer.decode(torch.argmax(logits[mask_positions[0]])).lower()
    if ner_token == 'none':
        pre_ner = None
    else:
        pre_ner = extract_number2(ner_token)
    
    return pre_ner, error

def predict_dependency_clm(model, tokenizer, sentence, error):
    """预测所有词的head和关系"""
    prompt = build_english_prompt_clm(sentence)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to('cuda') for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"].cuda(),
            attention_mask=inputs["attention_mask"].cuda(),
            max_length=inputs["input_ids"].shape[-1]+100,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=0,
            use_cache=True,
        )
        output_sentence = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0]

    return output_sentence

def find_targets(t):
    t = torch.tensor(t)
    entities = []
    t = t.view(-1)
    pos = torch.where(t % 2 != 0)[0]
    for p in pos:
        tmp_len = 0
        while (p + tmp_len + 1 < t.shape[0]) and (t[p + tmp_len + 1] == t[p] + 1):
            tmp_len += 1
        entities.append((t[p].item(), p.item(), tmp_len+1))
    return entities

def tag_sentence(output_str):
    # match = re.search(r'\[(.*?)\]', output_str).group(0)
    # tag_list = match.replace("'", "").replace("\n", "").strip()[1:-1].replace('), (', '),  (').split(',  ')
    output_str = output_str.replace("'", "").replace("\n", "").strip()
    tag_list = []
    stack = []
    for idx, char in enumerate(output_str):
        if char in ['{', '[', '(']:
            stack.append((char, idx))
        elif char in ['}', ']', ')']:
            if len(stack) != 0:
                tag_list.append(output_str[stack[-1][1]+1:idx])
            stack = []
    return tag_list

if __name__ == "__main__":
    (_, raw_test_data), _ = load_raw_data('conll2012en_entity')
    # tmp = "  ['today', 'date'], ['let', 'ordinal'], ['turn', 'work_of_art'], ['our', 'cardinal'], ['attention', 'event'], ['to', 'ordinal'], ['a', 'product'], ['road', 'norp'], ['cave', 'product'], ['in', 'ordinal'], ['accident', 'event'], ['that', 'ordinal'], ['happened', 'event'], ['in', 'location'], ['beijing"
    # tag_sentence(tmp)
    models_name_list= [
        "pretrained_models/bert-base-cased",
        "pretrained_models/bert-large-cased",
        "pretrained_models/roberta-base",
        "pretrained_models/roberta-large",
        "pretrained_models/electra-base-generator",
        "pretrained_models/electra-large-generator",
        "pretrained_models/Qwen2-1.5B-Instruct",
        "pretrained_models/Qwen2-7B-Instruct",
        "pretrained_models/gemma-2-2b-it",
        "pretrained_models/gemma-2-9b-it",
        "pretrained_models/Llama-3.2-3B-Instruct",
        "pretrained_models/Llama-3.1-8B-Instruct"
    ]
    model_name = models_name_list[6]
    print(model_name)
    

    config = AutoConfig.from_pretrained(
        model_name, 
        local_files_only=True
    )
    if_causal = is_causal_lm(config.model_type)
    model_class = getattr(__import__('transformers'), config.architectures[0])
    model = model_class.from_pretrained(
        model_name, config=config,
        local_files_only=True, 
        attn_implementation="eager"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    
    model = model.cuda()
    model.eval()

    acc = 0
    total_word_num = 0
    error = 0
    total_p, total_tp, total_fn, total_labels = 0, 0, 0, 0
    bar = tqdm(raw_test_data, desc="zeroshoting")
    if if_causal:
        for data in bar: 
            for sentence in data['sentences']:
                tokens, ner_tags = sentence['words'], sentence['named_entities']
                labels_entities_tuple = find_targets(ner_tags)
                if len(labels_entities_tuple) == 0:
                    continue
                labels_entities = []
                for tmp_le in labels_entities_tuple:
                    labels_entities.append(f"{' '.join(tokens[tmp_le[1]:tmp_le[1]+tmp_le[2]]).lower()}, {NER_LABELS2[(tmp_le[0]-1)//2].lower()}")
                
                sentence = " ".join(tokens)
                tmp = predict_dependency_clm(model, tokenizer, sentence, error).split('Response:')[-1].lower()
                preds_entities = tag_sentence(tmp)
                total_p += len(preds_entities)
                for p_en in preds_entities:
                    if p_en in labels_entities:
                        total_tp += 1
                precision = total_tp / total_p if total_p != 0 else 0
                for l_en in labels_entities:
                    if l_en not in preds_entities:
                        total_fn += 1
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) != 0 else 0
                f1 = 2 * precision * recall / (precision + recall + 1e-9)
            bar.set_postfix_str(f"error: {error}, F1: {f1:.4f}")
        print(f"error: {error}, F1: {f1:.4f}")
        print(model_name)
    else:
        for data in bar: 
            for sentence in data['sentences']:
                tokens, ner_tags = sentence['words'], sentence['named_entities']
                labels_entities = find_targets(ner_tags)
                if len(labels_entities) == 0:
                    continue
                total_labels += len(labels_entities)
                
                preds_entities = []
                sentence = " ".join(tokens)
                for begin_pos in range(len(tokens)):
                    for end_pos in range(begin_pos+1, len(tokens)+1):
                        tmp_words = " ".join(tokens[begin_pos:end_pos])
                        tmp, error = predict_dependency_mlm(model, tokenizer, sentence, tmp_words, error)
                        if tmp is not None and tmp != -1:
                            preds_entities.append((tmp, begin_pos, end_pos-begin_pos))
                for p_en in preds_entities:
                    if p_en in labels_entities:
                        total_tp += 1
                precision = total_tp / total_p if total_p != 0 else 0
                for l_en in labels_entities:
                    if l_en not in preds_entities:
                        total_fn += 1
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) != 0 else 0
                f1 = 2 * precision * recall / (precision + recall + 1e-9)
            bar.set_postfix_str(f"error: {error}, F1: {f1:.4f}")
        print(f"error: {error}, F1: {f1:.4f}")
        print(model_name)
            