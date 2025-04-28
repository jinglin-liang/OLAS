# from transformers import BertForMaskedLM, BertTokenizer
from transformers import AutoConfig, AutoTokenizer
import torch
import re
from tqdm import tqdm

from data_utils.data import (
    load_raw_data
)

# 预定义的依存关系标签列表
REL_LABELS = [
    '[PAD]', '[UNK]', '[BERT]', '<ROOT>', 'acl', 'acl:relcl', 'advcl', 'advcl:relcl',
    'advmod', 'amod', 'appos', 'aux', 'aux:pass', 'case', 'cc', 'cc:preconj', 'ccomp',
    'compound', 'compound:prt', 'conj', 'cop', 'csubj', 'csubj:outer', 'csubj:pass',
    'dep', 'det', 'det:predet', 'discourse', 'dislocated', 'expl', 'fixed', 'flat',
    'goeswith', 'iobj', 'list', 'mark', 'nmod', 'nmod:desc', 'nmod:poss', 'nmod:unmarked',
    'nsubj', 'nsubj:outer', 'nsubj:pass', 'nummod', 'obj', 'obl', 'obl:agent',
    'obl:unmarked', 'orphan', 'parataxis', 'punct', 'reparandum', 'root', 'vocative', 'xcomp'
]
REL_LABELS_STR = '0: [PAD]\n1: [UNK]\n2: [BERT]\n3: <ROOT>\n4: acl\n5: acl:relcl\n6: advcl\n7: advcl:relcl\n8: advmod\n9: amod\n10: appos\n11: aux\n12: aux:pass\n13: case\n14: cc\n15: cc:preconj\n16: ccomp\n17: compound\n18: compound:prt\n19: conj\n20: cop\n21: csubj\n22: csubj:outer\n23: csubj:pass\n24: dep\n25: det\n26: det:predet\n27: discourse\n28: dislocated\n29: expl\n30: fixed\n31: flat\n32: goeswith\n33: iobj\n34: list\n35: mark\n36: nmod\n37: nmod:desc\n38: nmod:poss\n39: nmod:unmarked\n40: nsubj\n41: nsubj:outer\n42: nsubj:pass\n43: nummod\n44: obj\n45: obl\n46: obl:agent\n47: obl:unmarked\n48: orphan\n49: parataxis\n50: punct\n51: reparandum\n52: root\n53: vocative\n54: xcomp\n'

def is_causal_lm(model_type: str) -> bool:
    causal_models = {"gpt2", "opt", "llama", "qwen2", "gemma2", "bloom", "mistral"}
    non_causal_models = {"bert", "roberta", "albert", "deberta-v2", "electra"}
    if model_type.lower() in causal_models:
        return True
    elif model_type.lower() in non_causal_models:
        return False
    else:
        raise ValueError(f"Model type {model_type} is not tested.")

def build_english_prompt_mlm(words, word, mask_str):
    # return f"""Analyze the grammatical structure of this sentence:
    # Sentence: {" ".join(words)}

    # For the word '{word}':
    # 1. Head must be its parent word's position index (integer between 1-{len(words)}) or 0.
    # 2. Dependency relation must be one of: {REL_LABELS}.

    # {word} → head: {mask_str}, rel: {mask_str}."""
    tmp = ''
    for idx, w in enumerate(words):
        tmp += f'{idx}: {w}\n'
    return f"""Act as a dependency relation analyzing tool. Find the head and dependency relation of the given word in a sentence according to these rules:
    1. Choose the correct head number from {tmp}.
    2. Choose the correct dependency relation number from {REL_LABELS_STR}.

    Sentence: {" ".join(words)}
    Word: {word}
    Response: the head number is {mask_str}, the dependency relation number is {mask_str}."""

def build_english_prompt_clm(words, word):
    tmp = ''
    for idx, w in enumerate(words):
        tmp += f'{idx}: {w}\n'
    return f"""Act as a dependency relation analyzing tool. Find the head and dependency relation of the given word in a sentence according to these rules:
    1. Choose the correct head number from {tmp}.
    2. Choose the correct dependency relation number from {REL_LABELS_STR}.
    3. Do not explain or add extra text.

    Sentence: {" ".join(words)}
    Word: {word}
    Response as a tuple which has exactly two elements: first element is the head number (as a int), second element is the dependency relation number (as a int), e.g. (<head>, <relation>)
    Response: """

def extract_number(s):
    """从预测文本中提取数字（处理如'4'或'##4'的情况）"""
    numbers = re.findall(r'\d+', s)
    return int(numbers[0]) if numbers else -1

def match_rel_label(pred_token, rel_labels=REL_LABELS):
    """匹配最接近的关系标签（不区分大小写）"""
    pred = pred_token.strip().lower()
    for label in rel_labels:
        if pred == label.lower():
            return label
    return "[UNK]"

def predict_dependency_mlm(model, tokenizer, word, words, error):
    """预测所有词的head和关系"""
    prompt = build_english_prompt_mlm(words, word, tokenizer.mask_token)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    # 定位所有[MASK]位置
    mask_positions = [i for i, token_id in enumerate(inputs['input_ids'][0]) 
                      if token_id == tokenizer.mask_token_id]
    if len(mask_positions) != 2:
        error += 1
        return [(-1, "[UNK]")], error
    
    # 预测
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0]
    
    # 解析结果
    predictions = []
    for i in range(0, len(mask_positions), 2):
        # 预测head
        head_token = tokenizer.decode(torch.argmax(logits[mask_positions[i]]))
        head = extract_number(head_token)
        
        # 预测rel
        rel_token = tokenizer.decode(torch.argmax(logits[mask_positions[i+1]]))
        rel = extract_number(rel_token)
        
        predictions.append((head, rel))
    
    return predictions, error

def predict_dependency_clm(model, tokenizer, word, words, error):
    """预测所有词的head和关系"""
    prompt = build_english_prompt_clm(words, word)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to('cuda') for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"].cuda(),
            attention_mask=inputs["attention_mask"].cuda(),
            max_length=inputs["input_ids"].shape[-1]+20,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=0,
            use_cache=True,
        )
        output_sentence = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0]

    return output_sentence

if __name__ == "__main__":
    (_, raw_test_data), _ = load_raw_data('ud_english_ewt')
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
    model_name = models_name_list[11]
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

    uas = las = 0
    total_word_num = 0
    error = 0
    bar = tqdm(raw_test_data, desc="zeroshoting")
    for data in bar:
        words, heads, rels = data.words, data.heads, data.rels
        sentence = " ".join(words)
        for idx, word in enumerate(words):
            total_word_num += 1

            if not if_causal:
                predictions, error = predict_dependency_mlm(model, tokenizer, word, sentence, error)  
                if predictions[0][0] == heads[idx]:
                    uas += 1
                    if predictions[0][1] == rels[idx]:
                        las += 1
            else:
                predictions = predict_dependency_clm(model, tokenizer, word, words, error)
                tmp = predictions.split('Response:')[-1].lower()
                pre_h, pre_r = extract_number(tmp.split(',')[0]), extract_number(tmp.split(',')[-1])
                # print(predictions.split('Output:')[-1])
                if heads[idx] == pre_h:
                    uas += 1
                    if rels[idx] == pre_r:
                        las += 1
            # bar.set_postfix_str(f"{word} → head: {str(predictions[0][0]).ljust(2)} | rel: {predictions[0][1]}, UAS: {(uas * 100 / total_word_num):.4f}, LAS: {(las * 100 / total_word_num):.4f}")
        bar.set_postfix_str(f"error: {error}, UAS: {(uas * 100 / total_word_num):.4f}, LAS: {(las * 100 / total_word_num):.4f}")
    print(model_name)
            