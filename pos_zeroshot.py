# from transformers import BertForMaskedLM, BertTokenizer
from transformers import AutoConfig, AutoTokenizer
import torch
import re
from tqdm import tqdm

from data_utils.data import (
    load_raw_data
)

# 预定义的依存关系标签列表
POS_LABELS = ["''", '#', '$', '(', ')', ',', '.', ':', '``', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
POS_LABELS_STR = "0: ''\n1: #\n2: $\n3: (\n4: )\n5: ,\n6: .\n7: :\n8: ``\n9: CC\n10: CD\n11: DT\n12: EX\n13: FW\n14: IN\n15: JJ\n16: JJR\n17: JJS\n18: MD\n19: NN\n20: NNP\n21: NNPS\n22: NNS\n23: PDT\n24: POS\n25: PRP\n26: PRP$\n27: RB\n28: RBR\n29: RBS\n30: RP\n31: SYM\n32: TO\n33: UH\n34: VB\n35: VBD\n36: VBG\n37: VBN\n38: VBP\n39: VBZ\n40: WDT\n41: WP\n42: WP$\n43: WRB\n"

def is_causal_lm(model_type: str) -> bool:
    causal_models = {"gpt2", "opt", "llama", "qwen2", "gemma2", "bloom", "mistral"}
    non_causal_models = {"bert", "roberta", "albert", "deberta-v2", "electra"}
    if model_type.lower() in causal_models:
        return True
    elif model_type.lower() in non_causal_models:
        return False
    else:
        raise ValueError(f"Model type {model_type} is not tested.")

def build_english_prompt_mlm(sentence, word, mask_str):
    return f"""Act as a part-of-speech (POS) tagging tool. Find the POS tag number of the given word in the given sentence by choosing the correct option number from {POS_LABELS_STR}.

    Sentence: {sentence}.
    Word: {word}.
    Response: The POS tag number is {mask_str}."""

def build_english_prompt_clm(sentence, word):
    return f"""Act as a part-of-speech (POS) tagging tool. Find the POS tag of the given word in the given sentence according to these rules:
    1. Choose the correct option number from {POS_LABELS_STR}.
    2. Do not explain or add extra text. Only provide the option number.

    Sentence: {sentence}.
    Word: {word}
    Response:"""

def extract_number(s):
    """从预测文本中提取数字（处理如'4'或'##4'的情况）"""
    numbers = re.findall(r'\d+', s)
    return int(numbers[0]) if numbers else -1

def predict_dependency_mlm(model, tokenizer, sentence, word, error):
    prompt = build_english_prompt_mlm(sentence, word, tokenizer.mask_token)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    # 定位所有[MASK]位置
    mask_positions = [i for i, token_id in enumerate(inputs['input_ids'][0]) 
                      if token_id == tokenizer.mask_token_id]
    if len(mask_positions) != 1:
        error += 1
        return -1, error
    
    # 预测
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0]
    
    # 解析结果
    pos_token = tokenizer.decode(torch.argmax(logits[mask_positions[0]]))
    pre_pos = extract_number(pos_token)
    
    return pre_pos, error

def predict_dependency_clm(model, tokenizer, sentence, word, error):
    """预测所有词的head和关系"""
    prompt = build_english_prompt_clm(sentence, word)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to('cuda') for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"].cuda(),
            attention_mask=inputs["attention_mask"].cuda(),
            max_length=inputs["input_ids"].shape[-1]+5,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=0,
            use_cache=True,
        )
        output_sentence = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0]

    return output_sentence

if __name__ == "__main__":
    (_, raw_test_data), _ = load_raw_data('conll2000_pos')
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
    bar = tqdm(raw_test_data, desc="zeroshoting")
    for data in bar:
        tokens, pos_tags = data['tokens'], data['pos_tags']
        
        sentence = " ".join(tokens)
        for widx, word in enumerate(tokens):
            total_word_num += 1
            if not if_causal:
                predictions, error = predict_dependency_mlm(model, tokenizer, sentence, word, error)  
                if predictions == pos_tags[widx]:
                    acc += 1
            else:
                predictions = predict_dependency_clm(model, tokenizer, sentence, word, error)
                tmp = predictions.split('Response:')[-1].lower()
                predict = extract_number(tmp)
                if predict == pos_tags[widx]:
                    acc += 1
                # pos_str_len = 1 if pos_tags[widx] < 10 else 2
                # pos_p = tmp.find(str(pos_tags[widx]))
                # if pos_p != -1:
                #     if not tmp[max(pos_p-1, 0)].isdigit() and not tmp[min(pos_p+pos_str_len, len(tmp)-1)].isdigit():
                        # acc += 1
        bar.set_postfix_str(f"error: {error}, ACC: {(acc * 100 / total_word_num):.4f}")
    print(f"error: {error}, ACC: {(acc * 100 / total_word_num):.4f}")
    print(model_name)
            