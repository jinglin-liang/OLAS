# from transformers import BertForMaskedLM, BertTokenizer
from transformers import AutoConfig, AutoTokenizer
import torch
import re
from tqdm import tqdm

from data_utils.data import (
    load_raw_data
)

# 预定义的依存关系标签列表
REL_LABELS = ['Cause-Effect(e1,e2)', 'Cause-Effect(e2,e1)', 'Component-Whole(e1,e2)', 'Component-Whole(e2,e1)', 'Content-Container(e1,e2)', 'Content-Container(e2,e1)', 'Entity-Destination(e1,e2)', 'Entity-Destination(e2,e1)', 'Entity-Origin(e1,e2)', 'Entity-Origin(e2,e1)', 'Instrument-Agency(e1,e2)', 'Instrument-Agency(e2,e1)', 'Member-Collection(e1,e2)', 'Member-Collection(e2,e1)', 'Message-Topic(e1,e2)', 'Message-Topic(e2,e1)', 'Product-Producer(e1,e2)', 'Product-Producer(e2,e1)', 'Other']
REL_LABELS_STR = '1. Cause-Effect(e1,e2)\n2. Cause-Effect(e2,e1)\n3. Component-Whole(e1,e2)\n4. Component-Whole(e2,e1)\n5. Content-Container(e1,e2)\n6. Content-Container(e2,e1)\n7. Entity-Destination(e1,e2)\n8. Entity-Destination(e2,e1)\n9. Entity-Origin(e1,e2)\n10. Entity-Origin(e2,e1)\n11. Instrument-Agency(e1,e2)\n12. Instrument-Agency(e2,e1)\n13. Member-Collection(e1,e2)\n14. Member-Collection(e2,e1)\n15. Message-Topic(e1,e2)\n16. Message-Topic(e2,e1)\n17. Product-Producer(e1,e2)\n18. Product-Producer(e2,e1)\n19. Other\n'

def is_causal_lm(model_type: str) -> bool:
    causal_models = {"gpt2", "opt", "llama", "qwen2", "gemma2", "bloom", "mistral"}
    non_causal_models = {"bert", "roberta", "albert", "deberta-v2", "electra"}
    if model_type.lower() in causal_models:
        return True
    elif model_type.lower() in non_causal_models:
        return False
    else:
        raise ValueError(f"Model type {model_type} is not tested.")

def build_english_prompt_mlm(sentence, e1, e2, mask_str):
    return f"""Act as a relation extraction tagging tool. Find the relationship between e1 and e2 in the given sentence by choosing the correct option number from {REL_LABELS_STR}.

    Sentence: {sentence}.
    e1: {e1}
    e2: {e2}
    Response: The relationship number is {mask_str}."""
    # return f"""What is the relationship between e1:"{e1}" and e2:"{e2}" in this sentence? Choose the correct option number.
    # Sentence: {sentence}
    # Options:
    # {REL_LABELS_STR}
    # Answer: {mask_str}."""

def build_english_prompt_clm(sentence, e1, e2):
    return f"""Act as a relation extraction tagging tool. Find the relationship between e1 and e2 in the given sentence according to these rules:
    1. Choose the correct option number from {REL_LABELS_STR}.
    2. Do not explain or add extra text. Only provide the option number.

    Sentence: {sentence}.
    e1: {e1}
    e2: {e2}
    Response:"""

def extract_number(s):
    """从预测文本中提取数字（处理如'4'或'##4'的情况）"""
    numbers = re.findall(r'\d+', s)
    return int(numbers[0]) if numbers else -1

def predict_dependency_mlm(model, tokenizer, sentence, error):
    """预测所有词的head和关系"""
    e1 = sentence.split('<e1>')[-1].split('</e1>')[0]
    e2 = sentence.split('<e2>')[-1].split('</e2>')[0]
    prompt = build_english_prompt_mlm(sentence, e1, e2, tokenizer.mask_token)
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
    rel_token = tokenizer.decode(torch.argmax(logits[mask_positions[0]]))
    pre_rel = extract_number(rel_token)
    
    return pre_rel, error

def predict_dependency_clm(model, tokenizer, sentence, error):
    """预测所有词的head和关系"""
    e1 = sentence.split('<e1>')[-1].split('</e1>')[0]
    e2 = sentence.split('<e2>')[-1].split('</e2>')[0]
    prompt = build_english_prompt_clm(sentence, e1, e2)
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
    (_, raw_test_data), _ = load_raw_data('semeval_re')
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
    model_name = models_name_list[8]
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
    total_sentence_num = 0
    error = 0
    bar = tqdm(raw_test_data, desc="zeroshoting")
    for data in bar:
        sentence, rel = data['sentence'], data['relation']
        total_sentence_num += 1

        if not if_causal:
            predictions, error = predict_dependency_mlm(model, tokenizer, sentence, error)  
            if predictions == rel:
                acc += 1
        else:
            predictions = predict_dependency_clm(model, tokenizer, sentence, error)
            tmp = predictions.split('Response:')[-1].lower().replace("e1", "").replace("e2", "")
            if tmp.find(str(rel)) != -1:
                if rel < 10:
                    if tmp.find(str(rel+10)) == -1:
                        acc += 1
                else:
                    acc += 1
        bar.set_postfix_str(f"error: {error}, ACC: {(acc * 100 / total_sentence_num):.4f}")
    print(model_name)
            