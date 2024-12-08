# Order-level Attention Analysis and Application

## Visualize OLA

```bash
python main.py configs/visual_llms.json
```
key arguments: 
```bash
visual_text_file
train_models_name_list
cutoff_len
use_orders
visual_annot_size
visual_label_size
```

## Train OLA Adapter

1. Train from raw dataset, only support training adapter for one LLM

```bash
python main.py configs/train_qwen_1b_conll2000pos.json
```
key arguments: 
```bash
train_models_name_list
adapter_architecture
dataset_name
cutoff_len
use_orders
logging_steps
num_train_epochs
learning_rate
per_device_train_batch_size
save_steps
```

2. Train from generated ola dataset

Firstly, generate ola data

```bash
python main.py configs/gendata_all_conll2000pos.json
```

key arguments: 
```bash
train_models_name_list
dataset_name
cutoff_len
use_orders
```

Secondly, train ola adapter using generated ola data

```bash
python main.py configs/train_qwen_1b_conll2000pos_ola.json
```
or
```bash
python main.py configs/train_qwen_1b_conll2000pos.json --use_generated_oladata true
```

key arguments: 
```bash
train_models_name_list
adapter_architecture
dataset_name
cutoff_len
use_orders
logging_steps
num_train_epochs
learning_rate
per_device_train_batch_size
save_steps
```

## Evaluate OLA Adapter

```bash
python main.py configs/eval_qwen_1b_conll2000pos.json
```

key arguments: 
```bash
eval_models_name_list
adapter_architecture
dataset_name
cutoff_len
per_device_eval_batch_size
use_orders
```

## Todo 

- [ ] 对text chunking任务实现P，R，F1评价指标
- [ ] 实现OLAS相似性文本分类的数据集生成代码
- [ ] 数据增强（模拟不同模型的olas扰动，增强跨模型泛化性）
- [ ] 跨语言迁移
