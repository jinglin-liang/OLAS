# Order-Level Attention Similarity Across Language Models: A Latent Commonality

 <p align='center'>
  <b>
    <a href="https://arxiv.org/abs/2511.05064">Paper</a>
    |
    <a href="https://neurips.cc/media/PosterPDFs/NeurIPS%202025/117799.png?t=1764063123.996938">Poster</a>
  </b>
</p> 

> **Order-Level Attention Similarity Across Language Models: A Latent Commonality**<br> Jinglin Liang, Jin Zhong, Shuangping Huang*, Yunqing Hu, Huiyuan Zhang, Huifang Li, Lixin Fan, Hanlin Gu  <br>


>**Abstract**: <br> In this paper, we explore an important yet previously neglected question: Do context aggregation patterns across Language Models (LMs) share commonalities? While some works have investigated context aggregation or attention weights in LMs, they typically focus on individual models or attention heads, lacking a systematic analysis across multiple LMs to explore their commonalities. In contrast, we focus on the commonalities among LMs, which can deepen our  nderstanding of LMs and even facilitate cross-model knowledge transfer. In this work, we introduce the Order-Level Attention (OLA) derived from the order-wise decomposition of Attention Rollout and reveal that the OLA at the same order across LMs exhibits significant similarities. Furthermore, we discover an implicit mapping between OLA and syntactic knowledge. Based on these two findings, we propose the Transferable OLA Adapter (TOA), a training-free cross-LM adapter transfer method. Specifically, we treat the OLA as a unified syntactic feature representation and train an adapter that takes OLA as input. Due to the similarities in OLA across LMs, the adapter generalizes to unseen LMs without requiring any parameter updates. Extensive experiments demonstrate that TOA’s cross-LM generalization effectively enhances the performance of unseen LMs. Code is available at https://github.com/jinglin-liang/OLAS. <br>

## 📢 Description
This repository is the official PyTorch implementation of:

[Order-Level Attention Similarity Across Language Models: A Latent Commonality](https://arxiv.org/abs/2511.05064) (NeurIPS 2025).

## 🔨 Requirement
### Environment
```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```
Then install CLIP from the official [CLIP](https://github.com/openai/CLIP) repository.

### Prepare Data
The program will automatically download the CIFAR-100 dataset. You only need to download the Tiny ImageNet dataset using the following commands.
```bash
cd data
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
python preprocess.py
cd ..
```

## 🍔 Pre-trained Model
We use the pretrained diffusion model from [LDM](https://github.com/CompVis/latent-diffusion) repository, you can simply use the following command to obtain the pre-trained model.
```bash
mkdir -p models/ldm/text2img-large
wget -O models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```

Please download bert-base-uncased from [here](https://huggingface.co/google-bert/bert-base-uncased), and put it in models/bert.

## Qualitative Empirical Evidence of OLAS

```bash
python main.py configs/visual_clms.json
```

## Quantitative Empirical Evidence of OLAS 

### Quantitative Analysis Based on Visual Model

python main.py configs/gendata_conll2012_mlm.json --do_classify_data_generate
python classify.py

###  Quantitative Analysis Based on Image Similarity Retrieval

python map_metric.py

##  Transferable OLA Adapter

### Training

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

## Testing

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
