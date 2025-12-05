import random
from tqdm import tqdm
import lmdb
import pickle
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torchvision import transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

from models.ola_utils import cal_maskes
from models.ola_model import preprocess_ola
from models.ola_augmentations import (
    RandomHightlightColumns,
    AddGuassianNoise,
    RandomTemperatureScaling
)


class ClassifyDataset(Dataset):
    def __init__(self, data_dirs, selected_orders, is_casual=False, use_augment=True, sentence_len=100, sentence_num_perdir=1000):
        self.len = 0
        self.data = []
        self.remove_outliers = True
        self.outliers_sigma_multiplier = 3.0
        self.is_casual = is_casual
        self._init_ola_augmentation(use_augment)
        self.transform = transforms.Compose([
            transforms.Resize(size=(sentence_len, sentence_len), antialias=True)
        ])

        for data_dir in data_dirs:
            self.env = lmdb.open(data_dir, max_readers=8, readonly=True, lock=False, readahead=True, meminit=True)
            with self.env.begin(write=False) as txn:
                self.num_samples = int(txn.get('num_samples'.encode('utf-8')).decode("utf-8"))
                if self.num_samples < sentence_num_perdir:
                    # raise ValueError(f"num_samples: {self.num_samples} < sentence_num_perdir: {sentence_num_perdir}")
                    sentence_num_perdir = self.num_samples
                self.len += sentence_num_perdir
                for idx in range(sentence_num_perdir):
                    data_id = str(idx).encode("utf-8")
                    data_byte = txn.get(data_id)
                    data = pickle.loads(data_byte)
                    tmp_attn_map = {k: attn for k, attn in data['ola'].items() if k in selected_orders}
                    self.data.append({'id': data['id'], 'attn_map': tmp_attn_map, 'attention_mask': data['attention_mask'], 'model': data_dir.split('/')[-2]})

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        attn = self.data[idx]['attn_map']
        attention_mask = torch.tensor(self.data[idx]['attention_mask']).unsqueeze(0)
        # calculate maskes from ola
        attn_mask = cal_maskes(
            attn, attention_mask, 
            is_casual=self.is_casual, 
            sigma_multiplier=self.outliers_sigma_multiplier
        )
        # augments ola
        if self.ola_augments is not None:
            for tmp_aug in self.ola_augments:
                attn, _ = tmp_aug(attn, attn_mask)
                attn_mask = cal_maskes(
                    attn, attention_mask, 
                    is_casual=self.is_casual, 
                    sigma_multiplier=self.outliers_sigma_multiplier
                )
        # preprocess ola
        stack_attn_tensor = preprocess_ola(
            attn, attn_mask, 
            remove_outliers=self.remove_outliers, 
            is_casual=self.is_casual,
            result_type="rm_ol"
        )
        stack_attn_tensor = self.transform(stack_attn_tensor)[0]
        return {'index': self.data[idx]['id'], 'attn_map': stack_attn_tensor, 'model': self.data[idx]['model']}
    
    def _init_ola_augmentation(self, use_augment):
        if use_augment:
            ola_augments = [
                        {
                            "class_name": "RandomHightlightColumns",
                            "params": {
                                "p": 0.3,
                                "min_columns": 1,
                                "max_columns": 3,
                                "ref_rank1": 0,
                                "ref_rank2": 1
                            }
                        },
                        {
                            "class_name": "AddGuassianNoise",
                            "params": {
                                "p": 0.15,
                                "std_ratio": 0.13
                            }
                        },
            ]
        else:
            ola_augments = None
        if ola_augments is not None:
            self.ola_augments = []
            for tmp_aug in ola_augments:
                class_name = tmp_aug["class_name"]
                params = tmp_aug["params"]
                self.ola_augments.append(
                    globals()[class_name](**params)
                )
        else:
            self.ola_augments = None


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument("--src_model", type=str, choices=[
            'bert-base-cased', 'bert-large-cased', 
            'roberta-base', 'roberta-large',
            'electra-base-generator', 'electra-large-generator',
            'Qwen2-1.5B-Instruct', 'Qwen2-7B-Instruct', 
            'gemma-2-2b-it', 'gemma-2-9b-it', 
            'Llama-3.2-3B-Instruct', 'Llama-3.1-8B-Instruct'
        ], help="source model")
    parser.add_argument("--tgt_model", type=str, choices=[
            'bert-base-cased', 'bert-large-cased', 
            'roberta-base', 'roberta-large',
            'electra-base-generator', 'electra-large-generator',
            'Qwen2-1.5B-Instruct', 'Qwen2-7B-Instruct', 
            'gemma-2-2b-it', 'gemma-2-9b-it', 
            'Llama-3.2-3B-Instruct', 'Llama-3.1-8B-Instruct'
        ], help="target model")
    parser.add_argument('--selected_orders', type=int, nargs='+', default=[1], help='selected OLA orders')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)

    model1_name = args.tgt_model
    model2_name = args.src_model
    selected_orders = args.selected_orders
    sentence_len = 50
    use_augment = False
    attn_type = 'ola'
    sentence_num_perdir = 1000

    data_dir_path1 = [f'datasets/conll2012_{attn_type}_en_entity_classify_len50_num2000_origin_seed2025/{model1_name}/train']
    data_dir_path2 = [f'datasets/conll2012_{attn_type}_en_entity_classify_len50_num2000_origin_seed2025/{model2_name}/train']
    dataset1 = ClassifyDataset(data_dir_path1, selected_orders, use_augment=use_augment, sentence_len=sentence_len, sentence_num_perdir=sentence_num_perdir)
    dataset2 = ClassifyDataset(data_dir_path2, selected_orders, use_augment=use_augment, sentence_len=sentence_len, sentence_num_perdir=sentence_num_perdir)
    print(f"dataset1_len = {len(dataset1)}, dataset2_len = {len(dataset2)}")

    metric_name = 'ssim'
    if metric_name == 'ssim':
        metric = StructuralSimilarityIndexMeasure().to('cuda')
    elif metric_name == 'psnr':
        metric = PeakSignalNoiseRatio().to('cuda')

    hit_num = [1, 5]
    acc = {f"hit@{hit}":0 for hit in hit_num}
    print(f"model1{model1_name}, model2{model2_name}, hit@{hit_num}")
    bar = tqdm(enumerate(dataset1), desc='datas')
    for idx1, data1 in bar:
        metric_value_list = []
        for idx2, data2 in enumerate(dataset2):
            attn1, attn2 = data1['attn_map'].unsqueeze(0).to('cuda'), data2['attn_map'].unsqueeze(0).to('cuda')
            metric_value = metric(attn1, attn2).item()
            metric_value_list.append(metric_value)
        target_value = metric_value_list[idx1]
        sorted_list = sorted(metric_value_list, reverse=True)
        target_pos = sorted_list.index(target_value) + 1
        for hit in hit_num:
            if target_pos <= hit:
                acc[f"hit@{hit}"] += 1
        postfix_str = ""
        for k, v in acc.items():
            postfix_str = postfix_str + f"{k}={v * 100 / (idx1 + 1):.4f} "
        bar.set_postfix_str(postfix_str)
    print(postfix_str)
    print(f"model1{model1_name}, model2{model2_name}, hit@{hit_num} finish.")
    print(f"hit@1/hit@5: {(acc['hit@1']*100/sentence_num_perdir):.2f}/{(acc['hit@5']*100/sentence_num_perdir):.2f}")
