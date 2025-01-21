import copy
import logging
import json
import os
from tqdm import tqdm
import numpy as np
import random

import lmdb, pickle

import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import set_seed

from convs.cifar_resnet import resnet32
from convs.resnet import resnet18, resnet34, resnet50
from convs.ucir_cifar_resnet import resnet32 as cosine_resnet32
from convs.ucir_resnet import resnet18 as cosine_resnet18
from convs.ucir_resnet import resnet34 as cosine_resnet34
from convs.ucir_resnet import resnet50 as cosine_resnet50
from convs.linears import SimpleLinear, SplitCosineLinear, CosineLinear
from convs.modified_represnet import resnet18_rep,resnet34_rep
from convs.resnet_cbam import resnet18_cbam,resnet34_cbam,resnet50_cbam

from typing import Tuple
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from transformers import (
    set_seed,
    HfArgumentParser,
)

from utils import (
    ADAPTERS_CKPT_NAME,
    save_arguments,
    visualize_attn_map,
    visualize_layer_attn_map,
    TextClsMetric,
    TokenClsMetric,
    MultiTokenMetric,
    ModelArguments, 
    DataArguments, 
    OLALMTrainingArguments as TrainingArguments,
    OLALMTrainer,
)
from utils.visualize import draw_attn_heatmap
from data_utils import (
    generate_save_ola_data,
    get_oladata_dir_path,
    DataManager,
)
from models.ola_model import OLAModel
from models.ola_utils import cal_maskes
from models.ola_model import preprocess_ola
from models.ola_augmentations import (
    RandomHightlightColumns,
    AddGuassianNoise,
    RandomTemperatureScaling
)


def args_parser() -> Tuple[ModelArguments, DataArguments, TrainingArguments]:
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    json_args_dict = {}
    other_args = []
    # if the first argument is a JSON file, parse it
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        with open(os.path.abspath(sys.argv[1]), 'r') as f:
            json_args_dict = json.load(f)
        other_args = sys.argv[2:]
    else:
        other_args = sys.argv[1:]
    # convert json args to list
    json_args_list = []
    for key, value in json_args_dict.items():
        if value is None:
            continue
        json_args_list.append(f"--{key}")
        if isinstance(value, list):
            json_args_list.extend(map(str, value))
        else:
            json_args_list.append(str(value))
    # combine json args and other args, other args have higher priority
    combined_args = json_args_list + other_args
    # parse arguments
    model_args, data_args, training_args = (
        parser.parse_args_into_dataclasses(combined_args)
    )
    return model_args, data_args, training_args


def get_convnet(args, pretrained=False):
    name = args["net"].lower()
    if name == "resnet32":
        return resnet32()
    elif name == "resnet18":
        return resnet18(pretrained=pretrained,args=args)
    elif name == "resnet34":
        return resnet34(pretrained=pretrained,args=args)
    elif name == "resnet50":
        return resnet50(pretrained=pretrained,args=args)
    elif name == "cosine_resnet18":
        return cosine_resnet18(pretrained=pretrained,args=args)
    elif name == "cosine_resnet32":
        return cosine_resnet32()
    elif name == "cosine_resnet34":
        return cosine_resnet34(pretrained=pretrained,args=args)
    elif name == "cosine_resnet50":
        return cosine_resnet50(pretrained=pretrained,args=args)
    elif name == "resnet18_rep":
        return resnet18_rep(pretrained=pretrained,args=args)
    elif name == "resnet18_cbam":
        return resnet18_cbam(pretrained=pretrained,args=args)
    elif name == "resnet34_cbam":
        return resnet34_cbam(pretrained=pretrained,args=args)
    elif name == "resnet50_cbam":
        return resnet50_cbam(pretrained=pretrained,args=args)
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args, num_classes=200, pretrained=None):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(args, pretrained)
        num_classes = num_classes
        self.fc = SimpleLinear(self.feature_dim, num_classes)

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)

        return out

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
    
class DataIter(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)
    
    def next(self):
        try:
            data = next( self._iter )
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next( self._iter )
        return data

class ClassifyDataset(Dataset):
    def __init__(self, data_dirs, selected_orders, is_casual=False, use_augment=True, sentence_len=100):
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
                self.len += self.num_samples
                for idx in range(self.num_samples):
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
            origin=False
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

def _compute_accuracy(model, loader, model_names):
    model.eval()
    correct, total = 0, 0
    # cor_model_dict = {"gemma-9b": 0, "bloomz-7b1": 0, "qwen-7b": 0, "gemma-2b": 0, "bloomz-3b": 0, "qwen-1b": 0}
    cor_model_dict = {model_name: 0 for model_name in model_names}
    for i, data in enumerate(loader):
        inputs, targets, model_names = data['attn_map'].cuda(), data['index'], data['model']
        with torch.no_grad():
            outputs = model(inputs)["logits"]
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == targets).sum()
        for cor_idx in torch.where(predicts.cpu() == targets)[0]:
            cor_model_dict[model_names[cor_idx]] += 1
        total += len(targets)
    return np.around(correct.cpu().data.numpy() * 100 / total, decimals=2), cor_model_dict

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
if __name__ == "__main__":
    setup_seed(2025)

    ams = {1:'bert-base-cased', 2:'bert-large-cased', 3:'roberta-base', 4:'roberta-large', 5:'electra-base-generator', 6:'electra-large-generator'}
    train_model_ids = [1,2,5,6]
    test_model_ids = [3,4]
    train_model_names = [ams[i] for i in train_model_ids]
    test_model_names = [ams[i] for i in test_model_ids]
    selected_orders = [1]
    num_classes = 1000
    sentence_len = 50
    use_augment = True
    attn_type = 'alti'
    lr = 0.003

    train_data_dir_paths = [f'datasets/conll2012_{attn_type}_en_entity_classify_len{sentence_len}/{model_name}/train' for model_name in train_model_names]
    test_data_dir_paths = [f'datasets/conll2012_{attn_type}_en_entity_classify_len{sentence_len}/{model_name}/train' for model_name in test_model_names]
    train_dataset = ClassifyDataset(train_data_dir_paths, selected_orders, use_augment=use_augment, sentence_len=sentence_len)
    test_dataset = ClassifyDataset(test_data_dir_paths, selected_orders, use_augment=use_augment, sentence_len=sentence_len)
    # print(train_dataset[0])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=20,
        shuffle=True,
        num_workers=8,
    )
    train_dataiter = DataIter(train_dataloader)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=200,
        shuffle=True,
        num_workers=8,
    )

    model_args = dict()
    model_args['net'] = "resnet18"
    model_args['input_channels'] = len(selected_orders)
    model = BaseNet(model_args, num_classes=num_classes)
        
    model.train().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=5e-4)

    train_acc = test_acc = best_acc = 0
    pbar = tqdm(range(100000), desc="train classifier")
    for step in pbar:
        data = train_dataiter.next()
        images, labels = data['attn_map'].cuda(), data['index'].cuda()

        output = model(images)["logits"]
        loss = F.cross_entropy(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix({"loss": loss.item(), "train_acc": train_acc, "test_acc": test_acc})

        if (step+1) % 100 == 0:
            train_acc, cor_model_dict = _compute_accuracy(model, train_dataloader, train_model_names)
            test_acc, cor_model_dict = _compute_accuracy(model, test_dataloader, test_model_names)

            if best_acc > test_acc:
                model.load_state_dict(best_w)
            else:
                best_w = model.state_dict()
                best_cor_model_dict = cor_model_dict
                print(cor_model_dict)

            best_acc = max(best_acc, test_acc)

    print(f"best_acc={best_acc}")
    print(best_cor_model_dict)
    print(f"attn_type: {attn_type}, lr: {lr}, use_augment: {use_augment}")
    
