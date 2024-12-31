import copy
import logging
import json
import os
from tqdm import tqdm
import numpy as np
import random

import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import set_seed

from utils import DataIter

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

import torch
from torch.utils.data import DataLoader
from transformers import (
    set_seed,
    HfArgumentParser,
)

from utils import (
    ADAPTERS_CKPT_NAME,
    save_arguments,
    evaluate_ola_adapter,
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
from data_utils import (
    generate_save_ola_data,
    get_oladata_dir_path,
    DataManager,
)
from models.ola_model import OLAModel


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

class mydataset(Dataset):
    def __init__(self, model_names, selected_orders, cut_len=200, transform=None, reverse=False):
        self.data = []
        for model_name in model_names:
            data_path = f'map_datas/remove_outliers/ro_maps_{model_name}.json'
            with open(data_path, 'r', encoding='utf-8') as file:
                tmp_data = json.load(file)
                tmp_data = tmp_data[:cut_len]
                for data in tmp_data:
                    data['attn_map'] = torch.tensor(data['attn_map'])[selected_orders].tolist()
                self.data.extend(tmp_data)
        if transform == None:
            self.transform = transforms.Compose([
                # transforms.ToTensor(),
                transforms.Resize((20, 20)),
                # transforms.Normalize(mean=[0.485], std=[0.229]),
            ])
        else:
            self.transform = transform
        self.reverse = reverse
    
    def __getitem__(self, index):
        data = self.data[index]

        if self.reverse:
            data['attn_map'] = torch.tensor(data['attn_map'][0])
            mask = torch.ones_like(data['attn_map']) - (torch.eye(data['attn_map'].shape[-1]).unsqueeze(0) * 0.5)
            data['attn_map'] = (data['attn_map'] + data['attn_map'].transpose(1, 2)) * mask
            data['attn_map'] = self.transform(np.array(data['attn_map'][0])).float()
        else:
            data['attn_map'] = self.transform(torch.tensor(data['attn_map'])).float()
        
        return data

    def __len__(self):
        return len(self.data)

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

    # train_datapath = "map_datas/remove_outliers/ro_maps_small.json"
    # test_datapath = "map_datas/remove_outliers/ro_maps_large.json"
    train_model_names = ['qwen-1b', 'gemma-2b', 'bloomz-3b', 'yi-6b', 'opt-1b3']
    test_model_names = ['qwen-7b', 'gemma-9b', 'bloomz-7b1', 'yi-9b', 'opt-2b7']
    selected_orders = [2]
    num_classes = 1500
    train_dataset = mydataset(train_model_names, selected_orders, cut_len=num_classes)
    test_dataset = mydataset(test_model_names, selected_orders, cut_len=num_classes)

    # parser arguments
    model_args, data_args, training_args = args_parser()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    # set random seed
    set_seed(training_args.seed)

    # create data manager
    data_manager = DataManager(
        dataset_name=data_args.dataset_name,
        cutoff_len=data_args.cutoff_len,
        train_model_name_or_paths=model_args.train_models_name_list,
        test_model_name_or_paths=model_args.eval_models_name_list,
        use_generated_oladata=data_args.use_generated_oladata,
        attn_type=data_args.attn_type,
        do_classify_data_generate=data_args.do_classify_data_generate
    )

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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=5e-4)

    train_acc = test_acc = best_acc = 0
    pbar = tqdm(range(90000), desc="train classifier")
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
                print(cor_model_dict)

            best_acc = max(best_acc, test_acc)

    print(best_acc)
    
