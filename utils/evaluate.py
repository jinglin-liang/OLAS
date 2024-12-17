from typing import Optional, List, Dict
from dataclasses import dataclass, field
import inspect
import json
import os
from tqdm import tqdm

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models import OLAModel


@dataclass
class TextClsMetric:
    samples_num: int = 0
    correct_num: int = 0
    accuracy: float = 0
    all_samples: List[dict] = field(default_factory=lambda: [])
    positive_samples: List[dict] = field(default_factory=lambda: [])
    negative_samples: List[dict] = field(default_factory=lambda: [])

    def judge(self, predictions: Tensor, labels: Tensor, data: Dict = None):
        '''
        predictions: (n, c)
        labels: (n)
        '''
        predictions = predictions.cpu()
        labels = labels.cpu() 
        pred = predictions.argmax(dim=-1)
        self.correct_num += (pred == labels).sum().item()
        self.samples_num += len(labels)
        self.accuracy = self.correct_num / self.samples_num
        if data is not None:
            batch_samples = [
                {
                    "text": data["text"][i], 
                    "labels": data["labels"][i].item(),
                    "pred": pred[i].item()
                } 
                for i in range(data["labels"].shape[0])
            ]
            self.all_samples += batch_samples
            self.positive_samples += [
                sample for sample in batch_samples if sample["labels"] == sample["pred"]
            ]
            self.negative_samples += [
                sample for sample in batch_samples if sample["labels"] != sample["pred"]
            ]

    def metric_string(self):
        return f"Accuracy={self.accuracy:.4f}. "


@dataclass
class TokenClsMetric(TextClsMetric):
    total_tokens_num: int = 0
    correct_tokens_num: int = 0
    line_level_accuracy: float = 0
    label_names: Optional[List[str]] = None
    tokenizer: Optional[AutoTokenizer] = None

    def judge(self, predictions: Tensor, labels: Tensor, data: Dict = None):
        '''
        predictions: (n, l, c)
        labels: (n, l)
        '''
        predictions = predictions.cpu()
        labels = labels.cpu()
        attention_mask = data["attention_mask"].cpu()
        preds = predictions.argmax(dim=-1)
        self.samples_num += labels.shape[0]
        self.total_tokens_num += attention_mask.sum().item()
        self.correct_tokens_num += ((preds == labels) * attention_mask).sum().item()
        self.correct_num += (
            ((preds == labels) * attention_mask).sum(-1) == attention_mask.sum(-1)
        ).sum().item()
        self.accuracy = self.correct_tokens_num / self.total_tokens_num
        self.line_level_accuracy = self.correct_num / self.samples_num
        if data is not None:
            for i in range(data["labels"].shape[0]):
                sample = {
                    "text": data["text"][i],
                    "token:label->pred": []
                }
                for j in range(data["labels"].shape[1]):
                    if attention_mask[i][j] == 0:
                        continue
                    sample["token:label->pred"].append(
                        f"{self.tokenizer.decode(data['input_ids'][i][j].item())}: {self.label_names[data['labels'][i][j].item()]} -> {self.label_names[preds[i][j].item()]}"
                    )
                self.all_samples.append(sample)
                if ((preds[i] == labels[i]) * attention_mask[i]).sum(-1) == attention_mask[i].sum(-1):
                    self.positive_samples.append(sample)
                else:
                    self.negative_samples.append(sample)

    def metric_string(self):
        return f"Accuracy={self.accuracy:.4f}, Line-level_Accuracy={self.line_level_accuracy:.4f}."


class EntityMetric(TokenClsMetric):
    total_tokens_num: int = 0
    correct_tokens_num: int = 0
    total_p: int = 0
    total_tp: int = 0
    total_fn: int = 0
    line_level_accuracy: float = 0
    label_names: Optional[List[str]] = None
    tokenizer: Optional[AutoTokenizer] = None

    def judge(self, predictions: Tensor, labels: Tensor, data: Dict = None):
        '''
        predictions: (n, l, c)
        labels: (n, l)
        '''
        predictions = predictions.cpu()
        labels = labels.cpu()
        attention_mask = data["attention_mask"].cpu()
        preds = predictions.argmax(dim=-1)
        self.samples_num += labels.shape[0]
        self.total_tokens_num += attention_mask.sum().item()
        self.correct_tokens_num += ((preds == labels) * attention_mask).sum().item()
        self.correct_num += (
            ((preds == labels) * attention_mask).sum(-1) == attention_mask.sum(-1)
        ).sum().item()
        self.accuracy = self.correct_tokens_num / self.total_tokens_num
        self.line_level_accuracy = self.correct_num / self.samples_num

        preds_entities = self.find_entities(preds)
        labels_entities = self.find_entities(labels)
        self.total_p += len(preds_entities)
        for p_en in preds_entities:
            if p_en in labels_entities:
                self.total_tp += 1
        self.precision = self.total_tp / self.total_p
        for l_en in labels_entities:
            if l_en not in preds_entities:
                self.total_fn += 1
        self.recall = self.total_tp / (self.total_tp + self.total_fn)
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall + 1e-9)

        if data is not None:
            for i in range(data["labels"].shape[0]):
                sample = {
                    "text": data["text"][i],
                    "token:label->pred": []
                }
                for j in range(data["labels"].shape[1]):
                    if attention_mask[i][j] == 0:
                        continue
                    sample["token:label->pred"].append(
                        f"{self.tokenizer.decode(data['input_ids'][i][j].item())}: {self.label_names[data['labels'][i][j].item()]} -> {self.label_names[preds[i][j].item()]}"
                    )
                self.all_samples.append(sample)
                if ((preds[i] == labels[i]) * attention_mask[i]).sum(-1) == attention_mask[i].sum(-1):
                    self.positive_samples.append(sample)
                else:
                    self.negative_samples.append(sample)

    def find_entities(self, t):
        entities = []
        t = t.view(-1)
        pos = torch.where(t % 2 != 0)[0]
        for p in pos:
            tmp_len = 0
            while (p + tmp_len + 1 < t.shape[0]) and (t[p + tmp_len + 1] == t[p] + 1):
                tmp_len += 1
            entities.append((t[p].item(), p.item(), tmp_len+1))
        return entities

    def metric_string(self):
        return f"Accuracy={self.accuracy:.4f}, Line-level_Accuracy={self.line_level_accuracy:.4f}, Precision={self.precision:.4f}, Recall={self.recall:.4f}, F1_score={self.f1:.4f}."


def evaluate_ola_adapter(
    eval_dataloader: DataLoader,
    eval_metric: TextClsMetric,
    eval_ola_model: OLAModel, 
    eval_adapter_ckpt: int,
    output_dir: str,
):
    # load the adapter checkpoint
    model = eval_ola_model.eval().cuda()
    model.load_adapter(eval_adapter_ckpt)
    # do evaluation
    interested_keys = inspect.signature(model.forward).parameters.keys()
    bar = tqdm(eval_dataloader, desc="Evaluating")
    for data in bar:
        if data['input_ids'].shape[-1] == 0:
            continue
        with torch.no_grad():
            output = model(
                **{k: v for k, v in data.items() if k in interested_keys}
            )
        eval_metric.judge(output.logits, data["labels"], data)
        bar.set_postfix_str(f"{eval_metric.metric_string()}")
    # save the results
    if len(eval_metric.all_samples) > 0:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"all_results_acc_{eval_metric.accuracy:.4f}.json"), "w") as f:
            json.dump(eval_metric.all_samples, f, indent=4)
        with open(os.path.join(output_dir, f"positive_results.json"), "w") as f:
            json.dump(eval_metric.positive_samples, f, indent=4)
        with open(os.path.join(output_dir, f"negative_results.json"), "w") as f:
            json.dump(eval_metric.negative_samples, f, indent=4)
    print(f"{eval_metric.metric_string()}\nResults are saved to {output_dir}")
