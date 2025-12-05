from typing import Optional, List, Dict
from dataclasses import dataclass, field
import inspect
import json
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

from models import OLAModel
from utils.visualize import draw_attn_heatmap
from data_utils import DataManager


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
class DependencyParsingMetric:
    total = 0
    correct_arcs = 0
    correct_rels = 0
    uas = 0
    las = 0
    all_samples: List[dict] = field(default_factory=lambda: [])
    positive_samples: List[dict] = field(default_factory=lambda: [])
    negative_samples: List[dict] = field(default_factory=lambda: [])
    tokenizer: Optional[AutoTokenizer] = None

    def judge(self, predictions: Tensor, labels: Tensor, data: Dict = None):
        '''
        predictions: (n, c)
        labels: (n)
        '''
        pred_begin_arc, pred_begin_rel = predictions
        heads = data["heads"].to(pred_begin_arc.device)
        dp_rels = data["dp_rels"].to(pred_begin_arc.device)
        heads = heads[heads.ne(-200)]
        dp_rels = dp_rels[dp_rels.ne(-200)]
        assert pred_begin_arc.shape[0] == pred_begin_rel.shape[0] == heads.shape[0] == dp_rels.shape[0]
        in_range_mask = heads.ne(-100)
        pred_begin_arc = pred_begin_arc[in_range_mask]
        pred_begin_rel = pred_begin_rel[in_range_mask, :]
        heads = heads[in_range_mask]
        dp_rels = dp_rels[in_range_mask]
        pred_arc = pred_begin_arc.argmax(dim=-1)
        pred_rel = pred_begin_rel[torch.arange(pred_arc.shape[0]), pred_arc, :].argmax(dim=-1)
        self.total += pred_arc.shape[0]
        correct_acr_mask = (pred_arc == heads)
        correct_rel_mask = (pred_rel == dp_rels) & correct_acr_mask
        self.correct_arcs += correct_acr_mask.sum().item()
        self.correct_rels += correct_rel_mask.sum().item()
        self.uas = self.correct_arcs / (self.total + 1e-10)
        self.las = self.correct_rels / (self.total + 1e-10)
        temp_samples = {
            "heads": heads.tolist(),
            "dp_rels": dp_rels.tolist(),
            "pred_arc": pred_arc.tolist(),
            "pred_rel": pred_rel.tolist(),
        }
        self.all_samples.append(temp_samples)
        if correct_acr_mask.all():
            self.positive_samples.append(temp_samples)
        else:
            self.negative_samples.append(temp_samples)

    def metric_string(self):
        return f"UAS={self.uas:.4f}, LAS={self.las:.4f}."


@dataclass
class TokenClsMetric(TextClsMetric):
    total_tokens_num: int = 0
    correct_tokens_num: int = 0
    line_level_accuracy: float = 0
    label_names: Optional[List[str]] = None
    tokenizer: Optional[AutoTokenizer] = None

    def __post_init__(self):
        self.confuse_matrix = np.zeros((len(self.label_names), len(self.label_names)))

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
                    self.confuse_matrix[data['labels'][i][j].item(), preds[i][j].item()] += 1
                self.all_samples.append(sample)
                if ((preds[i] == labels[i]) * attention_mask[i]).sum(-1) == attention_mask[i].sum(-1):
                    self.positive_samples.append(sample)
                else:
                    self.negative_samples.append(sample)

    def metric_string(self):
        return f"Accuracy={self.accuracy:.4f}, Line-level_Accuracy={self.line_level_accuracy:.4f}."
    
    def write_confusion_matrix(
            self, 
            csv_file_path: Optional[str] = None,
            pic_file_path: Optional[str] = None,
        ):
        if csv_file_path is not None:
            os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
            confuse_df = pd.DataFrame(self.confuse_matrix, columns=self.label_names, index=self.label_names)
            confuse_df.to_csv(csv_file_path)
        if pic_file_path is not None:
            os.makedirs(os.path.dirname(pic_file_path), exist_ok=True)
            cf_tensor = torch.tensor(self.confuse_matrix)[None, None, :, :]
            cf_tensor = cf_tensor / (cf_tensor.sum(-1, keepdim=True) + 1e-10)
            tick_ls = self.label_names
            draw_attn_heatmap(cf_tensor, tick_ls, tick_ls, "Confusion Matrix", annot_size=2, label_size=4)
            plt.gcf().set_size_inches(8, 8)
            plt.savefig(pic_file_path, dpi=450)
            plt.clf()


class MultiTokenMetric(TokenClsMetric):
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

        preds_entities = self.find_targets(preds)
        labels_entities = self.find_targets(labels)
        self.total_p += len(preds_entities)
        for p_en in preds_entities:
            if p_en in labels_entities:
                self.total_tp += 1
        self.precision = self.total_tp / self.total_p if self.total_p != 0 else 0
        for l_en in labels_entities:
            if l_en not in preds_entities:
                self.total_fn += 1
        self.recall = self.total_tp / (self.total_tp + self.total_fn) if (self.total_tp + self.total_fn) != 0 else 0
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

    def find_targets(self, t):
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
    output_dir: str,
    eval_adapter_ckpt: Optional[int] = None,
):
    # load the adapter checkpoint
    model = eval_ola_model.eval().cuda()
    if eval_adapter_ckpt is not None:
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
        eval_metric.judge(output.logits, data.get("labels"), data)
        bar.set_postfix_str(f"{eval_metric.metric_string()}")
    # save the results
    if len(eval_metric.all_samples) > 0:
        os.makedirs(output_dir, exist_ok=True)
        if isinstance(eval_metric, MultiTokenMetric):
            with open(os.path.join(output_dir, f"all_results_acc_{eval_metric.accuracy:.4f}_P_{eval_metric.precision:.4f}_R_{eval_metric.recall:.4f}_F1_{eval_metric.f1:.4f}.json"), "w") as f:
                json.dump(eval_metric.all_samples, f, indent=4)
        elif isinstance(eval_metric, DependencyParsingMetric):
            with open(os.path.join(output_dir, f"all_results_UAS_{eval_metric.uas:.4f}_LAS_{eval_metric.las:.4f}.json"), "w") as f:
                json.dump(eval_metric.all_samples, f, indent=4)
        else:
            with open(os.path.join(output_dir, f"all_results_acc_{eval_metric.accuracy:.4f}.json"), "w") as f:
                json.dump(eval_metric.all_samples, f, indent=4)
        with open(os.path.join(output_dir, f"positive_results.json"), "w") as f:
            json.dump(eval_metric.positive_samples, f, indent=4)
        with open(os.path.join(output_dir, f"negative_results.json"), "w") as f:
            json.dump(eval_metric.negative_samples, f, indent=4)
        # if hasattr(eval_metric, "write_confusion_matrix") and callable(getattr(eval_metric, "write_confusion_matrix")):
        #     eval_metric.write_confusion_matrix(os.path.join(output_dir, "confusion_matrix.csv"),
        #                                        os.path.join(output_dir, "confusion_matrix.png"))
    print(f"{eval_metric.metric_string()}\nResults are saved to {output_dir}")


def evaluate_ola_adapter_with_multi_llms(
    eval_models_name_list: List[str],
    eval_args: Dict,
    output_dir: str,
    data_manager: DataManager,
    task: str,
    batch_size: int,
    attn_type: str,
    use_generated_oladata: bool,
    eval_adapter_checkpoint: Optional[str] = None,
    model: Optional[OLAModel] = None,
):
    assert int(eval_adapter_checkpoint is not None) + int(model is not None) == 1, \
        "Only one of eval_adapter_checkpoint and model can be set."
    # evaluate each model
    for eval_model_name in eval_models_name_list:
        print(f"Evaluating model {eval_model_name}")
        # load eval dataset
        eval_dataset, data_collator = data_manager.get_dataset_collator(
            [eval_model_name], "test", task=task
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=data_collator,
            batch_size=batch_size,
            shuffle=False,
        )
        # load eval metric
        if data_manager.dataset_name.lower() in ["semeval_re"]:
            eval_metric = TextClsMetric()
        elif data_manager.dataset_name.lower() in ["conll2000_pos", "conll2012en_pos"]:
            if hasattr(eval_dataset.datasets[0], "features"):
                try:
                    label_names = eval_dataset.datasets[0].features["pos_tags"].feature.names
                except:
                    label_names = eval_dataset.datasets[0].features["pos_tags_names"]
                label_names.append("[None]")
            elif hasattr(eval_dataset.datasets[0], "pos_tags_names"):
                label_names = eval_dataset.datasets[0].pos_tags_names
                label_names.append("[None]")
            else:
                label_names = [str(i) for i in range(eval_args["num_classes"])]
            eval_metric = TokenClsMetric(
                label_names=label_names,
                tokenizer=data_manager.tokenizer_dict[eval_model_name],
            )
        elif data_manager.dataset_name.lower() in ["conll2012en_entity"]:
            if hasattr(eval_dataset.datasets[0], "features"):
                eval_metric = MultiTokenMetric(
                    label_names=eval_dataset.datasets[0].features["named_entities_names"],
                    tokenizer=data_manager.tokenizer_dict[eval_model_name],
                )
            else:
                eval_metric = MultiTokenMetric(
                    label_names=eval_dataset.datasets[0].named_entities_names,
                    tokenizer=data_manager.tokenizer_dict[eval_model_name],
                )
        elif data_manager.dataset_name.lower() in ["ud_english_ewt"]:
            eval_metric = DependencyParsingMetric(
                tokenizer=data_manager.tokenizer_dict[eval_model_name],
            )
        else:
            raise NotImplemented
        # create OLAModel
        if 'adapter_params' not in eval_args.keys(): # for ola checkpoints
            eval_args['adapter_params'] = {}
            eval_args['adapter_params']["axial_tf_layers"] = eval_args['num_layers']
            eval_args['adapter_params']["hidden_size"] = eval_args['adapter_hidden_size']

        if eval_adapter_checkpoint is not None:
            model = OLAModel(
                base_model_name_list=[eval_model_name,],
                adapter_architecture=eval_args["adapter_architecture"],
                num_classes=eval_args["num_classes"],
                use_orders=eval_args["use_orders"],
                remove_outliers=eval_args["remove_outliers"],
                outliers_sigma_multiplier=eval_args["outliers_sigma_multiplier"],
                attn_type=attn_type,
                abandom_base_lm=use_generated_oladata,
                **eval_args.get("adapter_params", {})
            )
        # set output_dir
        tmp_output_dir = os.path.join(
            output_dir, 
            f"eval_{os.path.basename(eval_model_name)}"
        )
        # evaluate
        evaluate_ola_adapter(
            eval_dataloader=eval_dataloader,
            eval_metric=eval_metric,
            eval_ola_model=model,
            eval_adapter_ckpt=eval_adapter_checkpoint,
            output_dir=tmp_output_dir,
        )
        # Explicitly delete the model and clear cache
        if eval_adapter_checkpoint is not None:
            del model
            torch.cuda.empty_cache()
