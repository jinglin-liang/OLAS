from typing import Optional, List
import inspect
import json
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from models import OLAModel


def evaluate_ola_adapter(
    eval_dataloader: DataLoader,
    eval_ola_model: OLAModel, 
    eval_adapter_ckpt: int,
    output_dir: str,
):
    # load the adapter checkpoint
    model = eval_ola_model.eval().cuda()
    model.load_adapter(eval_adapter_ckpt)

    # do evaluation
    total_num = 0
    total_correct = 0
    all_samples = []
    positive_samples = []
    negative_samples = []
    interested_keys = inspect.signature(model.forward).parameters.keys()
    bar = tqdm(eval_dataloader, desc="Evaluating")
    for data in bar:
        with torch.no_grad():
            output = model(
                **{k: v for k, v in data.items() if k in interested_keys}
            )
        pred = output.logits.argmax(dim=-1)
        total_correct += (pred.cpu() == data["labels"]).sum().item()
        total_num += len(data["labels"])
        batch_samples = [
            {
                "text": data["text"][i], 
                "labels": data["labels"][i].item(),
                "pred": pred[i].item()
            } 
            for i in range(data["labels"].shape[0])
        ]
        all_samples += batch_samples
        positive_samples += [
            sample for sample in batch_samples if sample["labels"] == sample["pred"]
        ]
        negative_samples += [
            sample for sample in batch_samples if sample["labels"] != sample["pred"]
        ]
        bar.set_postfix(Acc=f"{total_correct/total_num:.4f}")
    accuracy = total_correct / total_num
    # save the results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"all_results_acc_{accuracy:.4f}.json"), "w") as f:
        json.dump(all_samples, f, indent=4)
    with open(os.path.join(output_dir, f"positive_results.json"), "w") as f:
        json.dump(positive_samples, f, indent=4)
    with open(os.path.join(output_dir, f"negative_results.json"), "w") as f:
        json.dump(negative_samples, f, indent=4)
    print(f"Accuracy: {accuracy:.4f}, results are saved to {output_dir}")
