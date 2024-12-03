import os
import inspect
import lmdb
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from models import OLAModel
from data_utils.data import DATASET_NAME_TO_PATH


def get_oladata_dir_path(dataset_name, model_name_or_path, split):
    if os.path.isfile(DATASET_NAME_TO_PATH[dataset_name]):
        data_root_dir = os.path.dirname(DATASET_NAME_TO_PATH[dataset_name]) + "_ola"
    elif os.path.isdir(DATASET_NAME_TO_PATH[dataset_name]):
        data_root_dir = DATASET_NAME_TO_PATH[dataset_name] + "_ola"
    else:
        raise ValueError("Invalid dataset path")
    save_dir = os.path.join(
        data_root_dir, 
        os.path.basename(model_name_or_path),
        split
    )
    return save_dir


def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def generate_save_ola_data(
    model: OLAModel,
    dataset,
    data_collator,
    save_dir: str,
):
    # create dataloader
    dataloader = DataLoader(
        dataset,
        collate_fn=data_collator,
        batch_size=1,
        shuffle=False,
    )
    # set model to eval mode
    model = model.eval().cuda()
    # create lmdb
    os.makedirs(save_dir, exist_ok=True)
    env = lmdb.open(save_dir, map_size=1099511627776)
    # generate data
    interested_keys = inspect.signature(model.forward).parameters.keys()
    bar = tqdm(dataloader, desc="Generating OLA data")
    cnt = 0
    cache = {}
    for data in bar:
        with torch.no_grad():
            input_dict = {k: v for k, v in data.items() if k in interested_keys}
            input_dict["output_ola"] = True
            input_dict["labels"] = None
            output = model(
                **input_dict
            )
        ola = output.order_level_attention
        tmp_data = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                # "input_ids", "attention_mask", "labels"
                tmp_data[k] = v[0].tolist()
            else:
                # "token_pos_tags", "token_chunk_tags"
                tmp_data[k] = v[0]
        tmp_data["ola"] = {k: v.cpu() for k, v in ola.items()}
        data_byte = pickle.dumps(tmp_data)
        data_id = str(cnt).encode("utf-8")
        cache[data_id] = data_byte
        cnt += 1
        if cnt % 400 == 0:
            write_cache(env, cache)
            cache = {}
        # if cnt == 20:
        #     break
    cache["num_samples".encode('utf-8')] = str(cnt).encode("utf-8")
    write_cache(env, cache)
    print('save {} samples to {}'.format(cnt, save_dir))
    env.close()


class OLADataset:
    def __init__(self, data_dir):
        self.env = lmdb.open(data_dir, max_readers=8, readonly=True, lock=False, readahead=True, meminit=True)
        with self.env.begin(write=False) as txn:
            self.num_samples = int(txn.get('num_samples'.encode('utf-8')).decode("utf-8"))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            data_id = str(idx).encode("utf-8")
            data_byte = txn.get(data_id)
            data = pickle.loads(data_byte)
        return data
