from typing import Optional, List
from tqdm import tqdm
import os
import math

import torch
import matplotlib.pyplot as plt
import seaborn as sns

from models import OLAModel


def preprocess_attn_map(attn_map: torch.Tensor):
    return attn_map.mean(dim=1).squeeze(0).cpu().detach().numpy()


def draw_attn_heatmap(input_map: torch.Tensor, 
                      xtick_ls, ytick_ls, title, 
                      annot_size=1, label_size=3,
                      amplify=False):

    def custom_fmt(ori, prob):
        return '{:.3f}\n{:.2f}'.format(prob, ori).lstrip('0')
    
    input_map = preprocess_attn_map(input_map)
    prob_array = input_map / (input_map.sum(axis=1, keepdims=True) + 1e-10)
    if amplify:
        show_array = input_map / (input_map.max(axis=1, keepdims=True) + 1e-10) 
    else:
        show_array = input_map
    # plt.title(title)
    center = show_array.min() + 0.3 * (show_array.max() - show_array.min())
    ax = sns.heatmap(data=show_array, cmap='rainbow', center=center,
                     vmax=show_array.max(), vmin=show_array.min(), annot=False, 
                     square=True, linewidths=0, cbar_kws={"shrink":.2}, 
                     xticklabels=xtick_ls, yticklabels=ytick_ls)
    # add text
    # for i in range(prob_array.shape[0]):
    #     for j in range(prob_array.shape[1]):
    #         fw = 'normal'
    #         ax.text(j + 0.5, i + 0.5, 
    #                 custom_fmt(input_map[i, j], prob_array[i, j]),
    #                 ha='center', va='center', 
    #                 fontsize=annot_size, fontweight=fw)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=label_size)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=label_size)
    # plt.xlabel('K')
    # plt.ylabel('Q')


def visualize_attn_map(
    visual_models_name_list: List[str],
    use_orders: List[int],
    text_list: List[str],
    output_dir: str,
    ola_augments: Optional[List[dict]] = None,
    attn_type: str = 'ola',
    cutoff_len: int = 320,
    outliers_sigma_multiplier: float = 3.0,
    annot_size: int = 1,
    label_size: int = 3,
    load_method: str = 'origin'
):
    ola_dict = {}
    ola_mask_dict = {}
    model_dict = {}
    # calculate ola
    for tmp_model_name in visual_models_name_list:
        tmp_model = OLAModel(
            base_model_name_list=[tmp_model_name,],
            adapter_architecture="textcls_resnet18",
            num_classes=2,
            use_orders=use_orders,
            remove_outliers=False,
            outliers_sigma_multiplier=outliers_sigma_multiplier,
            ola_augments=ola_augments,
            attn_type=attn_type,
            load_method=load_method
        ).train().cuda()
        model_dict[tmp_model_name] = tmp_model
        ola_list = []
        ola_mask_list = []
        with torch.no_grad():
            for tmp_text in text_list:
                tmp_output = tmp_model.cal_ola_from_text([tmp_text], cutoff_len, attn_type, do_vis=True)
                ola, ola_mask = tmp_output.order_level_attention, tmp_output.ola_maskes
                ola = {k: (v / (v.sum(dim=-1, keepdim=True) + 1e-10)).cpu() for k, v in ola.items()}
                ola_list.append(ola)
                for k, v in ola_mask.items():
                    if isinstance(v, torch.Tensor):
                        ola_mask[k] = v.cpu()
                    elif isinstance(v, dict):
                        for kk, vv in v.items():
                            ola_mask[k][kk] = vv.cpu() if isinstance(vv, torch.Tensor) else vv
                ola_mask_list.append(ola_mask)
        tmp_model = tmp_model.cpu()
        ola_dict[tmp_model_name] = ola_list
        ola_mask_dict[tmp_model_name] = ola_mask_list
    # visualize
    for text_id in range(len(text_list)):
        for order in use_orders:
            title = f"Text: {text_list[text_id]}"
            title = title.split(" ")
            row_words = 40
            title = [" ".join(title[i*row_words:(i+1)*row_words]) for i in range(math.ceil(len(title)/row_words))]
            title = "\n".join(title)
            title += f"\nOrder: {order}"
            plt.suptitle(title)
            rows = 2
            columus = len(visual_models_name_list)
            for tmp_c in tqdm(range(columus), desc=f"Drawing text {text_id} order {order}"):
                tmp_model_name = visual_models_name_list[tmp_c]
                attn_map = ola_dict[tmp_model_name][text_id][order]
                outliers_mask = ola_mask_dict[tmp_model_name][text_id]['outliers_mask'][order]
                input_ids = model_dict[tmp_model_name].tokenizer(
                    text_list[text_id],
                    truncation=True,
                    max_length=cutoff_len,
                    padding=False,
                    return_tensors=None,
                )["input_ids"]
                tick_ls = [model_dict[tmp_model_name].tokenizer.decode(tmp_id)
                           for tmp_id in input_ids]
                # draw original attn map
                plt.subplot(rows, columus, tmp_c+1)
                sub_title = tmp_model_name
                draw_attn_heatmap(attn_map, tick_ls, tick_ls, sub_title, annot_size, label_size)
                # draw attn map without outliers
                plt.subplot(rows, columus, tmp_c+columus+1)
                sub_title = tmp_model_name + " rm_outliers"
                draw_attn_heatmap(attn_map*(1-outliers_mask), tick_ls, tick_ls, sub_title, annot_size, label_size)
            # save figure
            plt.gcf().set_size_inches(columus * 9 + 4, 22)
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"text_id_{text_id}_order_{order}.png")
            plt.savefig(save_path, dpi=450)
            plt.clf()
            print(f"Save to {save_path}")


def visualize_layer_attn_map(
    visual_models_name_list: List[str],
    use_orders: List[int],
    text_list: List[str],
    output_dir: str,
    ola_augments: Optional[List[dict]] = None,
    cutoff_len: int = 320,
    outliers_sigma_multiplier: float = 3.0,
    annot_size: int = 1,
    label_size: int = 3,
):
    layer_attn_dict = {}
    model_dict = {}
    # calculate ola
    for tmp_model_name in visual_models_name_list:
        tmp_model = OLAModel(
            base_model_name_list=[tmp_model_name,],
            adapter_architecture="textcls_resnet18",
            num_classes=2,
            use_orders=use_orders,
            remove_outliers=False,
            outliers_sigma_multiplier=outliers_sigma_multiplier,
            ola_augments=ola_augments,
        ).train().cuda()
        model_dict[tmp_model_name] = tmp_model
        layer_attn_list = []
        with torch.no_grad():
            for tmp_text in text_list:
                tmp_output = tmp_model.cal_ola_from_text([tmp_text], cutoff_len, do_vis=True)
                layer_attn = tmp_output.layer_attentions
                layer_attn = {k: v.mean(dim=1, keepdims=True).cpu() for k, v in enumerate(layer_attn)}
                layer_attn_list.append(layer_attn)
        tmp_model = tmp_model.cpu()
        layer_attn_dict[tmp_model_name] = layer_attn_list
    # calculate min num_layers
    min_num_layers = 1e10
    for _, v in layer_attn_dict.items():
        if max(list(v[0].keys())) < min_num_layers:
            min_num_layers = max(list(v[0].keys()))
    min_num_layers = int(min_num_layers + 1)
    # visualize
    for text_id in range(len(text_list)):
        for layer_id in range(min_num_layers):
            title = f"Text: {text_list[text_id]}"
            title = title.split(" ")
            row_words = 40
            title = [" ".join(title[i*row_words:(i+1)*row_words]) for i in range(math.ceil(len(title)/row_words))]
            title = "\n".join(title)
            title += f"\nLayer: {layer_id}"
            plt.suptitle(title)
            rows = 1
            columus = len(visual_models_name_list)
            for tmp_c in tqdm(range(columus), desc=f"Drawing text {text_id} layer {layer_id}"):
                tmp_model_name = visual_models_name_list[tmp_c]
                attn_map = layer_attn_dict[tmp_model_name][text_id][layer_id]
                input_ids = model_dict[tmp_model_name].tokenizer(
                    text_list[text_id],
                    truncation=True,
                    max_length=cutoff_len,
                    padding=False,
                    return_tensors=None,
                )["input_ids"]
                tick_ls = [model_dict[tmp_model_name].tokenizer.decode(tmp_id)
                           for tmp_id in input_ids]
                # draw original attn map
                plt.subplot(rows, columus, tmp_c+1)
                sub_title = tmp_model_name
                draw_attn_heatmap(attn_map, tick_ls, tick_ls, sub_title, annot_size, label_size)
            # save figure
            plt.gcf().set_size_inches(columus * 9 + 4, 22)
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"text_id_{text_id}_layer_{layer_id}.png")
            plt.savefig(save_path, dpi=450)
            plt.clf()
            print(f"Save to {save_path}")
