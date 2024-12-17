import itertools

import torch


def combine_attention_orders(attn_maps, attention_mask, use_orders):
    attn_maps = [(attn_map * attention_mask.unsqueeze(1).unsqueeze(-1)).mean(dim=1)
                 for attn_map in attn_maps]
    i_mat = torch.eye(
        attn_maps[0].size(-1)
    ).expand_as(attn_maps[0]).to(attn_maps[0])
    order_to_attn_map = {}
    for tmp_order in use_orders:
        selected_ids = list(itertools.combinations(list(range(len(attn_maps))), tmp_order))
        order_attn_map = torch.zeros_like(attn_maps[0])
        for tmp_selected_id in selected_ids:
            # order 0
            if len(tmp_selected_id) == 0:
                order_attn_map += i_mat
            # order 1
            elif len(tmp_selected_id) == 1:
                order_attn_map += attn_maps[tmp_selected_id[0]]
            # order >= 2
            else:
                tmp_selected_id = sorted(list(tmp_selected_id), reverse=True)
                tmp_selected_attn_maps = [attn_maps[i] for i in tmp_selected_id]
                multi_dot_mat = tmp_selected_attn_maps[0]
                for tmp_attn_map in tmp_selected_attn_maps[1:]:
                    multi_dot_mat = torch.matmul(multi_dot_mat, tmp_attn_map)
                order_attn_map += multi_dot_mat
        order_attn_map = order_attn_map / len(selected_ids)
        order_to_attn_map[tmp_order] = order_attn_map[:, None, :, :]
    return order_to_attn_map


def get_interested_mask(unpadding_mask, is_casual):
    if is_casual:
        causal_mask = torch.tril(torch.ones(*unpadding_mask.shape[-2:])).to(unpadding_mask)
        causal_mask = causal_mask.expand_as(unpadding_mask)
        return causal_mask * unpadding_mask
    else:
        return unpadding_mask


def get_outliers_mask(attn_map, interested_masked=None, sigma_multiplier=None):
    # get the mean and std of the attention map
    if interested_masked is None:
        interested_masked = torch.ones_like(attn_map)
    attn_map = attn_map * interested_masked
    if sigma_multiplier is not None:
        interested_num_for_each_row = interested_masked.sum(dim=-1, keepdim=True)
        mean = attn_map.sum(dim=-1, keepdim=True) / (interested_num_for_each_row + 1e-10)
        std = (
            ((attn_map - mean).pow(2) * interested_masked).sum(dim=-1, keepdim=True) / \
                torch.max(interested_num_for_each_row-1, torch.full_like(interested_num_for_each_row, 1e-10))
        ).sqrt()
        upper_bound = mean + sigma_multiplier * std
        outliers_mask = (attn_map > upper_bound)
    else:
        outliers_mask = torch.zeros_like(attn_map)
    argmax_mask = (attn_map == attn_map.max(dim=-1, keepdim=True)[0])
    outliers_mask = torch.logical_or(outliers_mask, argmax_mask).int()
    return outliers_mask.int()


def cal_maskes(attn_map, attention_mask, is_casual, sigma_multiplier=3.0):
    attention_mask = attention_mask.unsqueeze(-1)
    unpadding_mask = attention_mask.float() @ attention_mask.transpose(1, 2).float()
    unpadding_mask = unpadding_mask.unsqueeze(1)
    interested_mask = get_interested_mask(unpadding_mask, is_casual)
    outliers_mask = {}
    for k, v in attn_map.items():
        if k == 0:
            outliers_mask[k] = torch.zeros_like(v)
        else:
            outliers_mask[k] = get_outliers_mask(v, interested_mask, sigma_multiplier=sigma_multiplier)
    maskes = {
        "unpadding_mask": unpadding_mask,
        "interested_mask": interested_mask,
        "outliers_mask": outliers_mask,
    }
    return maskes

# def get_order_level_attention(attn_maps, attention_mask, is_casual, use_orders=None, sigma_multiplier=3.0):
#     if use_orders is None:
#         use_orders = [1, 2, 3]
#     order_to_attn_map = combine_attention_orders(attn_maps, attention_mask, use_orders)
#     # get unpadding mask, interested mask, outliers mask
#     maskes = cal_maskes_from_ola(order_to_attn_map, attention_mask, is_casual, sigma_multiplier)
#     return order_to_attn_map, maskes


def get_order_level_attention(attn_maps, attention_mask, use_orders=None):
    if use_orders is None:
        use_orders = [1, 2, 3]
    order_to_attn_map = combine_attention_orders(attn_maps, attention_mask, use_orders)
    return order_to_attn_map

def get_tandem_level_attention(attn_maps, attention_mask):
    attn_maps = [(attn_map * attention_mask.unsqueeze(1).unsqueeze(-1)).mean(dim=1) for attn_map in attn_maps]
    tmp_attn_map = attn_maps[0]
    tandem_attn_map = tmp_attn_map + torch.eye(tmp_attn_map.size(-1)).expand_as(tmp_attn_map).to(tmp_attn_map)
    for tmp_attn_map in attn_maps[1:]:
        tandem_attn_map = torch.matmul(tmp_attn_map, tandem_attn_map) + tandem_attn_map
    tandem_to_attn_map = {1:tandem_attn_map.unsqueeze(1)}
    return tandem_to_attn_map