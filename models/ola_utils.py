import itertools
import numpy as np
import networkx as nx

import torch

from utils.utils_contributions import normalize_contributions, compute_joint_attention

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

def get_flow_attention(attn_maps, attention_mask, input_tokens, layer=1, output_index=1):
    attn_maps = [(attn_map * attention_mask.unsqueeze(1).unsqueeze(-1)).mean(dim=1) for attn_map in attn_maps]
    attn_device = attn_maps[0].device
    full_att_mat = torch.cat(attn_maps).cpu().numpy()
    
    input_tokens = input_tokens
    res_att_mat = full_att_mat + np.eye(full_att_mat.shape[1])[None,...]
    res_att_mat = res_att_mat / res_att_mat.sum(axis=-1)[...,None]

    res_adj_mat, res_labels_to_index = get_adjmat(mat=res_att_mat, input_tokens=input_tokens)
    
    A = res_adj_mat
    res_G=nx.from_numpy_array(A, create_using=nx.DiGraph())
    for i in np.arange(A.shape[0]):
        for j in np.arange(A.shape[1]):
            nx.set_edge_attributes(res_G, {(i,j): A[i,j]}, 'capacity')

    layer = full_att_mat.shape[0] - 1
    output_nodes = ['L'+str(layer+1)+'_'+str(idx) for idx in range(full_att_mat.shape[-1])]
    input_nodes = []
    for key in res_labels_to_index:
        if res_labels_to_index[key] < full_att_mat.shape[-1]:
            input_nodes.append(key)
    
    flow_values = compute_node_flow(res_G, res_labels_to_index, input_nodes=input_nodes, output_nodes=output_nodes, length=full_att_mat.shape[-1])
    
    n_layers = full_att_mat.shape[0]
    length = full_att_mat.shape[-1]
    final_layer_attention = flow_values[(layer+1)*length:, layer*length:(layer+1)*length]
    flow_to_attn_map = {1:torch.tensor(final_layer_attention).unsqueeze(0).unsqueeze(0).float().to(attn_device)}

    return flow_to_attn_map

def get_adjmat(mat, input_tokens):
    n_layers, length, _ = mat.shape
    adj_mat = np.zeros(((n_layers+1)*length, (n_layers+1)*length))
    labels_to_index = {}
    for k in np.arange(length):
        labels_to_index[str(k)+"_"+str(input_tokens[0][k].item())] = k

    for i in np.arange(1,n_layers+1):
        for k_f in np.arange(length):
            index_from = (i)*length+k_f
            label = "L"+str(i)+"_"+str(k_f)
            labels_to_index[label] = index_from
            for k_t in np.arange(length):
                index_to = (i-1)*length+k_t
                adj_mat[index_from][index_to] = mat[i-1][k_f][k_t]
                
    return adj_mat, labels_to_index 

def compute_node_flow(G, labels_to_index, input_nodes, output_nodes,length):
    number_of_nodes = len(labels_to_index)
    flow_values=np.zeros((number_of_nodes,number_of_nodes))
    for key in output_nodes:
        if key not in input_nodes:
            current_layer = int(labels_to_index[key] / length)
            pre_layer = current_layer - 1
            u = labels_to_index[key]
            for inp_node_key in input_nodes:
                v = labels_to_index[inp_node_key]
                flow_value = nx.maximum_flow_value(G,u,v, flow_func=nx.algorithms.flow.edmonds_karp)
                flow_values[u][pre_layer*length+v ] = flow_value
            flow_values[u] /= flow_values[u].sum()
            
    return flow_values

def get_rolloutplus_attention(attentions, contributions_data):
    attn_device = attentions[0].device
    normalized_model_norms = normalize_contributions(contributions_data['transformed_vectors_norm'],scaling='sum_one')
    norms_mix = compute_joint_attention(normalized_model_norms)
    rolloutplus_to_attn_map = {1:norms_mix[-1].unsqueeze(0).unsqueeze(0).float().to(attn_device)}
    
    return rolloutplus_to_attn_map

def get_alti_attention(attentions, contributions_data):
    attn_device = attentions[0].device
    _attentions = [att.detach().cpu().numpy() for att in attentions]
    attentions_mat = np.asarray(_attentions)[:,0] # (num_layers,num_heads,src_len,src_len)
    # att_mat_sum_heads = attentions_mat.sum(axis=1)/attentions_mat.shape[1]
    # normalized_model_norms = normalize_contributions(contributions_data['transformed_vectors_norm'],scaling='sum_one')
    resultant_norm = resultants_norm = torch.norm(torch.squeeze(contributions_data['resultants']),p=1,dim=-1)
    # ALTI Requires scaling = min_sum
    normalized_contributions = normalize_contributions(contributions_data['contributions'],scaling='min_sum',resultant_norm=resultant_norm)
    contributions_mix = compute_joint_attention(normalized_contributions)
    alti_to_attn_map = {1:contributions_mix[-1].unsqueeze(0).unsqueeze(0).float().to(attn_device)}
    
    return alti_to_attn_map