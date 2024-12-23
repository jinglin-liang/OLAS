import torch
import torch.nn as nn


class RandomHightlightColumns(nn.Module):
    def __init__(
        self, 
        p: float = 0.5, 
        min_columns: int = 1, 
        max_columns: int = 6
    ):
        super(RandomHightlightColumns, self).__init__()
        self.p = p  # probability of applying this augmentation
        self.min_columns = min_columns
        self.max_columns = max_columns

    def forward(self, ola, ola_mask):
        if torch.rand(1) > self.p:
            return ola, ola_mask
        select_cols = self.select_columns(ola_mask["interested_mask"])
        if select_cols is not None:
            ola = self.heightlight_columns(ola, ola_mask["interested_mask"], select_cols)
        return ola, ola_mask
    
    def select_columns(self, interested_mask):
        num_selected_cols = torch.randint(self.min_columns, self.max_columns + 1, (1,)).item()
        weights = (interested_mask.sum(-2) > 0).float().squeeze(1)
        num_selected_cols = int(min(num_selected_cols, weights.sum(-1).min().item()))
        if num_selected_cols <= 0:
            return None
        else:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-10)
            return torch.multinomial(weights, num_selected_cols, replacement=False)
    
    def heightlight_columns(self, ola, interested_mask, select_cols):
        ret_ola = {}
        for tmp_o, tmp_ola in ola.items():
            top2_v, _ = torch.topk(tmp_ola, k=2, dim=-1)
            m1 = top2_v[..., 0].unsqueeze(-1)
            m1 = m1.expand(m1.shape[:-1] + select_cols.shape[-1:])
            m2 = top2_v[..., 1].unsqueeze(-1)
            m2 = m2.expand(m2.shape[:-1] + select_cols.shape[-1:])
            rand_f_shape = list(m2.shape)
            rand_f_shape[-2] = 1
            rand_f = torch.rand(*rand_f_shape).to(m1.device)
            sink_bias = m1 + (rand_f - 0.5) * (m1 - m2)
            sink_bias_map = torch.zeros_like(tmp_ola)
            for b_idx in range(select_cols.size(0)):
                for i, col in enumerate(select_cols[b_idx]):
                    sink_bias_map[b_idx, ..., col] = sink_bias[b_idx, ..., i]
            sink_bias_map = sink_bias_map * interested_mask
            tmp_ola = tmp_ola + sink_bias_map
            tmp_ola = tmp_ola / (tmp_ola.sum(-1, keepdim=True) + 1e-10)
            ret_ola[tmp_o] = tmp_ola
        return ret_ola


class AddGuassianNoise(nn.Module):
    def __init__(self, p: float = 0.5, std_ratio: float = 0.1):
        super(AddGuassianNoise, self).__init__()
        self.p = p
        self.std_ratio = std_ratio

    def forward(self, ola, ola_mask):
        if torch.rand(1) > self.p:
            return ola, ola_mask
        ret_ola = {}
        for tmp_o, tmp_ola in ola.items():
            # interested_mask = ola_mask["interested_mask"]
            interested_mask = ola_mask["interested_mask"] * (1 - ola_mask["outliers_mask"][tmp_o])
            noise_std = self.cal_noise_std(tmp_ola, interested_mask)
            noise = noise_std * torch.randn_like(tmp_ola)
            noise = noise * ola_mask["interested_mask"]
            ret_ola[tmp_o] = torch.relu(tmp_ola + noise) * ola_mask["interested_mask"]
            ret_ola[tmp_o] = ret_ola[tmp_o] / (ret_ola[tmp_o].sum(-1, keepdim=True) + 1e-10)
        return ret_ola, ola_mask
    
    def cal_noise_std(self, input_ola, interested_mask):
        mean = (input_ola * interested_mask).sum(-1) / (interested_mask.sum(-1) + 1e-10)
        d = ((input_ola - mean.unsqueeze(-1)).pow(2) * interested_mask).sum(-1) / (interested_mask.sum(-1) + 1e-10)
        std = torch.sqrt(d)
        return std.unsqueeze(-1) * self.std_ratio


class RandomTemperatureScaling(nn.Module):
    def __init__(self, p: float = 0.5, min_temp: float = 0.5, max_temp: float = 5):
        super(RandomTemperatureScaling, self).__init__()
        self.p = p
        self.min_temp = min_temp
        self.max_temp = max_temp

    def forward(self, ola, ola_mask):
        if torch.rand(1) > self.p:
            return ola, ola_mask
        # get temperature
        temperature = torch.rand(ola[list(ola.keys())[0]].size(0)) * 2 - 1
        temperature = temperature.to(ola[list(ola.keys())[0]].device)
        temperature[torch.where(temperature >= 0)[0]] = \
            temperature[torch.where(temperature >= 0)[0]] * (self.max_temp - 1) + 1
        temperature[torch.where(temperature < 0)[0]] = \
            temperature[torch.where(temperature < 0)[0]] * (self.min_temp - 1) + self.min_temp
        temperature = temperature[:, None, None, None].expand(ola[list(ola.keys())[0]].size())
        # scale ola
        ret_ola = {}
        for tmp_o, tmp_ola in ola.items():
            interested_mask = ola_mask["interested_mask"]
            logic = torch.log(tmp_ola + 1e-10) / temperature.expand(tmp_ola.size())
            min_dtype = torch.finfo(logic.dtype).min
            logic = logic.masked_fill(~interested_mask.bool(), min_dtype)
            ret_ola[tmp_o] = torch.softmax(logic, dim=-1) * interested_mask
        return ret_ola, ola_mask
