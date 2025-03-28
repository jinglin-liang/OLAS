from typing import Tuple, Optional, Union, List, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models import resnet18
from transformers import AutoConfig, AutoTokenizer, BertModel, DebertaV2Model
from transformers.modeling_outputs import ModelOutput
from transformers.data.data_collator import DataCollatorWithPadding

from models.ola_utils import get_order_level_attention, get_tandem_level_attention, get_flow_attention, get_alti_attention, get_rolloutplus_attention, cal_maskes
from models.adapters import AxialTransformerAdapter, AxialTransformerRnnAdapter, UNet, AxialTransformerReAdapter
from models.ola_augmentations import (
    RandomHightlightColumns,
    AddGuassianNoise,
    RandomTemperatureScaling
)
from utils.contributions import ModelWrapper, LMModelWrapperCaptum


def is_causal_lm(model_type: str) -> bool:
    causal_models = {"gpt2", "opt", "llama", "qwen2", "gemma2", "bloom", "mistral"}
    non_causal_models = {"bert", "roberta", "albert", "deberta-v2", "electra"}
    if model_type.lower() in causal_models:
        return True
    elif model_type.lower() in non_causal_models:
        return False
    else:
        raise ValueError(f"Model type {model_type} is not tested.")


def preprocess_ola(ola, ola_mask, remove_outliers=False, regularize=True, is_casual=False, result_type='cat'):
    assert result_type in ['cat', 'origin', 'rm_ol']
    stack_ola_tensor = torch.cat([v for _, v in ola.items()], dim=1)
    if remove_outliers:
        outliers_mask = ola_mask["outliers_mask"]
        outliers_mask_tensor = torch.cat([v for _, v in outliers_mask.items()], dim=1)
        stack_ola_tensor_wo_ol = stack_ola_tensor * (1 - outliers_mask_tensor)
        if result_type == 'cat':
            stack_ola_tensor = torch.cat([stack_ola_tensor, stack_ola_tensor_wo_ol], dim=1)
        elif result_type == 'origin':
            stack_ola_tensor = stack_ola_tensor
        elif result_type == 'rm_ol':
            stack_ola_tensor = stack_ola_tensor_wo_ol
    if regularize:
        stack_ola_tensor = stack_ola_tensor / (stack_ola_tensor.sum(dim=-1, keepdim=True) + 1e-10)
    if is_casual:
        mask_t = torch.triu(torch.ones(stack_ola_tensor.size(-1), stack_ola_tensor.size(-1)), diagonal=1).to(stack_ola_tensor)
        tensor_t = stack_ola_tensor.transpose(-1, -2) * mask_t.expand_as(stack_ola_tensor)
        stack_ola_tensor = stack_ola_tensor + tensor_t
    return stack_ola_tensor


@dataclass
class OLALMOutput(ModelOutput):
    """
    Custom model output class for OLALM.

    Args:
        logits (torch.FloatTensor): Logits output from the model.
        loss (Optional[torch.FloatTensor]): Loss value if applicable (e.g., during training).
        order_level_attention (Optional[Dict[int, torch.FloatTensor]]): 
            A dictionary containing order-level attention matrix for specific orders.
        layer_attentions (Optional[Tuple[torch.FloatTensor, ...]]): 
            Tuple of attention weights from different layers, if applicable.
    """

    logits: torch.FloatTensor = None
    loss: Optional[torch.FloatTensor] = None
    order_level_attention: Optional[Dict[int, torch.FloatTensor]] = None
    ola_maskes: Optional[Dict] = None
    layer_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class OLAModel(nn.Module):
    def __init__(
        self,
        base_model_name_list: List[str],
        adapter_architecture: str,
        num_classes: int,
        use_orders: List[int] = [1, 2, 3],
        remove_outliers: bool = False,
        local_files_only: bool = True,
        outliers_sigma_multiplier: float = 3.0,
        abandom_base_lm: bool = False,
        ola_augments: Optional[List[Dict]] = None,
        attn_type: str = "ola",
        load_method: str = "origin",
        **kwargs,
    ):
        super(OLAModel, self).__init__()
        self._init_base_model(base_model_name_list, local_files_only, abandom_base_lm, attn_type, load_method)
        self._init_ola_adaptor(adapter_architecture, num_classes, 
                               use_orders, remove_outliers, outliers_sigma_multiplier, attn_type, **kwargs)
        self._init_learnable_params()
        self._init_ola_augmentation(ola_augments)

    def _init_base_model(self, base_model_name_list, local_files_only, abandom_base_lm, attn_type, load_method):
        self.base_model_name_list = base_model_name_list
        self.all_tokenizer = {}
        is_casual_list = []
        for tmp_model_name in base_model_name_list:
            # load config
            config = AutoConfig.from_pretrained(
                tmp_model_name, 
                local_files_only=local_files_only
            )
            # is causal language model
            is_casual_list.append(is_causal_lm(config.model_type))
            # load tokenizer
            tmp_tokenizer = AutoTokenizer.from_pretrained(
                tmp_model_name, local_files_only=True)
            tmp_tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
            tmp_tokenizer.padding_side = "left"  # Allow batched inference
            self.all_tokenizer[tmp_model_name] = tmp_tokenizer
            if len(base_model_name_list) == 1 and not abandom_base_lm:
                if load_method in ["origin", "layer_disorder"]:
                    # load base model
                    if config.model_type == 'deberta-v2':
                        self.base_model = DebertaV2Model.from_pretrained(
                            tmp_model_name, config=config,
                            local_files_only=local_files_only, 
                            attn_implementation="eager"
                        )
                    elif config._name_or_path in ['pretrained_models/spanbert-base-cased', 'pretrained_models/spanbert-large-cased']:
                        self.base_model = BertModel.from_pretrained(
                            tmp_model_name, config=config,
                            local_files_only=local_files_only, 
                            attn_implementation="eager"
                        )
                    else:
                        model_class = getattr(__import__('transformers'), config.architectures[0])
                        self.base_model = model_class.from_pretrained(
                            tmp_model_name, config=config,
                            local_files_only=local_files_only, 
                            attn_implementation="eager"
                        )
                    if load_method == "layer_disorder":
                        if config.model_type == 'bert':
                            encoder_layers = self.base_model.bert.encoder.layer
                        elif config.model_type == 'roberta':
                            encoder_layers = self.base_model.roberta.encoder.layer
                        elif config.model_type == 'electra':
                            encoder_layers = self.base_model.electra.encoder.layer
                        elif config.model_type == 'qwen2':
                            encoder_layers = self.base_model.model.layers
                        elif config.model_type == 'gemma2':
                            encoder_layers = self.base_model.model.layers
                        elif config.model_type == 'llama':
                            encoder_layers = self.base_model.model.layers
                        num_layers = len(encoder_layers)
                        # 获取所有层的权重参数
                        layer_weights = []
                        for i in range(num_layers):
                            layer_params = {}
                            for name, param in encoder_layers[i].named_parameters():
                                layer_params[name] = param.data.clone() # 克隆以避免直接修改原始数据
                            layer_weights.append(layer_params)

                        import random
                        random.shuffle(layer_weights)

                        # 将打乱后的权重重新赋值给模型
                        for i in range(num_layers):
                            for name, param in encoder_layers[i].named_parameters():
                                if name in layer_weights[i]:
                                    param.data.copy_(layer_weights[i][name])
                elif load_method in ["random_all", "random_half"]:
                    model_class = getattr(__import__('transformers'), config.architectures[0])
                    self.base_model = model_class(config)
                if attn_type in ["alti", "rolloutplus"]:
                    self.wrapped_model = ModelWrapper(self.base_model)
                if attn_type in ["grad", "grad_input", "ig"]:
                    self.wrapped_model = LMModelWrapperCaptum(self.base_model)
            else:
                self.base_model = None
        assert all(is_casual_list) or all([not i for i in is_casual_list]), "All models should be the same type."
        self.is_casual = is_casual_list[0]
        self.tokenizer = self.all_tokenizer[base_model_name_list[0]]
            
    def _init_ola_adaptor(self, adapter_architecture, num_classes, use_orders, 
                          remove_outliers, outliers_sigma_multiplier, attn_type="ola", **kwargs):
        self.adapter_architecture = adapter_architecture
        self.remove_outliers = remove_outliers
        self.outliers_sigma_multiplier = outliers_sigma_multiplier
        self.use_orders = use_orders
        self.num_classes = num_classes
        if attn_type == "ola":
            ola_input_channal = len(use_orders) * (1 + int(remove_outliers))
        else:
            ola_input_channal = (1 + int(remove_outliers))
        if adapter_architecture == "textcls_resnet18":
            self.adaptor = resnet18(num_classes=num_classes)
            self.adaptor.conv1 = nn.Conv2d(
                ola_input_channal, self.adaptor.conv1.out_channels, 
                kernel_size=7, stride=2, padding=3, bias=False
            )
            nn.init.kaiming_normal_(
                self.adaptor.conv1.weight, 
                mode="fan_out", nonlinearity="relu"
            )
        elif adapter_architecture == "tokencls_axialtranformer":
            self.adaptor = AxialTransformerAdapter(ola_input_channal, num_classes, **kwargs)
        elif adapter_architecture == "tokencls_axialtranformerrnn":
            self.adaptor = AxialTransformerRnnAdapter(ola_input_channal, num_classes, **kwargs)
        elif adapter_architecture == "tokencls_unet":
            self.adaptor = UNet(ola_input_channal, num_classes, **kwargs)
        elif adapter_architecture == "re_axialtranformer":
            self.adaptor = AxialTransformerReAdapter(ola_input_channal, num_classes, **kwargs)
        else:
            raise NotImplementedError(f"Adapter architecture {adapter_architecture} is not supported.")

    def _init_learnable_params(self):
        # freeze base model's layers
        self.base_model_requires_grad_(False)
        # unfreeze prompt's parameters
        self.adapter_requires_grad_(True)

    def _init_ola_augmentation(self, ola_augments):
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

    def base_model_requires_grad_(self, requires_grad):
        if self.base_model is not None:
            for _, param in self.base_model.named_parameters():
                param.requires_grad = requires_grad

    def adapter_requires_grad_(self, requires_grad):
        for _, param in self.adaptor.named_parameters():
            param.requires_grad = requires_grad

    def save_adapter(self, path):
        torch.save(self.adaptor.state_dict(), path)

    def load_adapter(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.adaptor.load_state_dict(ckpt)
    
    @property
    def device(self):
        return next(self.adaptor.parameters()).device

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        ola: Optional[Dict[int, torch.FloatTensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_attn: Optional[bool] = None,
        task: Optional[str] = "pos",
        attn_type: str = "ola",
        output_tandem: Optional[bool] = False,
        do_vis: bool = False,
        calc_flop: bool = False,
        e1_s: Optional[torch.LongTensor] = None,
        e1_e: Optional[torch.LongTensor] = None,
        e2_s: Optional[torch.LongTensor] = None,
        e2_e: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[OLALMOutput]:
        assert attn_type in ["ola", "tandem", "first", "last", "flow", "rolloutplus", "alti", "grad", "grad_input", "ig"]
        attn = ola
        # move inputs to device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        if e1_s is not None:
            e1_s = e1_s.to(self.device)
            e1_e = e1_e.to(self.device)
            e2_s = e2_s.to(self.device)
            e2_e = e2_e.to(self.device)
        # calculate attn map
        if attn is None:
            # base model forward
            if self.base_model.config.model_type == 'deberta-v2' or self.base_model.config._name_or_path in ['pretrained_models/spanbert-base-cased', 'pretrained_models/spanbert-large-cased']:
                output = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=True,
                    output_hidden_states=False,
                    return_dict=True
                )
            else:
                output = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    labels=None,
                    output_attentions=True,
                    output_hidden_states=False,
                    return_dict=True
                )
            # get order level attention
            if attn_type == "ola":
                attn = get_order_level_attention(
                    output.attentions, 
                    attention_mask, 
                    use_orders=self.use_orders,
                )
            elif attn_type == "tandem":
                attn = get_tandem_level_attention(
                    output.attentions, 
                    attention_mask
                )
            elif attn_type == "flow":
                attn = get_flow_attention(
                    output.attentions, 
                    attention_mask,
                    input_ids
                )
            elif attn_type == "first":
                attn = {1:(output.attentions[0] * attention_mask.unsqueeze(1).unsqueeze(-1)).mean(dim=1).unsqueeze(1)}
            elif attn_type == "last":
                attn = {1:(output.attentions[-1] * attention_mask.unsqueeze(1).unsqueeze(-1)).mean(dim=1).unsqueeze(1)}
            elif attn_type == "rolloutplus":
                tmp_b = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids, 'labels': None}
                hidden_states, attentions, contributions_data = self.wrapped_model(tmp_b)
                attn = get_rolloutplus_attention(
                    attentions, contributions_data
                )
                if input_ids.shape[-1] == 1:
                    attn[1] = attn[1].unsqueeze(0).unsqueeze(0)
            elif attn_type == "alti":
                tmp_b = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids, 'labels': None}
                with torch.no_grad():
                    hidden_states, attentions, contributions_data = self.wrapped_model(tmp_b)
                attn = get_alti_attention(
                    attentions, contributions_data
                )
                if input_ids.shape[-1] == 1:
                    attn[1] = attn[1].unsqueeze(0).unsqueeze(0)
            # elif attn_type == "grad":
            #     grad_attributions = interpret_sentence(self.wrapped_model, tokenizer, input_ids, 'grad', target_idx)
        else:
            attn = {k: v.to(self.device) 
                   for k, v in attn.items() if ((attn_type != "ola") or (k in self.use_orders))}
        if calc_flop:
            return attn
        # calculate maskes from ola
        attn_mask = cal_maskes(
            attn, attention_mask, 
            is_casual=self.is_casual, 
            sigma_multiplier=self.outliers_sigma_multiplier
        )
        # augments ola
        if self.training and self.ola_augments is not None:
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
            is_casual=self.is_casual
        )
        if "textcls" in self.adapter_architecture:
            # adaptor forward
            if do_vis:
                prediction_scores = None
            else:
                prediction_scores = self.adaptor(stack_attn_tensor)
            # calculate loss
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(prediction_scores, labels)
            else:
                loss = None
        elif "tokencls" in self.adapter_architecture:
            # adaptor forward
            prediction_scores = self.adaptor(stack_attn_tensor)
            if labels is not None:
                if task != "entity":
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(prediction_scores.view(-1, self.num_classes), labels.view(-1))
                else:
                    loss_fct = nn.CrossEntropyLoss(reduction="none")
                    loss = loss_fct(prediction_scores.view(-1, self.num_classes), labels.view(-1))
                    loss_weight = torch.ones_like(labels.view(-1)) - 0.5 * (labels.view(-1) == 0)
                    loss = torch.mean(loss * loss_weight)
            else:
                loss = None
        elif "re" in self.adapter_architecture:
            # preprocess e1_s, e1_e, e2_s, e2_e
            pad_num = (attention_mask == 0).sum(-1)
            e1_e += pad_num
            e1_s += pad_num
            e2_e += pad_num
            e2_s += pad_num
            # adaptor forward
            prediction_scores = self.adaptor(stack_attn_tensor, e1_s, e1_e, e2_s, e2_e)
            # calculate loss
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(prediction_scores, labels)
            else:
                loss = None
        else:
            raise NotImplementedError(f"Adapter architecture {self.adapter_architecture} is not supported.")
        # return output
        return OLALMOutput(
            logits=prediction_scores,
            loss=loss,
            order_level_attention=attn if output_attn else None,
            ola_maskes=attn_mask if output_attn else None,
            layer_attentions=output.attentions if output_attentions else None
        )

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable: {100 * trainable_params / all_param}%"
        )

    def cal_ola_from_text(
        self, 
        text_list: List[str], 
        cutoff_len: int = 320,
        attn_type: str = 'ola',
        do_vis: bool = False
    ):
        tokenized_text_ls = [
            self.tokenizer(
                text,
                truncation=True,
                max_length=cutoff_len,
                padding=False,
                return_tensors=None,
            )
            for text in text_list
        ]
        collator_fn = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=False,
        )
        batch_input = collator_fn(tokenized_text_ls)
        if attn_type == 'ola':
            output = self(
                **batch_input, 
                output_attn=True, 
                output_attentions=True,
                do_vis=do_vis
            )
        elif attn_type == 'tandem':
            output = self(
                **batch_input, 
                output_attn=True, 
                attn_type=attn_type,
                do_vis=do_vis
            )
        return output
