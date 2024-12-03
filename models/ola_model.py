from typing import Tuple, Optional, Union, List, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models import resnet18
from transformers import AutoConfig, AutoTokenizer
from transformers.modeling_outputs import ModelOutput
from transformers.data.data_collator import DataCollatorWithPadding

from models.ola_utils import get_order_level_attention, cal_maskes_from_ola
from models.adapters import AxialTransformerAdapter


def is_causal_lm(model_type: str) -> bool:
    causal_models = {"gpt2", "opt", "llama", "qwen2", "gemma2"}
    non_causal_models = {"bert", "roberta", "albert"}
    if model_type.lower() in causal_models:
        return True
    elif model_type.lower() in non_causal_models:
        return False
    else:
        raise ValueError(f"Model type {model_type} is not tested.")


def preprocess_ola(ola, ola_mask, remove_outliers=False, regularize=True, is_casual=False):
    stack_ola_tensor = torch.cat([v for _, v in ola.items()], dim=1)
    if remove_outliers:
        outliers_mask = ola_mask["outliers_mask"]
        outliers_mask_tensor = torch.cat([v for _, v in outliers_mask.items()], dim=1)
        stack_ola_tensor_wo_ol = stack_ola_tensor * (1 - outliers_mask_tensor)
        stack_ola_tensor = torch.cat([stack_ola_tensor, stack_ola_tensor_wo_ol], dim=1)
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
        **kwargs,
    ):
        super(OLAModel, self).__init__()
        self._init_base_model(base_model_name_list, local_files_only)
        self._init_ola_adaptor(adapter_architecture, num_classes, 
                               use_orders, remove_outliers, outliers_sigma_multiplier)
        self._init_learnable_params()

    def _init_base_model(self, base_model_name_list, local_files_only):
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
            if len(base_model_name_list) == 1:
                # load base model
                model_class = getattr(__import__('transformers'), config.architectures[0])
                self.base_model = model_class.from_pretrained(
                    tmp_model_name, config=config,
                    local_files_only=local_files_only, 
                    attn_implementation="eager"
                )
            else:
                self.base_model = None
        assert all(is_casual_list) or all([not i for i in is_casual_list]), "All models should be the same type."
        self.is_casual = is_casual_list[0]
        self.tokenizer = self.all_tokenizer[base_model_name_list[0]]
            
    def _init_ola_adaptor(self, adapter_architecture, num_classes, use_orders, 
                          remove_outliers, outliers_sigma_multiplier):
        self.adapter_architecture = adapter_architecture
        self.remove_outliers = remove_outliers
        self.outliers_sigma_multiplier = outliers_sigma_multiplier
        self.use_orders = use_orders
        self.num_classes = num_classes
        ola_input_channal = len(use_orders) * (1 + int(remove_outliers))
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
            self.adaptor = AxialTransformerAdapter(ola_input_channal, num_classes)
        else:
            raise NotImplementedError(f"Adapter architecture {adapter_architecture} is not supported.")

    def _init_learnable_params(self):
        # freeze base model's layers
        self.base_model_requires_grad_(False)
        # unfreeze prompt's parameters
        self.adapter_requires_grad_(True)

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
        output_ola: Optional[bool] = None,
        **kwargs,
    ) -> Union[OLALMOutput]:
        # move inputs to device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        # calculate ola
        if ola is None:
            # base model forward
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
            ola = get_order_level_attention(
                output.attentions, 
                attention_mask, 
                use_orders=self.use_orders,
            )
        else:
            ola = {k: v.to(self.device) 
                   for k, v in ola.items() if k in self.use_orders}
        # calculate maskes from ola
        ola_mask = cal_maskes_from_ola(
            ola, attention_mask, 
            is_casual=self.is_casual, 
            sigma_multiplier=self.outliers_sigma_multiplier
        )
        # preprocess ola
        stack_ola_tensor = preprocess_ola(
            ola, ola_mask, 
            remove_outliers=self.remove_outliers, 
            is_casual=self.is_casual
        )
        if "textcls" in self.adapter_architecture:
            # adaptor forward
            prediction_scores = self.adaptor(stack_ola_tensor)
            # calculate loss
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(prediction_scores, labels)
            else:
                loss = None
        elif "tokencls" in self.adapter_architecture:
            # adaptor forward
            prediction_scores = self.adaptor(stack_ola_tensor)
            idx_tensor = torch.arange(prediction_scores.size(-1)).to(prediction_scores.device)
            prediction_scores = prediction_scores[:, :, idx_tensor, idx_tensor]
            prediction_scores = prediction_scores.transpose(1, 2).contiguous()
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(prediction_scores.view(-1, self.num_classes), labels.view(-1))
            else:
                loss = None
        else:
            raise NotImplementedError(f"Adapter architecture {self.adapter_architecture} is not supported.")
        # return output
        return OLALMOutput(
            logits=prediction_scores,
            loss=loss,
            order_level_attention=ola if output_ola else None,
            ola_maskes=ola_mask if output_ola else None,
            layer_attentions=output.attentions if output_attentions else None,
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
        cutoff_len: int = 320
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
        output = self(**batch_input, output_ola=True)
        output_ola = output.order_level_attention
        output_ola_mask = output.ola_maskes
        return output_ola, output_ola_mask
