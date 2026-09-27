import contextlib
import os

import torch
import torch.nn as nn
from transformers import BertTokenizer

from backends.bevdriver.modeling.base_model import BaseModel
from backends.bevdriver.modeling.qformer import BertConfig, BertLMHeadModel


class BEVLLMBase(BaseModel):
    @classmethod
    def init_tokenizer(cls, bert_model="bert-base-uncased", truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained(bert_model, truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        if self.device == torch.device("cpu"):
            return contextlib.nullcontext()
        return torch.cuda.amp.autocast(dtype=dtype)

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2, bert_model="bert-base-uncased"):
        encoder_config = BertConfig.from_pretrained(bert_model)
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        qformer = BertLMHeadModel.from_pretrained(bert_model, config=encoder_config)
        query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return qformer, query_tokens

    def load_from_pretrained(self, url_or_filename):
        if not os.path.isfile(url_or_filename):
            raise RuntimeError(f"checkpoint path is invalid: {url_or_filename}")
        checkpoint = torch.load(url_or_filename, map_location="cpu")
        state_dict = checkpoint["model"]
        return self.load_state_dict(state_dict, strict=False)


def disabled_train(self, mode=True):
    return self
