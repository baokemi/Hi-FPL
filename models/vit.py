# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin
from functools import reduce
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm,Parameter
from torch.nn.modules.utils import _pair
from scipy import ndimage
from models import configs
import torch.nn.functional as F
from .modeling_resnet import ResNetV2
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
import copy
import timm.models.vision_transformer
import torchvision.transforms as transforms
from torchvision import datasets
from utils import *

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def mul(a, b):
    "Same as a * b."
    return a * b

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention_pre(nn.Module):
    def __init__(self, config, vis,g_prompts):
        super(Attention_pre, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.g_prompts = g_prompts
        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)
        self.out = Linear(config.hidden_size, config.hidden_size)

        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Attention(nn.Module):
    def __init__(self, config, vis,g_prompts):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.g_prompts = g_prompts
        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)
        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        if self.g_prompts is not None:
            B = hidden_states.shape[0]
            ## prefix for  key and value respectively
            prompts =  self.g_prompts.expand(B, -1,2, -1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(torch.cat([hidden_states,prompts[:,:,0,:]],dim=1))
            mixed_value_layer = self.value(torch.cat([hidden_states,prompts[:,:,1,:]],dim=1))
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Block_pre(nn.Module):
    def __init__(self, config, vis,g_prompts= None):
        super(Block_pre, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention_pre(config, vis,g_prompts)
    
    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))

class Block(nn.Module):
    def __init__(self, config, vis,g_prompts= None):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis,g_prompts)
    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))

class Encoder_pre(nn.Module):
    def __init__(self, config, vis,prompt_dict):
        super(Encoder_pre, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for i in range(config.transformer["num_layers"]):
            if i in prompt_dict:
                layer = Block_pre(config, vis,prompt_dict[i])
            else:
                layer = Block_pre(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights
    
class Encoder(nn.Module):
    def __init__(self, config, vis,prompt_dict):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for i in range(config.transformer["num_layers"]):
            if i in prompt_dict:
                layer = Block(config, vis,prompt_dict[i])
            else:
                layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class Transformer_pre(nn.Module):
    def __init__(self, config, img_size, vis,prompt_dict = {}):
        super(Transformer_pre, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder_pre(config, vis,prompt_dict)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights

class Transformer(nn.Module):
    def __init__(self, config, img_size, vis,prompt_dict = {}):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis,prompt_dict)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)
    
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_cross_attention_heads: int = 12,
        attention_dropout: float = 0,
        ffn_dropout: float = 0
    ) -> None:
        super().__init__()

        self.pre_norm1 = nn.LayerNorm(embed_dim)
        self.pre_norm2 = nn.LayerNorm(embed_dim)
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_cross_attention_heads,
            dropout=attention_dropout,
        )
        self.pre_norm3 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, dropout=ffn_dropout)
        

    def forward(self, x, context):
        B, L, N = x.shape
        x = self.cross_attention(query=x, key=context, value=context)[0]
        x = self.pre_norm3(x)
        x = self.ffn(x) + x
        return x

class CrossAttention(nn.Module):
    def __init__(
        self,
        args,
        embed_dim: int = 768,
        num_cross_attention_heads: int = 1,
        attention_dropout: float = 0,
        ffn_dropout: float = 0
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        self.args = args
        for i in range(args.cross_depth):
            self.blocks.append(
                CrossAttentionBlock(
                    embed_dim=embed_dim,
                    num_cross_attention_heads=num_cross_attention_heads,
                    attention_dropout=attention_dropout,
                    ffn_dropout=ffn_dropout,
                )
            )
        self.knowledge_bank = nn.Parameter(torch.randn(int(args.n_parties*len(args.share_blocks_g)),args.n_prompt, embed_dim))
        self.new_prompt = torch.zeros_like(self.knowledge_bank)
    def forward(self, x):
        
        for block in self.blocks:
            print("x",x.shape)
            print("knowledge_bank",self.knowledge_bank.shape)
            x = block(x,self.knowledge_bank.data)
        
        return x
def prompt_generator(model, x, args,sample_size=None):
        B = x.shape[0]
        batch_imgs = x
        batch_imgs = batch_imgs.to(args.device)
        model.eval()
        model.pre_generate = True
        assert model.prompt_sample_type in ["random", "mean_pooling", "max_pooling", "kmeans"] 
        length = 12+1
        catcher = RepCatcher(model, list(range(length)))
        with torch.no_grad():
            pred = model(batch_imgs)
            x = catcher.get_features()
        
        if model.prompt_sample_type == "random":
            prompt = []
            for i in range(1,length):
                x_ = torch.reshape(x[i], shape=(-1, x[i].shape[-1]))
                L, D = x_.shape
                noise = torch.rand(L, requires_grad=False) 
                ids_shuffle = torch.argsort(noise, dim=0)
                ids_keep = ids_shuffle[:model.n_prompt]
                
                ids_keep = ids_keep.to(x_.device)  
                prompt_ = torch.gather(x_, dim=0, index=ids_keep.unsqueeze(-1).repeat(1, D)).cpu().numpy()
                prompt.append(prompt_)
            prompt = torch.as_tensor(np.array(prompt))
            print("prompt",prompt.shape)

            
        elif model.prompt_sample_type == "mean_pooling":
            prompt = []
            for i in range(1,length):
                pro = []
                B, L, D = x[i].shape
                window_size = L // (model.n_prompt - 1)
                remain_size = L % window_size
                for p_i in range(model.n_prompt-1):
                    x_ = x[i][:, p_i*window_size: (p_i+1)*window_size, :]
                    pro.append(x_.mean(dim=1).cpu().numpy())
                
                pro.append((x[i][:, -remain_size:, :]).mean(dim=1).cpu().numpy())
                x_ = torch.as_tensor(np.array(pro))
                assert x_.size() == (model.n_prompt, B, D)
                x_ = torch.reshape(x_, shape=(-1, D))
                L, D = x_.shape
                noise = torch.rand(L, requires_grad=False)
                ids_shuffle = torch.argsort(noise, dim=0)
                ids_keep = ids_shuffle[:model.n_prompt]
                prompt_ = torch.gather(x_, dim=0, index=ids_keep.unsqueeze(-1).repeat(1, D)).cpu().numpy()
                prompt.append(prompt_)
            prompt = torch.as_tensor(np.array(prompt))
            print("prompt",prompt.shape)

        elif model.prompt_sample_type == "max_pooling":
            prompt = []
            for i in range(1,length):
                pro = []
                B, L, D = x[i].shape
                window_size = L // (model.n_prompt - 1)
                remain_size = L % window_size
                for p_i in range(model.n_prompt-1):
                    x_ = x[i][:, p_i*window_size: (p_i+1)*window_size, :]
                    pro.append(torch.max(x_, dim=1).values.cpu().numpy())
                pro.append(torch.max(x[i][:, -remain_size:, :], dim=1).values.cpu().numpy())
                x_ = torch.as_tensor(np.array(pro))
                assert x_.size() == (model.n_prompt, B, D)
                x_ = torch.reshape(x_, shape=(-1, D))
                L, D = x_.shape
                noise = torch.rand(L, requires_grad=False)
                ids_shuffle = torch.argsort(noise, dim=0)
                ids_keep = ids_shuffle[:model.n_prompt]
                prompt_ = torch.gather(x_, dim=0, index=ids_keep.unsqueeze(-1).repeat(1, D)).cpu().numpy()
                prompt.append(prompt_)
            prompt = torch.as_tensor(np.array(prompt))
            print("prompt",prompt.shape)
        for i in model.share_blocks:
            model.prompt_common[model.share_blocks.index(i)].data.copy_(prompt[i])
        for j in model.share_blocks_g:
            model.prompt_uncommon[model.share_blocks_g.index(j)].data.copy_(prompt[j])
        model.pre_generate = False

class VisionTransformer_prompt(nn.Module):
    def __init__(self, config,img_size=224, num_classes=21843, vis=False, args = None):
        super(VisionTransformer_prompt, self).__init__()
        self.num_classes = num_classes
        self.classifier = config.classifier
        self.args =args
        self.share_blocks = args.share_blocks
        self.share_blocks_g = args.share_blocks_g
        self.hidden_size = config.hidden_size   

        self.prompt_sample_type = args.prompt_sample_type
        self.n_prompt = args.n_prompt
        self.config = config
        self.transformer = Transformer(config, img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)
        self.pre_generate = True
        if len(self.share_blocks) > 0:
            self.prompt_common = nn.ParameterList([nn.Parameter(torch.zeros(
                self.n_prompt, config.hidden_size)) for _ in range(len(self.share_blocks))])
        if len(self.share_blocks_g) > 0:
            self.prompt_uncommon = nn.ParameterList([nn.Parameter(torch.zeros(self.n_prompt, self.hidden_size)) for _ in range(len(self.share_blocks_g))])

        self.cross_blocks = CrossAttention(
            embed_dim=self.hidden_size,
            num_cross_attention_heads=self.args.num_cross_attention_heads,
            attention_dropout=self.args.attention_dropout,
            ffn_dropout=self.args.ffn_dropout,
            args = self.args
            )
    def load_prompt(self,prompt_commoning,prompt_uncommoning):
        for i in range(len(self.share_blocks)):
            self.prompt_common[i].data.copy_(prompt_commoning[i])
        for j in range(len(self.share_blocks_g)):
            self.prompt_uncommon[j].data.copy_(prompt_uncommoning[j])

    def extract_prompt(self):    
        prompt_commoning = []
        prompt_uncommoning = []
        for i in range(len(self.share_blocks)):
            prompt_commoning.append(self.prompt_common[i].data.cpu().numpy())
        for j in range(len(self.share_blocks_g)):
            prompt_uncommoning.append(self.prompt_uncommon[j].data.cpu().numpy())
        prompt_commoning = torch.as_tensor(np.array(prompt_commoning))
        prompt_uncommoning = torch.as_tensor(np.array(prompt_uncommoning))
        return prompt_commoning,prompt_uncommoning
    def forward(self, x,indexes=None,embedding_dict=None):
        B = x.shape[0]
        x = self.transformer.embeddings(x)
        output_dict ={}
        if self.pre_generate:
            for i,layer_block in enumerate(self.transformer.encoder.layer):
                x= layer_block(x)
        else:
            for i,layer_block in enumerate(self.transformer.encoder.layer):
                if i in self.share_blocks:
                    prompt_tokens = self.prompt_common[self.share_blocks.index(i)].unsqueeze(0).expand(B, -1, -1).to(x.device)
                    x=  torch.cat((
                                x[:, :self.args.n_prompt, :],
                                prompt_tokens,
                                x[:, self.args.n_prompt:, :]
                            ), dim=1)
                if i in self.share_blocks_g:
                    prompt_tokens = self.prompt_uncommon[self.share_blocks_g.index(i)].unsqueeze(0).expand(B, -1, -1).to(x.device)
                    x=  torch.cat((
                                x[:, :self.args.n_prompt, :],
                                prompt_tokens,
                                x[:, self.args.n_prompt:, :]
                            ), dim=1)
                x = layer_block(x)
        x = self.transformer.encoder.encoder_norm(x)
        cls_token = x[:,:len(self.share_blocks_g)+1].mean(1)
        logits = self.head(cls_token)
        output_dict['logits']  = logits

        return output_dict 
    def freeze(self):
        for k, p in self.transformer.named_parameters():
            if "prompt" not in k :
                p.requires_grad = False
    def train(self,mode=True):
        self.training = mode
        if mode:
            self.transformer.encoder.eval()
            self.transformer.embeddings.eval()
        else:
            for module in self.children():
                module.train(mode)
    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)
CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}
