from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
# from ..llava_arch import  LlavaMetaForCausalLM
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from PIL import Image
from collections import OrderedDict
import math
import requests
from io import BytesIO
from functools import partial
from PIL import Image
from typing import Callable, Optional, Sequence, Tuple, List, Union
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel
from transformers.models.clip.modeling_clip import (CLIPEncoderLayer, CLIPAttention, 
                                                    CLIPVisionEmbeddings,BaseModelOutputWithPooling,
                                                    CLIPEncoder, CLIPVisionTransformer, CLIPPreTrainedModel)
from typing import List, Optional, Tuple, Union
import os

import math

from torchvision.transforms import ToTensor
import torch

#-------------------------------------------------------#
#  预处理图像
#-------------------------------------------------------#
PATCH_SIZE       = 14
PATCH_NUM_WIDTH  = 24
PATCH_NUM_HEIGHT = 24
POSITION_EMBEDDING_LENGTH = 1024
# 576
MAX_PATCHES      = PATCH_NUM_WIDTH * PATCH_NUM_HEIGHT
# 
TOKEN_LENGTH     = 3 * PATCH_SIZE * PATCH_SIZE
# 336 336
IMAGE_WIDTH      = PATCH_SIZE * PATCH_NUM_WIDTH
IMAGE_HEIGHT     = PATCH_SIZE * PATCH_NUM_HEIGHT

def torch_extract_patches(image_tensor, patch_height, patch_width):
    """
    Utiliy function to extract patches from a given image tensor. Returns a tensor of shape (1, `patch_height`,
    `patch_width`, `num_channels`x `patch_height` x `patch_width`)

    Args:
        image_tensor (torch.Tensor):
            The image tensor to extract patches from.
        patch_height (int):
            The height of the patches to extract.
        patch_width (int):
            The width of the patches to extract.
    """

    image_tensor = image_tensor.unsqueeze(0)
    patches = torch.nn.functional.unfold(image_tensor, (patch_height, patch_width), stride=(patch_height, patch_width))
    patches = patches.reshape(image_tensor.size(0), image_tensor.size(1), patch_height, patch_width, -1)
    patches = patches.permute(0, 4, 2, 3, 1).reshape(
        image_tensor.size(2) // patch_height,
        image_tensor.size(3) // patch_width,
        image_tensor.size(1) * patch_height * patch_width,
    )
    return patches.unsqueeze(0)

# 用于计算adapt需要输入图片的大小
def adapt_size(originHeight:int,originWeight:int, \
            patchHeight:int = PATCH_SIZE,patchWidth:int = PATCH_SIZE, \
            maxPatches:int = MAX_PATCHES):
    ### 用于计算adapt的图片大小
    # 参数说明 
    # originHeight:              原图高度
    # originWidth:               原图宽度
    # patchHeight:               patch高度
    # patchWidth:                patch宽度
    # maxPatches:                patch数目上限
    # 返回值说明:
    # resized_height:            插值后图片高度
    # resized_width:             插值后图片宽度
    # resized_patch_height_num:  插值后图片垂直patch数目
    # resized_patch_width_num:   插值后图片水平patch数目
    scale = math.sqrt(maxPatches * (patchHeight / originHeight) * (patchWidth / originWeight))
    resized_patch_height_num = max(min(math.floor(scale * originHeight / patchHeight), maxPatches), 1)
    resized_patch_width_num = max(min(math.floor(scale * originWeight / patchWidth), maxPatches), 1)
    resized_height = max(resized_patch_height_num * PATCH_SIZE, 1)
    resized_width = max(resized_patch_width_num * PATCH_SIZE, 1)
    return resized_height, resized_width, resized_patch_height_num, resized_patch_width_num

def cal_num_of_slices(origin_image_width, origin_image_height):
    scale = origin_image_width*origin_image_height/(IMAGE_WIDTH*IMAGE_HEIGHT)  
    scale = math.ceil(scale)
    if scale > 6:
        scale = 6
    def factorize(n):
        factors = []
        for i in range(1, n + 1):
            if n % i == 0:
                factors.append((i/(n/i), i, n // i))
        return factors
    numbers = [1, 2, 3, 4, 5, 6, 7]
    factor_dict = {}
    for num in numbers:
        factor_dict[num] = factorize(num)
    log_origin_ratio = math.log(origin_image_width/origin_image_height)
    available_ratios = []
    if scale<=2:
        available_ratios = factor_dict[scale] + factor_dict[scale + 1]
    else :
        available_ratios = factor_dict[scale-1] + factor_dict[scale]+factor_dict[scale+1]
    min_dif = 1000 
    best_w = 0
    best_h = 0
    for (r,w_slice,h_slice) in available_ratios:
        log_r = math.log(r)
        if min_dif > abs(log_r - log_origin_ratio):
            min_dif = abs(log_r - log_origin_ratio)
            best_w = w_slice
            best_h = h_slice
    
    return best_w,best_h
# 做图片切片     
def get_patch_nums(origin_image_width, origin_image_height):
    # 输入原图的尺寸
    # 返回：
    # slice_w_num 切片的w方向有多少个patch
    # slice_h_num 切片的h方向有多少个patch
    # abstract_w_num 原图的w方向有多少个patch
    # abstract_h_num 原图的h方向有多少个patch
    
    best_w, best_h = cal_num_of_slices(origin_image_width,origin_image_height)
    slice_width = origin_image_width//best_w
    slice_height = origin_image_height//best_h
    _,_,slice_h_num,slice_w_num = adapt_size(slice_height,slice_width)
    _,_,abstract_h_num,abstract_w_num = adapt_size(origin_image_height,origin_image_width)

    return slice_w_num,slice_h_num,abstract_w_num,abstract_h_num

def slice_image(image):
    
    # slice the image according to our princeple
    # return an array of slices
    
    origin_image_width  = image.size[0]
    origin_image_height = image.size[1]

    best_w, best_h = cal_num_of_slices(origin_image_width=origin_image_width,origin_image_height=origin_image_height)
    
    slices = []
    # print(best_w,best_h)
    
    for j in range(best_h):
        for i in range(best_w):
            
            box = (i * origin_image_width//best_w, j * origin_image_height//best_h, (i + 1) * origin_image_width//best_w, (j + 1) * origin_image_height//best_h)
            # print(box)
            # 切割图片
            region = image.crop(box).convert("RGB")
            # 添加到列表
            slices.append(region)
          
    return slices



def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: (H, W)
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    # tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    # print("abs_pos_shape",abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2).shape)
    # abs_pos_shape torch.Size([1, 5120, 8, 8])
    # print("after F shape",F.interpolate(
    #     abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
    #     size=(tgt_size[0], tgt_size[1]),
    #     mode="bicubic",
    #     align_corners=False,
    # ).shape)
    # after F shape torch.Size([1, 5120, 64, 5120])
    # print("tgt_size",tgt_size)
    return F.interpolate(
        abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
        size=(tgt_size[0], tgt_size[1]),
        mode="bicubic",
        align_corners=False,
    ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)


# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
            self,
            grid_size,
            embed_dim,
            num_heads,
            kv_dim=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()
        self.num_queries = grid_size ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.trainbable = True
        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, grid_size)).float()
        ).requires_grad_(False)

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query, std=.02)

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)

        self.ln_post = norm_layer(embed_dim)
        self.proj = nn.Parameter((embed_dim ** -0.5) * torch.randn(embed_dim, embed_dim))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, tgt_size=(24,24), attn_mask=None):
        pos_embed = get_abs_pos(self.pos_embed, tgt_size)
        # print("pos_embed",pos_embed.shape)
        # print("1",x.shape)
        x = self.kv_proj(x)
        # print("2",x.shape)
        x = self.ln_kv(x).permute(1, 0, 2)
        # print("3",x.shape)
        
        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(
            self._repeat(q, N) + self.pos_embed.unsqueeze(1),
            x + pos_embed.unsqueeze(1),
            x,
            attn_mask=attn_mask)[0]
        x = out.permute(1, 0, 2)

        x = self.ln_post(x)
        x = x @ self.proj
        return x

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)


#---------------------------------------------------------------------------#
# 用来生成position embedding的层
#---------------------------------------------------------------------------#   
PATCH_SIZE       = 14
PATCH_NUM_WIDTH  = 24
PATCH_NUM_HEIGHT = 24
POSITION_EMBEDDING_LENGTH = 1024
# 196
MAX_PATCHES      = PATCH_NUM_WIDTH * PATCH_NUM_HEIGHT
# 768
TOKEN_LENGTH     = 3 * PATCH_SIZE * PATCH_SIZE
# 224 224
IMAGE_WIDTH      = PATCH_SIZE * PATCH_NUM_WIDTH
IMAGE_HEIGHT     = PATCH_SIZE * PATCH_NUM_HEIGHT

class adapt_CLIPVisionEmbeddings(nn.Module):
    def get_position_embedding(self,positional_embedding, patch_width_num:int, patch_height_num:int,method = 'bicubic'):
        patch_width_num  = int(patch_width_num)
        patch_height_num  = int(patch_height_num)
        position_embedding = positional_embedding.squeeze(0)
        position_for_class = position_embedding[0, :]  
        #----------------------------------------------------#
        # 插值获得 patch_width_num * patch_height_num 的位置编码
        #----------------------------------------------------#
            #----------------------------------------------------#
            # bicubic 插值
            #----------------------------------------------------#
        if method == 'bicubic':
            position_embedding = position_embedding[1:, :].reshape((PATCH_NUM_WIDTH, PATCH_NUM_HEIGHT, POSITION_EMBEDDING_LENGTH))
            position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(1)
            original_dtype = position_embedding.dtype  # 保存原始数据类型

            # 将数据类型更改为 float32 以进行插值
            position_embedding = position_embedding.to(torch.float32)

            # 执行双三次插值
            position_embedding = torch.nn.functional.interpolate(
                position_embedding, size=(int(patch_height_num), int(patch_width_num)),
                mode='bicubic', align_corners=False)

            # 将数据类型改回原来的类型
            position_embedding = position_embedding.to(original_dtype)
            
            position_embedding = position_embedding.squeeze(1).permute(1, 2, 0).reshape(patch_height_num*patch_width_num, POSITION_EMBEDDING_LENGTH)
            #----------------------------------------------------#
            # trilinear 插值
            #----------------------------------------------------#
        elif method == 'trilinear':
            position_embedding = position_embedding[1:, :].reshape((PATCH_NUM_WIDTH, PATCH_NUM_HEIGHT, POSITION_EMBEDDING_LENGTH)).unsqueeze(0).unsqueeze(0)
            m = torch.nn.Upsample(( patch_height_num, patch_width_num, POSITION_EMBEDDING_LENGTH), mode = 'trilinear')
            position_embedding = m(position_embedding).squeeze().view(patch_width_num*patch_height_num,POSITION_EMBEDDING_LENGTH)
        
        #-----------------------#
        # 用0补全位置编码缺少的部分
        #-----------------------#
        position_embedding = torch.nn.functional.pad(position_embedding, (0, 0, 0, MAX_PATCHES-patch_height_num*patch_width_num ))
        position_embedding = position_embedding.reshape(MAX_PATCHES, POSITION_EMBEDDING_LENGTH)
        position_embedding = torch.cat((position_for_class.reshape(1,POSITION_EMBEDDING_LENGTH),position_embedding))
        return position_embedding
    
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self,
        pixel_values: torch.FloatTensor,
        w_patch_num,
        h_patch_num) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        # torch.Size([16, 577, 1024])
        processed_position_embedding = torch.cat([
            self.get_position_embedding(
                self.position_embedding(self.position_ids),
                patch_width_num=dim[0],
                patch_height_num=dim[1]
            ).unsqueeze(0) for dim in list(zip(w_patch_num, h_patch_num))
        ])
        # print("origin_image_widths",origin_image_widths)
        # print("origin_image_heights",origin_image_heights)
        # print("pos_embedding_shape",processed_position_embedding.shape)
        embeddings = embeddings + processed_position_embedding
        # for i in range(32):
        #     if w_patch_num[i]*h_patch_num[i] == 576:
        #         print(embeddings[i][w_patch_num[i]*h_patch_num[i]][0].item(),0.0,end = "|")
        #     else:
        #         print(embeddings[i][w_patch_num[i]*h_patch_num[i]][0].item(),embeddings[i][w_patch_num[i]*h_patch_num[i]+1][0].item(),end = "|")
        # print(" ",w_patch_num,h_patch_num)
        return embeddings

class adapt_CLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = adapt_CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        w_patch_num = None,
        h_patch_num = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values = pixel_values,
            w_patch_num = w_patch_num,
            h_patch_num = h_patch_num)
        
        _sums = hidden_states.sum(dim=-1)
        _attentionMask = (_sums == 0.00)
        _attentionMask = _attentionMask.float()
        _attentionMask[_attentionMask == 1] = -float('inf')
        # _attentionMask[_attentionMask == 1] = -float('inf')

        # print("image 0 tensor sum",hidden_states[0].sum(dim = -1))
        # print("hidden_states[0][576][0]",hidden_states[0][576][0].item())
        # before layer torch.Size([32, 577, 1024])
        # after layer torch.Size([32, 577, 1024])
        hidden_states = self.pre_layrnorm(hidden_states)

        # print("after layernorm",hidden_states[0].sum(dim = -1))

        
        sums = hidden_states.sum(dim=-1)
        attentionMask = (sums == -1.0000)
        # attentionMask = (sums == 0)
        attentionMask = attentionMask.float()
        attentionMask[attentionMask == 1] = -float('inf')
        
        # for i in range(32):
            
        #     print(attentionMask[i][576].item(),end = " ")
        # print(" ")
        # attentionMask[attentionMask == 1] = -float('inf')

        # print(hidden_states.shape)
        # hidden_states torch.Size([32, 577, 1024])
        
        # print("hidden_states[0][576][0].item()",hidden_states[0][576][0].item())
        # print(attentionMask.shape)
        _true = True
        for i in range(577):
            if attentionMask[0][i] != _attentionMask[0][i]:
                _true = False
        # if _true:
        #     print("This mask is correct")
        # else:
        #     print("This mask is wrong")
        #     for i in range(577):
        #         print(attentionMask[0][i],"?",_attentionMask[0][i])
        # attentionMask torch.Size([32, 577])

        # 添加一个新维度并复制
        attentionMask = attentionMask.unsqueeze(1).unsqueeze(2).repeat(1, 1, 577, 1).to(torch.bfloat16)

        
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask = attentionMask,
            causal_attention_mask = attentionMask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        last_hidden_state = encoder_outputs[0]
        # print("last_hidden_state.shape",last_hidden_state.shape)
        
        _sums = last_hidden_state.sum(dim=-1)
        # print("_sum[0][576]",_sums[0][576].item())
        pooled_output = last_hidden_state[:, 0, :]
        # print("pooled_output.shape before layer",pooled_output.shape)
        pooled_output = self.post_layernorm(pooled_output)
        

        if not return_dict:
            # print("return dict")
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # print(" not return dict ")
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class adapt_CLIPVisionModel(CLIPVisionModel):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.vision_model = adapt_CLIPVisionTransformer(config)
        self.post_init()


    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        w_patch_num = None,
        h_patch_num = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        
        if pixel_values.shape[0] == 1:

            pixel_values = pixel_values.squeeze(0)


        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            w_patch_num = w_patch_num,
            h_patch_num = h_patch_num
        )


class adapt_CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            print("Loading adapt_CLIPVisionTower from", self.vision_tower_name)
            self.load_model()
        else:
            print("Delay loading adapt_CLIPVisionTower")
            self.load_model()
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = adapt_CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def unfreeze_position_embedding(self):
        self.vision_tower.vision_model.embeddings.requires_grad_(True)
    
    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features


    @torch.no_grad()
    def forward(self, images, origin_image_widths,origin_image_heights):
        image_features = []
        
        # images is generated by stacking the image sclices and the orignal image
        split_images = torch.chunk(images, chunks=images.shape[1] // 3, dim=1) # B x [3*(#slices + 1)] x 336 x 336
        slice_w_nums=[]
        slice_h_nums=[]
        abstract_w_nums=[]
        abstract_h_nums=[]
        
        for i in range(len(origin_image_widths)):
            slice_w_num,slice_h_num,abstract_w_num,abstract_h_num = get_patch_nums(origin_image_widths[i],origin_image_heights[i])
            slice_w_nums.append(slice_w_num)
            slice_h_nums.append(slice_h_num)
            abstract_w_nums.append(abstract_w_num)
            abstract_h_nums.append(abstract_h_num)
            
        for i, image in enumerate(split_images):
            image_forward_out = self.vision_tower(
                image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                output_hidden_states=True,
                w_patch_num = abstract_w_nums if i == len(split_images) - 1 else slice_w_nums,
                h_patch_num = abstract_h_nums if i == len(split_images) - 1 else slice_h_nums)
            
            image_feature = self.feature_select(image_forward_out).to(image.dtype)
            # print("image_feature.shape",image_feature.shape)
            # image_feature.shape torch.Size([4, 576, 1024])
            # print("image_features.shape",image_features.shape)
            image_features.append(image_feature)

        return image_features

    # @torch.no_grad()
    # def forward(self, images, origin_image_widths,origin_image_heights):
    #     if images.shape[1] == 24:
    #         image_features = []
    #         split_images = torch.chunk(images, chunks=8, dim=1) # B x (3*#slices) x 336 x 336
    #         slice_w_nums=[]
    #         slice_h_nums=[]
    #         abstract_w_nums=[]
    #         abstract_h_nums=[]
            
    #         for i in range(len(origin_image_widths)):
    #             slice_w_num,slice_h_num,abstract_w_num,abstract_h_num = get_patch_nums(origin_image_widths[i],origin_image_heights[i])
    #             slice_w_nums.append(slice_w_num)
    #             slice_h_nums.append(slice_h_num)
    #             abstract_w_nums.append(abstract_w_num)
    #             abstract_h_nums.append(abstract_h_num)
                
    #         for i, image in enumerate(split_images):
                
    #             if i == 7:
    #                 image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
    #                                                   output_hidden_states=True,
    #                                                   w_patch_num = abstract_w_nums,
    #                                                   h_patch_num = abstract_h_nums)
    #             else:
    #                 image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
    #                                                   output_hidden_states=True,
    #                                                   w_patch_num = slice_w_nums,
    #                                                   h_patch_num = slice_h_nums)
                
    #             image_feature = self.feature_select(image_forward_out).to(image.dtype)
    #             # print("image_feature.shape",image_feature.shape)
    #             # image_feature.shape torch.Size([4, 576, 1024])
    #             # print("image_features.shape",image_features.shape)
    #             image_features.append(image_feature)

    #     else:
    #         image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
    #                                                     output_hidden_states=True,
    #                                                     w_patch_num = origin_image_widths,
    #                                                     h_patch_num = origin_image_heights)

    #         image_features = self.feature_select(image_forward_outs).to(images.dtype)


    #     return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


def adapt_build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        print(f"Use {vision_tower} as the vision tower")
        return adapt_CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    else:
        print(f"Cannot load {vision_tower} as the vision tower!")
    raise ValueError(f'Unknown vision tower: {vision_tower}')

import torch
import torch.nn as nn
import math

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    
    target_sequence_length = 64
    grid_size = int(math.sqrt(target_sequence_length))
    resampler = Resampler(
        grid_size=grid_size,
        embed_dim = config.hidden_size,  # 保持与视觉模型输出的 embed_dim 一致. vicuna-13b是5120，vicuna-7b和llama3-8b都是4096
        num_heads = 1024 // 128,  # 保持与视觉模型输出的 num_heads 一致
        kv_dim=1024,  # 保持与视觉模型输出的 kv_dim 一致
    )
    
    return resampler

class UILlavaMetaModel:

    def __init__(self, config):
        super(UILlavaMetaModel, self).__init__(config)

        self.vision_tower = adapt_build_vision_tower(config, delay_load=True)
        self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = adapt_build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
   
        
class UILlavaMetaForCausalLM(ABC):
    

    @abstractmethod
    def get_model(self):
        pass
    
    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def encode_images(self, images,origin_image_widths,origin_image_heights):
        
        # print("len(images)",len(images))
        # print("images[0]",images[0].shape)
        image_features = self.get_model().get_vision_tower()(images,origin_image_widths,origin_image_heights)

        # for i in range(8):
        #     print(image_features[i][0][0][0].item(),end="|")
        # print(" ")

        # print("len(image_features)",len(image_features))
        # print("image_features[0].shape",image_features[0].shape)
        # len(image_features) 8
        # image_features[0].shape torch.Size([32, 576, 1024])

        if isinstance(image_features,list):
            # print("len(image_features)",len(image_features))
            image_features_list = []
            for image_feature in image_features:
                # print(image_feature)
                # 将维度为5120的向量是否全为0的布尔掩码
                # mask = torch.all(image_feature == 0, dim=2)

                # # 打印维度为5120的向量为0的位置
                # indices = torch.nonzero(mask)

                # print("维度为5120的向量为0的位置：")
                # print(indices)

                image_features_list.append(self.get_model().mm_projector(image_feature))
            # print("image_features_list[0].shape",image_features_list[0].shape)
            image_features = torch.concat( tuple(image_features_list) ,dim = 0)
            # print("image_features.shape",image_features.shape)
            # image_features.shape torch.Size([32, 64, 5120])


        else:
            # print("image_features.shape",image_features.shape)
            image_features = self.get_model().mm_projector(image_features)

        # print("image_features.shape",image_features.shape)
        # image_features.shape torch.Size([256, 64, 5120])

        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images,origin_image_widths,origin_image_heights
    ):
        # input_ids is a list of 1D input ids
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        image_features = self.encode_images(images,origin_image_widths,origin_image_heights).to(self.device)
        # print("image_features.shape",image_features.shape)
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []

            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])

            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim)) # llama3 outputs L x 4096
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])

                if i < num_images:
                    for j in range(8):
                        cur_image_features = image_features[cur_image_idx+j*4]
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        tokenizer_model_max_length = 4096


        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]



        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

#_____MY_DEBUG____belong to llava_llama.py
class LlavaConfig(LlamaConfig):
    model_type = "ui-llava"

class UILlavaLlamaModel(UILlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(UILlavaLlamaModel, self).__init__(config)

class UILlavaLlamaForCausalLM(LlamaForCausalLM, UILlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = UILlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model


    def process_image(self, image):
        origin_image_width  = image.size[0]
        origin_image_height = image.size[1]

        image = image.convert("RGB")
        
        slices = slice_image(image)
        
        # 计算resize之后的图片大小
        resized_height, resized_width, resized_patch_height, resized_patch_width = \
        adapt_size(origin_image_height,origin_image_width)
        
        
        if len(slices) == 1:
            image = slices[0]
            image_w = image.size[0]
            image_h = image.size[1]
            resized_height, resized_width, resized_patch_height, resized_patch_width = \
            adapt_size(image_h,image_w)     
            
            image = ToTensor()(image)
        
            image = torch.nn.functional.interpolate(
                    image.unsqueeze(0),
                    size=(resized_height, resized_width),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                ).squeeze(0)
            # 需要mask的patch数
            num_patches_to_pad = MAX_PATCHES - resized_patch_height*resized_patch_width
            # raprint("mask: ",num_patches_to_pad)
            # 切割resize好的图片
            image = torch_extract_patches(image,PATCH_SIZE, PATCH_SIZE)
            image = image.reshape([resized_patch_width*resized_patch_height,TOKEN_LENGTH])
            # 用0补全需要mask的图片部分
            image = torch.nn.functional.pad(image, [0, 0, 0, num_patches_to_pad]).float()  #torch.Size([196, 768])
            image = image.reshape(PATCH_NUM_WIDTH, PATCH_NUM_HEIGHT, PATCH_SIZE, PATCH_SIZE, 3).permute(0, 2, 1, 3, 4).reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 3).permute(2, 0 ,1)
            # print(image)
            return [image]
        
        else:
            images = []
            resized_patch_widths = []
            resized_patch_heights = []
            
            # 将原图加到列表末尾
            slices.append(image)
            
            for image in slices:
                image = ToTensor()(image)
        
                image = torch.nn.functional.interpolate(
                        image.unsqueeze(0),
                        size=(resized_height, resized_width),
                        mode="bilinear",
                        align_corners=False,
                        antialias=True,
                    ).squeeze(0)
                # 需要mask的patch数
                num_patches_to_pad = MAX_PATCHES - resized_patch_height*resized_patch_width
                # raprint("mask: ",num_patches_to_pad)
                # 切割resize好的图片
                image = torch_extract_patches(image,PATCH_SIZE, PATCH_SIZE)
                image = image.reshape([resized_patch_width*resized_patch_height,TOKEN_LENGTH])
                # 用0补全需要mask的图片部分
                image = torch.nn.functional.pad(image, [0, 0, 0, num_patches_to_pad]).float()  #torch.Size([196, 768])
                image = image.reshape(PATCH_NUM_WIDTH, PATCH_NUM_HEIGHT, PATCH_SIZE, PATCH_SIZE, 3).permute(0, 2, 1, 3, 4).reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 3).permute(2, 0 ,1)
                
                # print(image)
                images.append(image)
                resized_patch_widths.append(resized_patch_width)
                resized_patch_heights.append(resized_patch_height)
            return images

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        images: Optional[Image.Image] = None,
        origin_image_widths: List[int] = None,
        origin_image_heights: List[int] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,

        return_dict: Optional[bool] = None,
        **kwargs: dict
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            origin_image_widths, origin_image_heights, image_tensors = [], [], []
            for image in images:
                origin_image_widths.append(image.size[0])
                origin_image_heights.append(image.size[1])
                
                slices_and_image = self.process_image(image) # A list of image slice tensors plus the original image, all resized to 3 x 336 x 336
                
                image_tuple = tuple(slices_and_image)
                # print(image_tuple)
                image_tensor = torch.cat(image_tuple,dim = 0)
                image_tensors.append(image_tensor)
            
            image_tensors = torch.stack(image_tensors)

            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                image_tensors,
                origin_image_widths=origin_image_widths,
                origin_image_heights=origin_image_heights,
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

AutoConfig.register("ui-llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, UILlavaLlamaForCausalLM)
