import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
from config import configs
import random

class channel_embedding(nn.Module):
    def __init__(self, kernal=1, d_model=256, max_len=3, device=torch.device('cuda:0')):
        super().__init__()
        channel_pos = torch.arange(max_len)[None, None, :, None].to(device)
        self.register_buffer('channel_pos', channel_pos)
        self.emb_channel = nn.Embedding(max_len, d_model)
        self.linear = nn.Linear(kernal, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        [B, T_56, C, N]
        [B, T_56, C, N, D]
        """
        assert len(x.size()) == 4
        x = x[:,:,:,:,None]
        embedded_channel = self.emb_channel(self.channel_pos)
        x = self.linear(x) + embedded_channel
        return self.norm(x)
class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, nheads, attn, dropout):
        super().__init__()
        assert d_model % nheads == 0
        self.d_k = d_model // nheads
        self.nheads = nheads
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout)
        self.attn = attn

    def forward(self, query, key, value, mask=None):
        """
         [B, T_56, C, N, D]
         [B, T_56, C, N, D]
        """
        nbatches = query.size(0)
        ntime = query.size(1)
        nchannel = query.size(2)
        nnode = query.size(3)
        # [B, h, T_56, C, N, d_k]
        query, key, value = \
            [l(x).view(x.size(0), x.size(1), x.size(2), x.size(3), self.nheads, self.d_k).permute(0, 4, 1, 2, 3, 5)
             for l, x in zip(self.linears, (query, key, value))]

        # [B, h, T_56, C, N, d_k]
        x = self.attn(query, key, value, dropout=self.dropout)

        # [B, h, T_56, C, N, d_k]
        x = x.permute(0, 2, 3, 4, 1, 5).contiguous() \
             .view(nbatches, ntime, nchannel, nnode, self.nheads * self.d_k)
        return self.linears[-1](x)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))

def TimeAttention(query, key, value, dropout=None):
    """
    [B, h, T_56, C, N, d_k]
    [B, h, T_56, C, N, d_k]
    """
    d_k = query.size(-1)
    query = query.transpose(2, 4)
    key = key.transpose(2, 4)
    value = value.transpose(2, 4)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value).transpose(2, 4)

class AttentionLayer(nn.Module):
    def __init__(self, dim_feedforward=512,d_model=256, nheads=4, attn=TimeAttention, dropout=0.2):
        super().__init__()
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.time_attn = MultiHeadedAttention(d_model, nheads, attn, dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
            )

    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda x: self.time_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)





# class input_embedding(nn.Module):
#     def __init__(self, kernal=1, d_model=256, max_len=20, device=torch.device('cuda:1')):
#         super().__init__()
#
#         self.time_pos = torch.arange(max_len)[None, :, None].to(device)
#         self.emb_time = nn.Embedding(max_len, d_model)
#
#
#         self.linear = nn.Linear(kernal, d_model)
#         self.norm = nn.LayerNorm(d_model)
#
#     def forward(self, x):
#         """
#         Add temporal encoding and learnable spatial embedding to the input (after patch)
#         Args:
#             input x of size [Batch, Output length, Variables]
#         Returns:
#             embedded input [Batch, Output length, Variables]
#         """
#         assert len(x.size()) == 3
#         x = x[:,:,:,None]
#         embedded_time = self.emb_time(self.time_pos)
#         x = self.linear(x) + embedded_time
#         return self.norm(x)
#
# class MultiHeadedAttention(nn.Module):
#     def __init__(self, d_model, nheads, attn, dropout):
#         super().__init__()
#         assert d_model % nheads == 0
#         self.d_k = d_model // nheads
#         self.nheads = nheads
#         self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
#         self.dropout = nn.Dropout(p=dropout)
#         self.attn = attn
#
#     def forward(self, query, key, value, mask=None):
#         """
#         Transform the query, key, value into different heads, then apply the attention in parallel
#         Args:
#             query, key, value: size [Batch, Output length, Variables][B,L,N,D]
#         Returns:
#             [Batch, Output length, Variables][B,L,N,D]
#         """
#         nbatches = query.size(0)
#         ntime = query.size(1)
#         nnode = query.size(2)
#         # (N, h, L, N, d_k)
#         query, key, value = \
#             [l(x).view(x.size(0), x.size(1), x.size(2), self.nheads, self.d_k).permute(0, 3, 1, 2, 4)
#              for l, x in zip(self.linears, (query, key, value))]
#
#         # (N, h, L, N, d_k)
#         x = self.attn(query, key, value, dropout=self.dropout)
#
#         # (N, S, C, T, D)
#         x = x.permute(0, 2, 3, 1, 4).contiguous() \
#              .view(nbatches, ntime, nnode, self.nheads * self.d_k)
#         return self.linears[-1](x)
#
# def TimeAttention(query, key, value, dropout=None):
#     """
#     attention over the two space axes
#     Args:
#         query, key, value: linearly-transformed query, key, value # (N, h, L, N, d_k)
#     """
#     d_k = query.size(-1)
#     query = query.transpose(2, 3)  # (N, h, N, L, d_k)
#     key = key.transpose(2, 3)  # (N, h, N, L, d_k)
#     value = value.transpose(2, 3)  # (N, h, N, L, d_k)
#     scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)  # (N, h, N, L, L)
#     p_attn = F.softmax(scores, dim=-1)
#     if dropout is not None:
#         p_attn = dropout(p_attn)
#     return torch.matmul(p_attn, value).transpose(2, 3)  # (N, h, L, N, d_k)
#
# def clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
#
# class SublayerConnection(nn.Module):
#     def __init__(self, size, dropout):
#         super().__init__()
#         self.norm = nn.LayerNorm(size)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, sublayer):
#         return self.norm(x + self.dropout(sublayer(x)))
#
# class AttentionLayer(nn.Module):
#     def __init__(self, dim_feedforward=512,d_model=256, nheads=4, attn=TimeAttention, dropout=0.2):
#         super().__init__()
#         self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
#         self.time_attn = MultiHeadedAttention(d_model, nheads, attn, dropout)
#
#         self.feed_forward = nn.Sequential(
#             nn.Linear(d_model, dim_feedforward),
#             nn.ReLU(),
#             nn.Linear(dim_feedforward, d_model)
#             )
#
#
#     def forward(self, x, mask=None):
#         x = self.sublayer[0](x, lambda x: self.time_attn(x, x, x))
#         return self.sublayer[1](x, self.feed_forward)