import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from model.attention import channel_embedding, AttentionLayer


def FFT_for_Period_initdata(x, k=4):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = float('-inf')
    top_list_final = []
    for i in range(x.shape[2]):
        _, top_list = torch.topk(frequency_list[:, i], k)
        top_list_final.append(top_list)
    period_final = []
    for j in range(x.shape[2]):
        period = [1]
        for top in top_list_final[j]:
            period = np.concatenate((period, [math.ceil(x.shape[1] / top)]))
        period_final.append(period)
    period_final = torch.tensor(np.array(period_final))
    return period_final


class moving_avg_initdata(nn.Module):
    def __init__(self, kernel_size):
        super(moving_avg_initdata, self).__init__()
        self.kernel_size = kernel_size.tolist()
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=self.kernel_size, padding=0)

    def forward(self, x):
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class moving_max_initdata(nn.Module):
    def __init__(self, kernel_size):
        super(moving_max_initdata, self).__init__()
        self.kernel_size = kernel_size.tolist()
        self.max = nn.MaxPool1d(kernel_size=self.kernel_size, stride=self.kernel_size, padding=0)

    def forward(self, x):
        x = self.max(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class multi_scale_initdata(nn.Module):
    def __init__(self, kernel_size_metix, return_len):
        super(multi_scale_initdata, self).__init__()
        self.kernel_size_metix = kernel_size_metix
        self.max_len = return_len
        self.moving_avg, self.moving_max = [], []
        for i in range(3):
            self.moving_avg.append([moving_avg_initdata(kernel) for kernel in list(kernel_size_metix[i])])
            self.moving_max.append([moving_max_initdata(kernel) for kernel in list(kernel_size_metix[i])])
        self.w = nn.Parameter(torch.rand(1, return_len, 1, 1), requires_grad=True)

    def forward(self, x):
        for i in range(x.shape[2]):
            different_scale_x_avg = []
            for func in self.moving_avg[i]:
                moving_avg = func(x[:, :, i])
                different_scale_x_avg.append(moving_avg)
            multi_scale_x_avg = torch.cat(different_scale_x_avg, dim=1)
            if multi_scale_x_avg.shape[1] < self.max_len:  # padding
                padding = torch.zeros([multi_scale_x_avg.shape[0], (self.max_len - (multi_scale_x_avg.shape[1])),
                                       multi_scale_x_avg.shape[2]]).to(x.device)
                multi_scale_x_avg = torch.cat([multi_scale_x_avg, padding], dim=1)
            elif multi_scale_x_avg.shape[1] > self.max_len:  # trunc
                multi_scale_x_avg = multi_scale_x_avg[:, :self.max_len, :]
            multi_scale_x_avg_final = multi_scale_x_avg[:, :, None] if i == 0 else torch.cat(
                (multi_scale_x_avg_final, multi_scale_x_avg[:, :, None]), dim=2)

        for i in range(x.shape[2]):
            different_scale_x_max = []
            for func in self.moving_max[i]:
                moving_max = func(x[:, :, i])
                different_scale_x_max.append(moving_max)
            multi_scale_x_max = torch.cat(different_scale_x_max, dim=1)
            if multi_scale_x_max.shape[1] < self.max_len:  # padding
                padding = torch.zeros([multi_scale_x_max.shape[0], (self.max_len - (multi_scale_x_max.shape[1])),
                                       multi_scale_x_max.shape[2]]).to(x.device)
                multi_scale_x_max = torch.cat([multi_scale_x_max, padding], dim=1)
            elif multi_scale_x_max.shape[1] > self.max_len:  # trunc
                multi_scale_x_max = multi_scale_x_max[:, :self.max_len, :]
            multi_scale_x_max_final = multi_scale_x_max[:, :, None] if i == 0 else torch.cat(
                (multi_scale_x_max_final, multi_scale_x_max[:, :, None]), dim=2)

        multi_scale_x = multi_scale_x_avg_final * self.w.to(x.device) + multi_scale_x_max_final * (
                    1 - self.w.to(x.device))

        return multi_scale_x

def FFT_for_Period(x, k=4):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = float('-inf')
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = [1]
    for top in top_list:
        period = np.concatenate((period, [math.ceil(x.shape[1] / top)]))  #
    return period, abs(xf).mean(-1)[:, top_list]  #


class moving_avg(nn.Module):
    def __init__(self, kernel_size):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=kernel_size, padding=0)

    def forward(self, x):
        # batch seq_len channel
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class moving_max(nn.Module):

    def __init__(self, kernel_size):
        super(moving_max, self).__init__()
        self.kernel_size = kernel_size
        self.max = nn.MaxPool1d(kernel_size=kernel_size, stride=kernel_size, padding=0)

    def forward(self, x):
        x = self.max(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class multi_scale_data(nn.Module):
    def __init__(self, kernel_size, return_len):
        super(multi_scale_data, self).__init__()
        self.kernel_size = kernel_size
        self.max_len = return_len
        self.moving_avg = [moving_avg(kernel) for kernel in kernel_size]
        self.moving_max = [moving_max(kernel) for kernel in kernel_size]
        self.w = nn.Parameter(torch.rand(1, return_len, 1), requires_grad=True)

    def forward(self, x):
        # batch seq_len channel
        different_scale_x_avg = []
        for func in self.moving_avg:
            moving_avg = func(x)
            different_scale_x_avg.append(moving_avg)
        multi_scale_x_avg = torch.cat(different_scale_x_avg, dim=1)
        if multi_scale_x_avg.shape[1] < self.max_len:
            padding = torch.zeros([x.shape[0], (self.max_len - (multi_scale_x_avg.shape[1])), x.shape[2]]).to(x.device)
            multi_scale_x_avg = torch.cat([multi_scale_x_avg, padding], dim=1)
        elif multi_scale_x_avg.shape[1] > self.max_len:
            multi_scale_x_avg = multi_scale_x_avg[:, :self.max_len, :]

        different_scale_x_max = []
        for func in self.moving_max:
            moving_max = func(x)
            different_scale_x_max.append(moving_max)
        multi_scale_x_max = torch.cat(different_scale_x_max, dim=1)
        if multi_scale_x_max.shape[1] < self.max_len:
            padding = torch.zeros([x.shape[0], (self.max_len - (multi_scale_x_max.shape[1])), x.shape[2]]).to(x.device)
            multi_scale_x_max = torch.cat([multi_scale_x_max, padding], dim=1)
        elif multi_scale_x_max.shape[1] > self.max_len:
            multi_scale_x_max = multi_scale_x_max[:, :self.max_len, :]

        multi_scale_x = multi_scale_x_avg * self.w.to(x.device) + multi_scale_x_max * (1 - self.w.to(x.device))

        return multi_scale_x


class adj_adjust(nn.Module):
    def __init__(self, in_features, gnn_type):
        super(adj_adjust, self).__init__()
        self.gnn_type = gnn_type
        self.in_features = in_features

        self.a = nn.Parameter(torch.ones(size=(1, 1, 2 * self.in_features, 1)) * 1e-1, requires_grad=True)
        self.mlp = nn.Sequential(nn.Linear(int(self.in_features), int(self.in_features)), nn.Tanh(),
                                 nn.Linear(int(self.in_features), int(self.in_features)))

        # self.leakyrelu = nn.LeakyReLU(1e-1)
        self.leakyrelu = nn.Tanh()

    def forward(self, h, adj):
        h = h.clone().detach()
        if self.gnn_type == 'time':
            h = h.transpose(1, 2)

        h = self.mlp(h)
        e = self._prepare_attentional_mechanism_input(h)
        zero_vec = torch.zeros_like(e)
        attention = torch.where(abs(adj) > 0, e, zero_vec)
        return attention

    def _prepare_attentional_mechanism_input(self, h):
        # in: [B, T, N, DM]
        # out: [B, T, N, N]
        Wh1 = torch.matmul(h, self.a[:, :, :self.in_features, :])
        Wh2 = torch.matmul(h, self.a[:, :, self.in_features:, :])
        # e = Wh1 + Wh2.transpose(-1, -2)
        e = torch.matmul(Wh1, Wh2.transpose(-1, -2))
        return self.leakyrelu(e)


class nconv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, alpha=1e-2):
        super(nconv, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, (1, kernel_size), padding=(0, int((kernel_size - 1) / 2)))
        self.conv2 = nn.Conv2d(c_in, c_out, (1, kernel_size), padding=(0, int((kernel_size - 1) / 2)))
        self.conv3 = nn.Conv2d(c_in, c_out, (1, kernel_size), padding=(0, int((kernel_size - 1) / 2)))
        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, a):
        x = x.permute(0, 3, 1, 2)
        temp = self.conv1(x) + torch.sigmoid(self.conv2(x))
        x = F.relu(temp + self.conv3(x))
        x = x.permute(0, 2, 3, 1)
        h_prime = torch.matmul(a, x)
        return h_prime


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, gnn_type, kernel_size, order=2, alpha=1e-2):
        super(gcn, self).__init__()
        self.nconv = nconv(c_in, c_out, kernel_size, alpha)
        self.gnn_type = gnn_type
        self.c_in = (order + 1) * c_in
        self.mlp = nn.Linear(self.c_in, c_out)
        self.dropout = dropout
        self.order = order
        self.act = nn.GELU()

    def forward(self, x, a):
        # in: [B, T, N, DM]
        # out: [B, T, N, DM]
        if self.gnn_type == 'time':
            x = x.transpose(1, 2)

        out = [x]
        x1 = self.nconv(x, a)
        out.append(x1)
        for k in range(2, self.order + 1):
            x2 = self.nconv(x1, a)
            out.append(x2)
            x1 = x2
        h = torch.cat(out, dim=-1)
        h = self.mlp(h)
        h = self.act(h)
        h_prime = F.dropout(h, self.dropout, training=self.training)

        if self.gnn_type == 'time':
            h_prime = h_prime.transpose(1, 2)

        return h_prime


class single_scale_gnn(nn.Module):
    def __init__(self, configs):
        super(single_scale_gnn, self).__init__()

        self.tk = configs.tk
        self.nodes = configs.nodes
        self.scale_number = configs.scale_number
        self.use_tgcn = configs.use_tgcn
        self.use_ngcn = configs.use_ngcn
        self.init_seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.ln = nn.ModuleList()
        self.channels = configs.enc_in
        self.individual = configs.individual
        self.dropout = configs.dropout
        self.device = 'cuda:' + str(configs.gpu)
        self.GraphforPre = False
        self.tvechidden = configs.tvechidden
        self.tanh = nn.Tanh()
        self.d_model = configs.hidden
        self.start_linear = nn.Linear(1, self.d_model)
        self.seq_len = self.init_seq_len + self.init_seq_len  # max_len (i.e., multi-scale shape)
        self.timevec1 = nn.Parameter(torch.randn(self.seq_len, configs.tvechidden).to(self.device),
                                     requires_grad=True).to(self.device)
        self.timevec2 = nn.Parameter(torch.randn(configs.tvechidden, self.seq_len).to(self.device),
                                     requires_grad=True).to(self.device)
        self.tgcn = gcn(self.d_model, self.d_model, self.dropout, gnn_type='time', alpha=1e-2,
                        kernel_size=5)  # use cnn del nodes dim
        self.nodevec1 = nn.Parameter(torch.randn(self.channels, configs.nvechidden).to(self.device),
                                     requires_grad=True).to(self.device)
        self.nodevec2 = nn.Parameter(torch.randn(configs.nvechidden, self.channels).to(self.device),
                                     requires_grad=True).to(self.device)
        self.gconv = gcn(self.d_model, self.d_model, self.dropout, gnn_type='nodes', alpha=1e-2,
                         kernel_size=5)  # use cnn del time dim
        self.layer_norm = nn.LayerNorm(self.channels)
        self.grang_emb_len = math.ceil(self.d_model // 4)
        self.graph_mlp = nn.Linear(2 * self.tvechidden, self.grang_emb_len)
        self.act = nn.Tanh()
        self.channel_embedding = channel_embedding(kernal=1, d_model=256, max_len=3, device=self.device)
        self.channel_attn = AttentionLayer(d_model=256, nheads=4, dim_feedforward=512, attn=ChannelAttention,
                                           dropout=0.2)
        self.inti_mlp = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        self.time_mlp = nn.Sequential(nn.Linear(int(self.seq_len), int(self.seq_len)), nn.ReLU(),
                                      nn.Linear(int(self.seq_len), int(self.seq_len / 2)))
        self.channel_mlp = nn.Sequential(nn.Linear(3, 3), nn.ReLU(), nn.Linear(3, 1))
        self.time_adj_adjust = adj_adjust(in_features=self.d_model, gnn_type='time')
        self.node_adj_adjust = adj_adjust(in_features=self.d_model, gnn_type='nodes')
        if self.use_tgcn:
            dim_seq = 2 * self.d_model
            if self.GraphforPre:
                dim_seq = 2 * self.d_model + self.grang_emb_len
        else:
            dim_seq = 2 * self.seq_len
        self.Linear = nn.Linear(dim_seq, 3)

    def logits_warper_softmax(self, adj, indices_to_remove, filter_value=-float("Inf")):
        adj = F.softmax(adj.masked_fill(indices_to_remove, filter_value), dim=0)
        return adj

    def logits_warper(self, adj, indices_to_remove, mask_pos, mask_neg, filter_value=-float("Inf")):
        mask_pos_inverse = ~mask_pos
        mask_neg_inverse = ~mask_neg
        processed_pos = mask_pos * F.softmax(adj.masked_fill(mask_pos_inverse, filter_value), dim=-1)
        processed_neg = -1 * mask_neg * F.softmax((1 / (adj + 1)).masked_fill(mask_neg_inverse, filter_value), dim=-1)
        processed_adj = processed_pos + processed_neg
        return processed_adj

    def add_adjecent_connect(self, mask):
        s = np.arange(0, self.seq_len - 1)
        e = np.arange(1, self.seq_len)
        forahead = np.stack([s, e], 0)
        back = np.stack([e, s], 0)
        all = np.concatenate([forahead, back], 1)
        mask[all] = False
        return mask

    def add_cross_scale_connect(self, adj, periods):
        max_L = self.seq_len
        mask = torch.tensor([], dtype=bool).to(adj.device)
        k = self.tk
        min_total_corss_scale_neighbors = 5
        start = 0
        end = 0
        for period in periods:
            ls = self.init_seq_len // period
            end = start + ls
            if end > max_L:
                end = max_L
                ls = max_L - start
            kp = k // period
            kp = max(kp, min_total_corss_scale_neighbors)
            kp = min(kp, ls)
            mask = torch.cat([mask, adj[:, start:end] < torch.topk(adj[:, start:end], k=kp)[0][..., -1, None]], dim=1)
            start = end
            if start == max_L:
                break
        if start < max_L:
            mask = torch.cat([mask, torch.zeros(self.seq_len, max_L - start, dtype=bool).to(mask.device)], dim=1)
        return mask

    def add_cross_var_adj(self, adj):
        k = 4
        k = min(k, adj.shape[0])
        mask = (adj < torch.topk(adj, k=adj.shape[0] - k)[0][..., -1, None]) * (
                    adj > torch.topk(adj, k=adj.shape[0] - k)[0][..., -1, None])
        mask_pos = adj >= torch.topk(adj, k=k)[0][..., -1, None]
        mask_neg = adj <= torch.kthvalue(adj, k=k)[0][..., -1, None]
        return mask, mask_pos, mask_neg

    def get_time_adj(self, periods):
        adj = F.relu(torch.einsum('td,dm->tm', self.timevec1, self.timevec2))
        mask = self.add_cross_scale_connect(adj, periods)
        mask = self.add_adjecent_connect(mask)
        adj = self.logits_warper_softmax(adj=adj, indices_to_remove=mask)
        return adj

    def get_var_adj(self):
        adj = F.relu(torch.einsum('td,dm->tm', self.nodevec1, self.nodevec2))
        mask, mask_pos, mask_neg = self.add_cross_var_adj(adj)
        adj = self.logits_warper(adj, mask, mask_pos, mask_neg)
        return adj

    def get_time_adj_embedding(self, b):
        graph_embedding = torch.cat([self.timevec1, self.timevec2.transpose(0, 1)], dim=1)
        graph_embedding = self.graph_mlp(graph_embedding)
        graph_embedding = graph_embedding.unsqueeze(0).unsqueeze(2).expand([b, -1, self.channels, -1])
        return graph_embedding

    def expand_channel(self, x):
        x = x.unsqueeze(-1)
        x = self.start_linear(x)
        return x

    def forward(self, x):
        periods = FFT_for_Period_initdata(x, self.scale_number)
        multi_scale_func = multi_scale_initdata(kernel_size_metix=periods, return_len=self.seq_len)
        x = multi_scale_func(x)

        x = self.channel_embedding(x)
        x = self.channel_attn(x)
        x = self.inti_mlp(x)
        x = self.time_mlp(x.squeeze(-1).transpose(1, 3)).transpose(1, 3)
        x = self.channel_mlp(x.transpose(2, 3)).transpose(2, 3).squeeze(-2)

        periods, _ = FFT_for_Period(x, self.scale_number)
        multi_scale_func = multi_scale_data(kernel_size=periods, return_len=self.seq_len)
        x = multi_scale_func(x)
        x = self.expand_channel(x)
        batch_size = x.shape[0]
        x_ = x
        if self.use_tgcn:
            time_adp = self.get_time_adj(periods)
            time_adj_adjust = self.time_adj_adjust(x, time_adp)
            time_adp = time_adp + time_adj_adjust
            x = self.tgcn(x, time_adp) + x
        if self.use_ngcn:
            gcn_adp = self.get_var_adj()
            node_adj_adjust = self.node_adj_adjust(x, gcn_adp)
            gcn_adp = gcn_adp + node_adj_adjust
            x = self.gconv(x, gcn_adp) + x
        x = torch.cat([x_, x], dim=-1)
        if self.use_tgcn and self.GraphforPre:
            graph_embedding = self.get_time_adj_embedding(b=batch_size)
            x = torch.cat([x, graph_embedding], dim=-1)

        x = self.Linear(x).transpose(2, 3)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x[:, :self.init_seq_len, :]

def ChannelAttention(query, key, value, dropout=None):
    d_k = query.size(-1)
    query = query.transpose(3, 4)
    key = key.transpose(3, 4)
    value = value.transpose(3, 4)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value).transpose(3, 4)


def TimeAttention(query, key, value, dropout=None):
    d_k = query.size(-1)
    query = query.transpose(2, 4)
    key = key.transpose(2, 4)
    value = value.transpose(2, 4)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value).transpose(2, 4)

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.graph_encs = nn.ModuleList()
        self.enc_layers = configs.e_layers
        self.anti_ood = configs.anti_ood
        for i in range(self.enc_layers):
            self.graph_encs.append(single_scale_gnn(configs=configs))
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        self.scale_number = configs.scale_number

    def forward(self, x):
        # x: [B,T,C,N]
        if self.anti_ood:
            seq_last = x[:, -1:, :, :].detach()
            x = x - seq_last

        for i in range(self.enc_layers):
            x = self.graph_encs[i](x)

        x = self.Linear(x.transpose(1, 3)).transpose(1, 3)

        if self.anti_ood:
            x = x + seq_last

        return x