import torch
import torch.nn as nn
import torch.nn.functional as F

def ncc(y1,y2):
    y1 =y1.transpose(1, 3)
    y2 = y2.transpose(1, 3)
    B, N, C, T = y1.shape
    y1 = y1.reshape(B,-1)[:,None]
    y2 = y2.reshape(B,-1)[:,None]
    y = torch.cat([y1,y2],1)
    cor = torch.zeros(B,1)
    for i in range(B):
        cor[i,0] = torch.corrcoef(y[i])[0,1]
    return cor
    

def L_ashift(pred,true):
    bias = (pred-true).transpose(1,3)
    a = F.softmax(bias,-1)
    L_ashift = bias.shape[-1] * (1/bias.shape[-1] - a).sum(-1)
    return L_ashift.mean()
    

def L_phase(pred,true,p):
    y1 = pred.transpose(1, 3)
    y2 = true.transpose(1, 3)
    f1 = torch.fft.rfft(y1, dim=-1)
    f2 = torch.fft.rfft(y2, dim=-1)
    f1_max, f1_indices_max = abs(f1).topk(1, dim=-1, largest=True, sorted=True)
    f2_max, f2_indices_max = abs(f2).topk(1, dim=-1, largest=True, sorted=True)
    _, f1_indices_else = abs(f1).topk(f1.shape[-1]-1, dim=-1, largest=False, sorted=False)
    f_domain = f1_indices_max - f2_indices_max
    L_phase = torch.cat([f_domain, f1_indices_else], dim=-1).float()
    L_phase = F.normalize(L_phase, p=p, dim=-1)
    return L_phase.mean()
    

def L_amp(pred,true,p):
    ncc_r = ncc(true, pred)
    L_amp = F.normalize((1-ncc_r), p=p)
    return L_amp.mean()


def Trend_Loss(pred,true,p):
    r = L_ashift(pred, true) + L_phase(pred, true, p) + L_amp(pred, true, p)
    return r
    

class Trend_Loss(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(configs.nodes*configs.pred_len, 512),nn.ReLU(),nn.Linear(512, 3)).to(configs.device)

    def forward(self, pred, true, p):
        w = F.softmax(self.mlp(true.mean(0).flatten()),dim=0)
        # w = self.mlp(true.mean(0).flatten())
        r = w[0] * L_ashift(pred, true) + w[1] * L_phase(pred, true, p) + w[2] * L_amp(pred, true, p)
        return r
