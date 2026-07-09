import torch


class Configs:
    def __init__(self):
        pass


configs = Configs()

configs.seed = 5

# trainer related
configs.n_cpu = 0
configs.gpu = 0
configs.device = torch.device('cuda:'+str(configs.gpu))
configs.use_gpu = 1
configs.batch_size_test = 64


# data related
configs.input_dim = 3
configs.output_dim = 3
configs.nodes = 100

configs.seq_len = 28
configs.pred_len = 20

# model related
configs.hidden = 32
configs.scale_number = 4
configs.use_ngcn = 1
configs.use_tgcn = 1
configs.nvechidden = 4
configs.tvechidden = 4
configs.tk = 10
configs.dropout = 0.1
configs.e_layers = 3
configs.anti_ood = 0
configs.enc_in = configs.nodes
configs.dec_in = configs.nodes
configs.individual = False
configs.embed = 'timeF'


