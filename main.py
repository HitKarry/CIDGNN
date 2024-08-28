import torch
import random
import numpy as np
from model.DCIGNN import Model
from config import configs
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class dataset_package(Dataset):
    def __init__(self, train_x, train_y):
        super().__init__()
        self.input = train_x
        self.target = train_y

    def GetDataShape(self):
        return {'input': self.input.shape,
                'target': self.target.shape}

    def __len__(self, ):
        return self.input.shape[0]

    def __getitem__(self, idx):
        return self.input[idx], self.target[idx]

set_seed(configs.seed)
data = np.load('./data/input_example.npz')
dataset_test = dataset_package(train_x=data['x'], train_y=data['y'])
model = Model(configs).to(configs.device)
weight = torch.load('checkpoint/weights.chk')
model.load_state_dict(weight['net'])
model = torch.nn.DataParallel(model, device_ids=[0, 1])
model.eval()

data = DataLoader(dataset_test, batch_size=configs.batch_size_test, shuffle=False)

with torch.no_grad():
    for i, (input, target) in enumerate(data):
        pred_temp = model(input.float().to(configs.device))
        if i == 0:
            pred = pred_temp
            label = target
        else:
            pred = torch.cat((pred, pred_temp), 0)
            label = torch.cat((label, target), 0)

np.savez('./result/output_example.npz', pred=pred.cpu(), label=label.cpu())
