import numpy as np
import torch


def w_rmse(y_pred,y_true):
    RMSE = np.empty([y_pred.size(1)])
    for i in range(y_pred.size(1)):
        RMSE[i] = np.sqrt(((y_pred[:,i,:] - y_true[:,i,:])**2).mean([0,1]))
    return RMSE

def w_mae(y_pred,y_true):
    MAE = np.empty([y_pred.size(1)])
    for i in range(y_pred.size(1)):
        MAE[i] = (abs(y_pred[:, i, :] - y_true[:, i, :])).mean([0, 1])
    return MAE

def w_acc(y_pred,y_true):
    ACC = np.empty([y_pred.size(1)])
    for i in range(y_pred.size(1)):
        clim = y_true[:,i,:].mean(0)
        a = y_true[:,i,:] - clim
        a_prime = (a - a.mean())
        fa = y_pred[:,i,:] - clim
        fa_prime = (fa - fa.mean())
        ACC[i] = (
                torch.sum(fa_prime * a_prime) /
                torch.sqrt(
                    torch.sum(fa_prime ** 2) * torch.sum(a_prime ** 2)
                )
        )
    return ACC

def evaluate(data_path,time,std,mean):
    i=time
    data = np.load(data_path)
    y_pred, y_true = torch.tensor(data['pred']), torch.tensor(data['label'])
    y_pred, y_true = y_pred.reshape(y_pred.shape[0], y_pred.shape[1], 3, 100), y_true.reshape(y_true.shape[0], y_true.shape[1], 3, 100)
    y_pred_1, y_pred_2, y_pred_3, y_true_1, y_true_2, y_true_3 = y_pred[:,:,0,:]*std[0]+mean[0], y_pred[:,:,1,:]*std[1]+mean[1], y_pred[:,:,2,:]*std[2]+mean[2], y_true[:,:,0,:]*std[0]+mean[0], y_true[:,:,1,:]*std[1]+mean[1], y_true[:,:,2,:]*std[2]+mean[2]
    y_pred, y_true = np.concatenate([y_pred_1[:,:,None,:], y_pred_2[:,:,None,:], y_pred_3[:,:,None,:]],axis=2), np.concatenate([y_true_1[:,:,None,:], y_true_2[:,:,None,:], y_true_3[:,:,None,:]],axis=2)
    y_pred, y_true = y_pred[:,i,:,:], y_true[:,i,:,:]
    print('RMSE:', w_rmse(torch.tensor(y_pred),torch.tensor(y_true)))
    print('MAE: ', w_mae(torch.tensor(y_pred),torch.tensor(y_true)))
    print('ACC: ', w_acc(torch.tensor(y_pred),torch.tensor(y_true)))

data_info = np.load('./data/input_example.npz')
std, mean = torch.tensor(data_info['std']),torch.tensor(data_info['mean'])

for i in range(20):
    print('Lead Time: ',(i+1)/4)
    evaluate('./result/output_example.npz',i,std, mean)