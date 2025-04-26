import torch
import random
import h5py
import numpy as np
import os

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.98, patience=10, verbose=True)
# scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.9)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.991)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[10, 20, 30], gamma=0.1)
    
def set_seed(seed: int = 0):    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保PyTorch使用相同的初始化权重
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False  # 禁用 cudnn 自动优化器，保证确定性
    
def collate(X):
    # print(f"X: {len(X)}")
    # [input, t] = X[0]
    # dim = [B, T, N, S]
    
    N_max = max([input["node_pos"].shape[-2] for [input, t] in X])
    E_max = max([input["edges"].shape[-2] for [input, t] in X])
    
    N_all = torch.zeros(len(X))
    E_all = torch.zeros(len(X))
    
    
    mask = []
    
    for batch, [input, t] in enumerate(X):
        # node
        tensor = input['node_pos']
        N, S = tensor.shape
        input['node_pos'] = torch.cat([tensor, torch.zeros(N_max - N + 1, S)], dim=-2)
        N_all[batch] = N
        mask_i = torch.zeros(N_max)
        mask_i[:N] = 1
        mask.append(mask_i)
        
        gt = input['gt']
        if gt.ndim == 1:
            pass
        else:
            N, S = gt.shape 
            input['gt'] = torch.cat([gt, torch.zeros(N_max - N + 1, S)], dim=-2)
            N_all[batch] = N
        
        # edge
        edges = input['edges']
        E, S = edges.shape
        input['edges'] = torch.cat([edges, N_max * torch.ones(E_max - E + 1, S)], dim=-2)
        E_all[batch] = E

    # stack
    batch_in = {key: None for key in ['node_pos', 'edges', 'gt']}
    mask = torch.stack(mask, dim=0)

    for key in batch_in.keys():
        batch_in[key] = torch.stack([x[0][key] for x in X], dim=0)
    batch_in['mask'] = mask
        
    names = [x[1] for x in X]

    return batch_in, names