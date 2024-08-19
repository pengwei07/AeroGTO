import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import os

def get_l2_loss(output, target):
    # output.dim = (batch, N, c)
    # target.dim = (batch, N, c)   
    # output = output.squeeze(-1) 
    # target = target.squeeze(-1) 

    error = output - target
    
    norm_error_sample = torch.norm(error, dim=-2) / (torch.norm(target, dim=-2) + 1e-6)
    if norm_error_sample.shape[-1] == 1:
        norm_error_channnel = norm_error_sample.squeeze(-1) 
    else:
        norm_error_channnel = torch.mean(norm_error_sample, dim=-1)
    
    norm_error_batch = torch.mean(norm_error_channnel)
    
    return norm_error_batch

def get_val_loss(output_p_hat, p, if_rescale, info, flag, dragWeight=None):
    
    device = output_p_hat.device
    #################################
    p_target = p.to(device)
    if flag=="all":
        dragWeight = dragWeight.to(device)
    
    losses = {}
    
    losses['L2_p_norm'] = get_l2_loss(output_p_hat[...,:1], (p_target - info['p_mean']) / info['p_std']).item()
    ################
    if if_rescale:
        p_hat = output_p_hat[...,:1] * info['p_std'] + info['p_mean']
        
    losses['L2_p'] = get_l2_loss(p_hat, p_target).item()
    
    if flag=="all":
        drag_truth = abs(torch.sum(dragWeight * p_target))
        drag_pred = abs(torch.sum(dragWeight * p_hat))
        losses['MRE_drag'] = abs(drag_pred - drag_truth) / abs(drag_truth)
    
    return losses

def get_train_loss(output_p_hat, p, loss_flag, if_rescale, info):

    device = output_p_hat.device
    #################################
    p_target = p.to(device)
    
    losses = {}
    criterion = nn.MSELoss()
        
    if loss_flag == 'L2_loss_norm':
        losses['loss'] = get_l2_loss(output_p_hat[...,:1], (p_target - info['p_mean']) / info['p_std'])
    elif loss_flag == 'MSE_loss_norm':
        losses['loss'] = criterion(output_p_hat[...,:1], (p_target - info['p_mean']) / info['p_std'])

    losses['L2_p_norm'] = get_l2_loss(output_p_hat[...,:1], (p_target - info['p_mean']) / info['p_std']).item()
    
    if if_rescale:
        p_hat = output_p_hat[...,:1] * info['p_std'] + info['p_mean']

    if loss_flag == 'L2_loss':
        losses['loss'] = get_l2_loss(p_hat, p_target)
        
    elif loss_flag == 'MSE_loss':
        losses['loss'] = criterion(p_hat, p_target)

    losses['L2_p'] = get_l2_loss(p_hat, p_target).item()

    return losses

def train(args, model, train_dataloader, optim, device, scaler):
        
    model.train()    
    loss = 0
    L2_p = 0
    L2_p_norm = 0
    
    num = 0
    for i, [input,name] in enumerate(train_dataloader):
        # forward
        optim.zero_grad()
        
        p = input['pressure']        
        node_pos = input['node_pos']
        edges = input['edges']
        areas = input['areas']
        info = input['info']

        output_p_hat = model(node_pos.to(device), areas.to(device), edges, info.to(device)) 
        
        costs = get_train_loss(
            output_p_hat,
            p, 
            args.train["loss_flag"], 
            args.train["if_rescale"], 
            args.train["info"]
            )
        
        costs['loss'].backward()
        optim.step()
        
        loss = loss + costs['loss'].item()
        L2_p = L2_p + costs['L2_p']
        L2_p_norm = L2_p_norm + costs['L2_p_norm']
        #########################################
        num = num + 1
        
        # break
            
    batch_error = {}
    batch_error['loss'] = loss / num

    batch_error['L2_p'] = L2_p / num
    batch_error['L2_p_norm'] = L2_p_norm / num
    
    return batch_error, scaler

def validate(args, model, val_dataloader, device, flag):
    
    model.eval()
    
    L2_p = 0
    L2_p_norm = 0
    if flag=="all": 
        MRE_drag = 0
    
    num = 0
    with torch.no_grad():
          
        # for i, [input, t] in enumerate(tqdm(val_dataloader, desc="Validation")):
        for i, [input,_] in enumerate(val_dataloader): 
            
            if flag=="part":   
            
                p = input['pressure']        
                node_pos = input['node_pos']
                edges = input['edges']
                areas = input['areas']
                info = input['info']

                output_p_hat = model(node_pos.to(device), areas.to(device), edges, info.to(device)) 
                                    
                costs = get_val_loss(
                    output_p_hat,
                    p, 
                    args.train["if_rescale"], 
                    args.train["info"],
                    flag
                    )

                L2_p = L2_p + costs['L2_p']
                L2_p_norm = L2_p_norm + costs['L2_p_norm']

            elif flag=="all": 
                
                p = input['pressure']        
                node_pos = input['node_pos']
                edges = input['edges']
                areas = input['areas']
                info = input['info']
                dragWeight = input['dragWeight']

                lens = len(node_pos)
                
                p_outs = []
                
                for item in range(lens):
                    
                    output_p_hat_i = model(node_pos[item].to(device), areas[item].to(device), edges[item], info.to(device)) 
                    p_outs.append(output_p_hat_i)
                                    
                costs = get_val_loss(
                    torch.cat(p_outs,dim=-2),
                    torch.cat(p,dim=-2),
                    args.train["if_rescale"], 
                    args.train["info"],
                    flag,
                    torch.cat(dragWeight,dim=-2),
                    )

                L2_p = L2_p + costs['L2_p']
                L2_p_norm = L2_p_norm + costs['L2_p_norm']
                MRE_drag = MRE_drag + costs['MRE_drag'].item()
            #########################################
            num = num + 1
            # break
            
    batch_error = {}
    
    batch_error['L2_p'] = L2_p / num
    batch_error['L2_p_norm'] = L2_p_norm / num
    
    if flag=="all": 
        batch_error['MRE_drag'] = MRE_drag / num
    
    return batch_error


def infer(args, model, val_dataloader, device):
    
    model.eval()
    with torch.no_grad():
        for i, [input, name] in enumerate(val_dataloader):    
            
            node_pos = input['node_pos']
            edges = input['edges']
            areas = input['areas']
            info = input['info']

            output_p_hat = model(node_pos.to(device), areas.to(device), edges, info) 
            ################
            if args.train["if_rescale"]:
                p_hat = output_p_hat[...,:1] * args.train["info"]['p_std'] + args.train["info"]['p_mean']
                
            save_path = os.path.join(args.save_path, f"{args.name}_infer_result")
            os.makedirs(save_path, exist_ok=True)
            
            p_save = p_hat.detach().cpu().numpy()
            np.save(f"{save_path}/press_{name[0]}.npy", p_save.reshape(-1))
            
            break
            
    print("Finished!")

