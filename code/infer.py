import argparse
# import yaml
import json
from types import SimpleNamespace
import time
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import init
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
import torch
import numpy as np
import os
from tensorboardX import SummaryWriter
from torch.nn import init, DataParallel
import h5py

# load
from src.pressure_dataset import Pressure_Dataset
from src.velocity_dataset import Velocity_Dataset
from src.cd_dataset import Drag_Dataset


from src.AeroGTO import AeroGTO
from src.AeroGTO_cd_pred import AeroGTO_cd

from src.utils import set_seed
from src.train import infer
from src.utils import collate

device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', type=str, help='Path to config file')  # Change the default config file name if needed

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)  # Load JSON instead of YAML
    
    args = SimpleNamespace(**config)
    
    return args

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.MultiheadAttention):
        # 初始化 in_proj_weight 和 in_proj_bias
        torch.nn.init.xavier_uniform_(m.in_proj_weight)
        if m.in_proj_bias is not None:
            m.in_proj_bias.data.fill_(0.01)

        # out_proj 属于 nn.Linear，所以它有 weight 和 bias
        torch.nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            m.out_proj.bias.data.fill_(0.01)

def gather_tensor(tensor):
    """
    Gathers tensors from all processes and reduces them by summing up.
    """
    # Ensure the tensor is on the same device as specified for the operation
    tensor = tensor.to(device)
    # All-reduce: Sum the tensors from all processes
    return tensor


def get_model(args):
    if args.task == "cd_pred":  
        model = AeroGTO_cd(
            N_block = args.model["N_block"], 
            state_size = args.model["state_size"],  
            state_embedding_dim = args.model["state_embedding_dim"], 
            att_embedding_dim = args.model["att_embedding_dim"],
            n_head = args.model["n_head"],
            n_token = args.model["n_token"]
            ).to(device)
    else:
        model = AeroGTO(
            N_block = args.model["N_block"], 
            state_size = args.model["state_size"],  
            state_embedding_dim = args.model["state_embedding_dim"], 
            att_embedding_dim = args.model["att_embedding_dim"],
            n_head = args.model["n_head"],
            n_token = args.model["n_token"]
            ).to(device)
    return model


def main(args):
    model = get_model(args)
    model.load_state_dict(torch.load(args.check_point_path)["state_dict"])
    model = model.to(device)
    
    if args.train["if_multi_gpu"]:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = DataParallel(model)
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())    
    params = sum([np.prod(p.size()) for p in model_parameters])
    
    # load data
    if args.task == "cd_pred":
        test_dataset = Drag_Dataset(
            data_path = args.dataset["infer_data_path"],
            edge_type = args.dataset["edge_type"]
            )
    elif args.task == "p_pred":
        test_dataset = Pressure_Dataset(
            data_path = args.dataset["infer_data_path"],
            edge_type = args.dataset["edge_type"]
            )
    elif args.task == "v_pred":
        test_dataset = Velocity_Dataset(
            data_path = args.dataset["infer_data_path"],
            )
        
    test_dataloader = DataLoader(test_dataset, 
        batch_size=args.dataset["test"]["batchsize"], 
        shuffle=args.dataset["test"]["shuffle"], 
        num_workers=args.dataset["test"]["num_workers"],
        collate_fn = collate)
        
    print("#############")
    print("#params:", params)
    print(f"model name: {args.model['name']}")
    
    print(f"No. of test samples: {len(test_dataloader)}")
    print("#############")
        
    infer(args, model, test_dataloader, device=device)
        
            
if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    device = args.device
    if 'cuda' in device:
        assert torch.cuda.is_available()
        if device == 'cuda' and torch.cuda.device_count() > 1 and args.train["if_multi_gpu"]:
            use_multi_gpu = True
            print(f"lets use {torch.cuda.device_count()} gpus!")
        else:
            use_multi_gpu = False
            print(f"lets use 1 gpu!")
    else:
        use_multi_gpu = False
        print(f"lets use cpu!")
    device = torch.device(device)
    print('device:', device)
        
    # if args.seed is not None:
    #     set_seed(args.seed)
    set_seed(args.seed)
    
    # # train+val+test
    main(args)
    