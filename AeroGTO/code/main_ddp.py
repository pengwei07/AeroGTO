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
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import numpy as np
import os
import random
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler

# load
from src.pressure_dataset import Pressure_Dataset
from src.velocity_dataset import Velocity_Dataset
from src.cd_dataset import Drag_Dataset

from src.AeroGTO import AeroGTO
from src.AeroGTO_cd_pred import AeroGTO_cd

from src.utils import set_seed
from src.train import train, validate
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

def setup():
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    return rank, world_size

def gather_tensor(tensor, world_size):
    """
    Gathers tensors from all processes and reduces them by summing up.
    """
    # Ensure the tensor is on the same device as specified for the operation
    tensor = tensor.to(device)
    # All-reduce: Sum the tensors from all processes
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    # Only on rank 0, we scale the tensor to find the average
    if dist.get_rank() == 0:
        tensor /= world_size
    return tensor

def get_model(args, rank):
    if args.task == "cd_pred":  
        model = AeroGTO_cd(
            N_block = args.model["N_block"], 
            state_size = args.model["state_size"],  
            state_embedding_dim = args.model["state_embedding_dim"], 
            att_embedding_dim = args.model["att_embedding_dim"],
            n_head = args.model["n_head"],
            n_token = args.model["n_token"]
            ).to(rank)
    else:
        model = AeroGTO(
            N_block = args.model["N_block"], 
            state_size = args.model["state_size"],  
            state_embedding_dim = args.model["state_embedding_dim"], 
            att_embedding_dim = args.model["att_embedding_dim"],
            n_head = args.model["n_head"],
            n_token = args.model["n_token"]
            ).to(rank)
    return model

def main(args):
    # setting
    local_rank, world_size = setup()

    model = get_model(args, local_rank)

    if args.model["if_init"]:
        model.apply(init_weights)
    
    # checkpoint_path = "result/nn/block_4_epo_fine_epo_200_100.nn"   
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # model.load_state_dict(checkpoint['state_dict'])
        
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())    
    params = sum([np.prod(p.size()) for p in model_parameters])
    
    # load data
    if args.task == "cd_pred":
        train_dataset = Drag_Dataset(
            data_path = args.dataset["train_data_path"],
            edge_type = args.dataset["edge_type"]
            )
        test_dataset = Drag_Dataset(
            data_path = args.dataset["test_data_path"],
            edge_type = args.dataset["edge_type"]
            )
    elif args.task == "p_pred":
        train_dataset = Pressure_Dataset(
            data_path = args.dataset["train_data_path"],
            edge_type = args.dataset["edge_type"]
            )
        test_dataset = Pressure_Dataset(
            data_path = args.dataset["test_data_path"],
            edge_type = args.dataset["edge_type"]
            )
    elif args.task == "v_pred":
        train_dataset = Velocity_Dataset(
            data_path = args.dataset["train_data_path"],
            )
        test_dataset = Velocity_Dataset(
            data_path = args.dataset["test_data_path"],
            )
        
        
    # sampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, shuffle=True, seed=args.seed, rank=local_rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, shuffle=False, rank=local_rank)
        
    train_dataloader = DataLoader(train_dataset, 
                        batch_size=args.dataset["train"]["batchsize"], 
                        shuffle=args.dataset["train"]["shuffle"], 
                        sampler=train_sampler,
                        num_workers=args.dataset["train"]["num_workers"],
                        collate_fn = collate)
    
    test_dataloader = DataLoader(test_dataset, 
                        batch_size=args.dataset["test"]["batchsize"], 
                        shuffle=args.dataset["test"]["shuffle"], 
                        sampler= test_sampler,
                        num_workers=args.dataset["test"]["num_workers"],
                        collate_fn = collate)
    
    EPOCH = args.train["epoch"]

    if local_rank == 0:
        
        for i, [data, _] in enumerate(train_dataloader):
            for key in data.keys():
                print(key, data[key].shape)
            break
        print("#############")
        '''
        node_pos torch.Size([4, 3586, 3])
        edges torch.Size([4, 21504, 2])
        pressure torch.Size([4, 3586, 1])
        '''
        print(f"No. of train samples: {len(train_dataset)}, No. of test samples: {len(test_dataset)}")
        print(f"No. of train batches: {len(train_dataloader)}, No. of test batches: {len(test_dataloader)}")
        print("#############")
        print("#params:", params)
        print(f"EPOCH: {EPOCH}")
        print(f"model name: {args.model['name']}")
        print("#############")
        
        if not os.path.exists(f"{args.save_path}/record"):
            os.makedirs(f"{args.save_path}/record")
        
        with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
            file.write(f"No. of train samples: {len(train_dataset)}, No. of test samples: {len(test_dataset)}\n")
            file.write(f"No. of train batches: {len(train_dataloader)}, No. of test batches: {len(test_dataloader)}\n")
            file.write(f"Let's use {torch.cuda.device_count()} GPUs!\n")
            file.write(f"{args.name}, #params: {params}\n")
            file.write(f"EPOCH: {EPOCH}\n")
            
        log_dir = f"{args.save_path}/logs/{args.name}/rank_{local_rank}"
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        
    real_lr = float(args.train["lr"])
    optim = torch.optim.AdamW(model.parameters(), lr=real_lr)
    scheduler = CosineAnnealingLR(optim, T_max= EPOCH, eta_min = float(args.train["eta_min"]))
    
    
    for epoch in range(EPOCH):
        start_time = time.time()
        train_error = train(args, model, train_dataloader, optim, local_rank)
        end_time = time.time()
        
        # 获取当前的学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        training_time = (end_time - start_time)
        current_lr = torch.tensor(current_lr, device=device)
        train_loss = torch.tensor(train_error['loss'], device=device)
        L2 = torch.tensor(train_error['L2'], device=device)
        L2_norm = torch.tensor(train_error['L2_norm'], device=device)
        
        training_time = torch.tensor(training_time, device=device)
        current_lr = gather_tensor(current_lr, world_size)
        
        train_loss = gather_tensor(train_loss, world_size)
        L2 = gather_tensor(L2, world_size)
        L2_norm = gather_tensor(L2_norm, world_size)
        
        training_time = gather_tensor(training_time, world_size)
        
        if local_rank == 0:
            writer.add_scalar('lr/lr', current_lr, epoch)
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('L2/train_L2', L2, epoch)
            writer.add_scalar('L2/train_L2_norm', L2_norm, epoch)
            
            with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
                file.write(f"Epoch: {epoch + 1}/{EPOCH}, Train Loss: {train_loss:.4f}\n")
                file.write(f"L2_norm: {L2_norm:.4f}, L2: {L2:.4f}\n")
                file.write(f"time pre train epoch/s:{training_time:.2f}, current_lr:{current_lr:.4e}\n")
            
            if (epoch+1) % 1 == 0 or epoch == 0 or (epoch+1) == EPOCH:
                print(f"Epoch: {epoch + 1}/{EPOCH}, Train Loss: {train_loss:.4f}")
                print(f"L2_norm: {L2_norm:.4f}, L2: {L2:.4f}")
                print(f"time pre train epoch/s:{training_time:.2f}, current_lr:{current_lr:.4e}")
                print("#################")

        if (epoch+1) % 10 == 0 or epoch == 0 or (epoch+1) == EPOCH:
            # test
            start_time = time.time() 
            test_error =  validate(args, model, test_dataloader, device=local_rank)
            end_time = time.time()
            training_time1 = (end_time - start_time)
            
            test_L2 = torch.tensor(test_error['L2'], device=device)
            test_L2_norm = torch.tensor(test_error['L2_norm'], device=device)

            test_L2 = gather_tensor(test_L2, world_size)
            test_L2_norm = gather_tensor(test_L2_norm, world_size)
            
            if local_rank == 0:
                print(f"Epoch: {epoch + 1}/{EPOCH}, test_L2_norm: {test_L2_norm:.4f}, test_L2: {test_L2:.4f}")
                print(f"time pre test epoch/s:{training_time1:.2f}")
                print("#################")
                
                writer.add_scalar('L2/test_L2_norm', test_L2_norm, epoch)
                writer.add_scalar('L2/test_L2', test_L2, epoch)
                
                with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
                    file.write(f"Epoch: {epoch + 1}/{EPOCH}, test_L2_norm: {test_L2_norm:.4f}, test_L2: {test_L2:.4f}\n")
                    file.write(f"time pre test epoch/s:{training_time1:.2f}\n")
                    
        if (epoch+1) % 100 == 0 or epoch == 0 or (epoch+1) == EPOCH:
            if args.if_save:
                checkpoint = { 
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict() if args.train["if_multi_gpu"] else model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'learning_rate': scheduler.get_last_lr()[0]
                }
                nn_save_path = os.path.join(args.save_path, "nn")
                os.makedirs(nn_save_path, exist_ok=True)
                torch.save(checkpoint, f"{nn_save_path}/{args.name}_{epoch+1}.nn")

    if local_rank == 0:
        writer.close()
    

if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    # with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
    #     file.write(str(args) + "\n")
    #     file.write(f"time is {time.asctime(time.localtime(time.time()))}\n")
        
    if args.seed is not None:
        set_seed(args.seed)

    # main(args)
    if args.train["if_multi_gpu"]:
        world_size = torch.cuda.device_count()  # 获取GPU数量
        print(f"Let's use {world_size} GPUs!")
    
    main(args)
    
    # with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
    #     file.write(f"time is {time.asctime( time.localtime(time.time()) )}\n")