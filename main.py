import argparse
# import yaml
import json
from types import SimpleNamespace
import time
from torch.utils.data import DataLoader
import torch.nn as nn
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
from utils.dataset import Car_Dataset
from model.AeroGTO import Model
from utils.train import train, infer, validate
from model.IPOT import EncoderProcessorDecoder
from model.meshgrapnent import meshgrapnent
from model.transolver import transolver
from model.gnot import GNOT

# os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print(device) 

def set_seed(seed: int = 0):    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
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

def get_model(args):
    if args.model["name"] == "AeroGTO":
        model =  Model(
                N_block = args.model["N_block"],
                state_embedding_dim=args.model["state_embedding_dim"], 
                att_embedding_dim=args.model["att_embedding_dim"],
                n_head=args.model["n_head"],
                n_token=args.model["n_token"]
            )
    elif args.model["name"] == "IPOT":
        model =  EncoderProcessorDecoder(
            input_channel = args.model["input_channel"],
            pos_channel = args.model["pos_channel"],
            num_bands = args.model["num_bands"],
            max_resolution = args.model["max_resolution"],
            num_latents = args.model["num_latents"],
            latent_channel = args.model["latent_channel"],
            self_per_cross_attn = args.model["self_per_cross_attn"],
            cross_heads_num = args.model["cross_heads_num"],
            self_heads_num = args.model["self_heads_num"],
            cross_heads_channel = args.model["cross_heads_channel"],
            self_heads_channel = args.model["self_heads_channel"],
            ff_mult = args.model["ff_mult"],
            latent_init_scale = args.model["latent_init_scale"],
            output_scale = args.model["output_scale"],
            output_channel = args.model["output_channel"],
            position_encoding_type = args.model["position_encoding_type"],
        )
        
    elif args.model["name"] == "MGN":
        model =  meshgrapnent(
            N=args.model["N"], 
            state_size=args.model["state_size"],
        )
    elif args.model["name"] == "GNOT":
        model =  GNOT(
            trunk_size=args.model["trunk_size"],
            branch_sizes=args.model["branch_sizes"],
            space_dim=args.model["space_dim"],
            output_size=args.model["output_size"],
            n_layers=args.model["n_layers"],
            n_hidden=args.model["n_hidden"],
            n_head=args.model["n_head"],
            n_experts = args.model["n_experts"],
            n_inner = args.model["n_inner"],
            mlp_layers=args.model["mlp_layers"],
            attn_type=args.model["attn_type"],
            act = args.model["act"],
            ffn_dropout=args.model["ffn_dropout"],
            attn_dropout=args.model["attn_dropout"],
            horiz_fourier_dim = args.model["horiz_fourier_dim"],
        )
    elif args.model["name"] == "Transolver":
        model =  transolver(
            space_dim=args.model["space_dim"],
            n_layers=args.model["n_layers"],
            n_hidden=args.model["n_hidden"],
            dropout=args.model["dropout"],
            n_head=args.model["n_head"],
            act=args.model["act"],
            mlp_ratio=args.model["mlp_ratio"],
            fun_dim=args.model["fun_dim"],
            out_dim=args.model["out_dim"],
            slice_num=args.model["slice"],
            ref=args.model["ref"],
            unified_pos=args.model["unified_pos"]
        )
    return model

def main(rank, world_size, args):

    torch.autograd.set_detect_anomaly(True)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '54321' 
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model =  get_model(args).to(rank)
    
    if args.model["if_init"]:
        model.apply(init_weights)
        
    # checkpoint_path = "xxxxxxx.nn" 
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # model.load_state_dict(checkpoint['state_dict'])
    
    model = DDP(model, device_ids=[rank])
    # model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())    
    params = sum([np.prod(p.size()) for p in model_parameters])
    
    # load data
    train_dataset = Car_Dataset(
        data_path = args.dataset["data_path"],
        mode="train",
        if_sample = True,
        adj_num = args.dataset["adj_num"],
        edges_ratio = args.dataset["edges_ratio"],
        node_ratio = args.dataset["node_ratio"],
        )
    
    test_dataset_part = Car_Dataset( 
        data_path = args.dataset["data_path"],
        mode="test",
        if_sample = True,
        adj_num = args.dataset["adj_num"],
        edges_ratio = args.dataset["edges_ratio"],
        node_ratio = args.dataset["node_ratio"],
        )
    
    test_dataset_all = Car_Dataset( 
        data_path = args.dataset["data_path"],
        mode="test",
        if_sample = True,
        adj_num = args.dataset["adj_num"],
        edges_ratio = args.dataset["edges_ratio"],
        node_ratio = args.dataset["node_ratio"],
        if_zero_shot = True
        )

    # sampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, shuffle=True, seed=args.dataset["shuffle_seed"], rank=rank)
    
    test_sampler_part = DistributedSampler(test_dataset_part, num_replicas=world_size, shuffle=False, rank=rank)
    
    test_sampler_all = DistributedSampler(test_dataset_all, num_replicas=world_size, shuffle=False, rank=rank)
    
        
    train_dataloader = DataLoader(train_dataset, 
                        batch_size=args.dataset["train"]["batchsize"], 
                        shuffle=args.dataset["train"]["shuffle"], 
                        sampler=train_sampler,
                        num_workers=args.dataset["train"]["num_workers"])
    
    test_dataloader_part = DataLoader(test_dataset_part, 
                        batch_size=args.dataset["test"]["batchsize"], 
                        shuffle=args.dataset["test"]["shuffle"], 
                        sampler= test_sampler_part,
                        num_workers=args.dataset["test"]["num_workers"])
    
    test_dataloader_all = DataLoader(test_dataset_all, 
                        batch_size=args.dataset["test"]["batchsize"], 
                        shuffle=args.dataset["test"]["shuffle"], 
                        sampler= test_sampler_all,
                        num_workers=args.dataset["test"]["num_workers"])
    
    
    EPOCH = args.train["epoch"]

    if rank == 0:
        
        print("---train_dataloader---")
        for i, [data, _] in enumerate(train_dataloader):
            for key in data.keys():
                print(key, data[key].shape)
            break
        print("---test_dataloader_part---")
        for i, [data, _] in enumerate(test_dataloader_part):
            for key in data.keys():
                print(key, data[key].shape)
            break
        print("---test_dataloader_all---")
        for i, [data, _] in enumerate(test_dataloader_all):
            for key in data.keys():
                if key=="info":
                    print(key, data[key].shape)
                else:
                    print(key, torch.stack(data[key],dim=1).shape)
            break
        print("#############")
        '''
        node_pos torch.Size([4, 3586, 3])
        edges torch.Size([4, 21504, 2])
        pressure torch.Size([4, 3586, 1])
        cells torch.Size([4, 7168, 3])     
        '''
        print(f"No. of train samples: {len(train_dataset)}, No. of test samples part: {len(test_dataset_part)}, No. of test samples all: {len(test_dataset_all)}")
        print(f"No. of train batches: {len(train_dataloader)}, No. of test batches part: {len(test_dataloader_part)}, No. of test batches all: {len(test_dataloader_all)}")
        print("#############")
        print("#params:", int(params))
        print(f"EPOCH: {EPOCH}")
        print("#############")
        
        with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
            file.write(f"No. of train samples: {len(train_dataset)}, No. of test samples part: {len(test_dataset_part)}, No. of test samples all: {len(test_dataset_all)}\n")
            file.write(f"No. of train batches: {len(train_dataloader)}, No. of test batches part: {len(test_dataloader_part)}, No. of test batches all: {len(test_dataloader_all)}\n")
            file.write(f"Let's use {torch.cuda.device_count()} GPUs!\n")
            file.write(f"{args.name}, #params: {int(params)}\n")
            file.write(f"EPOCH: {EPOCH}\n")
            
        log_dir = f"{args.save_path}/logs/{args.name}/rank_{rank}"
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        
    real_lr = float(args.train["lr"])
    optim = torch.optim.AdamW(model.parameters(), lr=real_lr)
    scheduler = CosineAnnealingLR(optim, T_max= EPOCH, eta_min = float(args.train["eta_min"]))
    
    scaler = GradScaler()
    for epoch in range(EPOCH):
        start_time = time.time()
        train_error, scaler = train(args, model, train_dataloader, optim, rank, scaler)
        end_time = time.time()
        
        # 获取当前的学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        training_time = (end_time - start_time)
        current_lr = torch.tensor(current_lr, device=device)
        train_loss = torch.tensor(train_error['loss'], device=device)
        L2_p = torch.tensor(train_error['L2_p'], device=device)
        L2_p_norm = torch.tensor(train_error['L2_p_norm'], device=device)
        
        training_time = torch.tensor(training_time, device=device)
        current_lr = gather_tensor(current_lr, world_size)
        
        train_loss = gather_tensor(train_loss, world_size)
        L2_p = gather_tensor(L2_p, world_size)
        L2_p_norm = gather_tensor(L2_p_norm, world_size)
        
        training_time = gather_tensor(training_time, world_size)
        
        if rank == 0:
            writer.add_scalar('lr/lr', current_lr, epoch)
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('L2/train_L2_p', L2_p, epoch)
            writer.add_scalar('L2/train_L2_p_norm', L2_p_norm, epoch)
            
            with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
                file.write(f"Epoch: {epoch + 1}/{EPOCH}, Train Loss: {train_loss:.4f}\n")
                file.write(f"L2_p_norm: {L2_p_norm:.4f}, L2_p: {L2_p:.4f}\n")
                file.write(f"time pre train epoch/s:{training_time:.2f}, current_lr:{current_lr:.4e}\n")
            
            if (epoch+1) % 1 == 0 or epoch == 0 or (epoch+1) == EPOCH:
                print("---Train part---")
                print(f"Epoch: {epoch + 1}/{EPOCH}, Train Loss: {train_loss:.4f}")
                print(f"L2_p_norm: {L2_p_norm:.4f}, L2_p: {L2_p:.4f}")
                print(f"time pre train epoch/s:{training_time:.2f}, current_lr:{current_lr:.4e}")
                # print("#################")

        if (epoch+1) % 10 == 0 or epoch == 0 or (epoch+1) == EPOCH:
            # test part
            start_time = time.time() 
            flag="part"
            test_error_part =  validate(args, model, test_dataloader_part, rank, flag)
            end_time = time.time()
            training_time1 = (end_time - start_time)
            
            # test part
            start_time = time.time() 
            flag="all"
            test_error_all =  validate(args, model, test_dataloader_all, rank, flag)
            end_time = time.time()
            training_time2 = (end_time - start_time)
            
            ###############################################################
            test_L2_p_part = torch.tensor(test_error_part['L2_p'], device=device)
            test_L2_p_norm_part = torch.tensor(test_error_part['L2_p_norm'], device=device)
            
            test_L2_p_all = torch.tensor(test_error_all['L2_p'], device=device)
            test_L2_p_norm_all = torch.tensor(test_error_all['L2_p_norm'], device=device)
            test_MRE_drag = torch.tensor(test_error_all['MRE_drag'], device=device)
            
            ###############################################################
            test_L2_p_part = gather_tensor(test_L2_p_part, world_size)
            test_L2_p_norm_part = gather_tensor(test_L2_p_norm_part, world_size)
            
            test_L2_p_all = gather_tensor(test_L2_p_all, world_size)
            test_L2_p_norm_all = gather_tensor(test_L2_p_norm_all, world_size)
            test_MRE_drag = gather_tensor(test_MRE_drag, world_size)
            
            if rank == 0:
                print("---Test part---")
                print(f"Epoch: {epoch + 1}/{EPOCH}, test_L2_p_norm: {test_L2_p_norm_part:.4f}, test_L2_p: {test_L2_p_part:.4f}")
                print(f"time pre test epoch/s:{training_time1:.2f}")
                print("---Test all---")
                print(f"Epoch: {epoch + 1}/{EPOCH}, test_L2_p_norm: {test_L2_p_norm_all:.4f}, test_L2_p: {test_L2_p_all:.4f}, test_MRE_drag: {test_MRE_drag:.4f}")
                print(f"time pre test epoch/s:{training_time2:.2f}")
                writer.add_scalar('L2/test_L2_p_part', test_L2_p_part, epoch)
                writer.add_scalar('L2/test_L2_p_norm_part', test_L2_p_norm_part, epoch)
                
                writer.add_scalar('L2/test_L2_p_part', test_L2_p_all, epoch)
                writer.add_scalar('L2/test_L2_p_norm_part', test_L2_p_norm_all, epoch)
                writer.add_scalar('L2/test_MRE_drag', test_MRE_drag, epoch)
                
                with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
                    
                    file.write(f"#################\n")
                    file.write(f"---Test part---\n")
                    file.write(f"Epoch: {epoch + 1}/{EPOCH}, test_L2_p_norm: {test_L2_p_norm_part:.4f}, test_L2_p: {test_L2_p_part:.4f}\n")
                    file.write(f"time pre test epoch/s:{training_time1:.2f}\n")
                    file.write(f"---Test all---\n")
                    file.write(f"Epoch: {epoch + 1}/{EPOCH}, test_L2_p_norm: {test_L2_p_norm_all:.4f}, test_L2_p: {test_L2_p_all:.4f}, test_MRE_drag: {test_MRE_drag:.4f}\n")
                    file.write(f"time pre test epoch/s:{training_time2:.2f}\n")
                    file.write(f"#################\n")
                    
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

    if rank == 0:
        writer.close()
    
    ##################    
    # infer(args, model, test_dataloader, rank)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
        file.write(str(args) + "\n")
        file.write(f"time is {time.asctime(time.localtime(time.time()))}\n")
        
    if args.seed is not None:
        set_seed(args.seed)
    
    if args.train["if_multi_gpu"]:
        world_size = torch.cuda.device_count()
        print(f"Let's use {world_size} GPUs!")
        torch.multiprocessing.spawn(main, args=(world_size, args), nprocs=world_size)
    
    with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
        file.write(f"time is {time.asctime( time.localtime(time.time()) )}\n")