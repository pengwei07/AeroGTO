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
    assert args.task in ["cd_pred", "p_pred", "v_pred"]
    
    model = get_model(args)
    # model.load_state_dict(torch.load(args.resume_path)["state_dict"])
    
    if args.model["if_init"]:
        model.apply(init_weights)
    
    if args.train["if_multi_gpu"]:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = DataParallel(model)
    
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
        
        
    train_dataloader = DataLoader(train_dataset,
                        batch_size=args.dataset["train"]["batchsize"], 
                        shuffle=args.dataset["train"]["shuffle"], 
                        num_workers=args.dataset["train"]["num_workers"],
                        collate_fn = collate)

    test_dataloader = DataLoader(test_dataset, 
                            batch_size=args.dataset["test"]["batchsize"], 
                            shuffle=args.dataset["test"]["shuffle"], 
                            num_workers=args.dataset["test"]["num_workers"],
                            collate_fn = collate)

    EPOCH = args.train["epoch"]
    warmup_epochs = 5
        
    print("#############")
    print("#params:", params)
    print(f"EPOCH: {EPOCH}")
    print(f"model name: {args.model['name']}")
    
    print(f"No. of train samples: {len(train_dataset)}, No. of test samples: {len(test_dataset)}")
    print(f"No. of train batches: {len(train_dataloader)}, No. of test batches: {len(test_dataloader)}")
    print("#############")
    
    with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
        file.write(f"No. of train samples: {len(train_dataset)}, No. of test samples: {len(test_dataset)}\n")
        file.write(f"No. of train batches: {len(train_dataloader)}, No. of test batches: {len(test_dataloader)}\n")
        file.write(f"Let's use {torch.cuda.device_count()} GPUs!\n")
        file.write(f"{args.name}, #params: {params}\n")
        file.write(f"EPOCH: {EPOCH}\n")
        
    log_dir = f"{args.save_path}/logs/{args.name}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    real_lr = float(args.train["lr"])
    optim = torch.optim.AdamW(model.parameters(), lr=real_lr)
    # betas=(0.9,0.999), eps=1e08, weight_decay=0.01,amsgrad=False
    # cosine_scheduler = CosineAnnealingLR(optim, T_max= int(EPOCH // args.train["T_max"]), eta_min = float(args.train["eta_min"]))
    # scheduler = GradualWarmupScheduler(optim, multiplier=10, total_epoch=warmup_epochs, after_scheduler=cosine_scheduler)
    scheduler = CosineAnnealingLR(optim, T_max= EPOCH, eta_min = float(args.train["eta_min"]))
    
    for epoch in range(EPOCH):
        start_time = time.time()
        train_error = train(args, model, train_dataloader, optim, device)
        end_time = time.time()
        
        # 获取当前的学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]  # get_last_lr返回的是列表，我们需要第一个元素，即当前的学习率
        
        training_time = (end_time - start_time)
        current_lr = torch.tensor(current_lr, device=device)
        train_loss = torch.tensor(train_error['loss'], device=device)
        L2 = torch.tensor(train_error['L2'], device=device)
        L2_norm = torch.tensor(train_error['L2_norm'], device=device)
        training_time = torch.tensor(training_time, device=device)
        
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
            start_time = time.time() 
            test_error =  validate(args, model, test_dataloader, device=device)
            end_time = time.time()
            training_time1 = (end_time - start_time)
            
            test_L2 = torch.tensor(test_error['L2'], device=device)
            test_L2_norm = torch.tensor(test_error['L2_norm'], device=device)

            test_L2 = gather_tensor(test_L2)
            test_L2_norm = gather_tensor(test_L2_norm)
            
            print(f"Epoch: {epoch + 1}/{EPOCH}, test_L2_norm: {test_L2_norm:.4f}, test_L2: {test_L2:.4f}")
            print(f"time pre test epoch/s:{training_time1:.2f}")
            print("#################")
            
            writer.add_scalar('L2/test_L2', test_L2, epoch)
            writer.add_scalar('L2/test_L2_norm', test_L2_norm, epoch)
            
            with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
                file.write(f"Epoch: {epoch + 1}/{EPOCH}, test_L2_norm: {test_L2_norm:.4f}, test_L2: {test_L2:.4f}\n")
                file.write(f"time pre test epoch/s:{training_time1:.2f}\n")
        
        if (epoch+1) % 100 == 0 or epoch == 0 or (epoch+1) == EPOCH:
            if args.if_save:
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict() if args.train["if_multi_gpu"] else model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'learning_rate': scheduler.get_last_lr()[0],  # 获取当前学习率
                }
                nn_save_path = os.path.join(args.save_path, "nn")
                os.makedirs(nn_save_path, exist_ok=True)
                torch.save(checkpoint, f"{nn_save_path}/{args.name}_{epoch+1}.nn")

    writer.close()
            
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
    
    if not os.path.exists(f"{args.save_path}/record/"):
        os.makedirs(f"{args.save_path}/record/")
    with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
        file.write(str(args) + "\n")
        file.write(f"time is {time.asctime(time.localtime(time.time()))}\n")
        
    # if args.seed is not None:
    #     set_seed(args.seed)
    set_seed(args.seed)
    
    # # train+val+test
    main(args)
    
    with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
        file.write(f"time is {time.asctime( time.localtime(time.time()) )}\n")