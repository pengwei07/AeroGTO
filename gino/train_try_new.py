import matplotlib
import logging
matplotlib.use("Agg")  # Set the backend to Agg
import sys
sys.path.append("./neuraloperator")
import os
from typing import List, Tuple, Union
import numpy as np
import yaml
from timeit import default_timer
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import lightning as L
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

fabric = L.Fabric(accelerator="cuda", devices=[0,1,2,3,4,5,6,7], strategy="ddp")
# fabric = L.Fabric(accelerator="cuda", devices=-1, strategy="auto")
# fabric = L.Fabric(accelerator="cuda", devices=1, strategy="auto")
fabric.launch()

from src.data import instantiate_datamodule
from src.networks import instantiate_network
from src.utils.average_meter import AverageMeter, AverageMeterDict
from src.utils.dot_dict import DotDict, flatten_dict
from src.losses import LpLoss
from src.utils.loggers import init_logger
from src.optim.schedulers import instantiate_scheduler

def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def str2intlist(s: str) -> List[int]:
    return [int(item.strip()) for item in s.split(",")]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/FNOInterpAhmed.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",
        help="Device to use for training (cuda or cpu)",
    )
    parser.add_argument(
        "--data_path", type=str, default=None, help="Override data_path in config file"
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint file to resume training",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="1223_ahmed_pressure_551",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="log",
        help="Path to the log directory",
    )
    parser.add_argument("--logger_types", type=str, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for training")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--no_fabric", action="store_true",  help="If --no_fabric, fabric will not be using for parallel.")
    parser.add_argument(
        "--sdf_spatial_resolution",
        type=str2intlist,
        default=None,
        help="SDF spatial resolution. Use comma to separate the values e.g. 32,32,32.",
    )

    args = parser.parse_args()
    return args

def load_config(config_path):
    def include_constructor(loader, node):
        # Get the path of the current YAML file
        current_file_path = loader.name

        # Get the folder containing the current YAML file
        base_folder = os.path.dirname(current_file_path)

        # Get the included file path, relative to the current file
        included_file = os.path.join(base_folder, loader.construct_scalar(node))

        # Read and parse the included file
        with open(included_file, "r") as file:
            return yaml.load(file, Loader=yaml.Loader)

    # Register the custom constructor for !include
    yaml.Loader.add_constructor("!include", include_constructor)

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # Convert to dot dict
    config_flat = flatten_dict(config)
    config_flat = DotDict(config_flat)
    return config_flat

def train(config, device: Union[torch.device, str] = "cuda:1",**kw):
    
    logging.basicConfig(filename='result/training_log.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

    # Initialize the device
    device = torch.device(device)

    # Initialize the dataloaders
    datamodule = instantiate_datamodule(config)
    train_loader = datamodule.train_dataloader(batch_size=config.batch_size, shuffle=True)

    # loggers = init_logger(config)

    # Initialize the optimizer
    # Initialize the model

    model = instantiate_network(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-6)

    model, optimizer = fabric.setup(model, optimizer)
    train_loader = fabric.setup_dataloaders(train_loader)

    # scheduler = instantiate_scheduler(optimizer, config)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    # Initialize the loss function
    loss_fn = LpLoss(size_average=True)

    # N_sample = 1000
    loss_train = np.zeros(config.num_epochs)
    logging.info("#####################################")
    logging.info(f"starting train process {config.model}")
    logging.info("#####################################")
    for ep in range(config.num_epochs):
        model.train()
        t1 = default_timer()
        train_l2_meter = AverageMeter()

        # train_reg = 0
        for data_dict in train_loader:
            optimizer.zero_grad()
            pred, truth = model(data_dict)
            loss = loss_fn(pred, truth)
            # if kw['use_fabric'] :
            fabric.backward(loss)
            # else:loss.backward()
            optimizer.step()
            train_l2_meter.update(loss.item())
        
        scheduler.step()
        
        t2 = default_timer()

        # epoch end
        # logging.info(f'####################################')
        # logging.info(f'Epoch {ep + 1} completed')
        # logging.info(f'####################################')
        if (ep+1) % 10 == 0 or ep == config.num_epochs - 1 or (ep+1) == 1:
            print(f"Training epoch {ep} took {t2 - t1:.2f} seconds. L2 loss: {train_l2_meter.avg:.4f}")
            logging.info(f"Training epoch {ep} took {t2 - t1:.2f} seconds. L2 loss: {train_l2_meter.avg:.4f}")

        loss_train[ep] = train_l2_meter.avg
        
        # Save the weights
        if (ep+1) % 100 == 0 or ep == config.num_epochs - 1 or (ep+1) == 1:
            torch.save(model.state_dict(), f"result/model_new_iter_try-{config.model}-{ep}.pth")
        
        if ep == config.num_epochs - 1:
            np.save(f"result/model_new_train-{config.model}-{ep}.npy", loss_train)

# python train_try_new.py --config configs/GNOFNOGNOAhmed_try.yaml --data_path /workspace/data/ahmed-with-info --num_epochs 4

if __name__ == "__main__":
    args = parse_args()
    # print command line args
    print(args)
    config = load_config(args.config)

    # Update config with command line arguments
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            config[key] = value

    # pretty print the config
    for key, value in config.items():
        print(f"{key}: {value}")

    # Set the random seed
    if config.seed is not None:
        set_seed(config.seed)
        
    train(config, device=args.device)
