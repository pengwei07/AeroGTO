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
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

import pickle

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

@torch.no_grad()
def eval(device, model, datamodule, config, loss_fn=None):
    model.eval()
    test_loader = datamodule.test_dataloader(batch_size=config.batch_size, shuffle=False, num_workers=0)

    eval_meter = AverageMeterDict()
    visualize_data_dicts = []
    print(f"len(test_loader) : {len(test_loader)}")
    logging.info(f"len(test_loader) : {len(test_loader)}")

    test_l2 = np.empty((len(test_loader)))
    test_l2_decoded = np.empty((len(test_loader)))

    pressure_pred_decode = []
    pressure_truth_decode = []

    drag_pred = np.empty((len(test_loader)))
    drag_truth = np.empty((len(test_loader)))

    for i, data_dict in enumerate(test_loader):
        
        print(f"{i} th ahmed dataset")
        # logging.info(f"{i} th ahmed dataset")

        out_dict, truth_decode, pred_decode = model.eval_dict(device, data_dict, loss_fn=loss_fn, decode_fn=datamodule.decode)
        eval_meter.update(out_dict)

        test_l2[i] = out_dict["l2"].item()
        test_l2_decoded[i] = out_dict["l2_decoded"].item()

        pressure_pred_decode.append(pred_decode.detach().cpu())
        pressure_truth_decode.append(truth_decode.detach().cpu())

        drag_pred[i] = out_dict['drag_pred'].item()
        drag_truth[i] =  out_dict['drag_truth'].item()

        if i % config.test_plot_interval == 0:
            visualize_data_dicts.append(data_dict)

    # Merge all dictionaries
    merged_image_dict = {}
    if hasattr(model, "image_dict"):
        for i, data_dict in enumerate(visualize_data_dicts):
            image_dict = model.image_dict(data_dict)
            for k, v in image_dict.items():
                merged_image_dict[f"{k}_{i}"] = v

    model.train()

    return eval_meter.avg, merged_image_dict, test_l2, test_l2_decoded, drag_pred, drag_truth, pressure_pred_decode, pressure_truth_decode

def train(config, device: Union[torch.device, str] = "cuda:1"):
    
    logging.basicConfig(filename='result/testing_log.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

    # Initialize the device
    device = torch.device(device)

    # Initialize the dataloaders
    datamodule = instantiate_datamodule(config)
    
    # loggers = init_logger(config)

    # Initialize the optimizer
    # Initialize the model
    model = instantiate_network(config).to(device)

    checkpoint = torch.load(f"result/model_new_iter_try-{config.model}-{config.num_epochs - 1}.pth", map_location = device)
    model.load_state_dict(checkpoint)
    
    # Initialize the loss function
    loss_fn = LpLoss(size_average=True)
    logging.info("#####################################")
    logging.info(f"starting test process {config.model}")
    logging.info("#####################################")

    t1 = default_timer()
    eval_dict, eval_images, test_l2, test_l2_decoded, drag_pred, drag_truth, pressure_pred_decode, pressure_truth_decode = eval(device, model, datamodule, config, loss_fn)
    t2 = default_timer()

    print(f"Testing took {t2 - t1:.2f} seconds.")
    logging.info(f"Testing took {t2 - t1:.2f} seconds.")

    for k, v in eval_dict.items():
        print(f"{k}: {v:.4f}")
        logging.info(f"{k}: {v:.4f}")
    
    # print("the drags' MRE is:")
    # logging.info(f"the drags' MRE is:")

    number_drag = drag_pred.shape[0]
    mre_drag = np.empty((number_drag))

    for i in range(number_drag):
        drag_pred_i = drag_pred[i]
        drag_truth_i = drag_truth[i]
        mre_drag[i] = abs(drag_pred_i - drag_truth_i) / abs(drag_truth_i)
    
    print("#######################################")
    print(f"the test datasets' L2 error is: {test_l2.mean():.4f}")
    logging.info(f"the test datasets' L2 error is: {test_l2.mean():.4f}")

    print(f"the test datasets' L2 decoded error is: {test_l2_decoded.mean():.4f}")
    logging.info(f"the test datasets' L2 decoded error is: {test_l2_decoded.mean():.4f}")

    print(f"the drags' MRE is: {mre_drag.mean():.4f}")
    logging.info(f"the drags' MRE is: {mre_drag.mean():.4f}")
    
    #np.save(f"result/pressure_pred_decode_{config.num_epochs}.npy", pressure_pred_decode)
    #np.save(f"result/pressure_truth_decode_{config.num_epochs}.npy", pressure_truth_decode)

    torch.save(pressure_pred_decode, f"result/pressure/pressure_pred_decode_{config.model}_{config.num_epochs}.pt")
    torch.save(pressure_truth_decode, f"result/pressure/pressure_truth_decode_{config.model}_{config.num_epochs}.pt")
    
    np.save(f"result/gino_drag_pred_{config.model}_{config.num_epochs}.npy", drag_pred)
    np.save(f"result/gino_drag_truth_{config.model}_{config.num_epochs}.npy", drag_truth)

# python test_try_new.py --config configs/GNOFNOGNOAhmed_try.yaml --data_path /workspace/data/ahmed-with-info --device cuda:0 --num_epochs 10

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
