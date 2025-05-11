import sys

sys.path.append("/workspace/AeroGTO/paddle_project")
import argparse
import json
import os
import time
from types import SimpleNamespace

import h5py
import numpy as np
import paddle
from paddle_utils import *
from src.AeroGTO import AeroGTO
from src.AeroGTO_cd_pred import AeroGTO_cd
from src.cd_dataset import Drag_Dataset
from src.pressure_dataset import Pressure_Dataset
from src.train import infer
from src.utils import collate, set_seed
from src.velocity_dataset import Velocity_Dataset
from tensorboardX import SummaryWriter

device = "cuda" if paddle.device.cuda.device_count() >= 1 else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config.json", type=str, help="Path to config file"
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    args = SimpleNamespace(**config)
    return args


def init_weights(m):
    if isinstance(m, paddle.nn.Linear):
        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(m.weight)
        m.bias.data.fill_(value=0.01)
>>>>>>    elif isinstance(m, torch.nn.MultiheadAttention):
        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(m.in_proj_weight)
        if m.in_proj_bias is not None:
            m.in_proj_bias.data.fill_(0.01)
        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(m.out_proj.weight)
        if m.out_proj.bias is not None:
            m.out_proj.bias.data.fill_(value=0.01)


def gather_tensor(tensor):
    """
    Gathers tensors from all processes and reduces them by summing up.
    """
    tensor = tensor.to(device)
    return tensor


def get_model(args):
    if args.task == "cd_pred":
        model = AeroGTO_cd(
            N_block=args.model["N_block"],
            state_size=args.model["state_size"],
            state_embedding_dim=args.model["state_embedding_dim"],
            att_embedding_dim=args.model["att_embedding_dim"],
            n_head=args.model["n_head"],
            n_token=args.model["n_token"],
        ).to(device)
    else:
        model = AeroGTO(
            N_block=args.model["N_block"],
            state_size=args.model["state_size"],
            state_embedding_dim=args.model["state_embedding_dim"],
            att_embedding_dim=args.model["att_embedding_dim"],
            n_head=args.model["n_head"],
            n_token=args.model["n_token"],
        ).to(device)
    return model


def main(args):
    model = get_model(args)
    model.set_state_dict(
        state_dict=paddle.load(path=str(args.check_point_path))["state_dict"]
    )
    model = model.to(device)
    if args.train["if_multi_gpu"]:
        print(f"Let's use {paddle.device.cuda.device_count()} GPUs!")
        model = paddle.DataParallel(layers=model)
    model_parameters = filter(lambda p: not p.stop_gradient, model.parameters())
    params = sum([np.prod(tuple(p.shape)) for p in model_parameters])
    if args.task == "cd_pred":
        test_dataset = Drag_Dataset(
            data_path=args.dataset["infer_data_path"],
            edge_type=args.dataset["edge_type"],
        )
    elif args.task == "p_pred":
        test_dataset = Pressure_Dataset(
            data_path=args.dataset["infer_data_path"],
            edge_type=args.dataset["edge_type"],
        )
    elif args.task == "v_pred":
        test_dataset = Velocity_Dataset(data_path=args.dataset["infer_data_path"])
    test_dataloader = paddle.io.DataLoader(
        dataset=test_dataset,
        batch_size=args.dataset["test"]["batchsize"],
        shuffle=args.dataset["test"]["shuffle"],
        num_workers=args.dataset["test"]["num_workers"],
        collate_fn=collate,
    )
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
    if "cuda" in device:
        assert paddle.device.cuda.device_count() >= 1
        if (
            device == "cuda"
            and paddle.device.cuda.device_count() > 1
            and args.train["if_multi_gpu"]
        ):
            use_multi_gpu = True
            print(f"lets use {paddle.device.cuda.device_count()} gpus!")
        else:
            use_multi_gpu = False
            print(f"lets use 1 gpu!")
    else:
        use_multi_gpu = False
        print(f"lets use cpu!")
    device = device2str(device)
    print("device:", device)
    set_seed(args.seed)
    main(args)
