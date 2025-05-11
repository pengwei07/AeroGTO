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
from src.train import train, validate
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
        paddle.assign(paddle.full_like(m.bias, 0.01), m.bias)
    # elif isinstance(m, paddle.nn.MultiHeadAttention):
    #     init_XavierUniform = paddle.nn.initializer.XavierUniform()
    #     paddle.assign(paddle.full_like(m.weight_attr, 0.01), m.weight_attr)
    #     init_XavierUniform = paddle.nn.initializer.XavierUniform()
    #     init_XavierUniform(m.out_proj.weight)
    #     if m.out_proj.bias is not None:
    #         paddle.assign(paddle.full_like(m.out_proj.bias, 0.01), m.out_proj.bias)



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
    assert args.task in ["cd_pred", "p_pred", "v_pred"]
    model = get_model(args)
    if args.model["if_init"]:
        model.apply(init_weights)
    if args.train["if_multi_gpu"]:
        print(f"Let's use {paddle.device.cuda.device_count()} GPUs!")
        model = paddle.DataParallel(layers=model)
    model_parameters = filter(lambda p: not p.stop_gradient, model.parameters())
    params = sum([np.prod(tuple(p.shape)) for p in model_parameters])
    if args.task == "cd_pred":
        train_dataset = Drag_Dataset(
            data_path=args.dataset["train_data_path"],
            edge_type=args.dataset["edge_type"],
        )
        test_dataset = Drag_Dataset(
            data_path=args.dataset["test_data_path"],
            edge_type=args.dataset["edge_type"],
        )
    elif args.task == "p_pred":
        train_dataset = Pressure_Dataset(
            data_path=args.dataset["train_data_path"],
            edge_type=args.dataset["edge_type"],
        )
        test_dataset = Pressure_Dataset(
            data_path=args.dataset["test_data_path"],
            edge_type=args.dataset["edge_type"],
        )
    elif args.task == "v_pred":
        train_dataset = Velocity_Dataset(data_path=args.dataset["train_data_path"])
        test_dataset = Velocity_Dataset(data_path=args.dataset["test_data_path"])
    train_dataloader = paddle.io.DataLoader(
        dataset=train_dataset,
        batch_size=args.dataset["train"]["batchsize"],
        shuffle=args.dataset["train"]["shuffle"],
        num_workers=args.dataset["train"]["num_workers"],
        collate_fn=collate,
    )
    test_dataloader = paddle.io.DataLoader(
        dataset=test_dataset,
        batch_size=args.dataset["test"]["batchsize"],
        shuffle=args.dataset["test"]["shuffle"],
        num_workers=args.dataset["test"]["num_workers"],
        collate_fn=collate,
    )
    EPOCH = args.train["epoch"]
    warmup_epochs = 5
    print("#############")
    print("#params:", params)
    print(f"EPOCH: {EPOCH}")
    print(f"model name: {args.model['name']}")
    print(
        f"No. of train samples: {len(train_dataset)}, No. of test samples: {len(test_dataset)}"
    )
    print(
        f"No. of train batches: {len(train_dataloader)}, No. of test batches: {len(test_dataloader)}"
    )
    print("#############")
    with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
        file.write(
            f"""No. of train samples: {len(train_dataset)}, No. of test samples: {len(test_dataset)}
"""
        )
        file.write(
            f"""No. of train batches: {len(train_dataloader)}, No. of test batches: {len(test_dataloader)}
"""
        )
        file.write(f"Let's use {paddle.device.cuda.device_count()} GPUs!\n")
        file.write(f"{args.name}, #params: {params}\n")
        file.write(f"EPOCH: {EPOCH}\n")
    log_dir = f"{args.save_path}/logs/{args.name}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    real_lr = float(args.train["lr"])
    optim = paddle.optimizer.AdamW(
        parameters=model.parameters(), learning_rate=real_lr, weight_decay=0.0
    )
    tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(
        T_max=EPOCH, eta_min=float(args.train["eta_min"]), learning_rate=optim.get_lr()
    )
    optim.set_lr_scheduler(tmp_lr)
    scheduler = tmp_lr
    for epoch in range(EPOCH):
        start_time = time.time()
        train_error = train(args, model, train_dataloader, optim, device)
        end_time = time.time()
        scheduler.step()
        current_lr = scheduler.get_lr()
        training_time = end_time - start_time
        current_lr = paddle.to_tensor(data=current_lr, place=device)
        train_loss = paddle.to_tensor(data=train_error["loss"], place=device)
        L2 = paddle.to_tensor(data=train_error["L2"], place=device)
        L2_norm = paddle.to_tensor(data=train_error["L2_norm"], place=device)
        training_time = paddle.to_tensor(data=training_time, place=device)
        writer.add_scalar("lr/lr", float(current_lr), epoch)
        writer.add_scalar("Loss/train", float(train_loss), epoch)
        writer.add_scalar("L2/train_L2", float(L2), epoch)
        writer.add_scalar("L2/train_L2_norm", float(L2_norm), epoch)
        with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
            file.write(f"Epoch: {epoch + 1}/{EPOCH}, Train Loss: {train_loss:.4f}\n")
            file.write(f"L2_norm: {L2_norm:.4f}, L2: {L2:.4f}\n")
            file.write(
                f"time pre train epoch/s:{training_time:.2f}, current_lr:{current_lr:.4e}\n"
            )
        if (epoch + 1) % 1 == 0 or epoch == 0 or epoch + 1 == EPOCH:
            print(f"Epoch: {epoch + 1}/{EPOCH}, Train Loss: {train_loss:.4f}")
            print(f"L2_norm: {L2_norm:.4f}, L2: {L2:.4f}")
            print(
                f"time pre train epoch/s:{training_time:.2f}, current_lr:{current_lr:.4e}"
            )
            print("#################")
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == EPOCH:
            start_time = time.time()
            test_error = validate(args, model, test_dataloader, device=device)
            end_time = time.time()
            training_time1 = end_time - start_time
            test_L2 = paddle.to_tensor(data=test_error["L2"], place=device)
            test_L2_norm = paddle.to_tensor(data=test_error["L2_norm"], place=device)
            test_L2 = gather_tensor(test_L2)
            test_L2_norm = gather_tensor(test_L2_norm)
            print(
                f"Epoch: {epoch + 1}/{EPOCH}, test_L2_norm: {test_L2_norm:.4f}, test_L2: {test_L2:.4f}"
            )
            print(f"time pre test epoch/s:{training_time1:.2f}")
            print("#################")
            writer.add_scalar("L2/test_L2", float(test_L2), epoch)
            writer.add_scalar("L2/test_L2_norm", float(test_L2_norm), epoch)
            with open(
                f"{args.save_path}/record/{args.name}_training_log.txt", "a"
            ) as file:
                file.write(
                    f"""Epoch: {epoch + 1}/{EPOCH}, test_L2_norm: {test_L2_norm:.4f}, test_L2: {test_L2:.4f}
"""
                )
                file.write(f"time pre test epoch/s:{training_time1:.2f}\n")
        if (epoch + 1) % 100 == 0 or epoch == 0 or epoch + 1 == EPOCH:
            if args.if_save:
                checkpoint = {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict()
                    if args.train["if_multi_gpu"]
                    else model.state_dict(),
                    "optimizer": optim.state_dict(),
                    "learning_rate": scheduler.get_lr(),
                }
                nn_save_path = os.path.join(args.save_path, "nn")
                os.makedirs(nn_save_path, exist_ok=True)
                paddle.save(
                    obj=checkpoint, path=f"{nn_save_path}/{args.name}_{epoch + 1}.nn"
                )
    writer.close()


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
    if not os.path.exists(f"{args.save_path}/record/"):
        os.makedirs(f"{args.save_path}/record/")
    with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
        file.write(str(args) + "\n")
        file.write(f"time is {time.asctime(time.localtime(time.time()))}\n")
    set_seed(args.seed)
    main(args)
    with open(f"{args.save_path}/record/{args.name}_training_log.txt", "a") as file:
        file.write(f"time is {time.asctime(time.localtime(time.time()))}\n")
