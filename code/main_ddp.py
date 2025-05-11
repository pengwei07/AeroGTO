import argparse
import json
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import paddle
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


def setup():
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    paddle.distributed.init_parallel_env()
    return rank, world_size


def gather_tensor(tensor, world_size):
    """
    Gathers tensors from all processes and reduces them by summing up.
    """
    tensor = tensor.to(device)
    paddle.distributed.all_reduce(tensor=tensor, op=paddle.distributed.ReduceOp.SUM)
    if paddle.distributed.get_rank() == 0:
        tensor /= world_size
    return tensor


def get_model(args, rank):
    if args.task == "cd_pred":
        model = AeroGTO_cd(
            N_block=args.model["N_block"],
            state_size=args.model["state_size"],
            state_embedding_dim=args.model["state_embedding_dim"],
            att_embedding_dim=args.model["att_embedding_dim"],
            n_head=args.model["n_head"],
            n_token=args.model["n_token"],
        ).to(rank)
    else:
        model = AeroGTO(
            N_block=args.model["N_block"],
            state_size=args.model["state_size"],
            state_embedding_dim=args.model["state_embedding_dim"],
            att_embedding_dim=args.model["att_embedding_dim"],
            n_head=args.model["n_head"],
            n_token=args.model["n_token"],
        ).to(rank)
    return model


def main(args):
    local_rank, world_size = setup()
    model = get_model(args, local_rank)
    if args.model["if_init"]:
        model.apply(init_weights)
    model = paddle.DataParallel(layers=model, find_unused_parameters=True)
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
    train_sampler = paddle.io.DistributedBatchSampler(
        dataset=train_dataset,
        num_replicas=world_size,
        shuffle=True,
        rank=local_rank,
        batch_size=1,
    )
    test_sampler = paddle.io.DistributedBatchSampler(
        dataset=test_dataset,
        num_replicas=world_size,
        shuffle=False,
        rank=local_rank,
        batch_size=1,
    )
>>>>>>    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.dataset["train"]["batchsize"],
        shuffle=args.dataset["train"]["shuffle"],
        sampler=train_sampler,
        num_workers=args.dataset["train"]["num_workers"],
        collate_fn=collate,
    )
>>>>>>    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.dataset["test"]["batchsize"],
        shuffle=args.dataset["test"]["shuffle"],
        sampler=test_sampler,
        num_workers=args.dataset["test"]["num_workers"],
        collate_fn=collate,
    )
    EPOCH = args.train["epoch"]
    if local_rank == 0:
        for i, [data, _] in enumerate(train_dataloader):
            for key in data.keys():
                print(key, tuple(data[key].shape))
            break
        print("#############")
        """
        node_pos torch.Size([4, 3586, 3])
        edges torch.Size([4, 21504, 2])
        pressure torch.Size([4, 3586, 1])
        """
        print(
            f"No. of train samples: {len(train_dataset)}, No. of test samples: {len(test_dataset)}"
        )
        print(
            f"No. of train batches: {len(train_dataloader)}, No. of test batches: {len(test_dataloader)}"
        )
        print("#############")
        print("#params:", params)
        print(f"EPOCH: {EPOCH}")
        print(f"model name: {args.model['name']}")
        print("#############")
        if not os.path.exists(f"{args.save_path}/record"):
            os.makedirs(f"{args.save_path}/record")
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
        log_dir = f"{args.save_path}/logs/{args.name}/rank_{local_rank}"
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
        train_error = train(args, model, train_dataloader, optim, local_rank)
        end_time = time.time()
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        training_time = end_time - start_time
        current_lr = paddle.to_tensor(data=current_lr, place=device)
        train_loss = paddle.to_tensor(data=train_error["loss"], place=device)
        L2 = paddle.to_tensor(data=train_error["L2"], place=device)
        L2_norm = paddle.to_tensor(data=train_error["L2_norm"], place=device)
        training_time = paddle.to_tensor(data=training_time, place=device)
        current_lr = gather_tensor(current_lr, world_size)
        train_loss = gather_tensor(train_loss, world_size)
        L2 = gather_tensor(L2, world_size)
        L2_norm = gather_tensor(L2_norm, world_size)
        training_time = gather_tensor(training_time, world_size)
        if local_rank == 0:
            writer.add_scalar("lr/lr", current_lr, epoch)
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("L2/train_L2", L2, epoch)
            writer.add_scalar("L2/train_L2_norm", L2_norm, epoch)
            with open(
                f"{args.save_path}/record/{args.name}_training_log.txt", "a"
            ) as file:
                file.write(
                    f"Epoch: {epoch + 1}/{EPOCH}, Train Loss: {train_loss:.4f}\n"
                )
                file.write(f"L2_norm: {L2_norm:.4f}, L2: {L2:.4f}\n")
                file.write(
                    f"""time pre train epoch/s:{training_time:.2f}, current_lr:{current_lr:.4e}
"""
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
            test_error = validate(args, model, test_dataloader, device=local_rank)
            end_time = time.time()
            training_time1 = end_time - start_time
            test_L2 = paddle.to_tensor(data=test_error["L2"], place=device)
            test_L2_norm = paddle.to_tensor(data=test_error["L2_norm"], place=device)
            test_L2 = gather_tensor(test_L2, world_size)
            test_L2_norm = gather_tensor(test_L2_norm, world_size)
            if local_rank == 0:
                print(
                    f"Epoch: {epoch + 1}/{EPOCH}, test_L2_norm: {test_L2_norm:.4f}, test_L2: {test_L2:.4f}"
                )
                print(f"time pre test epoch/s:{training_time1:.2f}")
                print("#################")
                writer.add_scalar("L2/test_L2_norm", test_L2_norm, epoch)
                writer.add_scalar("L2/test_L2", test_L2, epoch)
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
                    "state_dict": model.module.state_dict()
                    if args.train["if_multi_gpu"]
                    else model.state_dict(),
                    "optimizer": optim.state_dict(),
                    "learning_rate": scheduler.get_last_lr()[0],
                }
                nn_save_path = os.path.join(args.save_path, "nn")
                os.makedirs(nn_save_path, exist_ok=True)
                paddle.save(
                    obj=checkpoint, path=f"{nn_save_path}/{args.name}_{epoch + 1}.nn"
                )
    if local_rank == 0:
        writer.close()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    if args.seed is not None:
        set_seed(args.seed)
    if args.train["if_multi_gpu"]:
        world_size = paddle.device.cuda.device_count()
        print(f"Let's use {world_size} GPUs!")
    main(args)
