#!/bin/bash

# 激活环境（如果需要）
# source activate your_env_name
# sudo iptables -A INPUT -p tcp --dport 5920 -j ACCEPT

export OMP_NUM_THREADS=32
export NCCL_P2P_DISABLE=1 
# export TORCH_DISTRIBUTED_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=1
# 设置 GPU 可见性
export CUDA_VISIBLE_DEVICES=0

# 运行 Python 脚本
torchrun --nproc_per_node=1 main_ddp.py --config ./config/AeroGTO_pressure.json
torchrun --nproc_per_node=1 main_ddp.py --config ./config/AeroGTO_velocity.json
torchrun --nproc_per_node=1 main_ddp.py --config ./config/AeroGTO_cd.json
