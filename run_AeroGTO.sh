# sudo iptables -A INPUT -p tcp --dport 5928 -j ACCEPT

export OMP_NUM_THREADS=32
export NCCL_P2P_DISABLE=1 
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=1

export CUDA_VISIBLE_DEVICES=0,1,2

python main.py --config ./configs/AeroGTO_4x.json