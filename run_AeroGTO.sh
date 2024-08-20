export OMP_NUM_THREADS=32
export NCCL_P2P_DISABLE=1 
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=1

export CUDA_VISIBLE_DEVICES=0,1,2
# ./configs/$case_name: stores the corresponding config files of a case

# 1. AeroGTO
python main.py --config ./configs/AeroGTO.json

# 2. GNOT
python main.py --config ./configs/GNOT.json

# 3. IPOT
python main.py --config ./configs/IPOT.json

# 4. MGN
python main.py --config ./configs/MGN.json

# 5. Transolver
python main.py --config ./configs/Transolver.json

# 6. GINO
# see ./gino/
