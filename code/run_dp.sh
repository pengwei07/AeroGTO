export CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

python main_dp.py --config ./config/AeroGTO_pressure.json
python main_dp.py --config ./config/AeroGTO_velocity.json
python main_dp.py --config ./config/AeroGTO_cd.json