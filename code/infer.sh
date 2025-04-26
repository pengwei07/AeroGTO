export CUDA_VISIBLE_DEVICES=0
python infer.py --config ./config/AeroGTO_pressure.json
python infer.py --config ./config/AeroGTO_velocity.json
python infer.py --config ./config/AeroGTO_cd.json
