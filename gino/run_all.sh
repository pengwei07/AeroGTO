# chmod +x run_all.sh

# train

#python train_try_new.py --config configs/GNOFNOGNOAhmed.yaml --data_path /workspace/ahmed-with-info --num_epochs 300

python train_try_new.py --config configs/GNOFNOGNOAhmed_wss.yaml --data_path /workspace/ahmed-with-info --num_epochs 300

#python test_try_new.py --config configs/GNOFNOGNOAhmed.yaml --data_path /workspace/ahmed-with-info --device cuda:0 --num_epochs 300

python test_try_new_wss.py --config configs/GNOFNOGNOAhmed_wss.yaml --data_path /workspace/ahmed-with-info --device cuda:0 --num_epochs 300