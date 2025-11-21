#!/bin/bash 
#SBATCH --job-name=clft-train                ### job name
#SBATCH -n 1
#SBATCH -p gpu                  ### use gpu
#SBATCH --gres=gpu:A100:1               ### specifying the GPU type
#SBATCH --constraint="A100-80"
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH -t 5-0                  ### time limit 
#SBATCH --mail-type=ALL
#SBATCH --account=project_autolab

## run job   
source venv/bin/activate 

# RGB Baseline
echo "Training RGB Baseline..."
python train_deeplabv3plus.py --config config/zod/deeplabv3/deeplabv3plus_rgb.json
python test_deeplabv3plus.py --config config/zod/deeplabv3/deeplabv3plus_rgb.json
python visualize_deeplabv3plus.py --config config/zod/deeplabv3/deeplabv3plus_rgb.json --upload

# Vehicle Only Specialization
echo "Training Vehicle Only Specialization..."
python train_deeplabv3plus.py --config config/zod/deeplabv3/deeplabv3plus_rgb_vehicle_only.json
python test_deeplabv3plus.py --config config/zod/deeplabv3/deeplabv3plus_rgb_vehicle_only.json
python visualize_deeplabv3plus.py --config config/zod/deeplabv3/deeplabv3plus_rgb_vehicle_only.json --upload

# Sign Only Specialization
echo "Training Sign Only Specialization..."
python train_deeplabv3plus.py --config config/zod/deeplabv3/deeplabv3plus_rgb_sign_only.json
python test_deeplabv3plus.py --config config/zod/deeplabv3/deeplabv3plus_rgb_sign_only.json
python visualize_deeplabv3plus.py --config config/zod/deeplabv3/deeplabv3plus_rgb_sign_only.json --upload

# Human Only Specialization
echo "Training Human Only Specialization..."
python train_deeplabv3plus.py --config config/zod/deeplabv3/deeplabv3plus_rgb_human_only.json
python test_deeplabv3plus.py --config config/zod/deeplabv3/deeplabv3plus_rgb_human_only.json
python visualize_deeplabv3plus.py --config config/zod/deeplabv3/deeplabv3plus_rgb_human_only.json --upload

# Vehicle + Sign Specialization
echo "Training Vehicle + Sign Specialization..."
python train_deeplabv3plus.py --config config/zod/deeplabv3/deeplabv3plus_rgb_vehicle_sign.json
python test_deeplabv3plus.py --config config/zod/deeplabv3/deeplabv3plus_rgb_vehicle_sign.json
python visualize_deeplabv3plus.py --config config/zod/deeplabv3/deeplabv3plus_rgb_vehicle_sign.json --upload

# Vehicle + Human Specialization
echo "Training Vehicle + Human Specialization..."
python train_deeplabv3plus.py --config config/zod/deeplabv3/deeplabv3plus_rgb_vehicle_human.json
python test_deeplabv3plus.py --config config/zod/deeplabv3/deeplabv3plus_rgb_vehicle_human.json
python visualize_deeplabv3plus.py --config config/zod/deeplabv3/deeplabv3plus_rgb_vehicle_human.json --upload

# Sign + Human Specialization
echo "Training Sign + Human Specialization..."
python train_deeplabv3plus.py --config config/zod/deeplabv3/deeplabv3plus_rgb_sign_human.json
python test_deeplabv3plus.py --config config/zod/deeplabv3/deeplabv3plus_rgb_sign_human.json
python visualize_deeplabv3plus.py --config config/zod/deeplabv3/deeplabv3plus_rgb_sign_human.json --upload
