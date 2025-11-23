#!/bin/bash 
#SBATCH --job-name=z-d-spec                ### job name
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

# Vehicle + Sign + Human Specialization
python train_deeplabv3plus.py --config config/zod/deeplabv3/rgb_baseline.json
python test_deeplabv3plus.py --config config/zod/deeplabv3/rgb_baseline.json
python visualize_deeplabv3plus.py --config config/zod/deeplabv3/rgb_baseline.json --upload

# Vehicle Only Specialization
python train_deeplabv3plus.py --config config/zod/deeplabv3/rgb_vehicle_only.json
python test_deeplabv3plus.py --config config/zod/deeplabv3/rgb_vehicle_only.json
python visualize_deeplabv3plus.py --config config/zod/deeplabv3/rgb_vehicle_only.json --upload

# Sign Only Specialization
python train_deeplabv3plus.py --config config/zod/deeplabv3/rgb_sign_only.json
python test_deeplabv3plus.py --config config/zod/deeplabv3/rgb_sign_only.json
python visualize_deeplabv3plus.py --config config/zod/deeplabv3/rgb_sign_only.json --upload

# Human Only Specialization
python train_deeplabv3plus.py --config config/zod/deeplabv3/rgb_human_only.json
python test_deeplabv3plus.py --config config/zod/deeplabv3/rgb_human_only.json
python visualize_deeplabv3plus.py --config config/zod/deeplabv3/rgb_human_only.json --upload

# Vehicle + Sign Specialization
python train_deeplabv3plus.py --config config/zod/deeplabv3/rgb_vehicle_sign.json
python test_deeplabv3plus.py --config config/zod/deeplabv3/rgb_vehicle_sign.json
python visualize_deeplabv3plus.py --config config/zod/deeplabv3/rgb_vehicle_sign.json --upload

# Vehicle + Human Specialization
python train_deeplabv3plus.py --config config/zod/deeplabv3/rgb_vehicle_human.json
python test_deeplabv3plus.py --config config/zod/deeplabv3/rgb_vehicle_human.json
python visualize_deeplabv3plus.py --config config/zod/deeplabv3/rgb_vehicle_human.json --upload

# Sign + Human Specialization
python train_deeplabv3plus.py --config config/zod/deeplabv3/rgb_sign_human.json
python test_deeplabv3plus.py --config config/zod/deeplabv3/rgb_sign_human.json
python visualize_deeplabv3plus.py --config config/zod/deeplabv3/rgb_sign_human.json --upload
