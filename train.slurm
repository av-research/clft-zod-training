#!/bin/bash 
#SBATCH --job-name=totahv-1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint="A100-80"
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH -t 5-0
#SBATCH --mail-type=ALL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate clft_py39_torch21

# Run job
python3 train.py -bb clft -m rgb
