#!/bin/bash
#SBATCH --job-name=clft-test
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint="A100-80"
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH -t 5-0
#SBATCH --mail-type=ALL

## run job
source venv/bin/activate 
python3 test.py --config $1
