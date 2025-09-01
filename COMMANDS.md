# ZOD CLFT

This project provides tools for processing and visualizing the Zenseact Open Dataset (ZOD) using camera and LiDAR data.

## Prerequisites

- Docker
- ZOD dataset files (mini or full version)

## Setup and Installation
### Start the Application
GPU-enabled (CUDA):
```  
docker compose run --rm --service-ports python-cuda bash
```

Train:
```  
python3 train.py -bb clft -m rgb
python3 test.py -bb clft -m rgb -p zod_dataset/splits_zod/all.txt
python3 visual_run.py -bb clft -m rgb -p zod_dataset/splits_zod/all.txt
```

Conda:
```  
conda env create -f clft_py39_torch21_env.yml
conda activate clft_py39_torch21
```

HPC:
```  
ssh totahv@base.hpc.taltech.ee
cd Projects/clft-zod-training/
sbatch -J train-1 train.slurm
watch -n1 squeue -u totahv
scp totahv@base.hpc.taltech.ee:/gpfs/mariana/home/totahv/Projects/clft-zod-training/model_path/rgbcheckpoint_59.pth .
scp -r totahv@base.hpc.taltech.ee:/gpfs/mariana/home/totahv/Projects/clft-zod-training/output/zod .
```
