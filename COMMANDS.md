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
```
