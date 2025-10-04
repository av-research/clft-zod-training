# ZOD CLFT

This project provides tools for processing and visualizing the Zenseact Open Dataset (ZOD) using camera and LiDAR data.

## Prerequisites

- Docker
- ZOD dataset files (mini or full version)

## Setup and Installation
### Start the Application
CPU-only (lightweight):
```
docker compose run --rm --service-ports python-lite bash
```

GPU-enabled (CUDA):
```  
docker compose run --rm --service-ports python-cuda bash
```

### Data Preparation
Copy the unpacked ZOD data files to the `/data` folder:
- `drives_mini.tar.gz`
- `frames_mini.tar.gz`
- `sequences_mini.tar.gz`

Run command to extract data:
```
sh scripts/extract_zod_data.sh
```

### Start jupyter
```
scripts/jupyter_startup.sh
```

## Setup Virtual Environment

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
