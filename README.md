# Fusion Training: SAM-Enhanced Semantic Segmentation on ZOD

This repository contains the implementation for the paper "SAM-Enhanced Semantic Segmentation on ZOD: Specialized Models for Vulnerable Road Users".

## Abstract

The Zenseact Open Dataset (ZOD) provides valuable multi-modal data for autonomous driving but lacks dense semantic segmentation annotations, limiting its use for pixel-level perception tasks. We introduce a preprocessing pipeline using the Segment Anything Model (SAM) to convert ZOD's 2D bounding box annotations into dense pixel-level segmentation masks, enabling semantic segmentation training on this dataset for the first time. Due to the imperfect nature of automated mask generation, only 36% of frames passed manual quality control and were included in the final dataset. We present a comprehensive comparison between transformer-based Camera-LiDAR Fusion Transformers (CLFT) and CNN-based DeepLabV3+ architectures for multi-modal semantic segmentation on ZOD across RGB, LiDAR, and fusion modalities under diverse weather conditions. Furthermore, we investigate model specialization techniques to address class imbalance, developing separate modules optimized for large-scale objects (vehicles) and small-scale vulnerable road users (pedestrians, cyclists, traffic signs). The specialized models significantly improve detection of underrepresented safety-critical classes while maintaining overall segmentation accuracy, providing practical insights for deploying multi-modal perception systems in autonomous vehicles. To enable reproducible research, we release the complete open-source implementation of our processing pipeline.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/av-research/fusion-training.git
   cd fusion-training
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the models, use the provided training scripts. For example:

- For CLFT models: `python train.py` (adjust configurations as needed)
- For DeepLabV3+ models: `python train_deeplabv3plus.py`

### Testing

Run tests with:
```bash
python test.py
```

### Visualization

Visualize results using:
```bash
python visualize.py
```

## Dataset

This project uses the Zenseact Open Dataset (ZOD). Ensure you have access to the dataset and place it in the appropriate directories as specified in the configuration files.

## Paper

For more details, refer to the full paper in the `paper/` directory.

## License

See LICENSE file for details.
