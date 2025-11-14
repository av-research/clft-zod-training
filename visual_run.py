#!/usr/bin/env python3
"""
This is the script to load input frames and visualize model predictions.
Updated to work with ZOD dataset and configurable class indexing.
Supports CLFT backbone.
"""
import os
import cv2
import sys
import torch
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import json
from clft.clft import CLFT
from utils.helpers import relabel_annotation

from utils.helpers import draw_test_segmentation_map, image_overlay


class OpenInput(object):
    def __init__(self, backbone, cam_mean, cam_std, lidar_mean, lidar_std, config):
        self.backbone = backbone
        self.cam_mean = cam_mean
        self.cam_std = cam_std
        self.lidar_mean = lidar_mean
        self.lidar_std = lidar_std
        self.config = config

    def open_rgb(self, image_path):
        clft_rgb_normalize = transforms.Compose(
            [transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.cam_mean,
                    std=self.cam_std)])

        rgb = Image.open(image_path).convert('RGB')
        # Visual run now supports CLFT backbone only; always use CLFT preprocessing
        rgb_norm = clft_rgb_normalize(rgb)
        return rgb_norm

    def open_anno(self, anno_path):
        # For ZOD dataset, annotations are already in PNG format with correct class indices
        anno = Image.open(anno_path)
        anno = np.array(anno)
        
        # Apply relabeling based on config
        anno = relabel_annotation(anno, self.config)
        
        # Convert to tensor and resize if needed
        anno_tensor = torch.from_numpy(anno).float()
        if self.backbone == 'clft':
            anno_tensor = transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.NEAREST)(anno_tensor)
        
        return anno_tensor.squeeze(0)

    def open_lidar(self, lidar_path):
        # For ZOD dataset, LiDAR is stored as PNG projection
        lidar_pil = Image.open(lidar_path)
        
        # Convert to tensor and normalize
        lidar_tensor = TF.to_tensor(lidar_pil)
        
        # Apply normalization (ZOD uses different normalization than Waymo)
        lidar_tensor = transforms.Normalize(
            mean=self.lidar_mean,
            std=self.lidar_std
        )(lidar_tensor)
        
        if self.backbone == 'clft':
            lidar_tensor = transforms.Resize((384, 384))(lidar_tensor)
        
        return lidar_tensor


def get_model_second_input(rgb_tensor, lidar_tensor, modality):
    """
    Get the correct second input (rgb or lidar) based on modality.
    
    Modality modes:
    - 'rgb': Use RGB for both inputs (RGB-only mode)
    - 'lidar': Use LiDAR as second input (LiDAR-only mode)  
    - 'cross_fusion': Use LiDAR as second input (cross-modal fusion)
    
    Args:
        rgb_tensor: RGB image tensor
        lidar_tensor: LiDAR image tensor
        modality: String specifying the modality mode
        
    Returns:
        Tensor to use as second input to the model
    """
    if modality == 'rgb':
        # RGB-only mode: use RGB for both inputs
        return rgb_tensor
    else:
        # LiDAR or cross_fusion modes: use LiDAR as second input
        return lidar_tensor


def run(modality, backbone, config, config_name):
    device = torch.device(config['General']['device']
                          if torch.cuda.is_available() else "cpu")
    
    # Calculate number of unique classes after relabeling
    unique_indices = set(cls['training_index'] for cls in config['Dataset']['classes'])
    num_unique_classes = len(unique_indices)
    
    open_input = OpenInput(backbone,
                           cam_mean=config['Dataset']['transforms']['image_mean'],
                           cam_std=config['Dataset']['transforms']['image_std'],
                           lidar_mean=config['Dataset']['transforms']['lidar_mean'],
                           lidar_std=config['Dataset']['transforms']['lidar_std'],
                           config=config)

    if backbone == 'clft':
        resize = config['Dataset']['transforms']['resize']
        model = CLFT(
            RGB_tensor_size=(3, resize, resize),
            XYZ_tensor_size=(3, resize, resize),
            patch_size=config['CLFT']['patch_size'],
            emb_dim=config['CLFT']['emb_dim'],
            resample_dim=config['CLFT']['resample_dim'],
            read=config['CLFT']['read'],
            hooks=config['CLFT']['hooks'],
            reassemble_s=config['CLFT']['reassembles'],
            nclasses=num_unique_classes,
            type=config['CLFT']['type'],
            model_timm=config['CLFT']['model_timm'],
            )
        print(f'Using backbone {backbone}')

        model_path = config['General']['model_path']
        model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
        model.to(device)
        model.eval()

    else:
        sys.exit("A backbone must be specified! (clft)")

    # Check if path is a single image file or a text file with multiple paths
    if args.path.endswith(('.png', '.jpg', '.jpeg')):
        # Single image path
        data_cam = [args.path]
    else:
        # Text file with multiple paths
        data_list = open(args.path, 'r')
        data_cam = np.array(data_list.read().splitlines())
        data_list.close()

    i = 1
    dataroot = config['Dataset']['dataset_root']
    for path in data_cam:
        # ZOD dataset structure
        cam_path = os.path.join(dataroot, path)
        
        # Determine annotation path based on modality
        if modality == 'rgb':
            anno_path = cam_path.replace('/camera', '/annotation_camera_only')
        elif modality == 'lidar':
            anno_path = cam_path.replace('/camera', '/annotation_lidar_only')
        else:  # cross_fusion
            anno_path = cam_path.replace('/camera', '/annotation_fusion')
        
        # LiDAR PNG path
        lidar_path = cam_path.replace('/camera', '/lidar_png')

        rgb_name = cam_path.split('/')[-1].split('.')[0]
        anno_name = anno_path.split('/')[-1].split('.')[0]
        lidar_name = lidar_path.split('/')[-1].split('.')[0]
        assert (rgb_name == lidar_name)
        assert (anno_name == lidar_name)

        rgb = open_input.open_rgb(cam_path).to(device, non_blocking=True)
        rgb = rgb.unsqueeze(0)  # add a batch dimension
        lidar = open_input.open_lidar(lidar_path).to(device, non_blocking=True)
        lidar = lidar.unsqueeze(0)

        if backbone == 'clft':
            with torch.no_grad():
                second_input = get_model_second_input(rgb, lidar, modality)
                _, output_seg = model(rgb, second_input, modality)
                segmented_image = draw_test_segmentation_map(output_seg, configs)
                
                # Resize to original image dimensions for overlay
                rgb_cv2 = cv2.imread(cam_path)
                h, w = rgb_cv2.shape[:2]
                seg_resize = cv2.resize(segmented_image, (w, h), interpolation=cv2.INTER_NEAREST)
                
                # Convert RGB to BGR for OpenCV
                seg_resize_bgr = cv2.cvtColor(seg_resize, cv2.COLOR_RGB2BGR)

                seg_path = cam_path.replace(dataroot, f'output/zod/{config_name}/segment')
                overlay_path = cam_path.replace(dataroot, f'output/zod/{config_name}/overlay')

                # Create output directories if they don't exist
                os.makedirs(os.path.dirname(seg_path), exist_ok=True)
                os.makedirs(os.path.dirname(overlay_path), exist_ok=True)

                print(f'saving segment result {i}...')
                cv2.imwrite(seg_path, seg_resize_bgr)

                overlay = image_overlay(rgb_cv2, seg_resize_bgr)
                print(f'saving overlay result {i}...')
                cv2.imwrite(overlay_path, overlay)

                # Create comparison image (green=correct, red=incorrect)
                # Load ground truth annotation
                gt_anno = cv2.imread(anno_path, cv2.IMREAD_GRAYSCALE)
                if gt_anno is None:
                    print(f'Warning: Could not load ground truth annotation for {anno_path}')
                else:
                    # Resize ground truth to match prediction dimensions
                    gt_anno_resized = cv2.resize(gt_anno, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    # Apply relabeling to ground truth to match training indices
                    gt_anno_tensor = torch.from_numpy(gt_anno_resized).unsqueeze(0).long()
                    gt_relabeled = relabel_annotation(gt_anno_tensor, configs)
                    gt_relabeled = gt_relabeled.squeeze(0).squeeze(0).numpy()
                    
                    # Get prediction labels
                    pred_labels = torch.argmax(output_seg.squeeze(), dim=0).detach().cpu().numpy()
                    
                    # Resize prediction to match ground truth dimensions
                    pred_labels = cv2.resize(pred_labels.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                    
                    # Create comparison image
                    comparison = np.zeros((h, w, 3), dtype=np.uint8)
                    
                    # Green for correct predictions, red for incorrect
                    correct_mask = (pred_labels == gt_relabeled)
                    incorrect_mask = (pred_labels != gt_relabeled)
                    
                    comparison[correct_mask] = [0, 255, 0]    # Green for correct
                    comparison[incorrect_mask] = [0, 0, 255]  # Red for incorrect
                    
                    # Save comparison image
                    compare_path = cam_path.replace(dataroot, f'output/zod/{config_name}/compare')
                    os.makedirs(os.path.dirname(compare_path), exist_ok=True)
                    print(f'saving comparison result {i}...')
                    cv2.imwrite(compare_path, comparison)

                    # Create correct_only image (only correct predictions visible)
                    correct_only = seg_resize.copy()
                    correct_only[incorrect_mask] = [0, 0, 0]  # Black out incorrect predictions
                    
                    correct_only_path = cam_path.replace(dataroot, f'output/zod/{config_name}/correct_only')
                    os.makedirs(os.path.dirname(correct_only_path), exist_ok=True)
                    correct_only_bgr = cv2.cvtColor(correct_only, cv2.COLOR_RGB2BGR)
                    print(f'saving correct_only result {i}...')
                    cv2.imwrite(correct_only_path, correct_only_bgr)

        # Only CLFT backbone is supported in this visual run script.

        else:
            sys.exit("A backbone must be specified! (clft)")
        i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visual run script')
    parser.add_argument('-p', '--path', type=str, required=False, default='zod_dataset/visualize.txt',
                        help='Path to image file (e.g., camera/frame_099988.png) or text file containing image paths (default: zod_dataset/visualize.txt)')
    parser.add_argument('-c', '--config', type=str, default='config/zod/config_4.json',
                        help='Path to config file (default: config/zod/config_4.json)')
    parser.add_argument('--model_path', type=str, default='logs/zod/config_4/progress_save/epoch_99_109a9b6a-402b-4759-b762-0f3c8a5425a7.pth',
                        help='Path to model checkpoint')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        configs = json.load(f)

    # Extract config name from path (e.g., 'config_1' from 'config/zod/config_1.json')
    config_name = os.path.splitext(os.path.basename(args.config))[0]

    # Set model path
    configs['General']['model_path'] = args.model_path

    # Always use CLFT backbone and read mode from config
    backbone = 'clft'
    mode = configs['CLI']['mode']

    run(mode, backbone, configs, config_name)
