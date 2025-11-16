#!/usr/bin/env python3
"""
This is the script to load Waymo frames and visualize model predictions.
Adapted from visual_run.py for Waymo dataset with PKL lidar files.
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
from utils.helpers import draw_test_segmentation_map, image_overlay, get_model_path
from utils.lidar_process import open_lidar, get_unresized_lid_img_val


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
        rgb_norm = clft_rgb_normalize(rgb)
        return rgb_norm

    def open_anno(self, anno_path):
        # Waymo annotations are already in PNG format
        anno = Image.open(anno_path)
        anno = np.array(anno)

        # Apply relabeling based on config
        anno = relabel_annotation(anno, self.config)

        # Convert to tensor and resize
        anno_tensor = torch.from_numpy(anno).float()
        if self.backbone == 'clft':
            anno_tensor = transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.NEAREST)(anno_tensor)

        return anno_tensor.squeeze(0)

    def open_lidar(self, lidar_path):
        # For Waymo dataset, LiDAR is stored as PKL file
        points_set, camera_coord = open_lidar(
            lidar_path,
            w_ratio=4, h_ratio=4,
            lidar_mean=self.lidar_mean,
            lidar_std=self.lidar_std
        )

        # Create lidar projection image
        X, Y, Z = get_unresized_lid_img_val(320, 480, points_set, camera_coord)

        # Convert to tensor
        lidar_tensor = torch.cat([
            TF.to_tensor(X),
            TF.to_tensor(Y),
            TF.to_tensor(Z)
        ], dim=0)

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

    # Use Waymo-specific lidar normalization
    lidar_mean = config['Dataset']['transforms'].get('lidar_mean_waymo', [-0.17263354, 0.85321806, 24.5527253])
    lidar_std = config['Dataset']['transforms'].get('lidar_std_waymo', [7.34546552, 1.17227659, 15.83745082])

    open_input = OpenInput(backbone,
                           cam_mean=config['Dataset']['transforms']['image_mean'],
                           cam_std=config['Dataset']['transforms']['image_std'],
                           lidar_mean=lidar_mean,
                           lidar_std=lidar_std,
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
    dataroot = os.path.abspath(config['Dataset']['dataset_root'])
    output_base = f'output/waymo/{config_name}'
    for path in data_cam:
        # Handle absolute or relative paths
        if os.path.isabs(path):
            cam_path = path
        else:
            cam_path = os.path.join(dataroot, path)

        # Waymo annotation path
        anno_path = cam_path.replace('/camera/', '/annotation/')

        # Waymo LiDAR path (PKL files)
        lidar_path = cam_path.replace('/camera/', '/lidar/').replace('.png', '.pkl')

        rgb_name = cam_path.split('/')[-1].split('.')[0]
        anno_name = anno_path.split('/')[-1].split('.')[0]
        lidar_name = lidar_path.split('/')[-1].split('.')[0]
        assert (rgb_name == anno_name)
        assert (rgb_name == lidar_name)

        rgb = open_input.open_rgb(cam_path).to(device, non_blocking=True)
        rgb = rgb.unsqueeze(0)  # add a batch dimension
        lidar = open_input.open_lidar(lidar_path).to(device, non_blocking=True)
        lidar = lidar.unsqueeze(0)

        if backbone == 'clft':
            with torch.no_grad():
                second_input = get_model_second_input(rgb, lidar, modality)
                _, output_seg = model(rgb, second_input, modality)
                segmented_image = draw_test_segmentation_map(output_seg, config)

                # Resize to original image dimensions for overlay
                rgb_cv2 = cv2.imread(cam_path)
                h, w = rgb_cv2.shape[:2]
                seg_resize = cv2.resize(segmented_image, (w, h), interpolation=cv2.INTER_NEAREST)

                # Convert RGB to BGR for OpenCV
                seg_resize_bgr = cv2.cvtColor(seg_resize, cv2.COLOR_RGB2BGR)

                # Use relative path for output
                rel_path = os.path.relpath(cam_path, dataroot)
                seg_path = os.path.join(output_base, 'segment', os.path.basename(cam_path))
                overlay_path = os.path.join(output_base, 'overlay', os.path.basename(cam_path))

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
                    gt_relabeled = relabel_annotation(gt_anno_tensor, config)
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
                    compare_path = os.path.join(output_base, 'compare', os.path.basename(cam_path))
                    os.makedirs(os.path.dirname(compare_path), exist_ok=True)
                    print(f'saving comparison result {i}...')
                    cv2.imwrite(compare_path, comparison)

                    # Create correct_only image (only correct predictions visible)
                    correct_only = np.zeros((h, w, 3), dtype=np.uint8)
                    correct_only[correct_mask] = seg_resize[correct_mask]

                    correct_only_path = os.path.join(output_base, 'correct_only', os.path.basename(cam_path))
                    os.makedirs(os.path.dirname(correct_only_path), exist_ok=True)
                    correct_only_bgr = cv2.cvtColor(correct_only, cv2.COLOR_RGB2BGR)
                    print(f'saving correct_only result {i}...')
                    cv2.imwrite(correct_only_path, correct_only_bgr)

        # Only CLFT backbone is supported in this visual run script.

        else:
            sys.exit("A backbone must be specified! (clft)")
        i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Waymo visual run script')
    parser.add_argument('-p', '--path', type=str, required=False, default='waymo_dataset/visual_run_demo.txt',
                        help='Path to image file or text file containing Waymo image paths')
    parser.add_argument('-c', '--config', type=str, default='',
                        help='Path to Waymo config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    # Extract config name from path
    config_name = os.path.splitext(os.path.basename(args.config))[0]

    # Get model path from config (or latest checkpoint if not specified)
    model_path = get_model_path(config)
    if not model_path:
        print("No model checkpoint found!")
        sys.exit(1)
    config['General']['model_path'] = model_path
    print(f"Using model: {model_path}")

    # Always use CLFT backbone and read mode from config
    backbone = 'clft'
    mode = config['CLI']['mode']

    run(mode, backbone, config, config_name)