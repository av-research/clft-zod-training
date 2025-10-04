#!/usr/bin/env python3
"""
Script to calculate mean and std for RGB images and LiDAR projections from ZOD dataset.
"""

import os
import numpy as np
from PIL import Image
import pickle
import argparse
from tqdm import tqdm

def load_lidar_pkl(pkl_path):
    """Load LiDAR point cloud from pickle file."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def project_lidar_to_image(points, w_ratio=8.84, h_ratio=8.825, img_width=1920, img_height=1280):
    """
    Project LiDAR points to image coordinates.
    Simplified version based on utils/lidar_process.py
    """
    # Assuming points is Nx3 array [X, Y, Z]
    # Project to image: this is a placeholder, need to match the actual projection
    # For simplicity, assume direct mapping or use the code from lidar_process.py

    # Placeholder: create a dummy projection
    # In reality, use the camera intrinsics and extrinsics from ZOD

    # For now, return zeros or something; user needs to implement proper projection
    projected = np.zeros((img_height, img_width, 3), dtype=np.float32)
    # TODO: Implement actual projection using camera params

    return projected

def calculate_image_stats(dataset_path, num_samples=None):
    """Calculate mean and std for RGB images."""
    camera_dir = os.path.join(dataset_path, 'camera')
    if not os.path.exists(camera_dir):
        print(f"Camera directory not found: {camera_dir}")
        return None, None

    image_files = [f for f in os.listdir(camera_dir) if f.endswith('.png')]
    if num_samples:
        image_files = image_files[:num_samples]

    print(f"Calculating stats for {len(image_files)} images...")

    # Accumulate sums
    sum_r = 0.0
    sum_g = 0.0
    sum_b = 0.0
    sum_sq_r = 0.0
    sum_sq_g = 0.0
    sum_sq_b = 0.0
    total_pixels = 0

    for img_file in tqdm(image_files):
        img_path = os.path.join(camera_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]

        # Accumulate
        sum_r += img_array[:, :, 0].sum()
        sum_g += img_array[:, :, 1].sum()
        sum_b += img_array[:, :, 2].sum()
        sum_sq_r += (img_array[:, :, 0] ** 2).sum()
        sum_sq_g += (img_array[:, :, 1] ** 2).sum()
        sum_sq_b += (img_array[:, :, 2] ** 2).sum()
        total_pixels += img_array.size // 3

    # Calculate mean
    mean_r = sum_r / total_pixels
    mean_g = sum_g / total_pixels
    mean_b = sum_b / total_pixels
    image_mean = [mean_r, mean_g, mean_b]

    # Calculate std
    var_r = (sum_sq_r / total_pixels) - (mean_r ** 2)
    var_g = (sum_sq_g / total_pixels) - (mean_g ** 2)
    var_b = (sum_sq_b / total_pixels) - (mean_b ** 2)
    std_r = np.sqrt(var_r)
    std_g = np.sqrt(var_g)
    std_b = np.sqrt(var_b)
    image_std = [std_r, std_g, std_b]

    return image_mean, image_std

def calculate_lidar_stats(dataset_path, num_samples=None):
    """Calculate mean and std for LiDAR projections."""
    lidar_dir = os.path.join(dataset_path, 'lidar')
    if not os.path.exists(lidar_dir):
        print(f"LiDAR directory not found: {lidar_dir}")
        return None, None

    lidar_files = [f for f in os.listdir(lidar_dir) if f.endswith('.pkl')]
    if num_samples:
        lidar_files = lidar_files[:num_samples]

    print(f"Calculating stats for {len(lidar_files)} LiDAR files...")

    # Accumulate sums for raw X, Y, Z
    sum_x = 0.0
    sum_y = 0.0
    sum_z = 0.0
    sum_sq_x = 0.0
    sum_sq_y = 0.0
    sum_sq_z = 0.0
    total_points = 0

    for lidar_file in tqdm(lidar_files):
        pkl_path = os.path.join(lidar_dir, lidar_file)
        with open(pkl_path, 'rb') as f:
            lidar_data = pickle.load(f)

        points3d = lidar_data['3d_points']
        camera_coord = lidar_data['camera_coordinates']

        # Select camera front
        mask = camera_coord[:, 0] == 0
        points3d = points3d[mask, :]

        if len(points3d) == 0:
            continue

        # Raw values: X=points3d[:,1], Y=points3d[:,2], Z=points3d[:,0]
        x_raw = points3d[:, 1]
        y_raw = points3d[:, 2]
        z_raw = points3d[:, 0]

        # Accumulate
        sum_x += x_raw.sum()
        sum_y += y_raw.sum()
        sum_z += z_raw.sum()
        sum_sq_x += (x_raw ** 2).sum()
        sum_sq_y += (y_raw ** 2).sum()
        sum_sq_z += (z_raw ** 2).sum()
        total_points += len(points3d)

    if total_points == 0:
        print("No points found.")
        return None, None

    # Calculate mean
    mean_x = sum_x / total_points
    mean_y = sum_y / total_points
    mean_z = sum_z / total_points
    lidar_mean = [mean_x, mean_y, mean_z]

    # Calculate std
    var_x = (sum_sq_x / total_points) - (mean_x ** 2)
    var_y = (sum_sq_y / total_points) - (mean_y ** 2)
    var_z = (sum_sq_z / total_points) - (mean_z ** 2)
    std_x = np.sqrt(var_x)
    std_y = np.sqrt(var_y)
    std_z = np.sqrt(var_z)
    lidar_std = [std_x, std_y, std_z]

    return lidar_mean, lidar_std

def main():
    parser = argparse.ArgumentParser(description='Calculate mean and std for ZOD dataset')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the ZOD dataset directory')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to use for calculation (default: all)')

    args = parser.parse_args()

    print("Calculating Image Mean and Std...")
    image_mean, image_std = calculate_image_stats(args.dataset_path, args.num_samples)
    if image_mean and image_std:
        print(f"Image Mean: {image_mean}")
        print(f"Image Std: {image_std}")

    print("\nCalculating LiDAR Mean and Std...")
    lidar_mean, lidar_std = calculate_lidar_stats(args.dataset_path, args.num_samples)
    if lidar_mean and lidar_std:
        print(f"LiDAR Mean: {lidar_mean}")
        print(f"LiDAR Std: {lidar_std}")

    print("\nStats calculation complete.")

if __name__ == "__main__":
    main()