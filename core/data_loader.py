#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data input handling for visualization.
"""
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from utils.helpers import relabel_annotation
from utils.lidar_process import open_lidar, get_unresized_lid_img_val


class DataLoader:
    """Handles loading and preprocessing of data for inference/visualization."""
    
    def __init__(self, config):
        self.config = config
        self.dataset_name = config['Dataset']['name']
        self.cam_mean = config['Dataset']['transforms']['image_mean']
        self.cam_std = config['Dataset']['transforms']['image_std']
        self._setup_lidar_normalization()
        self.resize = config['Dataset']['transforms']['resize']
    
    def _setup_lidar_normalization(self):
        """Setup dataset-specific LiDAR normalization."""
        if self.dataset_name == 'waymo':
            self.lidar_mean = self.config['Dataset']['transforms'].get(
                'lidar_mean_waymo', [-0.17263354, 0.85321806, 24.5527253]
            )
            self.lidar_std = self.config['Dataset']['transforms'].get(
                'lidar_std_waymo', [7.34546552, 1.17227659, 15.83745082]
            )
        else:  # ZOD
            self.lidar_mean = self.config['Dataset']['transforms']['lidar_mean']
            self.lidar_std = self.config['Dataset']['transforms']['lidar_std']
    
    def load_rgb(self, image_path):
        """Load and preprocess RGB image."""
        rgb_normalize = transforms.Compose([
            transforms.Resize((self.resize, self.resize), 
                            interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cam_mean, std=self.cam_std)
        ])
        
        rgb = Image.open(image_path).convert('RGB')
        return rgb_normalize(rgb)
    
    def load_annotation(self, anno_path):
        """Load and preprocess annotation."""
        anno = Image.open(anno_path)
        anno = np.array(anno)
        
        # Apply relabeling
        anno = relabel_annotation(anno, self.config)
        
        # Convert to tensor and resize
        anno_tensor = anno.float()
        anno_tensor = transforms.Resize(
            (self.resize, self.resize), 
            interpolation=transforms.InterpolationMode.NEAREST
        )(anno_tensor)
        
        return anno_tensor.squeeze(0)
    
    def load_lidar(self, lidar_path):
        """Load and preprocess LiDAR data (dataset-specific)."""
        if self.dataset_name == 'waymo':
            return self._load_waymo_lidar(lidar_path)
        else:  # ZOD
            return self._load_zod_lidar(lidar_path)
    
    def _load_zod_lidar(self, lidar_path):
        """Load ZOD LiDAR from PNG projection."""
        lidar_pil = Image.open(lidar_path)
        lidar_tensor = TF.to_tensor(lidar_pil)
        
        # Normalize
        lidar_tensor = transforms.Normalize(
            mean=self.lidar_mean, std=self.lidar_std
        )(lidar_tensor)
        
        lidar_tensor = transforms.Resize((self.resize, self.resize))(lidar_tensor)
        
        return lidar_tensor
    
    def _load_waymo_lidar(self, lidar_path):
        """Load Waymo LiDAR from PKL file."""
        points_set, camera_coord = open_lidar(
            lidar_path,
            w_ratio=4, h_ratio=4,
            lidar_mean=self.lidar_mean,
            lidar_std=self.lidar_std
        )
        
        # Create projection
        X, Y, Z = get_unresized_lid_img_val(320, 480, points_set, camera_coord)
        
        # Convert to tensor
        lidar_tensor = torch.cat([
            TF.to_tensor(X),
            TF.to_tensor(Y),
            TF.to_tensor(Z)
        ], dim=0)
        
        lidar_tensor = transforms.Resize((self.resize, self.resize))(lidar_tensor)
        
        return lidar_tensor
