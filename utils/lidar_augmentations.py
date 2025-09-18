"""
LiDAR Augmentation utilities extracted from the original train.py pipeline
Provides data augmentation functions specifically for LiDAR point cloud data
"""

import cv2
import random
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as TF
from torchvision.transforms.v2.functional import InterpolationMode
from PIL import Image


def lidar_dilation(X, Y, Z):
    """
    Apply morphological dilation to LiDAR X, Y, Z channels
    
    Args:
        X, Y, Z: PIL Images or numpy arrays representing LiDAR channels
        
    Returns:
        X_dilation, Y_dilation, Z_dilation: Dilated PIL Images
    """
    kernel = np.ones((3, 3), np.uint8)
    
    X_dilation = cv2.dilate(np.array(X).astype(np.float32), kernel, iterations=1)
    Y_dilation = cv2.dilate(np.array(Y).astype(np.float32), kernel, iterations=1)
    Z_dilation = cv2.dilate(np.array(Z).astype(np.float32), kernel, iterations=1)

    X_dilation = TF.to_pil_image(X_dilation.astype(np.float32))
    Y_dilation = TF.to_pil_image(Y_dilation.astype(np.float32))
    Z_dilation = TF.to_pil_image(Z_dilation.astype(np.float32))
    
    return X_dilation, Y_dilation, Z_dilation


class LiDARDataAugment:
    """
    LiDAR-specific data augmentation class
    Handles synchronized augmentation of RGB, annotation, and LiDAR data
    """
    
    def __init__(self, p_flip=0.5, p_crop=0.3, p_rot=0.4, rotation_range=20, img_size=384):
        self.p_flip = p_flip
        self.p_crop = p_crop  
        self.p_rot = p_rot
        self.rotation_range = rotation_range
        self.img_size = img_size
        
    def random_horizontal_flip(self, rgb, anno, lidar_tensor):
        """
        Apply random horizontal flip to RGB, annotation and LiDAR data
        
        Args:
            rgb: PIL Image
            anno: PIL Image or torch tensor
            lidar_tensor: torch tensor [3, H, W] containing X, Y, Z channels
            
        Returns:
            Augmented rgb, anno, lidar_tensor
        """
        if random.random() < self.p_flip:
            rgb = TF.hflip(rgb)
            anno = TF.hflip(anno)
            lidar_tensor = TF.hflip(lidar_tensor)
            
        return rgb, anno, lidar_tensor
    
    def random_crop(self, rgb, anno, lidar_tensor):
        """
        Apply random resized crop to RGB, annotation and LiDAR data
        
        Args:
            rgb: PIL Image
            anno: PIL Image or torch tensor 
            lidar_tensor: torch tensor [3, H, W] containing X, Y, Z channels
            
        Returns:
            Augmented rgb, anno, lidar_tensor
        """
        if random.random() < self.p_crop:
            # Force the crop to maintain the target image size
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                rgb, scale=[0.7, 1.0], ratio=[3. / 4., 4. / 3.])  # Less aggressive scaling
            
            rgb = TF.resized_crop(rgb, i, j, h, w, [self.img_size, self.img_size], InterpolationMode.BILINEAR)
            anno = TF.resized_crop(anno, i, j, h, w, [self.img_size, self.img_size], InterpolationMode.NEAREST)
            lidar_tensor = TF.resized_crop(lidar_tensor, i, j, h, w, [self.img_size, self.img_size], InterpolationMode.BILINEAR)
            
        return rgb, anno, lidar_tensor
    
    def random_rotate(self, rgb, anno, lidar_tensor):
        """
        Apply random rotation to RGB, annotation and LiDAR data
        
        Args:
            rgb: PIL Image
            anno: PIL Image or torch tensor
            lidar_tensor: torch tensor [3, H, W] containing X, Y, Z channels
            
        Returns:
            Augmented rgb, anno, lidar_tensor
        """
        if random.random() < self.p_rot:
            angle = (-self.rotation_range + 2 * self.rotation_range * torch.rand(1)[0]).item()
            
            rgb = TF.affine(rgb, angle, [0, 0], 1, 0, InterpolationMode.BILINEAR)
            anno = TF.affine(anno, angle, [0, 0], 1, 0, InterpolationMode.NEAREST)
            lidar_tensor = TF.affine(lidar_tensor, angle, [0, 0], 1, 0, InterpolationMode.NEAREST)
            
        return rgb, anno, lidar_tensor
    
    def apply_augmentations(self, rgb, anno, lidar_tensor):
        """
        Apply all augmentations in sequence
        
        Args:
            rgb: PIL Image
            anno: PIL Image or torch tensor
            lidar_tensor: torch tensor [3, H, W] containing X, Y, Z channels
            
        Returns:
            Augmented rgb, anno, lidar_tensor
        """
        # Apply augmentations in sequence
        rgb, anno, lidar_tensor = self.random_horizontal_flip(rgb, anno, lidar_tensor)
        rgb, anno, lidar_tensor = self.random_crop(rgb, anno, lidar_tensor)
        rgb, anno, lidar_tensor = self.random_rotate(rgb, anno, lidar_tensor)
        
        return rgb, anno, lidar_tensor


def apply_lidar_augmentations(rgb, anno, lidar_tensor, training=True, config=None):
    """
    Convenience function to apply LiDAR augmentations based on config
    
    Args:
        rgb: PIL Image
        anno: PIL Image or torch tensor
        lidar_tensor: torch tensor [3, H, W] 
        training: bool, whether in training mode
        config: dict with augmentation parameters
        
    Returns:
        Augmented rgb, anno, lidar_tensor
    """
    if not training:
        return rgb, anno, lidar_tensor
    
    # Default augmentation parameters (can be overridden by config)
    p_flip = 0.5
    p_crop = 0.3
    p_rot = 0.4
    rotation_range = 20
    img_size = 384
    
    if config:
        p_flip = config.get('p_flip', p_flip)
        p_crop = config.get('p_crop', p_crop) 
        p_rot = config.get('p_rot', p_rot)
        rotation_range = config.get('random_rotate_range', rotation_range)
        img_size = config.get('resize', img_size)
    
    augmenter = LiDARDataAugment(p_flip, p_crop, p_rot, rotation_range, img_size)
    return augmenter.apply_augmentations(rgb, anno, lidar_tensor)


def tensor_to_pil_channels(lidar_tensor):
    """
    Convert LiDAR tensor [3, H, W] to separate PIL Images for X, Y, Z channels
    
    Args:
        lidar_tensor: torch tensor [3, H, W]
        
    Returns:
        X, Y, Z: PIL Images
    """
    # Ensure tensor is on CPU and convert to numpy
    lidar_np = lidar_tensor.detach().cpu().numpy()
    
    X = Image.fromarray(lidar_np[0], mode='F')  # Float mode
    Y = Image.fromarray(lidar_np[1], mode='F')
    Z = Image.fromarray(lidar_np[2], mode='F')
    
    return X, Y, Z


def pil_channels_to_tensor(X, Y, Z):
    """
    Convert separate PIL Images back to LiDAR tensor [3, H, W]
    
    Args:
        X, Y, Z: PIL Images
        
    Returns:
        lidar_tensor: torch tensor [3, H, W]
    """
    # Convert PIL to numpy arrays first, then to tensors
    X_array = np.array(X, dtype=np.float32)
    Y_array = np.array(Y, dtype=np.float32)
    Z_array = np.array(Z, dtype=np.float32)
    
    # Ensure all arrays have the same shape
    assert X_array.shape == Y_array.shape == Z_array.shape, f"Shape mismatch: X={X_array.shape}, Y={Y_array.shape}, Z={Z_array.shape}"
    
    X_tensor = torch.from_numpy(X_array).unsqueeze(0)
    Y_tensor = torch.from_numpy(Y_array).unsqueeze(0)
    Z_tensor = torch.from_numpy(Z_array).unsqueeze(0)
    
    return torch.cat((X_tensor, Y_tensor, Z_tensor), 0)