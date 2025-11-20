#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import shutil
import datetime
import glob

label_colors_list = [
        (0, 0, 0),        # 0: background - black
        (128, 128, 128),  # 1: ignore - gray
        (255, 0, 0),      # 2: vehicle - blue
        (0, 0, 255),      # 3: sign - red
        (0, 255, 0),      # 4: cyclist - green
        (255, 255, 0)]    # 5: pedestrian - cyan


def creat_dir(config):
    logdir = config['Log']['logdir']
    if not os.path.exists(logdir):
        os.makedirs(logdir)
        print(f'Making log directory {logdir}...')
    if not os.path.exists(logdir + 'progress_save'):
        os.makedirs(logdir + 'progress_save')

def waymo_anno_class_relabel(annotation):
    """
    Reassign the indices of the objects in annotation(PointCloud);
    :parameter annotation: 0->ignore, 1->vehicle, 2->pedestrian, 3->sign,
                            4->cyclist, 5->background
    :return annotation: 0->background+sign, 1->vehicle
                            2->pedestrian+cyclist, 3->ignore
    """
    annotation = np.array(annotation)

    mask_ignore = annotation == 0
    mask_sign = annotation == 3
    mask_cyclist = annotation == 4
    mask_background = annotation == 5

    annotation[mask_sign] = 0
    annotation[mask_background] = 0
    annotation[mask_cyclist] = 2
    annotation[mask_ignore] = 3

    return torch.from_numpy(annotation).unsqueeze(0).long() # [H,W]->[1,H,W]


def waymo_anno_class_relabel_1(annotation):
    """
    Reassign the indices of the objects in annotation(PointCloud);
    Standardizes class indices to consistent mapping:
    :parameter annotation: 0->ignore, 1->vehicle, 2->pedestrian, 3->sign,
                            4->cyclist, 5->background
    :return annotation: 0->background, 1->vehicle, 2->pedestrian, 3->sign, 4->cyclist, 5->ignore
    """
    annotation = np.array(annotation)

    # Create a mapping array to standardize class indices
    # Original: 0->ignore, 1->vehicle, 2->pedestrian, 3->sign, 4->cyclist, 5->background
    # Target: 0->background, 1->vehicle, 2->pedestrian, 3->sign, 4->cyclist, 5->ignore
    
    mapping = np.full(6, 5, dtype=int)  # Default to ignore
    mapping[5] = 0  # background: 5 -> 0
    mapping[1] = 1  # vehicle: 1 -> 1
    mapping[2] = 2  # pedestrian: 2 -> 2
    mapping[3] = 3  # sign: 3 -> 3
    mapping[4] = 4  # cyclist: 4 -> 4
    # ignore: 0 -> 5 (already set as default)
    
    annotation = mapping[annotation]

    return torch.from_numpy(annotation).unsqueeze(0).long() # [H,W]->[1,H,W]


def zod_anno_class_relabel(annotation):
    """
    ZOD annotations are now already in the correct class mapping format.
    
    Class mapping:
    - 0: background
    - 1: ignore (LiDAR-only regions)
    - 2: vehicle (from SAM)
    - 3: sign (from SAM)
    - 4: cyclist (from SAM)
    - 5: pedestrian (from SAM)
    
    No relabeling needed.
    """
    annotation = np.array(annotation)
    
    # Annotations are already in the correct format, just return as tensor
    return torch.from_numpy(annotation).unsqueeze(0).long() # [H,W]->[1,H,W]


def relabel_annotation(annotation, config):
    """
    Relabel annotation based on configurable class indices from config.
    
    This allows for flexible class mapping, including merging classes by assigning
    them the same index value. Creates consecutive indices starting from 0.
    
    Args:
        annotation: numpy array or torch tensor with original class indices
        config: configuration dictionary containing Dataset.classes with 'color' and 'index' fields
        
    Returns:
        torch tensor with relabeled indices [1, H, W] using consecutive indices
    """
    annotation = np.array(annotation)
    
    classes = config['Dataset']['classes']
    
    # Create mapping from config index to consecutive training index
    # This handles class merging when multiple classes share the same config index
    config_index_to_training_index = {}
    unique_config_indices = sorted(set(cls['training_index'] for cls in classes))
    for training_index, config_index in enumerate(unique_config_indices):
        config_index_to_training_index[config_index] = training_index
    
    # Create the final mapping from original annotation values to training indices
    # Use a conservative size to handle all possible annotation values (ZOD uses 0-5)
    max_original_value = max(6, max(c['original_index'] for c in classes) + 1)  # At least 6 for ZOD
    original_to_training_mapping = np.full(max_original_value, 0, dtype=int)  # Default to 0 (background)
    
    for cls in classes:
        original_annotation_value = cls['original_index']
        config_index = cls['training_index']
        training_index = config_index_to_training_index[config_index]
        original_to_training_mapping[original_annotation_value] = training_index
    
    # Apply mapping
    relabeled = original_to_training_mapping[annotation]
    
    return torch.from_numpy(relabeled).unsqueeze(0).long()  # [H,W]->[1,H,W]


def draw_test_segmentation_map(outputs, config=None):
    """
    Create segmentation visualization with colors based on config class definitions.
    
    Args:
        outputs: Model output tensor
        config: Configuration dictionary with Dataset.classes. If None, uses hardcoded colors.
    """
    labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()
    
    # Create color mapping based on config or use default
    if config is not None:
        classes = config['Dataset']['classes']
        
        # Create mapping from training index to color index (original_index)
        training_to_color_index = {}
        for cls in classes:
            training_index = cls['training_index']
            color_index = cls['original_index']
            training_to_color_index[training_index] = color_index
        
        # Create color list for training indices
        unique_training_indices = sorted(training_to_color_index.keys())
        color_list = []
        
        for training_index in unique_training_indices:
            color_index = training_to_color_index[training_index]
            if color_index < len(label_colors_list):
                color_list.append(label_colors_list[color_index])
            else:
                # Fallback color
                color_list.append((255, 255, 255))  # white
        
        # Dataset-specific color adjustments
        if config.get('Dataset', {}).get('name') == 'waymo':
            # For Waymo, set background to black, vehicle to red, cyclist + pedestrian to yellow
            bg_training_index = None
            vehicle_training_index = None
            cyclist_ped_training_index = None
            for cls in config['Dataset']['classes']:
                if cls['name'] == 'background':
                    bg_training_index = cls['training_index']
                elif cls['name'] == 'vehicle':
                    vehicle_training_index = cls['training_index']
                elif cls['name'] == 'cyclist + pedestrian':
                    cyclist_ped_training_index = cls['training_index']
            if bg_training_index is not None and bg_training_index in unique_training_indices:
                bg_idx = unique_training_indices.index(bg_training_index)
                color_list[bg_idx] = (0, 0, 0)
            if vehicle_training_index is not None and vehicle_training_index in unique_training_indices:
                veh_idx = unique_training_indices.index(vehicle_training_index)
                color_list[veh_idx] = (255, 0, 0)
            if cyclist_ped_training_index is not None and cyclist_ped_training_index in unique_training_indices:
                cp_idx = unique_training_indices.index(cyclist_ped_training_index)
                color_list[cp_idx] = (255, 255, 0)
    else:
        # Use default hardcoded colors
        color_list = label_colors_list
    
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    for label_num in range(0, len(color_list)):
        idx = labels == label_num
        red_map[idx] = color_list[label_num][0]
        green_map[idx] = color_list[label_num][1]
        blue_map[idx] = color_list[label_num][2]

    segmented_image = np.stack([red_map, green_map, blue_map], axis=2)
    return segmented_image

def image_overlay(image, segmented_image):
    """
    Create overlay with transparent masks on original image.
    Only predicted classes are shown with transparency, background remains original.
    """
    # Create a copy of the original image
    overlay = image.copy().astype(np.float32)

    # Find non-black and non-background pixels in segmented image (predicted classes)
    # Background (class 5 for Waymo) is cyan (255,255,0), ignore (class 0) is black (0,0,0)
    mask = np.any(segmented_image != [0, 0, 0], axis=2) & np.any(segmented_image != [255, 255, 0], axis=2)

    # Apply alpha blending only to predicted regions
    alpha = 0.6  # transparency level
    overlay[mask] = alpha * segmented_image[mask].astype(np.float32) + (1 - alpha) * overlay[mask]

    return overlay.astype(np.uint8)

def get_model_path(config):
    model_path = config['General']['model_path']
    if model_path != '':
        return config['General']['model_path']
    # If model path not specified then take latest checkpoint
    files = glob.glob(config['Log']['logdir']+'progress_save/*.pth')
    if len(files) == 0:
        return False
    # Sort by checkpoint number (not by file creation time which can be unreliable)
    def get_checkpoint_num(filepath):
        try:
            filename = os.path.basename(filepath)
            # Handle both old format (checkpoint_0.pth) and new format (epoch_0_uuid.pth)
            if filename.startswith('checkpoint_'):
                num_str = filename.replace('checkpoint_', '').replace('.pth', '')
            elif filename.startswith('epoch_'):
                # Extract epoch number from epoch_0_uuid.pth format
                parts = filename.replace('epoch_', '').replace('.pth', '').split('_')
                num_str = parts[0] if parts else '0'
            else:
                num_str = '0'
            return int(num_str)
        except:
            return 0
    
    latest_file = max(files, key=get_checkpoint_num)
    return latest_file

def save_model_dict(config, epoch, model, optimizer, epoch_uuid=None):
    creat_dir(config)
    if epoch_uuid:
        filename = f"epoch_{epoch}_{epoch_uuid}.pth"
    else:
        filename = f"checkpoint_{epoch}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        config['Log']['logdir']+'progress_save/'+filename
    )

def adjust_learning_rate(config, optimizer, epoch):
    """Decay the learning rate based on schedule"""
    epoch_max = config['General']['epochs']
    momentum = config['CLFT']['lr_momentum']
    # lr = config['General']['dpt_lr'] * (1-epoch/epoch_max)**0.9
    lr = config['CLFT']['clft_lr'] * (momentum ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

class EarlyStopping(object):
    def __init__(self, config):
        self.patience = config['General']['early_stop_patience']
        self.config = config
        self.min_param = None
        self.early_stop_trigger = False
        self.count = 0

    def __call__(self, valid_param, epoch, model, optimizer, epoch_uuid=None):
        if self.min_param is None:
            self.min_param = valid_param
        elif valid_param >= self.min_param:
            self.count += 1
            print(f'Early Stopping Counter: {self.count} of {self.patience}')
            if self.count >= self.patience:
                self.early_stop_trigger = True
                print('Saving model for last epoch...')
                save_model_dict(self.config, epoch, model, optimizer, epoch_uuid)
                print('Saving Model Complete')
                print('Early Stopping Triggered!')
        else:
            print(f'Valid loss decreased from {self.min_param:.4f} ' + f'to {valid_param:.4f}')
            self.min_param = valid_param
            # Check if this epoch will also be saved as a regular checkpoint
            save_epoch = self.config['General']['save_epoch']
            if not (epoch == 0 or (epoch + 1) % save_epoch == 0):
                save_model_dict(self.config, epoch, model, optimizer, epoch_uuid)
                print('Saving Model...')
            else:
                print('Skipping early stopping save (regular checkpoint will be saved)')
            self.count = 0

def create_config_snapshot():
    source_file = 'config.json'
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    destination_file = f'config_{timestamp}.json'
    shutil.copy(source_file, destination_file)
    print(f'Config snapshot created {destination_file}')
