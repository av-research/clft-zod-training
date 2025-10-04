#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import torch
import numpy as np
import shutil
import datetime
import glob

label_colors_list = [
        (0, 0, 0),        # background
        (0, 255, 0),      # pedestrian
        (255, 0, 0),      # sign
        (255, 255, 0),    # cyclist
]

# all the classes that are present in the dataset
ALL_CLASSES = ['background', 'pedestrian', 'sign', 'cyclist']

"""
This (`class_values`) assigns a specific class label to each of the classes.
For example, `vehicle=0`, `human=1`, and so on.
"""
class_values = [ALL_CLASSES.index(cls.lower()) for cls in ALL_CLASSES]


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
    :parameter annotation: 0->ignore, 1->vehicle, 2->pedestrian, 3->sign,
                            4->cyclist, 5->background
    :return annotation: 0->background+vehicle+ignore, 1->pedestrian, 2->sign,
                            3->cyclist
    """
    annotation = np.array(annotation)

    # Merge background, vehicle, ignore to 0
    annotation[(annotation == 1) | (annotation == 5)] = 0

    # Remap pedestrian to 1
    annotation[annotation == 2] = 1

    # Remap sign to 2
    annotation[annotation == 3] = 2

    # Remap cyclist to 3
    annotation[annotation == 4] = 3

    return torch.from_numpy(annotation).unsqueeze(0).long() # [H,W]->[1,H,W]


def draw_test_segmentation_map(outputs):
    labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()
    # labels = outputs.squeeze().detach().cpu().numpy()
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    black_map = np.zeros_like(labels).astype(np.uint8)

    for label_num in range(0, len(label_colors_list)):
        if label_num in class_values:
            idx = labels == label_num
            red_map[idx] = np.array(label_colors_list)[label_num, 0]
            green_map[idx] = np.array(label_colors_list)[label_num, 1]
            black_map[idx] = np.array(label_colors_list)[label_num, 2]

    segmented_image = np.stack([red_map, green_map, black_map], axis=2)
    return segmented_image

def image_overlay(image, segmented_image):
    alpha = 0.4  # how much transparency to apply
    beta = 1 - alpha  # alpha + beta should equal 1
    gamma = 0  # scalar added to each sum
    cv2.addWeighted(segmented_image, alpha, image, beta, gamma, image)
    return image

def get_model_path(config):
    model_path = config['General']['model_path']
    if model_path != '':
        return config['General']['model_path']
    # If model path not specified then take latest checkpoint
    files = glob.glob(config['Log']['logdir']+'progress_save/*.pth')
    if len(files) == 0:
        return False
    latest_file = max(files, key=os.path.getctime)
    return latest_file

def save_model_dict(config, epoch, model, optimizer):
    creat_dir(config)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        config['Log']['logdir']+'progress_save/'+f"checkpoint_{epoch}.pth"
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

    def __call__(self, valid_param, epoch, model, optimizer):
        if self.min_param is None:
            self.min_param = valid_param
        elif valid_param >= self.min_param:
            self.count += 1
            print(f'Early Stopping Counter: {self.count} of {self.patience}')
            if self.count >= self.patience:
                self.early_stop_trigger = True
                print('Saving model for last epoch...')
                save_model_dict(self.config, epoch, model, optimizer)
                print('Saving Model Complete')
                print('Early Stopping Triggered!')
        else:
            print(f'Valid loss decreased from {self.min_param:.4f} ' + f'to {valid_param:.4f}')
            self.min_param = valid_param
            save_model_dict(self.config, epoch, model, optimizer)
            print('Saving Model...')
            self.count = 0

def create_config_snapshot():
    source_file = 'config.json'
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    destination_file = f'config_{timestamp}.json'
    shutil.copy(source_file, destination_file)
    print(f'Config snapshot created {destination_file}')
