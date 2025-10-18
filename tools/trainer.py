#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import time

from clfcn.fusion_net import FusionNet
from utils.metrics import find_overlap_1, zod_find_overlap_1
from clft.clft import CLFT
from utils.helpers import EarlyStopping, get_model_path
from utils.helpers import save_model_dict
from utils.helpers import adjust_learning_rate
from integrations.training_logger import log_epoch_results

writer = SummaryWriter()

class Trainer(object):
    def __init__(self, config, training_uuid=None, log_dir=None):
        super().__init__()
        self.config = config
        self.training_uuid = training_uuid
        self.log_dir = log_dir
        self.finished_epochs = 0
        self.device = torch.device(self.config['General']['device']
                                   if torch.cuda.is_available() else "cpu")
        print("device: %s" % self.device)

        if config['CLI']['backbone'] == 'clfcn':
            self.model = FusionNet()
            print(f"Using backbone {config['CLI']['backbone']}")
            self.optimizer_clfcn = torch.optim.Adam(self.model.parameters(), lr=config['CLFCN']['clfcn_lr'])
            self.scheduler_clfcn = ReduceLROnPlateau(self.optimizer_clfcn)

        elif config['CLI']['backbone'] == 'clft':
            resize = config['Dataset']['transforms']['resize']
            self.model = CLFT(
                RGB_tensor_size=(3, resize, resize),
                XYZ_tensor_size=(3, resize, resize),
                patch_size=config['CLFT']['patch_size'], # ?
                emb_dim=config['CLFT']['emb_dim'], # ?
                resample_dim=config['CLFT']['resample_dim'], # ?
                read=config['CLFT']['read'], # ?
                hooks=config['CLFT']['hooks'], # ?
                reassemble_s=config['CLFT']['reassembles'], # ?
                nclasses=len(config['Dataset']['classes']),
                type=config['CLFT']['type'], # ?
                model_timm=config['CLFT']['model_timm'], # ?
            )
            print(f"Using backbone {config['CLI']['backbone']}")
            self.optimizer_clft = torch.optim.Adam(self.model.parameters(), lr=config['CLFT']['clft_lr'])

        else:
            sys.exit("A backbone must be specified! (clft or clfcn)")

        self.model.to(self.device)

        self.nclasses = len(config['Dataset']['classes'])
        # Determine if we are running RGB-only mode (no separate lidar input)
        self.rgb_only = self.config.get('CLI', {}).get('mode', '') == 'rgb'
        
        # Read class_weights from config if available, otherwise use defaults
        if 'class_weights' in config['Dataset']:
            class_weights_config = config['Dataset']['class_weights']
            weight_loss = torch.Tensor(class_weights_config)
            print(f"Using class weights from config: {class_weights_config}")
        else:
            # Default weights if not specified in config
            weight_loss = torch.Tensor(self.nclasses).fill_(0)
            weight_loss[0] = 1 # background
            weight_loss[1] = 0 # vehicle (not present in ZOD)
            weight_loss[2] = 20 # sign
            weight_loss[3] = 68 # cyclist
            weight_loss[4] = 180 # pedestrian
            weight_loss[5] = 0 # ignore
            print(f"Using default class weights: {weight_loss.tolist()}")
        
        self.criterion = nn.CrossEntropyLoss(weight=weight_loss).to(self.device)

        # Choose appropriate IoU calculation function based on dataset
        self.dataset_name = config['Dataset']['name']
        if self.dataset_name == 'zod':
            self.find_overlap_func = zod_find_overlap_1
            print("Using ZOD-specific IoU calculation")
        else:
            self.find_overlap_func = find_overlap_1
            print("Using standard IoU calculation")

        if self.config['General']['resume_training'] is True:
            model_path = get_model_path(config)
            if model_path:
                print(f'Resume training on {model_path}')
                checkpoint = torch.load(model_path, map_location=self.device)

                if self.config['General']['reset_lr'] is True:
                    print('Reset the epoch to 0')
                    self.finished_epochs = 0
                else:
                    self.finished_epochs = checkpoint['epoch']
                    print( f"Finished epochs in previous training: {self.finished_epochs}")

                if self.config['General']['epochs'] <= self.finished_epochs:
                    print('Current epochs amount is smaller than finished epochs!!!')
                    print(f"Please setting the epochs bigger than {self.finished_epochs}")
                    sys.exit()
                else:
                    print('Loading trained model weights...')
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print('Loading trained optimizer...')
                    self.optimizer_clft.load_state_dict(checkpoint['optimizer_state_dict'])

        else:
            print('Training from the beginning')

    def train_clft(self, train_dataloader, valid_dataloader, modal):
        """
        The training of one epoch
        """
        epochs = self.config['General']['epochs']
        modality = modal
        early_stopping = EarlyStopping(self.config) # ?
        self.model.train()
        for epoch in range(self.finished_epochs, epochs):
            epoch_start_time = time.time()
            lr = adjust_learning_rate(self.config, self.optimizer_clft, epoch)
            print('Epoch: {:.0f}, LR: {:.6f}'.format(epoch, lr))
            print('Training...')
            train_loss = 0.0
            overlap_cum, pred_cum, label_cum, union_cum = 0, 0, 0, 0
            progress_bar = tqdm(train_dataloader)
            for i, batch in enumerate(progress_bar):
                batch['rgb'] = batch['rgb'].to(self.device, non_blocking=True)
                batch['lidar'] = batch['lidar'].to(self.device, non_blocking=True)
                batch['anno'] = batch['anno'].to(self.device, non_blocking=True) # ?

                self.optimizer_clft.zero_grad()

                # If running RGB-only, feed rgb into both rgb and lidar inputs
                lidar_input = batch['rgb'] if self.rgb_only else batch['lidar']
                _, output_seg = self.model(batch['rgb'], lidar_input, modality)

                # 1xHxW -> HxW
                output_seg = output_seg.squeeze(1)

                anno = batch['anno']

                batch_overlap, batch_pred, batch_label, batch_union = self.find_overlap_func(self.nclasses, output_seg, anno) # is it iou?

                overlap_cum += batch_overlap
                pred_cum += batch_pred
                label_cum += batch_label
                union_cum += batch_union

                loss = self.criterion(output_seg, batch['anno'])

                train_loss += loss.item()
                loss.backward()
                self.optimizer_clft.step()
                progress_bar.set_description(f'CLFT train loss:{loss:.4f}')

            # The IoU of one epoch
            train_epoch_IoU = overlap_cum / union_cum
            print(f'Training Vehicle IoU for Epoch: {train_epoch_IoU[0]:.4f}')
            print(f'Training Sign IoU for Epoch: {train_epoch_IoU[1]:.4f}')
            print(f'Training Cyclist IoU for Epoch: {train_epoch_IoU[2]:.4f}')
            print(f'Training Pedestrian IoU for Epoch: {train_epoch_IoU[3]:.4f}')
            # The loss_rgb of one epoch
            train_epoch_loss = train_loss / (i + 1)
            print(f'Average Training Loss for Epoch: {train_epoch_loss:.4f}')

            valid_epoch_loss, valid_epoch_IoU, valid_precision, valid_recall, valid_f1 = self.validate_clft(valid_dataloader, modality)

            epoch_time = time.time() - epoch_start_time

            # Compute additional metrics for training
            train_precision = overlap_cum / (pred_cum + 1e-6)
            train_recall = overlap_cum / (label_cum + 1e-6)
            train_f1 = 2 * train_precision * train_recall / (train_precision + train_recall + 1e-6)
            train_mean_iou = torch.mean(train_epoch_IoU).item()

            # Plot the train and validation loss in Tensorboard
            writer.add_scalars('Loss', {'train': train_epoch_loss, 'valid': valid_epoch_loss}, epoch)
            # Plot the train and validation IoU in Tensorboard
            writer.add_scalars('Vehicle_IoU', {'train': train_epoch_IoU[0], 'valid': valid_epoch_IoU[0]}, epoch)
            writer.add_scalars('Sign_IoU', {'train': train_epoch_IoU[1], 'valid': valid_epoch_IoU[1]}, epoch)
            writer.add_scalars('Cyclist_IoU', {'train': train_epoch_IoU[2], 'valid': valid_epoch_IoU[2]}, epoch)
            writer.add_scalars('Pedestrian_IoU', {'train': train_epoch_IoU[3], 'valid': valid_epoch_IoU[3]}, epoch)
            writer.close()

            if self.training_uuid and self.log_dir:
                results = {
                    "train": {
                        "vehicle": {
                            "iou": train_epoch_IoU[0].item(),
                            "precision": train_precision[0].item(),
                            "recall": train_recall[0].item(),
                            "f1": train_f1[0].item()
                        },
                        "sign": {
                            "iou": train_epoch_IoU[1].item(),
                            "precision": train_precision[1].item(),
                            "recall": train_recall[1].item(),
                            "f1": train_f1[1].item()
                        },
                        "cyclist": {
                            "iou": train_epoch_IoU[2].item(),
                            "precision": train_precision[2].item(),
                            "recall": train_recall[2].item(),
                            "f1": train_f1[2].item()
                        },
                        "pedestrian": {
                            "iou": train_epoch_IoU[3].item(),
                            "precision": train_precision[3].item(),
                            "recall": train_recall[3].item(),
                            "f1": train_f1[3].item()
                        },
                        "loss": train_epoch_loss,
                        "mean_iou": train_mean_iou
                    },
                    "val": {
                        "vehicle": {
                            "iou": valid_epoch_IoU[0].item(),
                            "precision": valid_precision[0].item(),
                            "recall": valid_recall[0].item(),
                            "f1": valid_f1[0].item()
                        },
                        "sign": {
                            "iou": valid_epoch_IoU[1].item(),
                            "precision": valid_precision[1].item(),
                            "recall": valid_recall[1].item(),
                            "f1": valid_f1[1].item()
                        },
                        "cyclist": {
                            "iou": valid_epoch_IoU[2].item(),
                            "precision": valid_precision[2].item(),
                            "recall": valid_recall[2].item(),
                            "f1": valid_f1[2].item()
                        },
                        "pedestrian": {
                            "iou": valid_epoch_IoU[3].item(),
                            "precision": valid_precision[3].item(),
                            "recall": valid_recall[3].item(),
                            "f1": valid_f1[3].item()
                        },
                        "loss": valid_epoch_loss,
                        "mean_iou": torch.mean(valid_epoch_IoU).item()
                    }
                }
                log_epoch_results(epoch, self.training_uuid, results, self.log_dir, learning_rate=lr, epoch_time=epoch_time)

            early_stop_index = round(valid_epoch_loss, 4)
            early_stopping(early_stop_index, epoch, self.model, self.optimizer_clft)
            if ((epoch + 1) % self.config['General']['save_epoch'] == 0 and epoch > 0):
                print('Saving model for every N epochs...')
                save_model_dict(self.config, epoch, self.model, self.optimizer_clft)
                print('Saving Model Complete')
            if early_stopping.early_stop_trigger is True:
                break
        print('Training Complete')

    def validate_clft(self, valid_dataloader, modal):
        """
            The validation of one epoch
        """
        self.model.eval()
        print('Validating...')
        valid_loss = 0.0
        overlap_cum, pred_cum, label_cum, union_cum = 0, 0, 0, 0
        with torch.no_grad():
            progress_bar = tqdm(valid_dataloader)
            for i, batch in enumerate(progress_bar):
                batch['rgb'] = batch['rgb'].to(self.device, non_blocking=True)
                batch['lidar'] = batch['lidar'].to(self.device, non_blocking=True)
                batch['anno'] = batch['anno'].to(self.device, non_blocking=True)

                lidar_input = batch['rgb'] if self.rgb_only else batch['lidar']
                _, output_seg = self.model(batch['rgb'], lidar_input, modal)
                # 1xHxW -> HxW
                output_seg = output_seg.squeeze(1)
                anno = batch['anno']

                batch_overlap, batch_pred, batch_label, batch_union = self.find_overlap_func(self.nclasses, output_seg, anno)

                overlap_cum += batch_overlap
                pred_cum += batch_pred
                label_cum += batch_label
                union_cum += batch_union

                loss = self.criterion(output_seg, batch['anno'])
                valid_loss += loss.item()
                progress_bar.set_description(f'valid fusion loss: {loss:.4f}')

        # The IoU of one epoch
        valid_epoch_IoU = overlap_cum / union_cum
        print(f'Validation Vehicle IoU for Epoch: {valid_epoch_IoU[0]:.4f}')
        print(f'Validation Sign IoU for Epoch: {valid_epoch_IoU[1]:.4f}')
        print(f'Validation Cyclist IoU for Epoch: {valid_epoch_IoU[2]:.4f}')
        print(f'Validation Pedestrian IoU for Epoch: {valid_epoch_IoU[3]:.4f}')
        # The loss_rgb of one epoch
        valid_epoch_loss = valid_loss / (i + 1)
        print(f'Average Validation Loss for Epoch: {valid_epoch_loss:.4f}')

        # Compute additional metrics
        valid_precision = overlap_cum / (pred_cum + 1e-6)
        valid_recall = overlap_cum / (label_cum + 1e-6)
        valid_f1 = 2 * valid_precision * valid_recall / (valid_precision + valid_recall + 1e-6)

        return valid_epoch_loss, valid_epoch_IoU, valid_precision, valid_recall, valid_f1
