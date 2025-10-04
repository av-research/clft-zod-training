#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import json
import datetime

from clfcn.fusion_net import FusionNet
from utils.metrics import find_overlap_1
from clft.clft import CLFT
from utils.helpers import EarlyStopping, get_model_path
from utils.helpers import save_model_dict
from utils.helpers import adjust_learning_rate

writer = SummaryWriter()

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class Trainer(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
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
        weight_loss = torch.Tensor(self.nclasses).fill_(0)
        # define weight of different classes
        weight_loss[0] = 0.1 # background
        weight_loss[1] = 15 # pedestrian
        weight_loss[2] = 12 # sign
        weight_loss[3] = 2 # cyclist
        # self.criterion = nn.CrossEntropyLoss(weight=weight_loss).to(self.device)
        self.criterion = FocalLoss(alpha=weight_loss, gamma=2).to(self.device)

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

                _, output_seg = self.model(batch['rgb'], batch['lidar'], modality)

                # 1xHxW -> HxW
                output_seg = output_seg.squeeze(1)

                anno = batch['anno']

                batch_overlap, batch_pred, batch_label, batch_union = find_overlap_1(self.nclasses, output_seg, anno) # is it iou?

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
            for i in range(self.nclasses):
                class_name = self.config['Dataset']['classes'][str(i)]['name']
                print(f'Training {class_name} IoU for Epoch: {train_epoch_IoU[i]:.4f}')
            # The loss_rgb of one epoch
            train_epoch_loss = train_loss / (i + 1)
            print(f'Average Training Loss for Epoch: {train_epoch_loss:.4f}')

            valid_epoch_loss, valid_epoch_IoU = self.validate_clft(valid_dataloader, modality)

            # self.scheduler_backbone.step(valid_epoch_loss)
            # self.scheduler_scratch.step(valid_epoch_loss)

            # Plot the train and validation loss in Tensorboard
            writer.add_scalars('Loss', {'train': train_epoch_loss, 'valid': valid_epoch_loss}, epoch)
            # Plot the train and validation IoU in Tensorboard
            writer.add_scalars('Pedestrian_IoU', {'train': train_epoch_IoU[1], 'valid': valid_epoch_IoU[1]}, epoch) 
            writer.add_scalars('Sign_IoU', {'train': train_epoch_IoU[2], 'valid': valid_epoch_IoU[2]}, epoch)
            writer.add_scalars('Cyclist_IoU', {'train': train_epoch_IoU[3], 'valid': valid_epoch_IoU[3]}, epoch)
            writer.close()

            # Save to JSON log
            log_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'epoch': int(epoch),
                'lr': float(lr),
                'train_loss': float(train_epoch_loss),
                'train_iou': {self.config['Dataset']['classes'][str(i)]['name']: float(train_epoch_IoU[i].item()) for i in range(self.nclasses)},
                'val_loss': float(valid_epoch_loss),
                'val_iou': {self.config['Dataset']['classes'][str(i)]['name']: float(valid_epoch_IoU[i].item()) for i in range(self.nclasses)}
            }
            log_file = os.path.join(self.config['Log']['logdir'], f'training_log_epoch_{epoch}.json')
            with open(log_file, 'w') as f:
                json.dump([log_data], f, indent=4)

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

                _, output_seg = self.model(batch['rgb'], batch['lidar'], modal)
                # 1xHxW -> HxW
                output_seg = output_seg.squeeze(1)
                anno = batch['anno']

                batch_overlap, batch_pred, batch_label, batch_union = find_overlap_1(self.nclasses, output_seg, anno)

                overlap_cum += batch_overlap
                pred_cum += batch_pred
                label_cum += batch_label
                union_cum += batch_union

                loss = self.criterion(output_seg, batch['anno'])
                valid_loss += loss.item()
                progress_bar.set_description(f'valid fusion loss: {loss:.4f}')

        # The IoU of one epoch
        valid_epoch_IoU = overlap_cum / union_cum
        for i in range(self.nclasses):
            class_name = self.config['Dataset']['classes'][str(i)]['name']
            print(f'Validation {class_name} IoU for Epoch: {valid_epoch_IoU[i]:.4f}')
        # The loss_rgb of one epoch
        valid_epoch_loss = valid_loss / (i + 1)
        print(f'Average Validation Loss for Epoch: {valid_epoch_loss:.4f}')

        return valid_epoch_loss, valid_epoch_IoU
