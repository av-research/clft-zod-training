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
from integrations.vision_service import create_training, send_epoch_results

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

        # Create training in vision service
        self.vision_training_id = None
        if self.training_uuid:
            training_name = f"{config['Dataset']['name']} - {config['CLI']['backbone']}"
            model_name = config['CLI']['backbone']
            dataset_name = config['Dataset']['name']
            description = f"Training {model_name} on {dataset_name} dataset"
            
            self.vision_training_id = create_training(
                uuid=self.training_uuid,
                name=training_name,
                model=model_name,
                dataset=dataset_name,
                description=description
            )
            if self.vision_training_id:
                print(f"Created training in vision service: {self.vision_training_id}")
            else:
                print("Failed to create training in vision service")

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
        
        self.criterion = nn.CrossEntropyLoss(weight=weight_loss).to(self.device)

        # Choose appropriate IoU calculation function based on dataset
        self.dataset_name = config['Dataset']['name']
        if self.dataset_name == 'zod':
            self.find_overlap_func = zod_find_overlap_1
            self.eval_classes = ['vehicle', 'sign', 'cyclist', 'pedestrian']
            print("Using ZOD-specific IoU calculation")
        else:
            self.find_overlap_func = find_overlap_1
            self.eval_classes = ['cyclist', 'pedestrian', 'sign']
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

    def compute_epoch_metrics(self, total_loss, num_batches):
        """
        Compute final metrics for an epoch.

        Args:
            total_loss: Sum of losses across all batches
            num_batches: Number of batches processed
            modal: Training mode
            dataloader: DataLoader

        Returns:
            dict: Dictionary containing all computed metrics
        """
        # Primary IoU
        epoch_IoU = self.overlap_cum / self.union_cum

        # Additional primary metrics
        precision = self.overlap_cum / (self.pred_cum + 1e-6)
        recall = self.overlap_cum / (self.label_cum + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        # Average loss
        epoch_loss = total_loss / num_batches

        metrics = {
            'epoch_IoU': epoch_IoU,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'epoch_loss': epoch_loss,
            'mean_iou': torch.mean(epoch_IoU).item()
        }

        return metrics

    def prepare_results_dict(self, train_metrics, val_metrics):
        """
        Prepare results dictionary for logging.

        Args:
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
            modal: Training mode
            dataloader: DataLoader (not used for ZOD)

        Returns:
            dict: Results dictionary for logging
        """
        results = {
            "train": {},
            "val": {}
        }

        # Add primary metrics
        for i, cls in enumerate(self.eval_classes):
            results["train"][cls] = {
                "iou": train_metrics["epoch_IoU"][i].item(),
                "precision": train_metrics["precision"][i].item(),
                "recall": train_metrics["recall"][i].item(),
                "f1": train_metrics["f1"][i].item()
            }
            results["val"][cls] = {
                "iou": val_metrics["epoch_IoU"][i].item(),
                "precision": val_metrics["precision"][i].item(),
                "recall": val_metrics["recall"][i].item(),
                "f1": val_metrics["f1"][i].item()
            }

        results["train"]["loss"] = train_metrics["epoch_loss"]
        results["train"]["mean_iou"] = train_metrics["mean_iou"]
        results["val"]["loss"] = val_metrics["epoch_loss"]
        results["val"]["mean_iou"] = val_metrics["mean_iou"]

        return results

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

            # Initialize metric accumulators
            self.overlap_cum = torch.zeros(self.nclasses - 2).to(self.device)
            self.pred_cum = torch.zeros(self.nclasses - 2).to(self.device)
            self.label_cum = torch.zeros(self.nclasses - 2).to(self.device)
            self.union_cum = torch.zeros(self.nclasses - 2).to(self.device)
            train_loss = 0.0

            progress_bar = tqdm(train_dataloader)
            for i, batch in enumerate(progress_bar):
                batch['rgb'] = batch['rgb'].to(self.device, non_blocking=True)
                batch['lidar'] = batch['lidar'].to(self.device, non_blocking=True)
                batch['anno'] = batch['anno'].to(self.device, non_blocking=True)

                self.optimizer_clft.zero_grad()

                # Prepare model inputs based on training mode
                if self.rgb_only or modality == 'rgb':
                    # RGB-only mode: use RGB data for both inputs
                    rgb_input = batch['rgb']
                    lidar_input = batch['rgb']
                elif modality == 'lidar':
                    # LiDAR-only mode: use LiDAR data for both inputs
                    rgb_input = batch['lidar']
                    lidar_input = batch['lidar']
                else:
                    # Cross-fusion mode: use RGB and LiDAR data
                    rgb_input = batch['rgb']
                    lidar_input = batch['lidar']

                # Forward pass through model
                _, output_seg = self.model(rgb_input, lidar_input, modality)

                # 1xHxW -> HxW
                output_seg = output_seg.squeeze(1)
                anno = batch['anno']

                # Calculate metrics for this batch
                batch_overlap, batch_pred, batch_label, batch_union = self.find_overlap_func(self.nclasses, output_seg, anno)

                # Accumulate metrics
                self.overlap_cum += batch_overlap
                self.pred_cum += batch_pred
                self.label_cum += batch_label
                self.union_cum += batch_union

                loss = self.criterion(output_seg, batch['anno'])
                train_loss += loss.item()
                loss.backward()
                self.optimizer_clft.step()
                progress_bar.set_description(f'CLFT train loss:{loss:.4f}')

            # Compute epoch metrics
            train_metrics = self.compute_epoch_metrics(train_loss, len(train_dataloader))

            # Print training metrics
            print(f'Training Mean IoU for Epoch: {train_metrics["mean_iou"]:.4f}')
            for i, cls in enumerate(self.eval_classes):
                print(f'Training {cls} IoU for Epoch: {train_metrics["epoch_IoU"][i]:.4f}')
            print(f'Average Training Loss for Epoch: {train_metrics["epoch_loss"]:.4f}')

            # Validate
            val_metrics = self.validate_clft(valid_dataloader, modality)

            epoch_time = time.time() - epoch_start_time

            # Tensorboard logging
            writer.add_scalars('Loss', {'train': train_metrics['epoch_loss'], 'valid': val_metrics['epoch_loss']}, epoch)
            for i, cls in enumerate(self.eval_classes):
                writer.add_scalars(f'{cls}_IoU', {'train': train_metrics['epoch_IoU'][i], 'valid': val_metrics['epoch_IoU'][i]}, epoch)
            writer.close()

            # Detailed logging
            if self.training_uuid and self.log_dir:
                results = self.prepare_results_dict(train_metrics, val_metrics)
                log_epoch_results(epoch, self.training_uuid, results, self.log_dir, learning_rate=lr, epoch_time=epoch_time)

            # Send epoch results to vision service
            if self.vision_training_id:
                results = self.prepare_results_dict(train_metrics, val_metrics)
                # Add learning rate and epoch time to results
                results["learning_rate"] = lr
                results["epoch_time"] = epoch_time
                success = send_epoch_results(self.vision_training_id, epoch, results)
                if success:
                    print(f"Sent epoch {epoch} results to vision service")
                else:
                    print(f"Failed to send epoch {epoch} results to vision service")

            early_stop_index = round(val_metrics['epoch_loss'], 4)
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
        
        # Initialize metric accumulators as tensors (same as training)
        overlap_cum = torch.zeros(self.nclasses - 2).to(self.device)
        pred_cum = torch.zeros(self.nclasses - 2).to(self.device)
        label_cum = torch.zeros(self.nclasses - 2).to(self.device)
        union_cum = torch.zeros(self.nclasses - 2).to(self.device)

        with torch.no_grad():
            progress_bar = tqdm(valid_dataloader)
            for i, batch in enumerate(progress_bar):
                batch['rgb'] = batch['rgb'].to(self.device, non_blocking=True)
                batch['lidar'] = batch['lidar'].to(self.device, non_blocking=True)
                batch['anno'] = batch['anno'].to(self.device, non_blocking=True)

                # Prepare model inputs based on training mode
                if self.rgb_only or modal == 'rgb':
                    # RGB-only mode: use RGB data for both inputs
                    rgb_input = batch['rgb']
                    lidar_input = batch['rgb']
                elif modal == 'lidar':
                    # LiDAR-only mode: use LiDAR data for both inputs
                    rgb_input = batch['lidar']
                    lidar_input = batch['lidar']
                else:
                    # Cross-fusion mode: use RGB and LiDAR data
                    rgb_input = batch['rgb']
                    lidar_input = batch['lidar']

                # Forward pass through model
                _, output_seg = self.model(rgb_input, lidar_input, modal)
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
        print(f'Validation Mean IoU for Epoch: {torch.mean(valid_epoch_IoU):.4f}')
        for i, cls in enumerate(self.eval_classes):
            print(f'Validation {cls} IoU for Epoch: {valid_epoch_IoU[i]:.4f}')
        # The loss_rgb of one epoch
        valid_epoch_loss = valid_loss / len(valid_dataloader)
        print(f'Average Validation Loss for Epoch: {valid_epoch_loss:.4f}')

        # Compute additional metrics
        valid_precision = overlap_cum / (pred_cum + 1e-6)
        valid_recall = overlap_cum / (label_cum + 1e-6)
        valid_f1 = 2 * valid_precision * valid_recall / (valid_precision + valid_recall + 1e-6)

        # Return metrics as dictionary (consistent with training metrics)
        val_metrics = {
            'epoch_IoU': valid_epoch_IoU,
            'precision': valid_precision,
            'recall': valid_recall,
            'f1': valid_f1,
            'epoch_loss': valid_epoch_loss,
            'mean_iou': torch.mean(valid_epoch_IoU).item()
        }

        return val_metrics
