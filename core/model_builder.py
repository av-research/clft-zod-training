#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model builder for creating and loading CLFT models.
"""
import torch
from clft.clft import CLFT


class ModelBuilder:
    """Handles model creation and checkpoint loading."""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.num_unique_classes = self._calculate_unique_classes()
    
    def _calculate_unique_classes(self):
        """Calculate number of training classes."""
        return len(self.config['Dataset']['train_classes'])
    
    def build_model(self):
        """Build model based on configuration."""

        resize = self.config['Dataset']['transforms']['resize']
        model = CLFT(
            RGB_tensor_size=(3, resize, resize),
            XYZ_tensor_size=(3, resize, resize),
            patch_size=self.config['CLFT']['patch_size'],
            emb_dim=self.config['CLFT']['emb_dim'],
            resample_dim=self.config['CLFT']['resample_dim'],
            read=self.config['CLFT']['read'],
            hooks=self.config['CLFT']['hooks'],
            reassemble_s=self.config['CLFT']['reassembles'],
            nclasses=self.num_unique_classes,
            type=self.config['CLFT']['type'],
            model_timm=self.config['CLFT']['model_timm']
        )
        
        print(f"Built {self.config['CLI']['backbone']} model with {self.num_unique_classes} classes")
        return model
    
    def load_checkpoint(self, model, checkpoint_path):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        epoch = checkpoint.get('epoch', 0)
        print(f"Loaded checkpoint from epoch {epoch}: {checkpoint_path}")
        return model, epoch
    
    def get_num_classes(self):
        """Get number of unique classes."""
        return self.num_unique_classes
