#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLabV3+ implementation for comparison with CLFT.
Supports RGB-only and late-fusion variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models import resnet101, ResNet101_Weights


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module."""
    
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()
        
        # Atrous convolutions with different dilation rates
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final projection
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        size = x.size()[2:]
        
        # Apply all branches
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = F.interpolate(self.global_pool(x), size=size, mode='bilinear', align_corners=True)
        
        # Concatenate and project
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        out = self.project(out)
        
        return out


class Decoder(nn.Module):
    """DeepLabV3+ decoder module."""
    
    def __init__(self, low_level_channels, num_classes, aspp_channels=256):
        super(Decoder, self).__init__()
        
        # Low-level feature projection
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Refinement convolutions
        self.refine = nn.Sequential(
            nn.Conv2d(aspp_channels + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Final classifier
        self.classifier = nn.Conv2d(256, num_classes, 1)
    
    def forward(self, x, low_level_feat, input_size):
        # Process low-level features
        low_level_feat = self.low_level_conv(low_level_feat)
        
        # Upsample high-level features to match low-level
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        
        # Concatenate and refine
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.refine(x)
        
        # Final classification
        x = self.classifier(x)
        
        # Upsample to input size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        return x


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ model for semantic segmentation.
    Supports RGB-only mode.
    """
    
    def __init__(self, num_classes, backbone='resnet101', pretrained=True):
        super(DeepLabV3Plus, self).__init__()
        
        # Load ResNet backbone
        if backbone == 'resnet101':
            if pretrained:
                resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            else:
                resnet = resnet101(weights=None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Extract backbone layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        
        # Apply dilation to layer4 for atrous convolutions
        self.layer4 = self._make_layer4_atrous(resnet.layer4)
        
        # ASPP module (output channels from layer4 = 2048)
        self.aspp = ASPP(in_channels=2048, out_channels=256)
        
        # Decoder (low-level features from layer1 = 256 channels)
        self.decoder = Decoder(low_level_channels=256, num_classes=num_classes)
        
    def _make_layer4_atrous(self, layer4):
        """Modify layer4 to use atrous convolutions."""
        for n, m in layer4.named_modules():
            if 'conv2' in n:
                m.dilation = (2, 2)
                m.padding = (2, 2)
                m.stride = (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        return layer4
    
    def forward(self, x):
        input_size = x.size()[2:]
        
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        low_level_feat = self.layer1(x)  # 1/4 resolution
        x = self.layer2(low_level_feat)   # 1/8 resolution
        x = self.layer3(x)                # 1/16 resolution
        x = self.layer4(x)                # 1/16 resolution (with atrous)
        
        # ASPP
        x = self.aspp(x)
        
        # Decoder
        x = self.decoder(x, low_level_feat, input_size)
        
        return x


class DeepLabV3PlusLateFusion(nn.Module):
    """
    DeepLabV3+ with late fusion for camera-LiDAR data.
    Uses two separate DeepLabV3+ networks and fuses predictions.
    """
    
    def __init__(self, num_classes, fusion_type='learned', pretrained=True):
        super(DeepLabV3PlusLateFusion, self).__init__()
        
        self.fusion_type = fusion_type
        
        # RGB branch
        self.rgb_branch = DeepLabV3Plus(num_classes, backbone='resnet101', pretrained=pretrained)
        
        # LiDAR branch (pretrained on ImageNet but will learn LiDAR features)
        self.lidar_branch = DeepLabV3Plus(num_classes, backbone='resnet101', pretrained=pretrained)
        
        # Fusion layer (if learned)
        if fusion_type == 'learned':
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(num_classes * 2, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(128, num_classes, 1)
            )
        elif fusion_type == 'weighted':
            # Learnable weights for weighted average
            self.rgb_weight = nn.Parameter(torch.tensor(0.5))
            self.lidar_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, rgb, lidar):
        # Get predictions from both branches
        rgb_pred = self.rgb_branch(rgb)
        lidar_pred = self.lidar_branch(lidar)
        
        # Fuse predictions
        if self.fusion_type == 'average':
            # Simple averaging
            out = (rgb_pred + lidar_pred) / 2
        elif self.fusion_type == 'weighted':
            # Learnable weighted average
            out = self.rgb_weight * rgb_pred + self.lidar_weight * lidar_pred
        elif self.fusion_type == 'learned':
            # Concatenate and learn fusion
            concat = torch.cat([rgb_pred, lidar_pred], dim=1)
            out = self.fusion_conv(concat)
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")
        
        return out, rgb_pred, lidar_pred
    
    def forward_single(self, x, modality='rgb'):
        """Forward pass for single modality (for evaluation)."""
        if modality == 'rgb':
            return self.rgb_branch(x)
        elif modality == 'lidar':
            return self.lidar_branch(x)
        else:
            raise ValueError(f"Invalid modality: {modality}")


def build_deeplabv3plus(num_classes, mode='rgb', fusion_type='learned', pretrained=True):
    """
    Build DeepLabV3+ model based on configuration.
    
    Args:
        num_classes (int): Number of output classes
        mode (str): 'rgb' for RGB-only, 'lidar' for LiDAR-only, 'fusion' for late fusion
        fusion_type (str): Type of fusion ('average', 'weighted', 'learned')
        pretrained (bool): Use ImageNet pretrained weights
    
    Returns:
        nn.Module: DeepLabV3+ model
    """
    if mode == 'rgb':
        print(f"Building DeepLabV3+ (RGB-only) with {num_classes} classes")
        return DeepLabV3Plus(num_classes, backbone='resnet101', pretrained=pretrained)
    elif mode == 'lidar':
        print(f"Building DeepLabV3+ (LiDAR-only) with {num_classes} classes")
        return DeepLabV3Plus(num_classes, backbone='resnet101', pretrained=pretrained)
    elif mode == 'fusion':
        print(f"Building DeepLabV3+ (Late Fusion - {fusion_type}) with {num_classes} classes")
        return DeepLabV3PlusLateFusion(num_classes, fusion_type=fusion_type, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


if __name__ == '__main__':
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test RGB-only model
    print("\nTesting RGB-only model:")
    model_rgb = build_deeplabv3plus(num_classes=4, mode='rgb').to(device)
    x_rgb = torch.randn(2, 3, 384, 384).to(device)
    out_rgb = model_rgb(x_rgb)
    print(f"Input shape: {x_rgb.shape}")
    print(f"Output shape: {out_rgb.shape}")
    
    # Test late fusion model
    print("\nTesting late fusion model:")
    model_fusion = build_deeplabv3plus(num_classes=4, mode='fusion', fusion_type='learned').to(device)
    x_lidar = torch.randn(2, 3, 384, 384).to(device)
    out_fusion, out_rgb_branch, out_lidar_branch = model_fusion(x_rgb, x_lidar)
    print(f"RGB input shape: {x_rgb.shape}")
    print(f"LiDAR input shape: {x_lidar.shape}")
    print(f"Fused output shape: {out_fusion.shape}")
    print(f"RGB branch output shape: {out_rgb_branch.shape}")
    print(f"LiDAR branch output shape: {out_lidar_branch.shape}")
    
    print("\nModel test passed!")
