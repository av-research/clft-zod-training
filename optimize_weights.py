#!/usr/bin/env python3
"""
Weight Optimization Script
Tests different weight configurations and finds the best one based on validation IoU.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from dataset import GenericDataset
from model import ViTSegmentation

def evaluate_weights(config, weights, num_epochs=50):
    """
    Train briefly with given weights and return validation metrics.
    """
    # Update config with test weights
    config = config.copy()
    config['class_weights'] = weights
    config['num_epochs'] = num_epochs

    # Quick training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = GenericDataset(config)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    model = ViTSegmentation(config['mode'], config['num_classes'])
    model.to(device)

    class_weights = torch.tensor(config['class_weights'], dtype=torch.float).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Train for a few epochs
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for rgb, lidar, anno in dataloader:
            rgb, lidar, anno = rgb.to(device), lidar.to(device), anno.to(device)

            optimizer.zero_grad()
            outputs = model(rgb, lidar)
            loss = criterion(outputs, anno)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

    # Quick evaluation on same data (for demo - use validation set in practice)
    model.eval()
    total_iou = []
    with torch.no_grad():
        for rgb, lidar, anno in tqdm(dataloader, desc="Evaluating"):
            rgb, lidar, anno = rgb.to(device), lidar.to(device), anno.to(device)
            outputs = model(rgb, lidar)
            pred = torch.argmax(outputs, dim=1)

            # Calculate IoU for each class
            batch_iou = []
            for cls in range(config['num_classes']):
                pred_cls = (pred == cls)
                true_cls = (anno == cls)

                intersection = (pred_cls & true_cls).sum().float()
                union = (pred_cls | true_cls).sum().float()

                if union > 0:
                    iou = (intersection / union).item()
                    batch_iou.append(iou)
                else:
                    batch_iou.append(1.0 if pred_cls.sum() == 0 else 0.0)

            total_iou.append(np.mean(batch_iou))

    return np.mean(total_iou), epoch_loss / len(dataloader)

def optimize_weights():
    """
    Test different weight configurations and find the best.
    """
    # Load base config
    with open('config_zod.json', 'r') as f:
        config = json.load(f)

    # Different weight strategies to test
    weight_configs = {
        'inverse_freq': [1.0, 23.6, 100.0, 100.0, 100.0],  # Current
        'uniform': [1.0, 1.0, 1.0, 1.0, 1.0],  # No weighting
        'moderate': [1.0, 10.0, 50.0, 50.0, 50.0],  # Moderate balancing
        'aggressive': [1.0, 50.0, 200.0, 200.0, 200.0],  # Strong balancing
        'conservative': [1.0, 5.0, 20.0, 20.0, 20.0],  # Light balancing
        'background_focus': [10.0, 1.0, 1.0, 1.0, 1.0],  # Focus on background
    }

    results = {}

    print("Testing different weight configurations...")
    for name, weights in weight_configs.items():
        print(f"\nTesting {name}: {weights}")
        try:
            mean_iou, final_loss = evaluate_weights(config, weights, num_epochs=20)
            results[name] = {
                'weights': weights,
                'mean_iou': mean_iou,
                'final_loss': final_loss
            }
            print(".4f")
        except Exception as e:
            print(f"Failed: {e}")
            continue

    # Find best configuration
    if results:
        best_config = max(results.items(), key=lambda x: x[1]['mean_iou'])
        print("\n=== BEST CONFIGURATION ===")
        print(f"Strategy: {best_config[0]}")
        print(f"Weights: {best_config[1]['weights']}")
        print(".4f")
        print(".4f")

        # Save best config
        best_weights = best_config[1]['weights']
        config['class_weights'] = best_weights

        with open('config_zod_optimized.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\nSaved optimized config to config_zod_optimized.json")

    return results

if __name__ == '__main__':
    optimize_weights()
