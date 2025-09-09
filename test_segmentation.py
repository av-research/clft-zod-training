import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import pickle

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json

from dataset import ZODDataset
from model import ViTSegmentation

# Metric Calculation Functions
def calculate_metrics(pred, target, num_classes):
    """
    Calculate IoU, Precision, Recall, Accuracy per class.
    pred: [H, W] predicted class indices
    target: [H, W] ground truth class indices
    """
    pred = pred.flatten()
    target = target.flatten()
    
    iou_per_class = []
    precision_per_class = []
    recall_per_class = []
    accuracy_per_class = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        pred_sum = pred_cls.sum().float()
        target_sum = target_cls.sum().float()
        
        # IoU
        iou = (intersection + 1e-6) / (union + 1e-6)
        iou_per_class.append(iou.item())
        
        # Precision
        precision = (intersection + 1e-6) / (pred_sum + 1e-6)
        precision_per_class.append(precision.item())
        
        # Recall
        recall = (intersection + 1e-6) / (target_sum + 1e-6)
        recall_per_class.append(recall.item())
        
        # Accuracy (per class)
        correct = (pred_cls == target_cls).sum().float()
        total = pred_cls.numel()
        accuracy = correct / total
        accuracy_per_class.append(accuracy.item())
    
    return {
        'iou': iou_per_class,
        'precision': precision_per_class,
        'recall': recall_per_class,
        'accuracy': accuracy_per_class
    }

# Test Function
def test_model(model, dataloader, num_classes=5, save_path=None, checkpoint_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    total_metrics = {
        'iou': np.zeros(num_classes),
        'precision': np.zeros(num_classes),
        'recall': np.zeros(num_classes),
        'accuracy': np.zeros(num_classes)
    }
    count = 0
    
    with torch.no_grad():
        for rgb, lidar, anno in tqdm(dataloader):
            rgb, lidar, anno = rgb.to(device), lidar.to(device), anno.to(device)
            
            outputs = model(rgb, lidar)
            pred = torch.argmax(outputs, dim=1)  # [batch, H, W]
            
            for i in range(pred.size(0)):
                metrics = calculate_metrics(pred[i], anno[i], num_classes)
                for key in total_metrics:
                    total_metrics[key] += np.array(metrics[key])
                count += 1
    
    # Average metrics
    for key in total_metrics:
        total_metrics[key] /= count
    
    # Prepare results for JSON
    class_names = ['background', 'vehicle', 'pedestrian', 'cyclist', 'sign']
    results = {
        'metadata': {
            'model': 'ViTSegmentation',
            'num_classes': num_classes,
            'checkpoint': checkpoint_path.replace('./model_path/', '') if checkpoint_path else 'unknown',
            'test_samples': count,
            'device': str(device)
        },
        'per_class_metrics': {},
        'mean_metrics': {}
    }
    
    # Per-class results
    for cls in range(num_classes):
        results['per_class_metrics'][class_names[cls]] = {
            'iou': round(total_metrics['iou'][cls], 4),
            'precision': round(total_metrics['precision'][cls], 4),
            'recall': round(total_metrics['recall'][cls], 4),
            'accuracy': round(total_metrics['accuracy'][cls], 4)
        }
    
    # Mean results
    results['mean_metrics'] = {
        'mean_iou': round(np.mean(total_metrics['iou']), 4),
        'mean_precision': round(np.mean(total_metrics['precision']), 4),
        'mean_recall': round(np.mean(total_metrics['recall']), 4),
        'mean_accuracy': round(np.mean(total_metrics['accuracy']), 4)
    }
    
    # Save to JSON if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {save_path}")
    
    # Print results
    print("\nTest Results:")
    print("-" * 50)
    for cls in range(num_classes):
        print(f"{class_names[cls]}:")
        print(f"  IoU: {total_metrics['iou'][cls]:.4f}")
        print(f"  Precision: {total_metrics['precision'][cls]:.4f}")
        print(f"  Recall: {total_metrics['recall'][cls]:.4f}")
        print(f"  Accuracy: {total_metrics['accuracy'][cls]:.4f}")
        print()
    
    # Mean metrics
    print(f"Mean IoU: {np.mean(total_metrics['iou']):.4f}")
    print(f"Mean Precision: {np.mean(total_metrics['precision']):.4f}")
    print(f"Mean Recall: {np.mean(total_metrics['recall']):.4f}")
    print(f"Mean Accuracy: {np.mean(total_metrics['accuracy']):.4f}")
    
    return results

# Main Script
if __name__ == '__main__':
    # Paths
    data_dir = './zod_dataset'
    split_file = './zod_dataset/splits_zod/all.txt'  # Or use a test split if available
    checkpoint_path = './model_path/checkpoint_epoch_1000_cross_fusion.pth'  # Use overtrained model
    results_save_path = './model_results/test_results.json'
    
    # Dataset and Dataloader
    dataset = ZODDataset(data_dir, split_file)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # Model
    model = ViTSegmentation(mode='cross_fusion', num_classes=5)

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}, using untrained model")
    
    # Test and save results
    test_model(model, dataloader, save_path=results_save_path, checkpoint_path=checkpoint_path)
