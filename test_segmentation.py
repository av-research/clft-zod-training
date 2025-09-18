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
import sys
import argparse
from datetime import datetime
import platform

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json

from dataset import GenericDataset
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
def test_model(model, dataloader, num_classes=5, save_path=None, checkpoint_path=None, class_names=None, mode='cross_fusion'):
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
            
            outputs = model(rgb, lidar, mode)
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
    if class_names is None:
        class_names = ['class_' + str(i) for i in range(num_classes)]
    
    timestamp = datetime.now().isoformat()
    results = {
        'metadata': {
            'timestamp': timestamp,
            'model': 'ViTSegmentation',
            'num_classes': num_classes,
            'checkpoint': checkpoint_path.replace('./model_path/', '') if checkpoint_path else 'unknown',
            'test_samples': count,
            'device': str(device),
            'pytorch_version': torch.__version__,
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'class_names': class_names
        },
        'training_info': {
            'epoch': None,  # Will be extracted from checkpoint name if available
            'mode': None
        },
        'per_class_metrics': {},
        'mean_metrics': {}
    }
    
    # Extract epoch and mode from checkpoint path if available
    if checkpoint_path:
        try:
            import re
            match = re.search(r'checkpoint_epoch_(\d+)_(\w+)\.pth', checkpoint_path)
            if match:
                results['training_info']['epoch'] = int(match.group(1))
                results['training_info']['mode'] = match.group(2)
        except:
            pass
    
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

# Comprehensive testing function for multiple conditions
def test_all_conditions(model, config, checkpoint_path, epoch=None, results_prefix="test_results"):
    """
    Test model on all 4 conditions and save separate results.
    
    Args:
        model: The model to test
        config: Configuration dictionary
        checkpoint_path: Path to the checkpoint file
        epoch: Optional epoch number for filename generation
        results_prefix: Prefix for results filename
    """
    # Define test conditions for comprehensive evaluation
    test_conditions = [
        'test_day_fair',
        'test_day_rain', 
        'test_night_fair',
        'test_night_rain'
    ]
    
    results_summary = {}
    
    for condition in test_conditions:
        print(f"\n{'='*60}")
        print(f"Testing on {condition.upper()}")
        print(f"{'='*60}")
        
        # Create test config for specific condition
        test_config = config.copy()
        test_config['split_file'] = test_config['split_file'].replace('train_all.txt', f'{condition}.txt')
        
        # Check if test file exists
        if not os.path.exists(test_config['split_file']):
            print(f"Warning: Test file {test_config['split_file']} not found, skipping {condition}")
            continue
        
        # Create test dataset and dataloader for this condition
        test_dataset = GenericDataset(test_config, training=False)
        test_dataloader = DataLoader(test_dataset, batch_size=config.get('batch_size', 2), shuffle=False, num_workers=4, pin_memory=True)
        
        # Generate condition-specific results file name
        if epoch is not None:
            results_save_path = f'./model_results/{results_prefix}_epoch_{epoch}_{config["mode"]}_{condition}.json'
        else:
            results_save_path = f'./model_results/{results_prefix}_{config["mode"]}_{condition}.json'
        
        # Run test for this condition
        print(f"Running evaluation on {len(test_dataset)} samples...")
        results = test_model(model, test_dataloader, 
                           num_classes=config['num_classes'], 
                           save_path=results_save_path, 
                           checkpoint_path=checkpoint_path, 
                           class_names=config['class_names'],
                           mode=config['mode'])
        
        results_summary[condition] = results['mean_metrics']
    
    # Print summary of all conditions
    if results_summary:
        print(f"\n{'='*60}")
        print("SUMMARY - All Test Conditions")
        print(f"{'='*60}")
        for condition, metrics in results_summary.items():
            print(f"{condition.upper()}:")
            print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
            print(f"  Mean Precision: {metrics['mean_precision']:.4f}")
            print(f"  Mean Recall: {metrics['mean_recall']:.4f}")
            print(f"  Mean Accuracy: {metrics['mean_accuracy']:.4f}")
            print()
    
    return results_summary

# Main Script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test segmentation model')
    parser.add_argument('--dataset', type=str, default='zod', choices=['zod', 'waymo'], help='Dataset to use (zod or waymo)')
    args = parser.parse_args()
    
    # Load config
    config_file = f'config_{args.dataset}.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Paths - Use proper test split for evaluation
    data_dir = config['data_dir']
    split_file = config['split_file']
    checkpoint_path = config['checkpoint_path']
    results_save_path = config['results_save_path']
    
    # Define test conditions for comprehensive evaluation
    test_conditions = [
        'test_day_fair',
        'test_day_rain', 
        'test_night_fair',
        'test_night_rain'
    ]
    
    # Model
    model = ViTSegmentation(config['mode'], config['num_classes'])

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        sys.exit(f"Checkpoint not found at {checkpoint_path}. Please provide a valid checkpoint path in config.")
    
    # Run comprehensive testing on all conditions
    test_all_conditions(model, config, checkpoint_path, results_prefix="test_results")
