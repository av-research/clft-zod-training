#!/usr/bin/env python3
"""
Evaluation script for CLFT training with PNG projections.

This script provides comprehensive evaluation metrics and comparison tools.
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

def evaluate_png_training(config_path, checkpoint_path=None, compare_with_pickle=False):
    """
    Comprehensive evaluation of PNG-based CLFT training.
    """

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Evaluating config: {config_path}")
    print(f"Dataset: {config['Dataset']['name']}")
    print(f"Mode: {config['CLI']['mode']}")

    # Import required modules - choose dataset loader based on mode
    modal = config['CLI']['mode']
    if modal == 'rgb':
        # RGB model was trained with pickle-based dataset
        from tools.dataset import Dataset
    else:
        # LiDAR/fusion models were trained with PNG-based dataset
        from tools.dataset_png import DatasetPNG as Dataset
    from clft.clft import CLFT
    from utils.metrics import zod_find_overlap_1

    # Setup device
    device = torch.device(config['General']['device'] if torch.cuda.is_available() else "cpu")

    # Load model
    resize = config['Dataset']['transforms']['resize']
    nclasses = len(config['Dataset']['classes'])

    model = CLFT(
        RGB_tensor_size=(3, resize, resize),
        XYZ_tensor_size=(3, resize, resize),
        patch_size=config['CLFT']['patch_size'],
        emb_dim=config['CLFT']['emb_dim'],
        resample_dim=config['CLFT']['resample_dim'],
        read=config['CLFT']['read'],
        hooks=config['CLFT']['hooks'],
        reassemble_s=config['CLFT']['reassembles'],
        nclasses=nclasses,
        type=config['CLFT']['type'],
        model_timm=config['CLFT']['model_timm']
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("⚠️  No checkpoint provided - evaluating randomly initialized model")

    model.to(device)
    model.eval()

    # Load validation dataset
    val_dataset = Dataset(config, 'val', config['Dataset']['val_split'])
    val_dataloader = DataLoader(val_dataset,
                               batch_size=config['General']['batch_size'],
                               shuffle=False,
                               pin_memory=True,
                               drop_last=False)

    print(f"Validation dataset size: {len(val_dataset)}")

    # Evaluation metrics
    metrics = {
        'loss': [],
        'iou_2d': {'overlap': 0, 'union': 0, 'pred': 0, 'label': 0},
        'precision': [],
        'recall': [],
        'f1': [],
        'per_class_iou': np.zeros(nclasses),
        'per_class_count': np.zeros(nclasses)
    }

    criterion = torch.nn.CrossEntropyLoss()

    print("Starting evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_dataloader)):
            batch['rgb'] = batch['rgb'].to(device, non_blocking=True)
            batch['lidar'] = batch['lidar'].to(device, non_blocking=True)
            batch['anno'] = batch['anno'].to(device, non_blocking=True)

            # Forward pass
            modal = config['CLI']['mode']
            lidar_input = batch['rgb'] if modal == 'rgb' else batch['lidar']
            _, output_seg = model(batch['rgb'], lidar_input, modal)

            output_seg = output_seg.squeeze(1)
            anno = batch['anno']

            # Compute loss
            loss = criterion(output_seg, anno)
            metrics['loss'].append(loss.item())

            # Compute 2D IoU (pixel-based)
            for cls in range(nclasses):
                pred_cls = (output_seg == cls)
                true_cls = (anno == cls)

                overlap = (pred_cls & true_cls).sum().item()
                union = (pred_cls | true_cls).sum().item()
                pred_count = pred_cls.sum().item()
                label_count = true_cls.sum().item()

                if union > 0:
                    metrics['per_class_iou'][cls] += overlap / union
                    metrics['per_class_count'][cls] += 1

                metrics['iou_2d']['overlap'] += overlap
                metrics['iou_2d']['union'] += union
                metrics['iou_2d']['pred'] += pred_count
                metrics['iou_2d']['label'] += label_count

    # Compute final metrics
    results = {}

    # Overall metrics
    results['mean_loss'] = np.mean(metrics['loss'])
    results['overall_iou_2d'] = metrics['iou_2d']['overlap'] / (metrics['iou_2d']['union'] + 1e-6)
    results['mean_precision'] = metrics['iou_2d']['overlap'] / (metrics['iou_2d']['pred'] + 1e-6)
    results['mean_recall'] = metrics['iou_2d']['overlap'] / (metrics['iou_2d']['label'] + 1e-6)
    results['mean_f1'] = 2 * results['mean_precision'] * results['mean_recall'] / (results['mean_precision'] + results['mean_recall'] + 1e-6)

    # Per-class metrics
    results['per_class_iou'] = {}
    class_names = [config['Dataset']['classes'][str(i)]['name'] for i in range(nclasses)]
    for i, name in enumerate(class_names):
        if metrics['per_class_count'][i] > 0:
            results['per_class_iou'][name] = metrics['per_class_iou'][i] / metrics['per_class_count'][i]
        else:
            results['per_class_iou'][name] = 0.0

    # Performance analysis
    results['performance'] = {
        'total_samples': len(val_dataset),
        'batch_size': config['General']['batch_size'],
        'device': str(device),
        'modal': modal
    }

    return results

def compare_experiments(results_png, results_pickle=None):
    """Compare PNG vs pickle-based training results."""

    print("\n" + "="*60)
    print("EVALUATION RESULTS - PNG TRAINING")
    print("="*60)

    print(f"Mean Loss: {results_png['mean_loss']:.4f}")
    print(f"Overall 2D IoU: {results_png['overall_iou_2d']:.4f}")
    print(f"Mean Precision: {results_png['mean_precision']:.4f}")
    print(f"Mean Recall: {results_png['mean_recall']:.4f}")
    print(f"Mean F1: {results_png['mean_f1']:.4f}")

    print("\nPer-Class IoU:")
    for cls, iou in results_png['per_class_iou'].items():
        print(f"  {cls:10s}: {iou:.4f}")

    print(f"\nPerformance: {results_png['performance']['total_samples']} samples, {results_png['performance']['modal']} mode")

    if results_pickle:
        print("\n" + "="*60)
        print("COMPARISON WITH PICKLE-BASED TRAINING")
        print("="*60)

        diff_iou = results_png['overall_iou_2d'] - results_pickle.get('overall_iou_2d', 0)
        diff_loss = results_png['mean_loss'] - results_pickle.get('mean_loss', 0)

        print(f"Pickle Loss: {results_pickle.get('mean_loss', 0):.4f}")
        print(f"IoU Difference: {diff_iou:+.4f} ({'better' if diff_iou > 0 else 'worse'})")
        print(f"Loss Difference: {diff_loss:+.4f} ({'better' if diff_loss < 0 else 'worse'})")

def main():
    parser = argparse.ArgumentParser(description='Evaluate CLFT training with PNG projections')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--checkpoint', help='Model checkpoint path')
    parser.add_argument('--output', default='evaluation_results.json', help='Output JSON file')
    parser.add_argument('--compare_pickle', help='Path to pickle-based results JSON for comparison')

    args = parser.parse_args()

    # Run evaluation
    results = evaluate_png_training(args.config, args.checkpoint)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")

    # Compare with pickle results if provided
    if args.compare_pickle and os.path.exists(args.compare_pickle):
        with open(args.compare_pickle, 'r') as f:
            pickle_results = json.load(f)
        compare_experiments(results, pickle_results)
    else:
        compare_experiments(results)

if __name__ == '__main__':
    main()