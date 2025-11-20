#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics calculation utilities.
"""
import torch
import math


class MetricsCalculator:
    """Calculate and aggregate training/testing metrics."""
    
    def __init__(self, config, num_eval_classes, find_overlap_func):
        self.config = config
        self.num_eval_classes = num_eval_classes
        self.find_overlap_func = find_overlap_func
        self.eval_classes = self._extract_eval_classes()
        self.eval_indices = self._extract_eval_indices()
    
    def _extract_eval_classes(self):
        """Extract evaluation class names from config (excludes background at index 0)."""
        train_classes = self.config['Dataset']['train_classes']
        # Get all classes with index > 0, sorted by index
        eval_classes = [
            cls['name'] 
            for cls in sorted(train_classes, key=lambda x: x['index'])
            if cls['index'] > 0
        ]
        return eval_classes
    
    def _extract_eval_indices(self):
        """Extract evaluation class indices from config (excludes background at index 0)."""
        train_classes = self.config['Dataset']['train_classes']
        # Get all indices > 0, sorted
        eval_indices = [
            cls['index'] 
            for cls in sorted(train_classes, key=lambda x: x['index'])
            if cls['index'] > 0
        ]
        
        return eval_indices
    
    def create_accumulators(self, device):
        """Create metric accumulator tensors."""
        return {
            'overlap': torch.zeros(self.num_eval_classes).to(device),
            'pred': torch.zeros(self.num_eval_classes).to(device),
            'label': torch.zeros(self.num_eval_classes).to(device),
            'union': torch.zeros(self.num_eval_classes).to(device)        }
    
    def update_accumulators(self, accumulators, output_seg, anno, num_classes):
        """Update metric accumulators with batch results."""
        batch_overlap, batch_pred, batch_label, batch_union = \
            self.find_overlap_func(num_classes, output_seg, anno)
        
        accumulators['overlap'] += batch_overlap
        accumulators['pred'] += batch_pred
        accumulators['label'] += batch_label
        accumulators['union'] += batch_union
        
        return batch_overlap, batch_union
    
    def compute_epoch_metrics(self, accumulators, total_loss, num_batches):
        """Compute final metrics for an epoch."""
        # IoU and metrics are already in correct order (class 1, 2, 3, ... -> indices 0, 1, 2, ...)
        epoch_IoU = accumulators['overlap'] / accumulators['union']
        
        # Additional metrics
        precision = accumulators['overlap'] / (accumulators['pred'] + 1e-6)
        recall = accumulators['overlap'] / (accumulators['label'] + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        # Average loss
        epoch_loss = total_loss / num_batches
        
        return {
            'epoch_IoU': epoch_IoU,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'epoch_loss': epoch_loss,
            'mean_iou': torch.mean(epoch_IoU).item()
        }
    
    def sanitize_value(self, value):
        """Replace NaN/Inf with 0."""
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return 0.0
        return value
    
    def prepare_results_dict(self, train_metrics, val_metrics):
        """Prepare results dictionary for logging."""
        results = {"train": {}, "val": {}}
        
        for i, cls in enumerate(self.eval_classes):
            results["train"][cls] = {
                "iou": self.sanitize_value(train_metrics["epoch_IoU"][i].item()),
                "precision": self.sanitize_value(train_metrics["precision"][i].item()),
                "recall": self.sanitize_value(train_metrics["recall"][i].item()),
                "f1": self.sanitize_value(train_metrics["f1"][i].item())
            }
            results["val"][cls] = {
                "iou": self.sanitize_value(val_metrics["epoch_IoU"][i].item()),
                "precision": self.sanitize_value(val_metrics["precision"][i].item()),
                "recall": self.sanitize_value(val_metrics["recall"][i].item()),
                "f1": self.sanitize_value(val_metrics["f1"][i].item())
            }
        
        results["train"]["loss"] = self.sanitize_value(train_metrics["epoch_loss"])
        results["train"]["mean_iou"] = self.sanitize_value(train_metrics["mean_iou"])
        results["val"]["loss"] = self.sanitize_value(val_metrics["epoch_loss"])
        results["val"]["mean_iou"] = self.sanitize_value(val_metrics["mean_iou"])
        
        return results
    
    def print_metrics(self, metrics, prefix="", accumulators=None):
        """Print metrics in a formatted way."""
        print(f'{prefix}Mean IoU: {metrics["mean_iou"]:.4f}')
        for i, cls in enumerate(self.eval_classes):
            print(f'{prefix}{cls} IoU: {metrics["epoch_IoU"][i]:.4f}')
        print(f'{prefix}Average Loss: {metrics["epoch_loss"]:.4f}')
