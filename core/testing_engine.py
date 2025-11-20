#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing engine for model evaluation.
"""
import time
import torch
import numpy as np
from tqdm import tqdm
from utils.helpers import relabel_annotation
from utils.metrics import auc_ap


class TestingEngine:
    """Handles model testing and evaluation."""
    
    def __init__(self, model, metrics_calculator, config, device):
        self.model = model
        self.metrics_calc = metrics_calculator
        self.config = config
        self.device = device
        self.dataset_name = config['Dataset']['name']
    
    def test(self, dataloader, modality, num_classes):
        """Run testing on a dataloader and return results."""
        self.model.eval()
        
        # Initialize accumulators
        accumulators = self.metrics_calc.create_accumulators(self.device)
        
        # Initialize per-class metrics for AP calculation
        num_eval_classes = len(self.metrics_calc.eval_classes)
        class_pre = torch.zeros((len(dataloader), num_eval_classes), dtype=torch.float)
        class_rec = torch.zeros((len(dataloader), num_eval_classes), dtype=torch.float)
        
        # Track inference time
        total_inference_time = 0.0
        total_samples = 0
        total_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader)
            for i, batch in enumerate(progress_bar):
                # Move data to device
                rgb = batch['rgb'].to(self.device, non_blocking=True)
                lidar = batch['lidar'].to(self.device, non_blocking=True)
                anno = batch['anno'].to(self.device, non_blocking=True)
                
                # Prepare inputs
                rgb_input, lidar_input = self._prepare_inputs(rgb, lidar, modality)
                
                # Synchronize for accurate timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Time the forward pass
                inference_start = time.time()
                
                # Forward pass
                _, output_seg = self.model(rgb_input, lidar_input, modality)
                output_seg = output_seg.squeeze(1)
                
                # Synchronize and record time
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_time = time.time() - inference_start
                total_inference_time += inference_time
                total_samples += rgb.size(0)
                total_batches += 1
                
                # Relabel annotation
                anno = relabel_annotation(anno.cpu(), self.config).squeeze(0).to(self.device)
                
                # Update accumulators
                batch_overlap, batch_union = self.metrics_calc.update_accumulators(
                    accumulators, output_seg, anno, num_classes
                )
                
                # Calculate batch metrics for AP
                batch_IoU = 1.0 * batch_overlap / (np.spacing(1) + batch_union)
                batch_precision = 1.0 * batch_overlap / (np.spacing(1) + accumulators['pred'])
                batch_recall = 1.0 * batch_overlap / (np.spacing(1) + accumulators['label'])
                
                # Store metrics for eval classes
                array_indices = self._get_array_indices()
                for j, array_idx in enumerate(array_indices):
                    class_pre[i, j] = batch_precision[array_idx]
                    class_rec[i, j] = batch_recall[array_idx]
                
                # Update progress bar
                progress_desc = ' '.join([
                    f'{cls.upper()}:IoU->{batch_IoU[array_indices[j]]:.4f}'
                    for j, cls in enumerate(self.metrics_calc.eval_classes)
                ])
                progress_bar.set_description(progress_desc)
        
        # Compute final metrics
        results = self._compute_final_results(accumulators, class_pre, class_rec)
        
        # Print results
        self._print_results(results)
        
        # Return results and inference stats
        inference_stats = {
            'total_inference_time': total_inference_time,
            'total_samples': total_samples,
            'total_batches': total_batches
        }
        
        return results, inference_stats
    
    def _prepare_inputs(self, rgb, lidar, modality):
        """Prepare model inputs based on modality."""
        if modality == 'rgb':
            return rgb, rgb
        elif modality == 'lidar':
            return lidar, lidar
        else:  # cross_fusion
            return rgb, lidar
    
    def _get_array_indices(self):
        """Get array indices for eval classes."""
        if self.dataset_name in ['zod', 'waymo']:
            return [idx - 1 for idx in self.metrics_calc.eval_indices]
        else:
            return self.metrics_calc.eval_indices
    
    def _compute_final_results(self, accumulators, class_pre, class_rec):
        """Compute final test results."""
        cum_IoU = accumulators['overlap'] / accumulators['union']
        cum_precision = accumulators['overlap'] / accumulators['pred']
        cum_recall = accumulators['overlap'] / accumulators['label']
        
        # Filter to eval classes
        array_indices = self._get_array_indices()
        eval_IoU = cum_IoU[array_indices]
        eval_precision = cum_precision[array_indices]
        eval_recall = cum_recall[array_indices]
        
        # Calculate F1 and AP
        results = {}
        for i, cls in enumerate(self.metrics_calc.eval_classes):
            iou = self.metrics_calc.sanitize_value(eval_IoU[i].item())
            precision = self.metrics_calc.sanitize_value(eval_precision[i].item())
            recall = self.metrics_calc.sanitize_value(eval_recall[i].item())
            
            # Calculate F1
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            f1 = self.metrics_calc.sanitize_value(f1)
            
            # Calculate AP
            ap = auc_ap(class_pre[:, i], class_rec[:, i])
            ap = self.metrics_calc.sanitize_value(ap)
            
            results[cls] = {
                'iou': iou,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'ap': ap
            }
        
        return results
    
    def _print_results(self, results):
        """Print test results."""
        print('-----------------------------------------')
        for cls, metrics in results.items():
            print(f'{cls.upper()}: IoU->{metrics["iou"]:.4f} '
                  f'Precision->{metrics["precision"]:.4f} '
                  f'Recall->{metrics["recall"]:.4f} '
                  f'F1->{metrics["f1_score"]:.4f} '
                  f'AP->{metrics["ap"]:.4f}')
        print('-----------------------------------------')
