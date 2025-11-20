# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import torch
import numpy as np
from tqdm import tqdm

from utils.helpers import get_model_path
from utils.metrics import find_overlap_1
from utils.metrics import auc_ap
from utils.metrics import zod_find_overlap_1
from utils.helpers import relabel_annotation
from clft.clft import CLFT


class Tester(object):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")
        print("device: %s" % self.device)
        
        # Determine if we are running RGB-only mode
        self.rgb_only = self.config.get('CLI', {}).get('mode', '') == 'rgb'
        if self.rgb_only:
            print("Using RGB-only mode (no separate LiDAR input)")

        if config['CLI']['backbone'] == 'clft':
            # Determine actual number of unique classes after relabeling
            unique_indices = set(cls['training_index'] for cls in config['Dataset']['classes'])
            num_unique_classes = len(unique_indices)
            
            resize = config['Dataset']['transforms']['resize']
            self.model = CLFT(RGB_tensor_size=(3, resize, resize),
                              XYZ_tensor_size=(3, resize, resize),
                              patch_size=config['CLFT']['patch_size'],
                              emb_dim=config['CLFT']['emb_dim'],
                              resample_dim=config['CLFT']['resample_dim'],
                              read=config['CLFT']['read'],
                              hooks=config['CLFT']['hooks'],
                              reassemble_s=config['CLFT']['reassembles'],
                              nclasses=num_unique_classes,
                              type=config['CLFT']['type'],
                              model_timm=config['CLFT']['model_timm'], )
            print(f"Using backbone {config['CLI']['backbone']} with {num_unique_classes} classes")

            model_path = get_model_path(config)
            if not model_path:
                sys.exit("No model checkpoint found! Please specify a model path in config['General']['model_path'] or ensure checkpoints exist in the log directory.")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'])
            self.model.to(self.device)  # Ensure model is on the correct device

        else:
            sys.exit("A backbone must be specified! (clft or clfcn)")

        self.nclasses = num_unique_classes
        
        # Extract eval classes information from consolidated classes array
        classes_config = config['Dataset']['classes']
        if isinstance(classes_config, list):
            # New consolidated structure
            # Create mapping from config index to consecutive training index
            unique_indices = sorted(set(cls['training_index'] for cls in classes_config))
            config_index_to_training_index = {}
            for training_index, config_index in enumerate(unique_indices):
                config_index_to_training_index[config_index] = training_index
            
            seen_indices = set()
            self.eval_classes = []
            self.eval_indices = []
            for cls in classes_config:
                if cls.get('evaluate', False) and cls['training_index'] not in seen_indices:
                    self.eval_classes.append(cls['name'])
                    # Use consecutive training index, not config index
                    config_index = cls['training_index']
                    training_index = config_index_to_training_index[config_index]
                    self.eval_indices.append(training_index)
                    seen_indices.add(cls['training_index'])
        else:
            # Legacy structure - extract from separate fields
            self.eval_classes = config['Dataset'].get('eval_classes', [])
            self.eval_indices = config['Dataset'].get('eval_indices', [])
        
        print(f"Using eval classes from config: {self.eval_classes}")
        print(f"Eval indices: {self.eval_indices}")
        
        # Choose appropriate IoU calculation function based on dataset
        self.dataset_name = config['Dataset']['name']
        if self.dataset_name == 'zod':
            self.find_overlap_func = zod_find_overlap_1
            print("Using ZOD-specific IoU calculation")
        elif self.dataset_name == 'waymo':
            self.find_overlap_func = find_overlap_1
            print("Using Waymo IoU calculation")
        else:
            self.find_overlap_func = find_overlap_1
            print("Using standard IoU calculation")
        
        self.model.eval()

    def test_clft(self, test_dataloader, modal, result_file):
        print('Testing...')
        overlap_cum, pred_cum, label_cum, union_cum = 0, 0, 0, 0
        modality = modal
        
        # Initialize per-class metrics tensors for eval classes
        num_eval_classes = len(self.eval_classes)
        class_pre = torch.zeros((len(test_dataloader), num_eval_classes), dtype=torch.float)
        class_rec = torch.zeros((len(test_dataloader), num_eval_classes), dtype=torch.float)
        
        with torch.no_grad():
            progress_bar = tqdm(test_dataloader)

            for i, batch in enumerate(progress_bar):
                batch['rgb'] = batch['rgb'].to(self.device, non_blocking=True)
                batch['lidar'] = batch['lidar'].to(self.device, non_blocking=True)
                batch['anno'] = batch['anno'].to(self.device, non_blocking=True)

                # Use RGB as second input if in RGB-only mode, otherwise use LiDAR
                lidar_input = batch['rgb'] if self.rgb_only else batch['lidar']
                _, output_seg = self.model(batch['rgb'], lidar_input, modality)

                # 1xHxW -> HxW
                output_seg = output_seg.squeeze(1)
                anno = batch['anno']

                # Relabel annotations to match model output classes
                anno = relabel_annotation(anno.cpu(), self.config).squeeze(0).to(self.device)

                batch_overlap, batch_pred, batch_label, batch_union = self.find_overlap_func(self.nclasses, output_seg, anno)
                overlap_cum += batch_overlap
                pred_cum += batch_pred
                label_cum += batch_label
                union_cum += batch_union

                batch_IoU = 1.0 * batch_overlap / (np.spacing(1) + batch_union)
                batch_precision = 1.0 * batch_overlap / (np.spacing(1) + batch_pred)
                batch_recall = 1.0 * batch_overlap / (np.spacing(1) + batch_label)

                # Store metrics for eval classes only
                # Convert eval_indices (training indices) to metrics array indices
                if self.dataset_name == 'zod':
                    array_indices = [idx - 1 for idx in self.eval_indices]
                elif self.dataset_name == 'waymo':
                    array_indices = [idx - 1 for idx in self.eval_indices]
                else:
                    array_indices = self.eval_indices
                
                for j, array_idx in enumerate(array_indices):
                    class_pre[i, j] = batch_precision[array_idx]
                    class_rec[i, j] = batch_recall[array_idx]

                # Create progress bar description with eval class names
                progress_desc = ' '.join([f'{cls.upper()}:IoU->{batch_IoU[array_indices[j]]:.4f}'
                                        for j, cls in enumerate(self.eval_classes)])
                progress_bar.set_description(progress_desc)

            print('Overall Performance Computing...')
            cum_IoU = overlap_cum / union_cum
            cum_precision = overlap_cum / pred_cum
            cum_recall = overlap_cum / label_cum

            # Filter to eval classes only
            # Convert eval_indices (training indices) to metrics array indices
            if self.dataset_name == 'zod':
                array_indices = [idx - 1 for idx in self.eval_indices]
            elif self.dataset_name == 'waymo':
                array_indices = [idx - 1 for idx in self.eval_indices]
            else:
                array_indices = self.eval_indices
            
            eval_IoU = cum_IoU[array_indices]
            eval_precision = cum_precision[array_indices]
            eval_recall = cum_recall[array_indices]

            # Calculate AP for each eval class
            average_precision = []
            for j in range(num_eval_classes):
                ap = auc_ap(class_pre[:, j], class_rec[:, j])
                average_precision.append(ap)

            print('-----------------------------------------')
            for j, cls in enumerate(self.eval_classes):
                print(f'{cls.upper()}:CUM_IoU->{eval_IoU[j]:.4f} '
                      f'CUM_Precision->{eval_precision[j]:.4f} '
                      f'CUM_Recall->{eval_recall[j]:.4f}')
            print('-----------------------------------------')
            print('Testing of the subset completed')
            results = self.get_test_results(eval_IoU, eval_precision, eval_recall)
            self.save_test_results(results, result_file)

    def get_test_results(self, cum_IoU, cum_precision, cum_recall):
        """Generate test results dictionary without saving to file"""
        import datetime
        import math
        
        # Convert tensors to lists if needed
        cum_IoU = cum_IoU.cpu().numpy().tolist() if isinstance(cum_IoU, torch.Tensor) else cum_IoU
        cum_precision = cum_precision.cpu().numpy().tolist() if isinstance(cum_precision, torch.Tensor) else cum_precision
        cum_recall = cum_recall.cpu().numpy().tolist() if isinstance(cum_recall, torch.Tensor) else cum_recall
        
        # Helper function to replace NaN with 0
        def sanitize_value(value):
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return 0.0
            return value
        
        # Sanitize all metric arrays
        cum_IoU = [sanitize_value(x) for x in cum_IoU]
        cum_precision = [sanitize_value(x) for x in cum_precision]
        cum_recall = [sanitize_value(x) for x in cum_recall]
        
        # Calculate F1 scores
        f1_scores = []
        for prec, rec in zip(cum_precision, cum_recall):
            if prec + rec > 0:
                f1 = 2 * (prec * rec) / (prec + rec)
            else:
                f1 = 0.0
            f1_scores.append(sanitize_value(f1))
        
        # Calculate overall metrics
        mean_iou = sum(cum_IoU) / len(cum_IoU) if cum_IoU else 0.0
        mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        
        # Sanitize overall metrics
        mean_iou = sanitize_value(mean_iou)
        mean_f1 = sanitize_value(mean_f1)
        
        # Create results dictionary using eval class names
        class_results = {}
        for j, cls in enumerate(self.eval_classes):
            class_results[cls] = {
                'iou': cum_IoU[j],
                'precision': cum_precision[j],
                'recall': cum_recall[j],
                'f1_score': f1_scores[j]
            }
        return class_results

    def save_test_results(self, results, result_file):
        """Save test results dictionary to JSON file"""
        # Save to JSON file
        json_path = result_file.replace('.txt', '.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {json_path}")
        print(f"Mean IoU: {results['overall_metrics']['mean_iou']:.4f}, Mean F1: {results['overall_metrics']['mean_f1']:.4f}")

    def test_clft_return_results(self, test_dataloader, modal):
        """Run testing and return results without saving to file"""
        print('Testing...')
        overlap_cum, pred_cum, label_cum, union_cum = 0, 0, 0, 0
        modality = modal
        
        # Initialize per-class metrics tensors for eval classes
        num_eval_classes = len(self.eval_classes)
        class_pre = torch.zeros((len(test_dataloader), num_eval_classes), dtype=torch.float)
        class_rec = torch.zeros((len(test_dataloader), num_eval_classes), dtype=torch.float)
        
        with torch.no_grad():
            progress_bar = tqdm(test_dataloader)

            for i, batch in enumerate(progress_bar):
                batch['rgb'] = batch['rgb'].to(self.device, non_blocking=True)
                batch['lidar'] = batch['lidar'].to(self.device, non_blocking=True)
                batch['anno'] = batch['anno'].to(self.device, non_blocking=True)

                # Use RGB as second input if in RGB-only mode, otherwise use LiDAR
                lidar_input = batch['rgb'] if self.rgb_only else batch['lidar']
                _, output_seg = self.model(batch['rgb'], lidar_input, modality)

                # 1xHxW -> HxW
                output_seg = output_seg.squeeze(1)
                anno = batch['anno']

                # Relabel annotations to match model output classes
                anno = relabel_annotation(anno.cpu(), self.config).squeeze(0).to(self.device)

                batch_overlap, batch_pred, batch_label, batch_union = self.find_overlap_func(self.nclasses, output_seg, anno)
                overlap_cum += batch_overlap
                pred_cum += batch_pred
                label_cum += batch_label
                union_cum += batch_union

                batch_IoU = 1.0 * batch_overlap / (np.spacing(1) + batch_union)
                batch_precision = 1.0 * batch_overlap / (np.spacing(1) + batch_pred)
                batch_recall = 1.0 * batch_overlap / (np.spacing(1) + batch_label)

                # Store metrics for eval classes only
                # Convert eval_indices (training indices) to metrics array indices
                if self.dataset_name == 'zod':
                    array_indices = [idx - 1 for idx in self.eval_indices]
                elif self.dataset_name == 'waymo':
                    array_indices = [idx - 1 for idx in self.eval_indices]
                else:
                    array_indices = self.eval_indices
                
                for j, array_idx in enumerate(array_indices):
                    class_pre[i, j] = batch_precision[array_idx]
                    class_rec[i, j] = batch_recall[array_idx]

                # Create progress bar description with eval class names
                progress_desc = ' '.join([f'{cls.upper()}:IoU->{batch_IoU[array_indices[j]]:.4f}'
                                        for j, cls in enumerate(self.eval_classes)])
                progress_bar.set_description(progress_desc)

            print('Overall Performance Computing...')
            cum_IoU = overlap_cum / union_cum
            cum_precision = overlap_cum / pred_cum
            cum_recall = overlap_cum / label_cum

            # Filter to eval classes only
            # Convert eval_indices (training indices) to metrics array indices
            if self.dataset_name == 'zod':
                array_indices = [idx - 1 for idx in self.eval_indices]
            elif self.dataset_name == 'waymo':
                array_indices = [idx - 1 for idx in self.eval_indices]
            else:
                array_indices = self.eval_indices
            
            eval_IoU = cum_IoU[array_indices]
            eval_precision = cum_precision[array_indices]
            eval_recall = cum_recall[array_indices]

            print('-----------------------------------------')
            for j, cls in enumerate(self.eval_classes):
                print(f'{cls.upper()}:CUM_IoU->{eval_IoU[j]:.4f} '
                      f'CUM_Precision->{eval_precision[j]:.4f} '
                      f'CUM_Recall->{eval_recall[j]:.4f}')
            print('-----------------------------------------')
            print('Testing of the subset completed')
            
            # Return results instead of saving
            return self.get_test_results(eval_IoU, eval_precision, eval_recall)
