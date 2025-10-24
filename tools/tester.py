# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from clfcn.fusion_net import FusionNet
from utils.helpers import get_model_path
from utils.metrics import find_overlap
from utils.metrics import find_overlap_1
from utils.metrics import auc_ap
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

        if config['CLI']['backbone'] == 'clfcn':
            self.model = FusionNet()
            print(f"Using backbone {config['CLI']['backbone']}")
            self.optimizer_fcn = torch.optim.Adam(self.model.parameters(), lr=config['CLFCN']['clfcn_lr'])
            self.scheduler_fcn = ReduceLROnPlateau(self.optimizer_fcn)

        elif config['CLI']['backbone'] == 'clft':
            resize = config['Dataset']['transforms']['resize']
            self.model = CLFT(RGB_tensor_size=(3, resize, resize),
                              XYZ_tensor_size=(3, resize, resize),
                              patch_size=config['CLFT']['patch_size'],
                              emb_dim=config['CLFT']['emb_dim'],
                              resample_dim=config['CLFT']['resample_dim'],
                              read=config['CLFT']['read'],
                              hooks=config['CLFT']['hooks'],
                              reassemble_s=config['CLFT']['reassembles'],
                              nclasses=len(config['Dataset']['classes']),
                              type=config['CLFT']['type'],
                              model_timm=config['CLFT']['model_timm'], )
            print(f"Using backbone {config['CLI']['backbone']}")

            model_path = get_model_path(config)
            if not model_path:
                sys.exit("No model checkpoint found! Please specify a model path in config['General']['model_path'] or ensure checkpoints exist in the log directory.")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'])

        else:
            sys.exit("A backbone must be specified! (clft or clfcn)")

        self.model.to(self.device)
        self.nclasses = len(config['Dataset']['classes'])
        self.model.eval()

    def test_clft(self, test_dataloader, modal, result_file):
        print('Testing...')
        overlap_cum, pred_cum, label_cum, union_cum = 0, 0, 0, 0
        modality = modal
        with torch.no_grad():
            progress_bar = tqdm(test_dataloader)

            cyclist_pre = torch.zeros((len(progress_bar)), dtype=torch.float)
            cyclist_rec = torch.zeros((len(progress_bar)), dtype=torch.float)
            pedestrian_pre = torch.zeros((len(progress_bar)), dtype=torch.float)
            pedestrian_rec = torch.zeros((len(progress_bar)), dtype=torch.float)
            sign_pre = torch.zeros((len(progress_bar)), dtype=torch.float)
            sign_rec = torch.zeros((len(progress_bar)), dtype=torch.float)

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

                batch_overlap, batch_pred, batch_label, batch_union = find_overlap_1(self.nclasses, output_seg, anno)
                overlap_cum += batch_overlap
                pred_cum += batch_pred
                label_cum += batch_label
                union_cum += batch_union

                batch_IoU = 1.0 * batch_overlap / (np.spacing(1) + batch_union)
                batch_precision = 1.0 * batch_overlap / (np.spacing(1) + batch_pred)
                batch_recall = 1.0 * batch_overlap / (np.spacing(1) + batch_label)

                cyclist_pre[i] = batch_precision[0]
                cyclist_rec[i] = batch_recall[0]
                pedestrian_pre[i] = batch_precision[1]
                pedestrian_rec[i] = batch_recall[1]
                sign_pre[i] = batch_precision[2]
                sign_rec[i] = batch_recall[2]

                progress_bar.set_description(f'CYCLIST:IoU->{batch_IoU[0]:.4f} '
                                             f'PEDESTRIAN:IoU->{batch_IoU[1]:.4f} '
                                             f'SIGN:IoU->{batch_IoU[2]:.4f} ')

            print('Overall Performance Computing...')
            cum_IoU = overlap_cum / union_cum
            cum_precision = overlap_cum / pred_cum
            cum_recall = overlap_cum / label_cum

            cyclist_AP = auc_ap(cyclist_pre, cyclist_rec)
            pedestrian_AP = auc_ap(pedestrian_pre, pedestrian_rec)
            sign_AP = auc_ap(sign_pre, sign_rec)
            average_precision = [cyclist_AP, pedestrian_AP, sign_AP]

            print('-----------------------------------------')
            print(f'CYCLIST:CUM_IoU->{cum_IoU[0]:.4f} '
                  f'CUM_Precision->{cum_precision[0]:.4f} '
                  f'CUM_Recall->{cum_recall[0]:.4f} '
                  f'Average Precision->{cyclist_AP:.4f} \n')
            print(f'PEDESTRIAN:CUM_IoU->{cum_IoU[1]:.4f} '
                  f'CUM_Precision->{cum_precision[1]:.4f} '
                  f'CUM_Recall->{cum_recall[1]:.4f} '
                  f'Average Precision->{pedestrian_AP:.4f} ')
            print(f'SIGN:CUM_IoU->{cum_IoU[2]:.4f} '
                  f'CUM_Precision->{cum_precision[2]:.4f} '
                  f'CUM_Recall->{cum_recall[2]:.4f} '
                  f'Average Precision->{sign_AP:.4f} ')
            print('-----------------------------------------')
            print('Testing of the subset completed')
            self.save_test_results(cum_IoU, cum_precision, cum_recall, average_precision, result_file)

    def save_test_results(self, cum_IoU, cum_precision, cum_recall, average_precision, result_file):
        """Save test results in JSON format with F1 score"""
        # Convert tensors to lists if needed
        cum_IoU = cum_IoU.cpu().numpy().tolist() if isinstance(cum_IoU, torch.Tensor) else cum_IoU
        cum_precision = cum_precision.cpu().numpy().tolist() if isinstance(cum_precision, torch.Tensor) else cum_precision
        cum_recall = cum_recall.cpu().numpy().tolist() if isinstance(cum_recall, torch.Tensor) else cum_recall
        average_precision = [ap.item() if isinstance(ap, torch.Tensor) else ap for ap in average_precision]
        
        # Calculate F1 scores
        f1_scores = []
        for prec, rec in zip(cum_precision, cum_recall):
            if prec + rec > 0:
                f1 = 2 * (prec * rec) / (prec + rec)
            else:
                f1 = 0.0
            f1_scores.append(f1)
        
        # Create results dictionary
        results = {
            'cyclist': {
                'iou': cum_IoU[0],
                'precision': cum_precision[0],
                'recall': cum_recall[0],
                'f1_score': f1_scores[0],
                'average_precision': average_precision[0]
            },
            'pedestrian': {
                'iou': cum_IoU[1],
                'precision': cum_precision[1],
                'recall': cum_recall[1],
                'f1_score': f1_scores[1],
                'average_precision': average_precision[1]
            },
            'sign': {
                'iou': cum_IoU[2],
                'precision': cum_precision[2],
                'recall': cum_recall[2],
                'f1_score': f1_scores[2],
                'average_precision': average_precision[2]
            }
        }
        
        # Save to JSON file
        json_path = result_file.replace('.txt', '.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
