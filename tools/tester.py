#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import torch
import numpy as np
from tqdm import tqdm
from utils.metrics import find_overlap_zod

from clft.clft import CLFT


class Tester(object):
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.args = args

        self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")
        print("device: %s" % self.device)

        if args.backbone == 'clft':
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
            print(f'Using backbone {args.backbone}')

            model_path = config['General']['model_path']
            self.model.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'])

        else:
            sys.exit("A backbone must be specified! (clft or clfcn)")

        self.model.to(self.device)
        self.nclasses = len(config['Dataset']['classes'])
        self.model.eval()

    def test_clft(self, test_dataloader, modal):
        print('CLFT Model Testing...')
        overlap_cum = torch.zeros(self.nclasses, dtype=torch.float)
        pred_cum = torch.zeros(self.nclasses, dtype=torch.float)
        label_cum = torch.zeros(self.nclasses, dtype=torch.float)
        union_cum = torch.zeros(self.nclasses, dtype=torch.float)
        modality = modal
        with torch.no_grad():
            progress_bar = tqdm(test_dataloader)
            for i, batch in enumerate(progress_bar):
                batch['rgb'] = batch['rgb'].to(self.device, non_blocking=True)
                batch['lidar'] = batch['lidar'].to(self.device, non_blocking=True)
                batch['anno'] = batch['anno'].to(self.device, non_blocking=True)

                _, output_seg = self.model(batch['rgb'], batch['lidar'], modality)
                # 1xHxW -> HxW
                output_seg = output_seg.squeeze(1)
                anno = batch['anno']

                batch_overlap, batch_pred, batch_label, batch_union = find_overlap_zod(self.nclasses, output_seg, anno)
                overlap_cum += batch_overlap
                pred_cum += batch_pred
                label_cum += batch_label
                union_cum += batch_union

                batch_IoU = 1.0 * batch_overlap / (np.spacing(1) + batch_union)
                batch_precision = 1.0 * batch_overlap / (np.spacing(1) + batch_pred)
                batch_recall = 1.0 * batch_overlap / (np.spacing(1) + batch_label)

                # Print per-class metrics for this batch
                desc = []
                for cls_idx, cls_name in enumerate(self.config['Dataset']['classes']):
                    desc.append(f'{cls_name}:IoU->{batch_IoU[cls_idx]:.4f} P->{batch_precision[cls_idx]:.4f} R->{batch_recall[cls_idx]:.4f}')
                progress_bar.set_description(' | '.join(desc))

            print('Overall Performance Computing...')
            cum_IoU = overlap_cum / (union_cum + np.spacing(1))
            cum_precision = overlap_cum / (pred_cum + np.spacing(1))
            cum_recall = overlap_cum / (label_cum + np.spacing(1))
            print('-----------------------------------------')
            for cls_idx, cls_name in enumerate(self.config['Dataset']['classes']):
                print(f'{cls_name}: CUM_IoU->{cum_IoU[cls_idx]:.4f} CUM_Precision->{cum_precision[cls_idx]:.4f} CUM_Recall->{cum_recall[cls_idx]:.4f}')
            print('-----------------------------------------')
            print('Testing of the subset completed')
