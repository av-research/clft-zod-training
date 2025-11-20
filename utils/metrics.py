#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Model evaluation metrics Python scripts

Created on June 18th, 2021
'''
import torch
import numpy as np

def find_overlap_exclude_bg_ignore(n_classes, output, anno):
    '''
    Universal IoU calculation that excludes background class.
    Works for both ZOD and Waymo datasets.
    
    Assumes:
    - Class 0: background (excluded from evaluation)
    - Classes 1+: evaluation classes
    
    :param n_classes: Total number of unique classes after relabeling
    :param output: Model output batch (B, C, H, W)
    :param anno: Annotation batch (B, H, W) - already relabeled
    :return: histogram statistic of overlap, prediction, annotation, union
    '''
    _, pred_indices = torch.max(output, dim=1)

    # Exclude background (0) from evaluation
    # Set predictions to 0 where annotation is 0
    pred_indices[anno == 0] = 0

    # Calculate overlap where prediction matches annotation
    overlap = pred_indices * (pred_indices == anno).long()

    # Calculate metrics for classes 1 to n_classes-1 (excluding background)
    num_eval_classes = n_classes - 1
    # Use bins from 0.5 to n_classes-0.5 to capture integer values 1 to n_classes-1
    area_overlap = torch.histc(overlap.float(), bins=num_eval_classes, min=0.5, max=n_classes - 0.5)
    area_pred = torch.histc(pred_indices.float(), bins=num_eval_classes, min=0.5, max=n_classes - 0.5)
    area_label = torch.histc(anno.float(), bins=num_eval_classes, min=0.5, max=n_classes - 0.5)
    area_union = area_pred + area_label - area_overlap

    # Add small epsilon to avoid division by zero
    area_union = torch.clamp(area_union, min=1e-6)

    assert (area_overlap <= area_label + 1e-6).all(), "Intersection area should be smaller than Union area"

    return area_overlap, area_pred, area_label, area_union

def auc_ap(precision, recall):
    '''
    Calculating AUC AP as defined in PASCAL VOC 2010
    :param precision: The tensor of precision of each batch for one class.
    :param recall: The tensor of recall of each batch for one class.
    :return auc_ap: Area-Under-Curve Average Precision
    '''
    # Reorganize precision-recall as the ascending order of recall values
    precision_list = precision.cpu().numpy()
    recall_list = recall.cpu().numpy()
    reordering_indices = np.argsort(recall_list)
    recall_ordered = recall_list[reordering_indices]
    precision_ordered = precision_list[reordering_indices]

    # Concatenate the point (0,0) and (1,0) to PR-curve, in case the minimum
    # (recall, precision) is far from (0,0). For example, if minimum point is
    # (0.9, 0.9), auc_ap should include area (0-0.9, 0-0.9).
    recall_concat = np.concatenate([[0.0], recall_ordered, [1.0]])
    prec_concat = np.concatenate([[0.0], precision_ordered, [0.0]])

    # Refer to VOC 2010 devkit_doc 3.4.1 "setting the precision for recall r
    # to the maximum precision obtained for any recall r′ ≥ r.
    
    for i in range(len(prec_concat)-1, 0, -1):
        prec_concat[i-1] = np.maximum(prec_concat[i-1], prec_concat[i])

    diff_idx = np.where(recall_concat[1:] != recall_concat[:-1])[0]

    auc_ap = np.sum((recall_concat[diff_idx + 1] - recall_concat[diff_idx]) *
                    prec_concat[diff_idx + 1])

    return auc_ap
