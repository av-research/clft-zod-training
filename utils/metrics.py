#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Model evaluation metrics Python scripts

Created on June 18th, 2021
'''
import torch
import numpy as np

'''
                         annotation
           |-----------|------------|------------
           |           | vehicle(1) | human(2)
prediction |vehicle(1) |     a      |     b
           |human(2)   |     c      |     d

IoU(1) = a / (a + b + c)
IoU(2) = d / (b + c + d)

precision(1) = a / (a + b)
precision(2) = d / (c + d)

recall(1) = a / (a + c)
recall(2) = d / (b + d)
'''


def find_overlap(n_classes, output, anno):
    '''
    This is the function for the IEEE-TIV jounrnal paper, which only consider the Vehicle and Human(pedestrian+cyclist)
    :param n_classes: Number of classes
    :param output: 'fusion' output batch (8, 4, 160, 480)
    :param anno: annotation batch (8, 160, 480)
    :return: histogram statistic of overlap, prediction and annotation, union
    '''
    # 0 -> background, 1-> vehicle,  2-> human (ped+cyclist), 3 -> ignore,
    # 4 -> cyclist.
    n_classes = n_classes - 1
    # Return each pixel value as either 0 or 1 or 2 or 3, which
    # represent different classes.
    _, pred_indices = torch.max(output, dim=1)  # (8, 160, 480)

    pred_indices[anno == 3] = 0

    # If pixel value in anno is 4(ignore), then it is False;
    # if pixel value in anno is 2 or 3, then it is True.
#    labeled = (anno > 1) * (anno <= n_classes)  # (8, 160, 480)
    # For 'label.long()', True will be 1, False will be 0.
    # In prediction, if value is 2 or 3, then no change;
    #                if value is 4, then change to 0
#    pred_indices = pred_indices * labeled.long()
    # If pixel value in prediction is same as anno, then keep it same;
    # if pixel value in prediction is not same as anno, then change to 0
    overlap = pred_indices * (pred_indices == anno).long()  # (8, 160, 480)

    # (a, d)
    area_overlap = torch.histc(overlap.float(),
                               bins=n_classes-1, max=2, min=1)
    # ((a + b), (c + d)
    area_pred = torch.histc(pred_indices.float(),
                            bins=n_classes-1, max=2, min=1)
    # ((a + c), (b + d)
    area_label = torch.histc(anno.float(),
                             bins=n_classes-1, max=2, min=1)
    # ((a + b + c), (b + c + d)
    area_union = area_pred + area_label - area_overlap

#     area_overlap = overlap.float().sum((1, 2))
#     area_pred = pred_indices.float().sum((1,2))
#     area_label = anno.float().sum((1,2))
#     area_union = (anno | pred_indices).float().sum((1,2))
    assert (area_overlap <= area_label).all(),\
        "Intersection area should be smaller than Union area"

    return area_overlap, area_pred, area_label, area_union


def zod_find_overlap_1(n_classes, output, anno):
    '''
    ZOD-specific IoU calculation optimized for sparse annotations.
    Key differences from standard version:
    - More lenient evaluation for sparse datasets
    - Different handling of ignore regions
    - Focus on rare classes (cyclists, pedestrians)
    
    Evaluates vehicle, sign, cyclist, pedestrian (classes 2-5).
    :param n_classes: Number of classes (6)
    :param output: 'fusion' output batch
    :param anno: annotation batch
    :return: histogram statistic of overlap, prediction and annotation, union
    '''
    # ZOD mapping: 0->background, 1->ignore, 2->vehicle, 3->sign, 4->cyclist, 5->pedestrian
    
    _, pred_indices = torch.max(output, dim=1)

    # ZOD-specific: For sparse datasets, be more careful with ignore handling
    # If annotation is ignore (1), force prediction to background (0) for IoU
    pred_indices[anno == 1] = 0

    # ZOD-specific: For very sparse annotations, we might want to consider
    # predictions that are "close" to correct (e.g., predicting vehicle instead of cyclist)
    # But for now, keep strict evaluation
    overlap = pred_indices * (pred_indices == anno).long()

    # Calculate metrics for classes 2 to n_classes-1 (excluding background 0 and ignore 1)
    # For n_classes=6: evaluates classes 2-5 (vehicle, sign, cyclist, pedestrian)
    num_eval_classes = n_classes - 2
    area_overlap = torch.histc(overlap.float(), bins=num_eval_classes, max=num_eval_classes + 2, min=2)
    area_pred = torch.histc(pred_indices.float(), bins=num_eval_classes, max=num_eval_classes + 2, min=2)
    area_label = torch.histc(anno.float(), bins=num_eval_classes, max=num_eval_classes + 2, min=2)
    area_union = area_pred + area_label - area_overlap

    # ZOD-specific: Add small epsilon to avoid division by zero in sparse cases
    area_union = torch.clamp(area_union, min=1e-6)

    assert (area_overlap <= area_label + 1e-6).all(), "Intersection area should be smaller than Union area"

    return area_overlap, area_pred, area_label, area_union


def zod_point_overlap(n_classes, output, anno, camera_coords):
    '''
    ZOD-specific point-based IoU calculation for LIDAR validation.
    Computes IoU only on the LIDAR points, not the entire 2D grid.
    This is more appropriate for sparse LIDAR data.
    
    :param n_classes: Number of classes (6)
    :param output: 'fusion' output batch (H, W)
    :param anno: annotation batch (H, W)
    :param camera_coords: batched coordinates (batch_size, max_len, 2)
    :return: histogram statistic of overlap, prediction and annotation, union
    '''
    _, pred_indices = torch.max(output, dim=1)  # (batch_size, H, W)
    
    all_pred_classes = []
    all_gt_classes = []
    
    batch_size = camera_coords.shape[0]
    for b in range(batch_size):
        # Filter valid coordinates (not padded)
        valid_coords = camera_coords[b][camera_coords[b][:, 0] != -1]
        
        if valid_coords.numel() == 0:
            continue
        
        # Vectorized indexing
        rows = valid_coords[:, 1].long()
        cols = valid_coords[:, 0].long()
        
        # Clamp to bounds
        rows = torch.clamp(rows, 0, anno.shape[1] - 1)
        cols = torch.clamp(cols, 0, anno.shape[2] - 1)
        
        pred = pred_indices[b, rows, cols]
        gt = anno[b, rows, cols]
        
        all_pred_classes.extend(pred.tolist())
        all_gt_classes.extend(gt.tolist())
    
    if not all_pred_classes:
        # No points, return zeros
        num_eval_classes = n_classes - 2
        return torch.zeros(num_eval_classes), torch.zeros(num_eval_classes), torch.zeros(num_eval_classes), torch.zeros(num_eval_classes)
    
    pred_tensor = torch.tensor(all_pred_classes, dtype=torch.float)
    gt_tensor = torch.tensor(all_gt_classes, dtype=torch.float)
    
    # ZOD-specific: If annotation is ignore (1), force prediction to background (0) for IoU
    pred_tensor[gt_tensor == 1] = 0
    
    overlap = pred_tensor * (pred_tensor == gt_tensor).long()
    
    # Calculate metrics for classes 2 to n_classes-1
    num_eval_classes = n_classes - 2
    area_overlap = torch.histc(overlap.float(), bins=num_eval_classes, max=num_eval_classes + 2, min=2)
    area_pred = torch.histc(pred_tensor.float(), bins=num_eval_classes, max=num_eval_classes + 2, min=2)
    area_label = torch.histc(gt_tensor.float(), bins=num_eval_classes, max=num_eval_classes + 2, min=2)
    area_union = area_pred + area_label - area_overlap
    
    # Add small epsilon to avoid division by zero
    area_union = torch.clamp(area_union, min=1e-6)
    
    return area_overlap, area_pred, area_label, area_union


def find_overlap_1(n_classes, output, anno):
    '''
    This is the function for the conference summary paper, ignore the car, consider pedestrian, cyclist and sign.
    :param n_classes: Number of classes
    :param output: 'fusion' output batch (8, 4, 160, 480)
    :param anno: annotation batch (8, 160, 480)
    :return: histogram statistic of overlap, prediction and annotation, union
    '''
    # 0->background, 1-> cyclist 2->pedestrain, 3->sign, 4->ignore+vehicle.
    n_classes = n_classes - 1
    # Return each pixel value as either 0 or 1 or 2 or 3 or 4, which
    # represent different classes.
    _, pred_indices = torch.max(output, dim=1)  # (8, 160, 480)

    pred_indices[anno == 4] = 0

    # If pixel value in anno is 4(ignore), then it is False;
    # if pixel value in anno is 2 or 3, then it is True.
#    labeled = (anno > 1) * (anno <= n_classes)  # (8, 160, 480)
    # For 'label.long()', True will be 1, False will be 0.
    # In prediction, if value is 2 or 3, then no change;
    #                if value is 4, then change to 0
#    pred_indices = pred_indices * labeled.long()
    # If pixel value in prediction is same as anno, then keep it same;
    # if pixel value in prediction is not same as anno, then change to 0
    overlap = pred_indices * (pred_indices == anno).long()  # (8, 160, 480)

    # (a, d)
    area_overlap = torch.histc(overlap.float(),
                               bins=n_classes-1, max=3, min=1)
    # ((a + b), (c + d)
    area_pred = torch.histc(pred_indices.float(),
                            bins=n_classes-1, max=3, min=1)
    # ((a + c), (b + d)
    area_label = torch.histc(anno.float(),
                             bins=n_classes-1, max=3, min=1)
    # ((a + b + c), (b + c + d)
    area_union = area_pred + area_label - area_overlap

#     area_overlap = overlap.float().sum((1, 2))
#     area_pred = pred_indices.float().sum((1,2))
#     area_label = anno.float().sum((1,2))
#     area_union = (anno | pred_indices).float().sum((1,2))
    assert (area_overlap <= area_label).all(), "Intersection area should be smaller than Union area"

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
