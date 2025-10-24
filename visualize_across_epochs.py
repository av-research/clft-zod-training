#!/usr/bin/env python3
"""
Visualize model predictions across multiple checkpoints for a single frame

This script allows you to:
1. Select a frame from the Waymo dataset
2. Run inference on multiple checkpoints
3. Generate comparison visualizations showing how predictions improve over training epochs
"""

import os
import cv2
import torch
import argparse
import numpy as np
import json
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from tqdm import tqdm

from clft.clft import CLFT
from utils.helpers import waymo_anno_class_relabel_1, image_overlay, draw_test_segmentation_map
from utils.lidar_process import open_lidar, crop_pointcloud, get_unresized_lid_img_val
from tools.dataset import lidar_dilation


class FrameVisualizer:
    def __init__(self, config_path, dataset_root='./waymo_dataset'):
        self.dataset_root = dataset_root
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def load_frame(self, frame_path):
        """Load RGB image and annotation for a Waymo frame"""
        rgb_path = os.path.join(self.dataset_root, frame_path)
        anno_path = rgb_path.replace('/camera/', '/annotation/')
        lidar_path = rgb_path.replace('/camera/', '/lidar/').replace('.png', '.pkl')
        
        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"RGB not found: {rgb_path}")
        if not os.path.exists(anno_path):
            raise FileNotFoundError(f"Annotation not found: {anno_path}")
        if not os.path.exists(lidar_path):
            raise FileNotFoundError(f"LiDAR not found: {lidar_path}")
        
        # Load images
        rgb_pil = Image.open(rgb_path).convert('RGB')
        anno_pil = Image.open(anno_path)
        
        return rgb_pil, anno_pil, lidar_path, rgb_path
    
    def preprocess_frame(self, rgb_pil, anno_pil, lidar_path):
        """Preprocess frame for model input"""
        resize = self.config['Dataset']['transforms']['resize']
        
        # RGB preprocessing
        w_orig, h_orig = rgb_pil.size
        delta = int(h_orig / 2)
        top_crop_rgb = TF.crop(rgb_pil, delta, 0, h_orig - delta, w_orig)
        
        rgb_transform = transforms.Compose([
            transforms.Resize((resize, resize), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config['Dataset']['transforms']['image_mean'],
                std=self.config['Dataset']['transforms']['image_std']
            )
        ])
        rgb_tensor = rgb_transform(top_crop_rgb)
        
        # LiDAR preprocessing
        points_set, camera_coord = open_lidar(
            lidar_path,
            w_ratio=4,
            h_ratio=4,
            lidar_mean=self.config['Dataset']['transforms']['lidar_mean_waymo'],
            lidar_std=self.config['Dataset']['transforms']['lidar_std_waymo']
        )
        
        top_crop_points_set, top_crop_camera_coord, _ = crop_pointcloud(
            points_set, camera_coord, 160, 0, 160, 480
        )
        X, Y, Z = get_unresized_lid_img_val(160, 480, top_crop_points_set, top_crop_camera_coord)
        X, Y, Z = lidar_dilation(X, Y, Z)
        
        X = transforms.Resize((resize, resize))(X)
        Y = transforms.Resize((resize, resize))(Y)
        Z = transforms.Resize((resize, resize))(Z)
        
        X = TF.to_tensor(np.array(X))
        Y = TF.to_tensor(np.array(Y))
        Z = TF.to_tensor(np.array(Z))
        
        lidar_tensor = torch.cat((X, Y, Z), 0)
        
        # Annotation preprocessing
        w_anno, h_anno = anno_pil.size
        delta_anno = int(h_anno / 2)
        top_crop_anno = TF.crop(anno_pil, delta_anno, 0, h_anno - delta_anno, w_anno)
        
        anno_np = np.array(top_crop_anno)
        anno_tensor = waymo_anno_class_relabel_1(anno_np)
        
        anno_transform = transforms.Compose([
            transforms.Resize((resize, resize), interpolation=transforms.InterpolationMode.NEAREST),
        ])
        anno_tensor = anno_transform(anno_tensor).squeeze(0)
        
        return rgb_tensor, lidar_tensor, anno_tensor, rgb_pil
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        resize = self.config['Dataset']['transforms']['resize']
        model = CLFT(
            RGB_tensor_size=(3, resize, resize),
            XYZ_tensor_size=(3, resize, resize),
            patch_size=self.config['CLFT']['patch_size'],
            emb_dim=self.config['CLFT']['emb_dim'],
            resample_dim=self.config['CLFT']['resample_dim'],
            read=self.config['CLFT']['read'],
            hooks=self.config['CLFT']['hooks'],
            reassemble_s=self.config['CLFT']['reassembles'],
            nclasses=len(self.config['Dataset']['classes']),
            type=self.config['CLFT']['type'],
            model_timm=self.config['CLFT']['model_timm'],
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def get_model_second_input(self, rgb_tensor, lidar_tensor, modality):
        """
        Get the correct second input (rgb or lidar) based on modality.
        
        Modality modes:
        - 'rgb': Use RGB for both inputs (RGB-only mode)
        - 'lidar': Use LiDAR as second input (LiDAR-only mode)
        - 'cross_fusion': Use LiDAR as second input (cross-modal fusion)
        
        Args:
            rgb_tensor: RGB image tensor
            lidar_tensor: LiDAR image tensor
            modality: String specifying the modality mode
            
        Returns:
            Tensor to use as second input to the model
        """
        if modality == 'rgb':
            # RGB-only mode: use RGB for both inputs
            return rgb_tensor
        else:
            # LiDAR or cross_fusion modes: use LiDAR as second input
            return lidar_tensor
    
    def predict_frame(self, rgb_tensor, lidar_tensor, checkpoint_path, modality='rgb'):
        """Run inference on frame"""
        model = self.load_checkpoint(checkpoint_path)
        
        rgb_tensor = rgb_tensor.unsqueeze(0).to(self.device)
        lidar_tensor = lidar_tensor.unsqueeze(0).to(self.device)
        
        # Get correct input based on modality
        second_input = self.get_model_second_input(rgb_tensor, lidar_tensor, modality)
        
        with torch.no_grad():
            _, pred_logits = model(rgb_tensor, second_input, modality)
        
        # Make sure pred_logits is on CPU before converting to numpy
        pred_logits = pred_logits.squeeze(0).cpu()
        pred_labels = torch.argmax(pred_logits, dim=0).numpy().astype(np.uint8)
        
        return pred_labels
    
    def create_segmentation_map(self, labels):
        """Create colored segmentation map from class labels"""
        # Waymo class colors (relabeled): background, cyclist, pedestrian, sign, ignore
        colors = [
            [0, 0, 0],        # 0: background - black
            [255, 0, 0],      # 1: cyclist - blue (BGR)
            [0, 0, 255],      # 2: pedestrian - red (BGR)
            [0, 255, 255],    # 3: sign - yellow (BGR)
            [128, 128, 128]   # 4: ignore - gray
        ]
        
        h, w = labels.shape
        seg_map = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in enumerate(colors):
            mask = labels == class_id
            seg_map[mask] = color
        
        return seg_map
    
    def create_visualizations(self, frame_path, checkpoint_dir, output_dir, modality='rgb', num_checkpoints=None):
        """Create visualizations across checkpoints"""
        
        print(f"Loading frame: {frame_path}")
        rgb_pil, anno_pil, lidar_path, rgb_path = self.load_frame(frame_path)
        rgb_tensor, lidar_tensor, anno_tensor, _ = self.preprocess_frame(rgb_pil, anno_pil, lidar_path)
        
        # Load ground truth annotation (original image)
        rgb_cv2 = cv2.imread(rgb_path)
        rgb_cv2_top = rgb_cv2[160:320, 0:480]  # Crop top half
        
        # Get ground truth segmentation
        anno_np = np.array(anno_pil)
        anno_tensor_relabel = waymo_anno_class_relabel_1(anno_np)
        anno_relabel = anno_tensor_relabel.squeeze(0).numpy()
        
        # Resize to match visualization size
        anno_resized = cv2.resize(anno_relabel.astype(np.uint8), (480, 160), interpolation=cv2.INTER_NEAREST)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all checkpoints
        checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pth')])
        
        if num_checkpoints:
            # Select evenly distributed checkpoints
            indices = np.linspace(0, len(checkpoint_files) - 1, num_checkpoints, dtype=int)
            checkpoint_files = [checkpoint_files[i] for i in indices]
        
        print(f"\nProcessing {len(checkpoint_files)} checkpoints...")
        
        predictions = {}
        
        for checkpoint_file in tqdm(checkpoint_files, desc="Running inference"):
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
            epoch = checkpoint_file.replace('checkpoint_', '').replace('.pth', '')
            
            try:
                pred_labels = self.predict_frame(rgb_tensor, lidar_tensor, checkpoint_path, modality)
                predictions[epoch] = pred_labels
                
                # Create segmentation map visualization
                seg_map = self.create_segmentation_map(pred_labels)
                
                # Resize to match image dimensions
                seg_map_resized = cv2.resize(seg_map, (480, 160))
                
                # Create overlay
                overlay = image_overlay(rgb_cv2_top.copy(), seg_map_resized)
                
                # Save
                output_path = os.path.join(output_dir, f'epoch_{int(epoch):03d}_prediction.png')
                cv2.imwrite(output_path, overlay)
                
            except Exception as e:
                print(f"Error processing checkpoint {checkpoint_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save ground truth
        gt_seg = self.create_segmentation_map(anno_resized)
        gt_overlay = image_overlay(rgb_cv2_top.copy(), gt_seg)
        gt_path = os.path.join(output_dir, 'ground_truth.png')
        cv2.imwrite(gt_path, gt_overlay)
        
        print(f"\nVisualizations saved to: {output_dir}")
        print(f"Total frames generated: {len(predictions) + 1}")
        
        return predictions


def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions across checkpoints')
    parser.add_argument('--frame', type=str, 
                        default='labeled/day/not_rain/camera/segment-10485926982439064520_4980_000_5000_000_with_camera_labels_0000000062.png',
                        help='Frame path (e.g., labeled/day/not_rain/camera/segment-xxx.png)')
    parser.add_argument('--checkpoint_dir', type=str, 
                        default='/media/tom/ml/logs/waymo/config_1/progress_save',
                        help='Directory containing checkpoint files')
    parser.add_argument('--config', type=str, default='config/waymo/config_1.json',
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='./logs/waymo_epoch_comparison',
                        help='Output directory for visualizations')
    parser.add_argument('--modality', type=str, default='rgb', choices=['rgb', 'lidar', 'cross_fusion'],
                        help='Fusion modality')
    parser.add_argument('--num_checkpoints', type=int, default=5,
                        help='Number of evenly distributed checkpoints to use (default: 5)')
    parser.add_argument('--dataset_root', type=str, default='./waymo_dataset',
                        help='Root directory of Waymo dataset')
    
    args = parser.parse_args()
    
    visualizer = FrameVisualizer(args.config, args.dataset_root)
    visualizer.create_visualizations(
        args.frame,
        args.checkpoint_dir,
        args.output_dir,
        args.modality,
        args.num_checkpoints
    )


if __name__ == '__main__':
    main()
