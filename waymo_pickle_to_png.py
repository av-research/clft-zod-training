#!/usr/bin/env python3
"""
Script to convert Waymo LiDAR pickle files to L2D (LiDAR-to-2D) projections.
Creates PNG visualizations of LiDAR point clouds projected onto camera images.

USES CLFT-CONFERENCE APPROACH:
- Uses pre-computed camera coordinates from pickle files (CLFT-conference method)
- Scales coordinates to match 480x320 output resolution
- Fixed output size to match camera images and annotations

This script handles:
1. Loading LiDAR data from pickle files
2. Using pre-computed camera coordinates for projection
3. Creating RGB visualizations of X, Y, Z channels
4. Saving as PNG files for dataset preparation

Camera Intrinsics (Waymo Front Camera - scaled for 480x320 output):
- fx: 315.0 (focal length x, scaled from 1260)
- fy: 315.0 (focal length y, scaled from 1260)
- cx: 240.0 (principal point x, scaled from 960)
- cy: 135.0 (principal point y, scaled from 540)

Output Size: Fixed at 480x320 to match camera images and annotations

LiDAR Normalization (from training config):
- mean: [-0.17263354, 0.85321806, 24.5527253]
- std: [7.34546552, 1.17227659, 15.83745082]
    # Process from all.txt file (default)
    python waymo_pickle_to_png.py

    # Process from custom file list
    python waymo_pickle_to_png.py --input /path/to/camera_paths.txt

    # Process directory (legacy)
    python waymo_pickle_to_png.py --input /path/to/pickle/dir --output /path/to/output/dir

    # Process single file
    python waymo_pickle_to_png.py --input /path/to/file.pkl --output /path/to/output/dir

Output Files (per pickle file):
    lidar_png/filename.png              - Combined 3-channel RGB PNG (for training - matches ZOD format) [ALWAYS CREATED]
    lidar_png_visualize/filename_overlay.png    - Camera image with LiDAR projection overlay for alignment verification [--visualize]
"""
import os
import pickle
import numpy as np
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2


class WaymoL2DProjector:
    """Handles LiDAR to 2D camera projection for Waymo dataset."""

    def __init__(self, output_width=480, output_height=320, w_ratio=4, h_ratio=4):
        """
        Initialize the projector.

        Args:
            output_width: Width of output projection image (fixed at 480 to match camera/annotation size)
            output_height: Height of output projection image (fixed at 320 to match camera/annotation size)
            w_ratio: Width downsampling ratio for camera coordinates
            h_ratio: Height downsampling ratio for camera coordinates
        """
        self.output_width = output_width
        self.output_height = output_height
        self.w_ratio = w_ratio
        self.h_ratio = h_ratio

        # CORRECTED: Waymo front camera intrinsic parameters (scaled for 480x320 output)
        # Original Waymo camera is ~1920x1280, intrinsics scaled by 0.25 for 480x320
        # fx/fy = 1260 * (320/1280) = 1260 * 0.25 = 315
        # cx/cy = 960 * (480/1920) = 960 * 0.25 = 240, 540 * (320/1280) = 540 * 0.25 = 135
        self.camera_intrinsics = {
            'fx': 315.0,   # focal length x (scaled)
            'fy': 315.0,   # focal length y (scaled)
            'cx': 240.0,   # principal point x (scaled)
            'cy': 135.0,   # principal point y (scaled)
        }

        # LiDAR normalization parameters (from existing code)
        self.lidar_mean = np.array([-0.17263354, 0.85321806, 24.5527253])
        self.lidar_std = np.array([7.34546552, 1.17227659, 15.83745082])

    def load_pickle_data(self, pickle_path):
        """
        Load LiDAR data from pickle file.

        Args:
            pickle_path: Path to pickle file

        Returns:
            points3d: 3D LiDAR points (N, 3)
            camera_coord: 2D camera coordinates (N, 2) - required for CLFT approach
        """
        with open(pickle_path, 'rb') as f:
            lidar_data = pickle.load(f)

        points3d = lidar_data['3d_points']

        # Check if camera coordinates are already projected (required for CLFT approach)
        if 'camera_coordinates' in lidar_data:
            camera_coord = lidar_data['camera_coordinates']
            # Select front camera (camera 1)
            mask = camera_coord[:, 0] == 1
            points3d = points3d[mask, :]
            camera_coord = camera_coord[mask, 1:3]  # u, v coordinates
            return points3d, camera_coord
        else:
            raise ValueError(f"No pre-computed camera coordinates found in {pickle_path}. CLFT approach requires camera_coordinates in pickle file.")

    def normalize_lidar_points(self, points3d):
        """
        Normalize LiDAR points using dataset statistics.

        Args:
            points3d: 3D points (N, 3) with columns [Z, X, Y]

        Returns:
            normalized_points: Normalized points (N, 3) as [X, Y, Z]
        """
        # Reorder to [X, Y, Z] and normalize
        x_lid = (points3d[:, 1] - self.lidar_mean[0]) / self.lidar_std[0]
        y_lid = (points3d[:, 2] - self.lidar_mean[1]) / self.lidar_std[1]
        z_lid = (points3d[:, 0] - self.lidar_mean[2]) / self.lidar_std[2]

        return np.stack([x_lid, y_lid, z_lid], axis=1)

    def create_l2d_projection(self, points3d, camera_coord):
        """
        Create L2D projection image from LiDAR points using pre-computed coordinates.

        Args:
            points3d: 3D LiDAR points (N, 3)
            camera_coord: Pre-computed camera coordinates (N, 2)

        Returns:
            X_img, Y_img, Z_img: PIL Images for each channel
        """
        # Normalize points
        normalized_points = self.normalize_lidar_points(points3d)

        # Use pre-projected coordinates (CLFT-conference approach)
        camera_coord = camera_coord.copy().astype(np.float32)  # Ensure float type for operations

        # Automatically scale coordinates to output space if needed
        max_u, max_v = camera_coord[:, 0].max(), camera_coord[:, 1].max()
        if max_u < 100 and max_v < 100:  # Likely normalized coordinates (0-1)
            camera_coord[:, 0] *= self.output_width
            camera_coord[:, 1] *= self.output_height
        elif max_u > self.output_width * 2:  # Likely in original camera space (~1920x1280)
            camera_coord[:, 0] *= self.output_width / 1920.0
            camera_coord[:, 1] *= self.output_height / 1280.0

        # Create projection images
        X = np.zeros((self.output_height, self.output_width), dtype=np.float32)
        Y = np.zeros((self.output_height, self.output_width), dtype=np.float32)
        Z = np.zeros((self.output_height, self.output_width), dtype=np.float32)

        # Get valid coordinates within image bounds
        rows = np.floor(camera_coord[:, 1]).astype(int)
        cols = np.floor(camera_coord[:, 0]).astype(int)

        # Filter valid coordinates
        valid_mask = (rows >= 0) & (rows < self.output_height) & \
                    (cols >= 0) & (cols < self.output_width)

        rows = rows[valid_mask]
        cols = cols[valid_mask]
        points = normalized_points[valid_mask]

        # Project points to image
        X[rows, cols] = points[:, 0]
        Y[rows, cols] = points[:, 1]
        Z[rows, cols] = points[:, 2]

        # Convert to PIL Images (convert to 8-bit grayscale for PNG)
        # Normalize to 0-255 range
        X_norm = ((X - X.min()) / (X.max() - X.min() + 1e-6) * 255).astype(np.uint8)
        Y_norm = ((Y - Y.min()) / (Y.max() - Y.min() + 1e-6) * 255).astype(np.uint8)
        Z_norm = ((Z - Z.min()) / (Z.max() - Z.min() + 1e-6) * 255).astype(np.uint8)

        X_img = Image.fromarray(X_norm, mode='L')
        Y_img = Image.fromarray(Y_norm, mode='L')
        Z_img = Image.fromarray(Z_norm, mode='L')

        return X_img, Y_img, Z_img

    def create_overlay_visualization(self, camera_path, lidar_png_path, output_path, points3d, camera_coord, alpha=0.9):
        """
        Create overlay visualization of LiDAR projection on camera image for alignment verification.

        Args:
            camera_path: Path to camera image
            lidar_png_path: Path to LiDAR PNG projection
            output_path: Path to save overlaid image
            points3d: 3D LiDAR points (N, 3)
            camera_coord: Pre-computed camera coordinates (N, 2)
            alpha: Transparency alpha for LiDAR overlay (0-1)
        """
        # Load camera image
        camera_img = Image.open(camera_path).convert('RGBA')

        # Calculate distances for each point
        distances = np.linalg.norm(points3d, axis=1)  # Euclidean distance from sensor

        # Create distance-based image
        distance_img = np.zeros((self.output_height, self.output_width), dtype=np.float32)

        # Convert camera coordinates to image indices (scale to output size)
        # Original camera is ~1920x1280, output is 480x320, so scale by 0.25
        scale_x = self.output_width / 1920.0  # 480/1920 = 0.25
        scale_y = self.output_height / 1280.0  # 320/1280 = 0.25

        rows = np.round(camera_coord[:, 1] * scale_y).astype(int)
        cols = np.round(camera_coord[:, 0] * scale_x).astype(int)

        # Filter valid coordinates (within image bounds)
        valid_mask = (rows >= 0) & (rows < self.output_height) & (cols >= 0) & (cols < self.output_width)
        rows = rows[valid_mask]
        cols = cols[valid_mask]
        valid_distances = distances[valid_mask]

        # Project distances to image with balanced contrast
        if valid_distances.size > 0:
            # Use logarithmic scaling for better contrast across distance range
            # Add small epsilon to avoid log(0)
            log_distances = np.log(valid_distances + 1.0)

            # Normalize logarithmic distances to 0-255 range
            dist_min, dist_max = log_distances.min(), log_distances.max()
            if dist_max > dist_min:
                distance_norm = ((log_distances - dist_min) / (dist_max - dist_min) * 255).astype(np.uint8)
            else:
                distance_norm = np.full_like(log_distances, 127, dtype=np.uint8)

            # Create distance image
            dist_img = np.zeros((self.output_height, self.output_width), dtype=np.uint8)
            dist_img[rows, cols] = distance_norm
        else:
            dist_img = np.zeros((self.output_height, self.output_width), dtype=np.uint8)

        # Apply TURBO colormap for excellent perceptual uniformity and contrast
        lidar_colored = cv2.applyColorMap(dist_img, cv2.COLORMAP_TURBO)
        lidar_colored_pil = Image.fromarray(cv2.cvtColor(lidar_colored, cv2.COLOR_BGR2RGB)).convert('RGBA')

        # Apply distance-based alpha with balanced visibility
        r, g, b, a = lidar_colored_pil.split()

        # Create alpha channel that ensures both close and far objects are visible
        intensity = np.array(r)  # Use red channel as intensity proxy
        intensity_norm = intensity.astype(np.float32) / 255.0
        # Balanced alpha: moderate base visibility with slight emphasis on closer objects
        alpha_values = np.where(intensity > 0,
                               0.5 + 0.4 * intensity_norm,  # Range from 0.5 to 0.9
                               0.0)
        a = Image.fromarray((alpha_values * 255).astype(np.uint8))
        lidar_overlay = Image.merge('RGBA', (r, g, b, a))

        # Resize overlay to match camera if needed
        if lidar_overlay.size != camera_img.size:
            lidar_overlay = lidar_overlay.resize(camera_img.size, Image.Resampling.LANCZOS)

        # Composite camera + LiDAR overlay
        overlaid = Image.alpha_composite(camera_img, lidar_overlay)

        # Save
        overlaid.convert('RGB').save(output_path)

    def create_combined_lidar_png(self, X_img, Y_img, Z_img, save_path):
        """
        Create a combined 3-channel PNG file (like ZOD format) for training.

        Args:
            X_img, Y_img, Z_img: Individual channel PIL Images
            save_path: Path to save the combined PNG
        """
        # Convert to numpy arrays
        X = np.array(X_img).astype(np.float32) / 255.0  # Normalize to [0,1]
        Y = np.array(Y_img).astype(np.float32) / 255.0
        Z = np.array(Z_img).astype(np.float32) / 255.0

        # Stack into 3-channel image (H, W, 3)
        combined_array = np.stack([X, Y, Z], axis=2)

        # Convert to uint8 for PNG saving (0-255 range)
        combined_uint8 = (combined_array * 255).astype(np.uint8)

        # Create PIL Image
        combined_img = Image.fromarray(combined_uint8, mode='RGB')
        combined_img.save(save_path)

    def process_pickle_file(self, pickle_path, output_dir, create_visualization=False):
        """
        Process a single pickle file and save projections.

        Args:
            pickle_path: Path to input pickle file
            output_dir: Directory to save output images
            create_visualization: Whether to create RGB visualization
        """
        # Load data
        points3d, camera_coord = self.load_pickle_data(pickle_path)

        if len(points3d) == 0:
            print(f"Warning: No points found in {pickle_path}")
            return

        # Create projections using pre-computed coordinates
        X_img, Y_img, Z_img = self.create_l2d_projection(points3d, camera_coord)

        # Get output filename
        pickle_name = Path(pickle_path).stem
        output_base = os.path.join(output_dir, pickle_name)

        # Create combined 3-channel PNG (matches ZOD format expected by training)
        self.create_combined_lidar_png(X_img, Y_img, Z_img, f"{output_base}.png")

        # Note: Overlay visualization is created in process_from_file_list when camera paths are available

    def process_from_file_list(self, file_list_path, dataset_root='', output_root='', create_visualization=False):
        """
        Process pickle files listed in a text file (one path per line).

        Args:
            file_list_path: Path to text file containing camera image paths
            dataset_root: Root directory of the dataset (optional, will be inferred if not provided)
            output_root: Root directory for output files (optional, will use dataset_root if not provided)
            create_visualization: Whether to create RGB visualizations
        """
        # Read file list
        with open(file_list_path, 'r') as f:
            camera_paths = [line.strip() for line in f if line.strip()]

        if not camera_paths:
            print(f"No paths found in {file_list_path}")
            return

        print(f"Found {len(camera_paths)} camera paths to process")

        # Infer dataset root if not provided
        if not dataset_root:
            file_list_path_obj = Path(file_list_path)
            # Assume the file list is in dataset_root/splits_clft/
            if 'splits_clft' in str(file_list_path_obj):
                dataset_root = str(file_list_path_obj.parent.parent)
            else:
                # Fallback: assume dataset root is the parent directory of the file list
                dataset_root = str(file_list_path_obj.parent)

        print(f"Using dataset root: {dataset_root}")
        if not output_root:
            output_root = dataset_root
        print(f"Using output root: {output_root}")

        # Convert camera paths to lidar paths and determine output paths
        pickle_paths = []
        output_dirs = []
        
        for cam_path in camera_paths:
            # Prepend dataset root to camera path
            full_cam_path = os.path.join(dataset_root, cam_path)
            
            # Replace /camera/ with /lidar/ and .png with .pkl to get pickle path
            lidar_path = full_cam_path.replace('/camera/', '/lidar/').replace('.png', '.pkl')
            pickle_paths.append(lidar_path)
            
            # Create output path: replace /camera/ with /lidar_png/ and change to .png
            output_path = full_cam_path.replace('/camera/', '/lidar_png/').replace('.png', '.png')
            # Replace dataset_root with output_root in the output path
            if output_root != dataset_root:
                output_path = output_path.replace(dataset_root, output_root)
            output_dir = os.path.dirname(output_path)
            output_dirs.append(output_dir)
            
            # Create output directory for training data
            os.makedirs(output_dir, exist_ok=True)
            
            # Create output directory for visualizations if needed
            if create_visualization:
                visualize_output_path = full_cam_path.replace('/camera/', '/lidar_png_visualize/').replace('.png', '_overlay.png')
                if output_root != dataset_root:
                    visualize_output_path = visualize_output_path.replace(dataset_root, output_root)
                visualize_dir = os.path.dirname(visualize_output_path)
                os.makedirs(visualize_dir, exist_ok=True)

        # Process each pickle file
        processed_count = 0
        for i, pickle_path in enumerate(tqdm(pickle_paths, desc="Processing pickle files")):
            try:
                if os.path.exists(pickle_path):
                    output_dir = output_dirs[i]
                    self.process_pickle_file(pickle_path, output_dir, create_visualization)

                    # Create overlay visualization if requested
                    if create_visualization:
                        camera_path = os.path.join(dataset_root, camera_paths[i])
                        lidar_png_path = os.path.join(output_dir, f"{Path(pickle_path).stem}.png")
                        # Reconstruct the full camera path for this index
                        current_full_cam_path = os.path.join(dataset_root, camera_paths[i])
                        visualize_output_path = current_full_cam_path.replace('/camera/', '/lidar_png_visualize/').replace('.png', '_overlay.png')
                        if output_root != dataset_root:
                            visualize_output_path = visualize_output_path.replace(dataset_root, output_root)

                        if os.path.exists(camera_path) and os.path.exists(lidar_png_path):
                            # Load the 3D points data for distance-based visualization
                            points3d_vis, camera_coord_vis = self.load_pickle_data(pickle_path)
                            self.create_overlay_visualization(camera_path, lidar_png_path, visualize_output_path, points3d_vis, camera_coord_vis)
                        else:
                            print(f"Warning: Missing files for overlay - camera: {camera_path}, lidar: {lidar_png_path}")

                    processed_count += 1
                else:
                    print(f"Warning: Pickle file not found: {pickle_path}")
            except Exception as e:
                print(f"Error processing {pickle_path}: {e}")
                continue

        print(f"Processing complete. Processed {processed_count}/{len(pickle_paths)} files.")


    def process_directory(self, input_dir, output_dir, create_visualization=False):
        """
        Process all pickle files in a directory.

        Args:
            input_dir: Directory containing pickle files
            output_dir: Directory to save output images
            create_visualization: Whether to create RGB visualizations
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Find all pickle files
        pickle_files = list(Path(input_dir).glob('**/*.pkl')) + \
                      list(Path(input_dir).glob('**/*.pickle'))

        if not pickle_files:
            print(f"No pickle files found in {input_dir}")
            return

        print(f"Found {len(pickle_files)} pickle files to process")

        # Process each file
        for pickle_file in tqdm(pickle_files, desc="Processing pickle files"):
            try:
                self.process_pickle_file(str(pickle_file), output_dir, create_visualization)
            except Exception as e:
                print(f"Error processing {pickle_file}: {e}")
                continue

        print(f"Processing complete. Output saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Convert Waymo LiDAR pickle files to L2D projections')
    parser.add_argument('--input', '-i', default='waymo_dataset/splits_clft/all.txt',
                       help='Input: text file with camera paths (default: waymo_dataset/splits_clft/all.txt), directory containing pickle files, or single pickle file')
    parser.add_argument('--output', '-o',
                       help='Output directory (for directory/pickle input) or output root (for file list input)')
    parser.add_argument('--dataset-root', '-d', default='waymo_dataset',
                       help='Dataset root directory (default: waymo_dataset)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create RGB visualization in addition to combined PNG')

    args = parser.parse_args()

    # Initialize projector with fixed 480x320 output size (matches camera/annotation dimensions)
    projector = WaymoL2DProjector(
        output_width=480,
        output_height=320
    )

    # Print camera intrinsics info
    print("Using CLFT-conference approach with pre-computed camera coordinates:")
    print(f"Output size: {projector.output_width}x{projector.output_height} (fixed to match camera/annotation dimensions)")
    for key, value in projector.camera_intrinsics.items():
        print(f"  {key}: {value}")
    print(f"LiDAR mean: {projector.lidar_mean}")
    print(f"LiDAR std: {projector.lidar_std}")
    print()

    # Check if input is a file or directory
    input_path = Path(args.input)
    if input_path.is_file():
        # Check if it's a text file (contains camera paths) or a pickle file
        if input_path.suffix.lower() in ['.txt']:
            # Process from file list - by default only create combined PNGs for training
            print(f"Processing from file list: {input_path}")
            output_root = args.output if args.output else ''
            dataset_root = args.dataset_root
            projector.process_from_file_list(str(input_path), dataset_root=dataset_root, output_root=output_root, create_visualization=args.visualize)
        else:
            # Process single pickle file - requires output directory
            if not args.output:
                parser.error("--output is required when processing a single pickle file")
            os.makedirs(args.output, exist_ok=True)
            print(f"Processing single pickle file: {input_path}")
            projector.process_pickle_file(str(input_path), args.output, args.visualize)
    else:
        # Process directory (legacy mode) - requires output directory
        if not args.output:
            parser.error("--output is required when processing a directory")
        print(f"Processing directory: {input_path}")
        projector.process_directory(args.input, args.output, args.visualize)


if __name__ == '__main__':
    main()