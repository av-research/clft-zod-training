import json
import pickle
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


def process_single_annotation_file(file_path):
    """Process a single annotation file and return pixel counts."""
    try:
        with Image.open(file_path) as img:
            mask = np.array(img, dtype=np.uint8)
            counts = np.bincount(mask.flatten(), minlength=256)
            pixel_counts = {}
            for val in range(256):
                if counts[val] > 0:
                    pixel_counts[val] = int(counts[val])
            total_pixels = mask.size
            return pixel_counts, total_pixels
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {}, 0


def analyze_annotations(annotation_input, class_names: dict, max_workers=None):
    """Analyze annotation masks for pixel class distribution using multiprocessing."""
    print("Analyzing annotation pixel distributions...")

    if isinstance(annotation_input, list):
        annotation_files = annotation_input
    else:
        annotation_files = list(annotation_input.glob("frame_*.png"))
    
    if not annotation_files:
        print(f"No annotation files found")
        return {}

    if max_workers is None:
        max_workers = min(mp.cpu_count(), 4)  # Limit to 4 workers to avoid memory issues

    print(f"Using {max_workers} worker processes...")

    pixel_counts = Counter()
    total_pixels = 0

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_annotation_file, file_path): file_path
                         for file_path in annotation_files}

        # Process results as they complete
        for future in tqdm(as_completed(future_to_file), total=len(annotation_files),
                          desc="Processing annotations"):
            file_pixel_counts, file_pixels = future.result()
            for val, count in file_pixel_counts.items():
                pixel_counts[val] += count
            total_pixels += file_pixels

    # Convert to percentages and structured results
    results = {
        "total_frames": len(annotation_files),
        "total_pixels": total_pixels,
        "class_distribution": {}
    }

    for class_id, count in pixel_counts.items():
        percentage = (count / total_pixels) * 100
        class_name = class_names.get(class_id, f"class_{class_id}")
        results["class_distribution"][class_name] = {
            "class_id": class_id,
            "pixel_count": count,
            "percentage": round(percentage, 4)
        }

    # Sort class_distribution by class_id
    sorted_class_distribution = {}
    for class_name in sorted(results["class_distribution"].keys(), 
                           key=lambda x: results["class_distribution"][x]["class_id"]):
        sorted_class_distribution[class_name] = results["class_distribution"][class_name]
    results["class_distribution"] = sorted_class_distribution

    print(f"Processed {len(annotation_files)} annotation files")
    print(f"Total pixels: {total_pixels:,}")
    print("Class distribution:")
    for class_name, data in results["class_distribution"].items():
        print(f"{class_name}: {data['pixel_count']:,} pixels ({data['percentage']:.2f}%)")

    return results


def process_single_camera_file(file_path):
    """Process a single camera file and return statistics."""
    try:
        with Image.open(file_path) as img:
            img = img.convert('RGB')  # Ensure RGB format for consistent statistics
            img_array = np.array(img, dtype=np.float32)  # Use float32 instead of float64
            
            if img_array.ndim == 3 and img_array.shape[2] == 3:
                pixel_sum = np.sum(img_array, axis=(0, 1))  # Sum over height and width
                pixel_sum_sq = np.sum(img_array ** 2, axis=(0, 1))  # Sum of squares
                total_pixels = img_array.shape[0] * img_array.shape[1]
                return pixel_sum, pixel_sum_sq, total_pixels
            else:
                return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 0
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 0


def analyze_camera_statistics(camera_input, max_workers=None):
    """Calculate mean and std statistics for camera images using multiprocessing."""
    print("Analyzing camera image statistics...")

    if isinstance(camera_input, list):
        camera_files = camera_input
    else:
        camera_files = list(camera_input.glob("frame_*.png"))
    
    if not camera_files:
        print(f"No camera files found")
        return {}

    if max_workers is None:
        max_workers = min(mp.cpu_count(), 4)  # Limit to 4 workers

    print(f"Using {max_workers} worker processes...")

    # Initialize accumulators
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sum_sq = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_single_camera_file, file_path): file_path
                         for file_path in camera_files}

        for future in tqdm(as_completed(future_to_file), total=len(camera_files),
                          desc="Processing camera images"):
            file_pixel_sum, file_pixel_sum_sq, file_pixels = future.result()
            pixel_sum += file_pixel_sum
            pixel_sum_sq += file_pixel_sum_sq
            total_pixels += file_pixels

    if total_pixels == 0:
        return {}

    # Calculate mean and std
    means = pixel_sum / total_pixels
    variances = (pixel_sum_sq / total_pixels) - (means ** 2)
    stds = np.sqrt(np.maximum(variances, 0))  # Ensure non-negative

    results = {
        "total_images": len(camera_files),
        "total_pixels": total_pixels,
        "mean_rgb": [round(float(m), 4) for m in means],
        "std_rgb": [round(float(s), 4) for s in stds],
        "normalized_mean_rgb": [round(float(m)/255, 4) for m in means],
        "normalized_std_rgb": [round(float(s)/255, 4) for s in stds]
    }

    print(f"Processed {len(camera_files)} camera images")
    print(f"Mean RGB: {results['mean_rgb']}")
    print(f"Std RGB: {results['std_rgb']}")

    return results


def process_single_lidar_file(file_path):
    """Process a single LiDAR file and return per-dimension statistics."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict) and '3d_points' in data:
            # Extract 3D points array
            points = data['3d_points']
            if isinstance(points, np.ndarray) and points.shape[1] == 3:
                # Calculate per-dimension statistics
                dim_sums = np.sum(points, axis=0)  # Sum along each dimension
                dim_sum_sq = np.sum(points ** 2, axis=0)  # Sum of squares along each dimension
                total_points = points.shape[0]  # Number of points
                return dim_sums, dim_sum_sq, total_points
        elif isinstance(data, np.ndarray) and data.shape[1] == 3:
            # Fallback for direct numpy arrays
            dim_sums = np.sum(data, axis=0)
            dim_sum_sq = np.sum(data ** 2, axis=0)
            total_points = data.shape[0]
            return dim_sums, dim_sum_sq, total_points
        
        return np.zeros(3), np.zeros(3), 0
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros(3), np.zeros(3), 0


def analyze_lidar_statistics(lidar_input, max_workers=None):
    """Calculate mean and std statistics for LiDAR data using multiprocessing."""
    print("Analyzing LiDAR data statistics...")

    if isinstance(lidar_input, list):
        lidar_files = lidar_input
    else:
        lidar_files = list(lidar_input.glob("frame_*.pkl"))
    
    if not lidar_files:
        print(f"No LiDAR files found")
        return {}

    if max_workers is None:
        max_workers = min(mp.cpu_count(), 4)  # Limit to 4 workers

    print(f"Using {max_workers} worker processes...")

    # Initialize accumulators for per-dimension statistics
    total_dim_sums = np.zeros(3, dtype=np.float64)
    total_dim_sum_sq = np.zeros(3, dtype=np.float64)
    total_points = 0

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_single_lidar_file, file_path): file_path
                         for file_path in lidar_files}

        for future in tqdm(as_completed(future_to_file), total=len(lidar_files),
                          desc="Processing LiDAR files"):
            file_dim_sums, file_dim_sum_sq, file_points = future.result()
            total_dim_sums += file_dim_sums
            total_dim_sum_sq += file_dim_sum_sq
            total_points += file_points

    if total_points == 0:
        return {}

    # Calculate per-dimension mean and std
    dim_means = total_dim_sums / total_points
    dim_variances = (total_dim_sum_sq / total_points) - (dim_means ** 2)
    dim_stds = np.sqrt(np.maximum(dim_variances, 0))

    results = {
        "total_files": len(lidar_files),
        "total_points": total_points,
        "lidar_mean": [round(float(m), 3) for m in dim_means],
        "lidar_std": [round(float(s), 3) for s in dim_stds]
    }

    print(f"Processed {len(lidar_files)} LiDAR files")
    print(f"Total points: {total_points:,}")
    print(f"LiDAR mean: {results['lidar_mean']}")
    print(f"LiDAR std: {results['lidar_std']}")

    return results
