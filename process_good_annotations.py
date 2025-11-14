import os
import numpy as np
from PIL import Image
import json
from tqdm import tqdm

# Load config
with open('config/zod/config_1.json') as f:
    config = json.load(f)

classes = config['Dataset']['classes']
class_names = [c['name'] for c in classes]
original_indices = [c['original_index'] for c in classes]  # Use original annotation indices

# Read good.txt
with open('zod_dataset/good.txt') as f:
    good_files = [line.strip() for line in f]

# Initialize counts
pixel_counts = {name: 0 for name in class_names}
total_pixels = 0

print(f"Processing {len(good_files)} annotation files...")
for file_path in tqdm(good_files):
    # file_path is like camera/frame_099988.png
    frame_name = os.path.basename(file_path)  # frame_099988.png
    anno_path = os.path.join('zod_dataset', 'annotation_camera_only', frame_name)
    if os.path.exists(anno_path):
        img = Image.open(anno_path)
        img_array = np.array(img)
        unique, counts = np.unique(img_array, return_counts=True)
        for val, count in zip(unique, counts):
            if val in original_indices:
                idx = original_indices.index(val)
                pixel_counts[class_names[idx]] += count
        total_pixels += img_array.size
    else:
        print(f"Annotation not found: {anno_path}")

# Write distribution summary
with open('distribution_summary.txt', 'w') as f:
    f.write("Pixel Distribution Summary:\n")
    for name, count in pixel_counts.items():
        percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
        f.write(f"{name}: {count} pixels ({percentage:.2f}%)\n")

print("Distribution summary written to distribution_summary.txt")

# Now, split into train and val
# Shuffle the good_files
np.random.seed(42)
np.random.shuffle(good_files)

n_train = int(0.8 * len(good_files))
train_files = good_files[:n_train]
val_files = good_files[n_train:]

# Check train/val split distribution
print("\nChecking train/validation split distribution...")

def check_split_distribution(split_files, split_name):
    split_counts = {name: 0 for name in class_names}
    split_total = 0
    cyclist_ped_samples = 0
    
    for file_path in split_files:
        frame_name = os.path.basename(file_path)
        anno_path = os.path.join('zod_dataset', 'annotation_camera_only', frame_name)
        if os.path.exists(anno_path):
            img = Image.open(anno_path)
            img_array = np.array(img)
            unique, counts = np.unique(img_array, return_counts=True)
            
            has_cyclist_ped = False
            for val, count in zip(unique, counts):
                if val in original_indices:
                    idx = original_indices.index(val)
                    split_counts[class_names[idx]] += count
                    if class_names[idx] == 'cyclist + pedestrian':
                        has_cyclist_ped = True
            if has_cyclist_ped:
                cyclist_ped_samples += 1
            split_total += img_array.size
    
    print(f"{split_name} split ({len(split_files)} files):")
    for name, count in split_counts.items():
        percentage = (count / split_total) * 100 if split_total > 0 else 0
        print(f"  {name}: {count} pixels ({percentage:.2f}%)")
    print(f"  Samples with cyclist+pedestrian: {cyclist_ped_samples}/{len(split_files)}")
    return split_counts

train_counts = check_split_distribution(train_files, "Train")
val_counts = check_split_distribution(val_files, "Validation")

# Write train.txt and validation.txt
with open('zod_dataset/train.txt', 'w') as f:
    for file in train_files:
        f.write(file + '\n')

with open('zod_dataset/validation.txt', 'w') as f:
    for file in val_files:
        f.write(file + '\n')

print(f"Train split ({len(train_files)} files) written to zod_dataset/train.txt")
print(f"Validation split ({len(val_files)} files) written to zod_dataset/validation.txt")