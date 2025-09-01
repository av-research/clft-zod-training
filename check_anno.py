from PIL import Image
import numpy as np
import glob

# List of annotation files to check (edit the pattern as needed)
anno_files = glob.glob('./zod_dataset/labeled/annotation/*.png')

for anno_path in anno_files:
    anno = Image.open(anno_path)
    anno_np = np.array(anno)
    print(f"File: {anno_path}")
    print("  PIL shape:", anno.size)  # (width, height)
    print("  Numpy shape:", anno_np.shape)
    print("  Unique values:", np.unique(anno_np))
    print("  Value counts:", {v: np.sum(anno_np == v) for v in np.unique(anno_np)})
    print()