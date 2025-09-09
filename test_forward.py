import torch
from model import ViTSegmentation

model = ViTSegmentation('rgb', 5)
rgb = torch.randn(1, 3, 384, 384)
lidar = torch.randn(1, 3, 384, 384)
out = model(rgb, lidar)
print(f"Output shape: {out.shape}")
print("Forward pass successful")
