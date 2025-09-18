import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pickle
from tqdm import tqdm
from utils.lidar_augmentations import apply_lidar_augmentations, lidar_dilation, tensor_to_pil_channels, pil_channels_to_tensor

# LiDAR processing functions
def open_lidar(lidar_path, w_ratio, h_ratio, lidar_mean, lidar_std, camera_mask_value=0):
    mean_lidar = np.array(lidar_mean)
    std_lidar = np.array(lidar_std)

    file = open(lidar_path, 'rb')
    lidar_data = pickle.load(file)
    file.close()

    points3d = lidar_data['3d_points']
    camera_coord = lidar_data['camera_coordinates']

    mask = camera_coord[:, 0] == camera_mask_value
    points3d = points3d[mask, :]
    camera_coord = camera_coord[mask, 1:3]

    x_lid = (points3d[:, 1] - mean_lidar[0])/std_lidar[0]
    y_lid = (points3d[:, 2] - mean_lidar[1])/std_lidar[1]
    z_lid = (points3d[:, 0] - mean_lidar[2])/std_lidar[2]

    camera_coord[:, 1] = (camera_coord[:, 1]/h_ratio).astype(int)
    camera_coord[:, 0] = (camera_coord[:, 0]/w_ratio).astype(int)
    
    # Clamp coordinates to image bounds
    camera_coord[:, 1] = np.clip(camera_coord[:, 1], 0, 383)  # h-1
    camera_coord[:, 0] = np.clip(camera_coord[:, 0], 0, 383)  # w-1

    points_set = np.stack((x_lid, y_lid, z_lid), axis=1).astype(np.float32)

    return points_set, camera_coord

def get_unresized_lid_img_val(h, w, points_set, camera_coord):
    X = np.zeros((h, w))
    Y = np.zeros((h, w))
    Z = np.zeros((h, w))

    rows = camera_coord[:, 1].astype(int)
    cols = camera_coord[:, 0].astype(int)

    X[rows, cols] = points_set[:, 0]
    Y[rows, cols] = points_set[:, 1]
    Z[rows, cols] = points_set[:, 2]

    # Fix deprecated to_tensor() calls
    X = torch.from_numpy(X.astype(np.float32)).unsqueeze(0)
    Y = torch.from_numpy(Y.astype(np.float32)).unsqueeze(0)
    Z = torch.from_numpy(Z.astype(np.float32)).unsqueeze(0)

    return X, Y, Z

class LiDARCameraDataset(Dataset):
    def __init__(self, data_dir, split_file, rgb_mean, rgb_std, lidar_mean, lidar_std, 
                 camera_mask_value=0, preload=False, training=False, augment_config=None):
        self.data_dir = data_dir
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.lidar_mean = lidar_mean
        self.lidar_std = lidar_std
        self.camera_mask_value = camera_mask_value
        self.training = training
        self.augment_config = augment_config or {}
        
        # Load samples
        with open(split_file, 'r') as f:
            self.samples = [line.strip() for line in f]
        
        self.resize = transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.NEAREST)
        
        self.data = None
        if preload:
            self.data = self._preload_data()

    def _preload_data(self):
        print("Preloading data...")
        data = []
        for sample in tqdm(self.samples):
            rgb, lidar, anno = self._load_sample(sample)
            data.append((rgb, lidar, anno))
        return data

    def _get_paths(self, sample):
        rgb_path = os.path.join(self.data_dir, sample)
        lidar_path = rgb_path.replace('/camera/', '/lidar/').replace('.png', '.pkl')
        anno_path = rgb_path.replace('/camera/', '/annotation/').replace('.png', '.png')
        return rgb_path, lidar_path, anno_path

    def _load_rgb(self, rgb_path, apply_augment=False):
        rgb = Image.open(rgb_path).convert('RGB')
        if not apply_augment:
            # Standard processing for non-augmented data
            rgb = self.resize(rgb)
            rgb = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float() / 255.0  # Convert to tensor [C,H,W]
            rgb = transforms.Normalize(self.rgb_mean, self.rgb_std)(rgb)
        return rgb

    def _load_lidar(self, lidar_path, apply_augment=False):
        points_set, camera_coord = open_lidar(lidar_path, w_ratio=4, h_ratio=4,
                                              lidar_mean=self.lidar_mean,
                                              lidar_std=self.lidar_std,
                                              camera_mask_value=self.camera_mask_value)
        X, Y, Z = get_unresized_lid_img_val(384, 384, points_set, camera_coord)
        
        if not apply_augment:
            # Standard processing for non-augmented data
            lidar = torch.cat((X, Y, Z), 0)
            return lidar
        else:
            # Return as tensor for augmentation
            lidar = torch.cat((X, Y, Z), 0)
            return lidar

    def _load_anno(self, anno_path, apply_augment=False):
        anno = Image.open(anno_path)
        if not apply_augment:
            # Standard processing for non-augmented data
            anno = self.resize(anno)
            anno = torch.from_numpy(np.array(anno)).long()
        return anno

    def __len__(self):
        return len(self.samples) if self.data is None else len(self.data)

    def _load_sample(self, sample):
        rgb_path, lidar_path, anno_path = self._get_paths(sample)
        
        # Apply augmentations
        if self.training:
            # Load raw data for augmentation
            rgb = self._load_rgb(rgb_path, apply_augment=True)
            lidar = self._load_lidar(lidar_path, apply_augment=True)
            anno = self._load_anno(anno_path, apply_augment=True)
            
            # Apply LiDAR augmentations
            rgb, anno, lidar = apply_lidar_augmentations(
                rgb, anno, lidar, 
                training=True, 
                config=self.augment_config
            )
            
            # Apply LiDAR dilation (from original training)
            X, Y, Z = tensor_to_pil_channels(lidar)
            X, Y, Z = lidar_dilation(X, Y, Z)
            lidar = pil_channels_to_tensor(X, Y, Z)
            
            # Ensure consistent target size after augmentations
            rgb = rgb.resize((384, 384), Image.BILINEAR)
            anno = anno.resize((384, 384), Image.NEAREST)
            lidar = torch.nn.functional.interpolate(lidar.unsqueeze(0), size=(384, 384), mode='bilinear', align_corners=False).squeeze(0)
            
            # Final processing
            rgb = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float() / 255.0  # Convert to tensor [C,H,W]
            rgb = transforms.Normalize(self.rgb_mean, self.rgb_std)(rgb)
            
            anno = torch.from_numpy(np.array(anno)).long()
        else:
            # Standard loading without augmentations
            rgb = self._load_rgb(rgb_path, apply_augment=False)
            lidar = self._load_lidar(lidar_path, apply_augment=False)
            anno = self._load_anno(anno_path, apply_augment=False)
        
        # Debug: Print shapes to understand inconsistencies (disabled for performance)
        # if sample.endswith('frame_007674.png'):  # Debug only first sample
        #     print(f"Sample {sample}: RGB {rgb.shape}, LiDAR {lidar.shape}, Anno {anno.shape}")
        
        return rgb, lidar, anno

    def __getitem__(self, idx):
        if self.data is not None:
            return self.data[idx]
        
        sample = self.samples[idx]
        
        try:
            return self._load_sample(sample)
        except Exception as e:
            print(f"Error loading sample {sample}: {e}")
            raise e

# Alias for backward compatibility
GenericDataset = LiDARCameraDataset

# Config-based wrapper for backward compatibility
class ConfigBasedDataset(LiDARCameraDataset):
    def __init__(self, config, training=True):
        augment_config = {
            'p_flip': 0.5,
            'p_crop': 0.3,
            'p_rot': 0.4,
            'random_rotate_range': 20,
            'resize': 384
        }
        
        super().__init__(
            data_dir=config['data_dir'],
            split_file=config['split_file'],
            rgb_mean=config['rgb_mean'],
            rgb_std=config['rgb_std'],
            lidar_mean=config['lidar_mean'],
            lidar_std=config['lidar_std'],
            camera_mask_value=config['camera_mask_value'],
            preload=config.get('preload', False),
            training=training,
            augment_config=augment_config if training else {}
        )

# Override the alias to use the config-based version
GenericDataset = ConfigBasedDataset
