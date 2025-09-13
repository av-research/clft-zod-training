import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pickle
from tqdm import tqdm

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

    X = transforms.ToTensor()(X.astype(np.float32))
    Y = transforms.ToTensor()(Y.astype(np.float32))
    Z = transforms.ToTensor()(Z.astype(np.float32))

    return X, Y, Z

class GenericDataset(Dataset):
    def __init__(self, config, transform=None):
        self.data_dir = config['data_dir']
        self.transform = transform
        with open(config['split_file'], 'r') as f:
            self.samples = f.read().splitlines()
        
        # Normalization stats
        self.lidar_mean = config['lidar_mean']
        self.lidar_std = config['lidar_std']
        self.rgb_mean = config['rgb_mean']
        self.rgb_std = config['rgb_std']
        self.camera_mask_value = config['camera_mask_value']
        
        # Resize transform
        self.resize = transforms.Resize((384, 384))
        
        self.preload = config['preload']
        if self.preload and len(self.samples) <= 100:  # Preload only for small datasets
            # Preload all data into memory for speed
            self.data = []
            print("Preloading dataset into memory...")
            for sample in tqdm(self.samples, desc="Loading samples"):
                try:
                    rgb_path, lidar_path, anno_path = self._get_paths(sample)
                    rgb = self._load_rgb(rgb_path)
                    lidar = self._load_lidar(lidar_path)
                    anno = self._load_anno(anno_path)
                    self.data.append((rgb, lidar, anno))
                except Exception as e:
                    print(f"Error loading sample {sample}: {e}. Skipping.")
            print(f"Preloaded {len(self.data)} samples.")
        else:
            self.data = None  # Will load on-the-fly

    def _get_paths(self, sample):
        rgb_path = os.path.join(self.data_dir, sample)
        lidar_path = sample.replace('/camera/', '/lidar/').replace('.png', '.pkl')
        lidar_path = os.path.join(self.data_dir, lidar_path)
        anno_path = sample.replace('/camera/', '/annotation/')
        anno_path = os.path.join(self.data_dir, anno_path)
        return rgb_path, lidar_path, anno_path

    def _load_rgb(self, rgb_path):
        rgb = Image.open(rgb_path).convert('RGB')
        rgb = self.resize(rgb)
        rgb = transforms.ToTensor()(rgb)
        rgb = transforms.Normalize(self.rgb_mean, self.rgb_std)(rgb)
        return rgb

    def _load_lidar(self, lidar_path):
        points_set, camera_coord = open_lidar(lidar_path, w_ratio=4, h_ratio=4,
                                              lidar_mean=self.lidar_mean,
                                              lidar_std=self.lidar_std,
                                              camera_mask_value=self.camera_mask_value)
        X, Y, Z = get_unresized_lid_img_val(384, 384, points_set, camera_coord)
        lidar = torch.cat((X, Y, Z), 0)
        lidar = transforms.Normalize(self.lidar_mean, self.lidar_std)(lidar)
        return lidar

    def _load_anno(self, anno_path):
        anno = Image.open(anno_path)
        anno = self.resize(anno)
        anno = torch.from_numpy(np.array(anno)).long()
        return anno

    def __len__(self):
        return len(self.samples) if self.data is None else len(self.data)

    def _load_sample(self, sample):
        rgb_path, lidar_path, anno_path = self._get_paths(sample)
        rgb = self._load_rgb(rgb_path)
        lidar = self._load_lidar(lidar_path)
        anno = self._load_anno(anno_path)
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
