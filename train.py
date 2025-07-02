import json
import sys
import argparse
import numpy as np
import torch

from torch.utils.data import DataLoader

from tools.trainer import Trainer
from tools.dataset import Dataset

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

with open('config.json', 'r') as f:
    config = json.load(f)

parser = argparse.ArgumentParser(description='CLFT and CLFCN Training')
parser.add_argument('-bb', '--backbone', required=True,
                    choices=['clft'],
                    help='Use the backbone of training, clft')
parser.add_argument('-m', '--mode', type=str, required=True,
                    choices=['rgb', 'lidar', 'cross_fusion'],
                    help='Output mode (lidar, rgb or cross_fusion)')
args = parser.parse_args()
np.random.seed(config['General']['seed'])
trainer = Trainer(config, args)

train_data = Dataset(config, 'train', './zod_dataset/splits_zod/train_all.txt')
train_dataloader = DataLoader(train_data,
                              batch_size=config['General']['batch_size'],
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)

valid_data = Dataset(config, 'val', './zod_dataset/splits_zod/early_stop_valid.txt')
valid_dataloader = DataLoader(valid_data,
                              batch_size=config['General']['batch_size'],
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)

if args.backbone == 'clft':
    trainer.train_clft(train_dataloader, valid_dataloader, modal=args.mode)

else:
    sys.exit("A backbone must be specified! (clft or clfcn)")
