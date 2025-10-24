#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import numpy as np
import argparse

from torch.utils.data import DataLoader

from tools.trainer import Trainer
from tools.dataset import Dataset
from integrations.training_logger import generate_training_uuid

parser = argparse.ArgumentParser(description='CLFT and CLFCN Training')
parser.add_argument('-c', '--config', type=str, required=False, default='config.json', help='The path of the config file')
args = parser.parse_args()
config_file = args.config

with open(config_file, 'r') as f:
    config = json.load(f)

training_uuid = generate_training_uuid()
print(f"Training UUID: {training_uuid}")

np.random.seed(config['General']['seed'])
trainer = Trainer(config, training_uuid, config['Log']['logdir'])

train_data = Dataset(config, 'train', config['Dataset']['train_split'])
train_dataloader = DataLoader(train_data,
                              batch_size=config['General']['batch_size'],
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)

valid_data = Dataset(config, 'val', config['Dataset']['val_split'])
valid_dataloader = DataLoader(valid_data,
                              batch_size=config['General']['batch_size'],
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)

trainer.train_clft(train_dataloader, valid_dataloader, modal=config['CLI']['mode'])
