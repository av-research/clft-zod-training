
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import numpy as np
import argparse
import datetime
import re
import uuid

from torch.utils.data import DataLoader

from tools.tester import Tester
from tools.dataset import Dataset
from tools.dataset_png import DatasetPNG
from integrations.vision_service import send_test_results_from_file
from utils.helpers import get_model_path

parser = argparse.ArgumentParser(description='CLFT and CLFCN Testing')
parser.add_argument('-c', '--config', type=str, required=False, default='config.json', help='The path of the config file')
args = parser.parse_args()
config_file = args.config

with open(config_file, 'r') as f:
    config = json.load(f)

np.random.seed(config['General']['seed'])

tester = Tester(config)

test_data_path = config['CLI']['path']
test_data_files = [
    'test_day_fair.txt',
    'test_night_fair.txt',
    'test_day_rain.txt',
    'test_night_rain.txt'
]

# Collect results from all test files
all_results = {}

for file in test_data_files:
    path = test_data_path + file
    print(f"Testing with the path {path}")

    # Use DatasetPNG for ZOD, Dataset for others
    if config['Dataset']['name'] == 'zod':
        test_data = DatasetPNG(config, 'test', path)
    else:
        test_data = Dataset(config, 'test', path)

    test_dataloader = DataLoader(test_data,
                                batch_size=config['General']['batch_size'],
                                shuffle=False,
                                pin_memory=True,
                                drop_last=True)

    # Get results for this test file
    results = tester.test_clft_return_results(test_dataloader, config['CLI']['mode'])
    
    # Extract weather condition from filename (remove 'test_' prefix)
    weather_condition = file.replace('test_', '').replace('.txt', '')
    
    # Store results with weather condition as key
    all_results[weather_condition] = results
    
    print(f'Testing completed for {weather_condition}')
    print()

# Save all results to a single JSON file
# Generate test UUID
test_uuid = str(uuid.uuid4())

# Extract epoch number and UUID from model path
model_path = get_model_path(config)
if not model_path:
    print("No model checkpoint found. Please train the model first.")
    exit(1)
epoch_match = re.search(r'epoch_(\d+)_([a-f0-9\-]+)\.pth', model_path)
if epoch_match:
    epoch_num = epoch_match.group(1)
    epoch_uuid = epoch_match.group(2)
else:
    # Fallback for old checkpoint format
    epoch_match = re.search(r'checkpoint_(\d+)\.pth', model_path)
    epoch_num = epoch_match.group(1) if epoch_match else '0'
    epoch_uuid = test_uuid  # Use test_uuid as fallback

# Create simplified results structure
combined_results = {
    'timestamp': datetime.datetime.now().isoformat(),
    'epoch': int(epoch_num),
    'epoch_uuid': epoch_uuid,
    'test_uuid': test_uuid,
    'test_results': all_results
}

# Save to combined results file with epoch number and UUID
import os
test_results_dir = config['Log']['logdir'] + 'test_results'
os.makedirs(test_results_dir, exist_ok=True)

if epoch_uuid:
    combined_json_path = os.path.join(test_results_dir, f'epoch_{epoch_num}_{epoch_uuid}.json')
else:
    combined_json_path = os.path.join(test_results_dir, f'epoch_{epoch_num}_test_results.json')
    
with open(combined_json_path, 'w') as f:
    json.dump(combined_results, f, indent=2)

print(f'All test results saved to: {combined_json_path}')
if epoch_uuid:
    print(f'Epoch UUID: {epoch_uuid}')
print(f'Test UUID: {test_uuid}')
print(f'Completed testing on {len(test_data_files)} test files')

# Upload test results to vision service
print("Uploading test results to vision service...")
upload_success = send_test_results_from_file(combined_json_path)
if upload_success:
    print("✅ Test results successfully uploaded to vision service")
else:
    print("❌ Failed to upload test results to vision service")
