import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import argparse

from dataset import GenericDataset
from model import ViTSegmentation
from test_segmentation import test_model

# Training Function
def train_model(model, dataloader, config, starting_epoch=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    class_weights = torch.tensor(config['class_weights'], dtype=torch.float).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    for epoch in range(starting_epoch, config['num_epochs']):
        model.train()
        epoch_loss = 0
        batch_count = 0
        for rgb, lidar, anno in tqdm(dataloader):
            rgb, lidar, anno = rgb.to(device), lidar.to(device), anno.to(device)
            
            # Debug: Check shapes and values
            if epoch == 0 and batch_count == 0:  # First batch
                print(f"RGB shape: {rgb.shape}, Lidar shape: {lidar.shape}, Anno shape: {anno.shape}")
                print(f"RGB range: {rgb.min().item():.3f} to {rgb.max().item():.3f}")
                print(f"Lidar range: {lidar.min().item():.3f} to {lidar.max().item():.3f}")
                print(f"Anno unique: {torch.unique(anno)}")
            
            optimizer.zero_grad()
            outputs = model(rgb, lidar)
            
            # Debug: Check output
            if epoch == 0 and batch_count == 0:
                print(f"Output shape: {outputs.shape}, range: {outputs.min().item():.3f} to {outputs.max().item():.3f}")
            
            loss = criterion(outputs, anno)
            
            if torch.isnan(loss):
                print(f"NaN loss at epoch {epoch+1}, batch {batch_count+1}")
                print(f"Outputs: {outputs}")
                print(f"Anno: {anno}")
                break
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{config["num_epochs"]}, Loss: {avg_loss:.4f}')
        
        # Periodic gradient checks during training
        if (epoch + 1) % config['gradient_check_interval'] == 0:  # Every X epochs
            print("Gradient checks at epoch", epoch + 1)
            vit_grads = []
            fusion_grads = []
            head_grads = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if 'vit' in name:
                        vit_grads.append(grad_norm)
                    elif 'fusion' in name:
                        fusion_grads.append(grad_norm)
                    elif 'head' in name:
                        head_grads.append(grad_norm)
            
            if vit_grads:
                print(f"  ViT: avg_grad_norm = {sum(vit_grads)/len(vit_grads):.6f}, max = {max(vit_grads):.6f}")
            if fusion_grads:
                print(f"  Fusion: avg_grad_norm = {sum(fusion_grads)/len(fusion_grads):.6f}, max = {max(fusion_grads):.6f}")
            if head_grads:
                print(f"  Head: avg_grad_norm = {sum(head_grads)/len(head_grads):.6f}, max = {max(head_grads):.6f}")
            print()
        
        if torch.isnan(torch.tensor(avg_loss)):
            print("Training stopped due to NaN loss")
            break
        
        # Step the scheduler
        scheduler.step()
        
        # Save checkpoint every checkpoint_interval epochs
        if (epoch + 1) % config['checkpoint_interval'] == 0:
            checkpoint_path = os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch+1}_{config["mode"]}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'mode': config['mode']
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
            
            # Run test and save results
            print(f"Running test at epoch {epoch + 1}...")
            results_save_path = f'./model_results/test_results_epoch_{epoch+1}_{config["mode"]}.json'
            test_model(model, dataloader, num_classes=config['num_classes'], save_path=results_save_path, checkpoint_path=checkpoint_path, class_names=config['class_names'])

# Main Script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('--dataset', type=str, default='zod', choices=['zod', 'waymo'], help='Dataset to use (zod or waymo)')
    args = parser.parse_args()
    
    os.makedirs('./model_results', exist_ok=True)
    
    # Load dataset configuration from JSON
    config_file = f'config_{args.dataset}.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    data_dir = config['data_dir']
    split_file = config['split_file']
    preload = config['preload']
    batch_size = config['batch_size']
    checkpoint_path = config['checkpoint_path']
    num_classes = config['num_classes']
    checkpoint_interval = config['checkpoint_interval']
    class_weights = config['class_weights']
    lidar_mean = config['lidar_mean']
    lidar_std = config['lidar_std']
    rgb_mean = config['rgb_mean']
    rgb_std = config['rgb_std']
    num_epochs = config['num_epochs']
    lr = config['lr']
    save_dir = config['save_dir']
    mode = config['mode']
    camera_mask_value = config['camera_mask_value']
    
    # Dataset and Dataloader
    dataset = GenericDataset(config)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    
    # Hardcoded mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ViTSegmentation(mode, num_classes=num_classes)
    
    # Resume from checkpoint if exists
    starting_epoch = config.get('starting_epoch', 0)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        # Note: starting_epoch is set from config, not from checkpoint
        print(f"Loaded model from checkpoint, starting from epoch {starting_epoch}")
    
    print(f"Training on {args.dataset.upper()} dataset with mode: {config['mode']}")
    
    train_model(model, dataloader, config, starting_epoch=starting_epoch)
