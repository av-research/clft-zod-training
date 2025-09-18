import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import argparse
import glob

# Enable TensorFloat32 for better performance on Ampere GPUs
torch.set_float32_matmul_precision('high')

# Set optimal number of threads for CPU operations
torch.set_num_threads(8)

from dataset import GenericDataset
from model import ViTSegmentation
from test_segmentation import test_all_conditions

def delete_old_checkpoints(save_dir, current_checkpoint_path):
    """
    Delete all previous checkpoint files, keeping only the current one.
    """
    checkpoint_pattern = os.path.join(save_dir, 'checkpoint_epoch_*.pth')
    existing_checkpoints = glob.glob(checkpoint_pattern)
    
    if existing_checkpoints:
        # Sort by modification time, oldest first
        existing_checkpoints.sort(key=os.path.getmtime)
        # Remove all but keep the current one we just saved
        for old_checkpoint in existing_checkpoints:
            if old_checkpoint != current_checkpoint_path and os.path.exists(old_checkpoint):
                try:
                    os.remove(old_checkpoint)
                    print(f'Previous checkpoint deleted: {old_checkpoint}')
                except OSError as e:
                    print(f'Warning: Could not delete checkpoint {old_checkpoint}: {e}')

def check_gradients(model, epoch):
    """
    Check gradient norms for different parts of the model.
    """
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

def save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, config):
    """
    Save model checkpoint with all necessary state information.
    Automatically delete previous checkpoint to save disk space.
    """
    checkpoint_path = os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch+1}_{config["mode"]}.pth')

    # Delete all previous checkpoints
    delete_old_checkpoints(config['save_dir'], checkpoint_path)

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_loss,
        'mode': config['mode']
    }, checkpoint_path)
    print(f'Checkpoint saved: {checkpoint_path}')
    
    
    return checkpoint_path

# Training Function
def train_model(model, dataloader, config, starting_epoch=0, scaler=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    class_weights = torch.tensor(config['class_weights'], dtype=torch.float).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    
    # Use exponential decay like the old training (more stable)
    def lr_lambda(epoch):
        return config['lr_momentum'] ** epoch
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
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
            
            # Mixed precision forward pass
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(rgb, lidar, config['mode'])
                    
                    # Debug: Check output (only first batch)
                    if epoch == 0 and batch_count == 0:
                        print(f"Output shape: {outputs.shape}, range: {outputs.min().item():.3f} to {outputs.max().item():.3f}")
                    
                    loss = criterion(outputs, anno)
                
                if torch.isnan(loss):
                    print(f"NaN loss at epoch {epoch+1}, batch {batch_count+1}")
                    break
                
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training without mixed precision
                outputs = model(rgb, lidar, config['mode'])
                
                # Debug: Check output (only first batch)
                if epoch == 0 and batch_count == 0:
                    print(f"Output shape: {outputs.shape}, range: {outputs.min().item():.3f} to {outputs.max().item():.3f}")
                
                loss = criterion(outputs, anno)
                
                if torch.isnan(loss):
                    print(f"NaN loss at epoch {epoch+1}, batch {batch_count+1}")
                    break
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{config["num_epochs"]}, Loss: {avg_loss:.4f}')
        
        # Periodic gradient checks during training
        if (epoch + 1) % config['checkpoint_interval'] == 0:
            check_gradients(model, epoch)
        
        if torch.isnan(torch.tensor(avg_loss)):
            print("Training stopped due to NaN loss")
            break
        
        # Step the scheduler
        scheduler.step()
        
        # Save checkpoint every checkpoint_interval epochs
        if (epoch + 1) % config['checkpoint_interval'] == 0:
            checkpoint_path = save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, config)
            
            # Run comprehensive test and save results for all conditions
            print(f"Running comprehensive test at epoch {epoch + 1}...")
            test_all_conditions(model, config, checkpoint_path, epoch=epoch+1, results_prefix="test_results")

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

    # Dataset and Dataloader - Set training=True to enable augmentations
    dataset = GenericDataset(config, training=True)  # Augmentations enabled
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    
    # Hardcoded mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Additional performance optimizations
    if device.type == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for matrix operations
        torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for cuDNN operations
    
    model = ViTSegmentation(mode, num_classes=num_classes)
    
    # Enable mixed precision for faster training
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # Compile model for PyTorch 2.0+ optimization (if available)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress compilation warnings
            model = torch.compile(model, mode='default')
        print("Model compiled for optimization")
    except Exception as e:
        print(f"Model compilation not available, continuing with standard model")
    
    # Resume from checkpoint if exists
    starting_epoch = config.get('starting_epoch', 0)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        # Note: starting_epoch is set from config, not from checkpoint
        print(f"Loaded model from checkpoint, starting from epoch {starting_epoch}")
    
    print(f"Training on {args.dataset.upper()} dataset with mode: {config['mode']}")
    
    train_model(model, dataloader, config, starting_epoch=starting_epoch, scaler=scaler)
