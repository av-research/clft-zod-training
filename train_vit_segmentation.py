import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from dataset import ZODDataset
from model import ViTSegmentation
from test_segmentation import test_model

# Training Function
def train_model(model, dataloader, num_epochs, lr, save_dir, mode, starting_epoch=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Formula: weight_i = 1 / (freq_i / total_pixels), then normalized so sum(weights) = num_classes
    # Classes (in order): [background, vehicle, pedestrian, cyclist, sign]
    # Frequencies: [~95%, ~2.8%, ~1.7%, ~2.3%, ~1.9%] (approximate from ZOD dataset analysis)
    class_weights = torch.tensor([0.0120, 0.0575, 0.3580, 0.2603, 0.3122]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(starting_epoch, num_epochs):
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
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
        
        # Periodic gradient checks during training
        if (epoch + 1) % 100 == 0:  # Every 10 epochs
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
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 100 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}_{mode}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'mode': mode
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
            
            # Run test and save results
            print(f"Running test at epoch {epoch + 1}...")
            results_save_path = f'./model_results/test_results_epoch_{epoch+1}_{mode}.json'
            test_model(model, dataloader, num_classes=5, save_path=results_save_path, checkpoint_path=checkpoint_path)

# Main Script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ViT Segmentation Model')
    args = parser.parse_args()
    
    os.makedirs('./model_results', exist_ok=True)
    
    # Paths (adjust as needed)
    data_dir = './zod_dataset'
    split_file = './zod_dataset/splits_zod/all.txt'  # Use all.txt for full dataset
    
    # Dataset and Dataloader
    dataset = ZODDataset(data_dir, split_file)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    
    # Hardcoded mode
    mode = 'cross_fusion'
    model = ViTSegmentation(mode, num_classes=5)
    
    # Resume from checkpoint if exists
    checkpoint_path = './model_path/checkpoint_epoch_430_cross_fusion.pth'
    starting_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        starting_epoch = checkpoint.get('epoch', 0)
        print(f"Resumed training from epoch {starting_epoch}")
    
    print(f"Training with mode: {mode}")
    
    train_model(model, dataloader, num_epochs=1000, lr=8e-5, save_dir='./model_path', mode=mode, starting_epoch=starting_epoch)
