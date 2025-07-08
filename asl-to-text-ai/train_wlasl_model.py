#!/usr/bin/env python3
"""
WLASL Model Training Script
Trains a neural network on the real WLASL dataset.
"""

import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import torchvision.transforms as transforms

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WLASLDataset(Dataset):
    """Dataset for WLASL video data with data augmentation."""

    def __init__(self, metadata_path, data_root, augment=True):
        self.data_root = Path(data_root)
        self.augment = augment

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Load vocabulary
        vocab_path = self.data_root / 'WLASL' / 'vocabulary.json'
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)

        # Data augmentation transforms
        if self.augment:
            self.spatial_transforms = transforms.Compose([
                transforms.RandomRotation(degrees=10),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
            ])

        logger.info(f"Loaded WLASL dataset: {len(self.metadata)} samples, {self.vocab['num_classes']} classes")
        logger.info(f"Data augmentation: {'ENABLED' if augment else 'DISABLED'}")

    def __len__(self):
        return len(self.metadata)

    def apply_temporal_augmentation(self, frames):
        """Apply temporal augmentation to video frames."""
        num_frames = frames.shape[0]

        # Random temporal cropping (take random subset of frames)
        if num_frames > 20:
            start_idx = random.randint(0, num_frames - 20)
            frames = frames[start_idx:start_idx + 20]

            # Pad back to 30 frames
            while frames.shape[0] < 30:
                frames = torch.cat([frames, frames[-1:]], dim=0)

        # Random frame dropout (replace some frames with previous frame)
        if random.random() < 0.3:
            dropout_indices = random.sample(range(1, frames.shape[0]),
                                          k=min(3, frames.shape[0] - 1))
            for idx in dropout_indices:
                frames[idx] = frames[idx - 1]

        return frames

    def apply_spatial_augmentation(self, frames):
        """Apply spatial augmentation to each frame."""
        augmented_frames = []

        for frame in frames:
            # Convert from (C, H, W) to PIL format for transforms
            frame_pil = transforms.ToPILImage()(frame)

            # Apply spatial transforms
            frame_aug = self.spatial_transforms(frame_pil)

            # Convert back to tensor
            frame_tensor = transforms.ToTensor()(frame_aug)
            augmented_frames.append(frame_tensor)

        return torch.stack(augmented_frames)

    def __getitem__(self, idx):
        item = self.metadata[idx]

        # Load frames
        frames_path = self.data_root / item['frames_path']
        frames = np.load(frames_path)  # Shape: (30, 224, 224, 3)

        # Convert to tensor and rearrange for PyTorch (T, C, H, W)
        frames = torch.FloatTensor(frames).permute(0, 3, 1, 2)

        # Apply augmentation during training
        if self.augment:
            # Apply temporal augmentation
            frames = self.apply_temporal_augmentation(frames)

            # Apply spatial augmentation
            frames = self.apply_spatial_augmentation(frames)

        label = item['label']

        return frames, label

class ASLVideoClassifier(nn.Module):
    """3D CNN for ASL video classification."""
    
    def __init__(self, num_classes, num_frames=30):
        super(ASLVideoClassifier, self).__init__()
        
        self.num_frames = num_frames
        
        # 3D CNN layers
        self.conv3d_1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        self.conv3d_2 = nn.Conv3d(64, 128, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1))
        
        self.conv3d_3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv3d_4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(512)
        self.pool4 = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # x shape: (batch_size, num_frames, channels, height, width)
        # Rearrange to (batch_size, channels, num_frames, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        
        # 3D CNN layers
        x = self.relu(self.bn1(self.conv3d_1(x)))
        x = self.pool1(x)
        
        x = self.relu(self.bn2(self.conv3d_2(x)))
        x = self.pool2(x)
        
        x = self.relu(self.bn3(self.conv3d_3(x)))
        x = self.pool3(x)
        
        x = self.relu(self.bn4(self.conv3d_4(x)))
        x = self.pool4(x)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        
        return output

def train_model():
    """Main training function."""
    
    # Configuration
    config = {
        'batch_size': 4,  # Small batch size for video data
        'learning_rate': 1e-4,
        'num_epochs': 25,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_every': 5,
        'val_split': 0.2
    }
    
    logger.info(f"Training configuration: {config}")
    logger.info(f"Using device: {config['device']}")
    
    # Check if dataset exists
    metadata_path = Path('data/real_asl/WLASL/processed_metadata.json')
    if not metadata_path.exists():
        logger.error("WLASL dataset not found. Run download_preprocessed_wlasl.py first!")
        return None, None
    
    # Load dataset
    dataset = WLASLDataset(metadata_path, 'data/real_asl')
    
    if len(dataset) < 10:
        logger.error(f"Dataset too small: {len(dataset)} samples. Need at least 10.")
        return None, None
    
    # Split dataset
    val_size = max(1, int(len(dataset) * config['val_split']))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = ASLVideoClassifier(num_classes=dataset.vocab['num_classes'])
    model = model.to(config['device'])
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    # Training loop
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    logger.info("Starting training...")
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
        
        for batch_idx, (data, targets) in enumerate(train_pbar):
            data, targets = data.to(config['device']), targets.to(config['device'])
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Val]")
            
            for data, targets in val_pbar:
                data, targets = data.to(config['device']), targets.to(config['device'])
                outputs = model(data)
                
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                val_pbar.set_postfix({
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        
        scheduler.step()
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config,
                'vocab': dataset.vocab
            }, 'models/best_wlasl_model.pth')
            logger.info(f"New best model saved: {val_acc:.2f}%")
        
        # Save checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config,
                'vocab': dataset.vocab
            }, f'models/wlasl_checkpoint_epoch_{epoch+1}.pth')
    
    logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save final results
    results = {
        'best_val_accuracy': best_val_acc,
        'final_val_accuracy': val_accuracies[-1],
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'config': config,
        'num_classes': dataset.vocab['num_classes'],
        'total_samples': len(dataset),
        'dataset_type': 'WLASL_Real'
    }
    
    with open('models/wlasl_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, results

if __name__ == "__main__":
    # Create models directory
    Path('models').mkdir(exist_ok=True)
    
    # Start training
    model, results = train_model()
    
    if results:
        print("\n" + "="*60)
        print("WLASL TRAINING COMPLETE!")
        print("="*60)
        print(f"Best Validation Accuracy: {results['best_val_accuracy']:.2f}%")
        print(f"Classes: {results['num_classes']}")
        print(f"Total Samples: {results['total_samples']}")
        print(f"Model saved to: models/best_wlasl_model.pth")
        print(f"Dataset: Real WLASL videos (not synthetic)")
        print("="*60)
    else:
        print("\nTraining failed. Please check the dataset.")
