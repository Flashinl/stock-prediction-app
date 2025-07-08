#!/usr/bin/env python3
"""
PyTorch ASL Model Training Script
Trains a neural network on the mega ASL dataset with 982 classes and 25,000 samples.
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ASLDataset(Dataset):
    """Custom dataset for ASL hand landmark data."""
    
    def __init__(self, metadata_path, data_root):
        self.data_root = Path(data_root)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load vocabulary
        vocab_path = self.data_root / 'vocabulary.json'
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        
        logger.info(f"Loaded dataset with {len(self.metadata)} samples and {self.vocab['num_classes']} classes")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]

        # Load the numpy array (video frames)
        data_path = self.data_root / item['video_path']
        video_frames = np.load(data_path)  # Shape: (30, 224, 224, 3)

        # Convert to tensor and rearrange dimensions for PyTorch
        # From (T, H, W, C) to (T, C, H, W)
        video_frames = torch.FloatTensor(video_frames).permute(0, 3, 1, 2)

        # Get label
        label = item['label']

        return video_frames, label

class ASLVideoClassifier(nn.Module):
    """3D CNN-based ASL video classification model."""

    def __init__(self, num_classes, num_frames=30):
        super(ASLVideoClassifier, self).__init__()

        self.num_frames = num_frames

        # 3D CNN backbone
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

        # Flatten
        x = x.view(x.size(0), -1)

        # Classification
        output = self.classifier(x)

        return output

def train_model():
    """Main training function."""
    
    # Configuration
    config = {
        'batch_size': 8,  # Reduced for video data
        'learning_rate': 1e-4,
        'num_epochs': 30,  # Reduced for faster training
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_every': 5,
        'val_split': 0.2
    }
    
    logger.info(f"Training configuration: {config}")
    logger.info(f"Using device: {config['device']}")
    
    # Load dataset
    dataset = ASLDataset('../data/mega_asl/metadata.json', '../data/mega_asl')
    
    # Split dataset
    val_size = int(len(dataset) * config['val_split'])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
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
            
            # Update progress bar
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
        
        # Update learning rate
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
                'config': config
            }, 'models/best_asl_model.pth')
            logger.info(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        
        # Save checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, f'models/checkpoint_epoch_{epoch+1}.pth')
    
    logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save final training results
    results = {
        'best_val_accuracy': best_val_acc,
        'final_val_accuracy': val_accuracies[-1],
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'config': config,
        'num_classes': dataset.vocab['num_classes'],
        'total_samples': len(dataset)
    }
    
    with open('models/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, results

if __name__ == "__main__":
    # Create models directory
    Path('models').mkdir(exist_ok=True)
    
    # Start training
    model, results = train_model()
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"✅ Best Validation Accuracy: {results['best_val_accuracy']:.2f}%")
    print(f"📊 Classes: {results['num_classes']}")
    print(f"📊 Total Samples: {results['total_samples']}")
    print(f"💾 Model saved to: models/best_asl_model.pth")
    print("="*60)
