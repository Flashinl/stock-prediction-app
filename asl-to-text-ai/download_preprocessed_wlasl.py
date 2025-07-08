#!/usr/bin/env python3
"""
Download Pre-processed WLASL Dataset from Kaggle
Uses the already processed WLASL dataset with 25 samples per word.
"""

import os
import json
import numpy as np
from pathlib import Path
import subprocess
import sys
import zipfile
import shutil
from collections import defaultdict

class PreprocessedWLASLDownloader:
    """Download and organize pre-processed WLASL dataset."""
    
    def __init__(self, data_dir="data/real_asl"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        print("Pre-processed WLASL Dataset Downloader")
        print("=" * 50)
        print("Using Kaggle pre-processed WLASL dataset")
        print("Target: 25 samples per word, 2000 words")
        print("=" * 50)
    
    def check_kaggle_setup(self):
        """Check if Kaggle API is set up."""
        try:
            result = subprocess.run(['kaggle', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("Kaggle API found:", result.stdout.strip())
                return True
            else:
                print("Kaggle API not working properly")
                return False
        except FileNotFoundError:
            print("Kaggle API not installed")
            return False
    
    def install_kaggle(self):
        """Install Kaggle API."""
        print("Installing Kaggle API...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kaggle'])
            print("Kaggle API installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("Failed to install Kaggle API")
            return False
    
    def setup_kaggle_credentials(self):
        """Guide user to set up Kaggle credentials."""
        print("\nKaggle API Setup Required:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Download kaggle.json file")
        print("4. Place it in: C:\\Users\\{username}\\.kaggle\\kaggle.json")
        print("5. Or set environment variables:")
        print("   KAGGLE_USERNAME=your_username")
        print("   KAGGLE_KEY=your_api_key")
        
        input("\nPress Enter after setting up Kaggle credentials...")
        
        # Test credentials
        try:
            result = subprocess.run(['kaggle', 'datasets', 'list', '--max-size', '1'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("Kaggle credentials working!")
                return True
            else:
                print("Kaggle credentials not working:", result.stderr)
                return False
        except Exception as e:
            print(f"Error testing Kaggle credentials: {e}")
            return False
    
    def download_dataset(self):
        """Download the pre-processed WLASL dataset."""
        print("\nDownloading pre-processed WLASL dataset...")
        
        dataset_name = "risangbaskoro/wlasl-processed"
        download_dir = self.data_dir / "kaggle_download"
        download_dir.mkdir(exist_ok=True)
        
        try:
            # Download dataset
            cmd = ['kaggle', 'datasets', 'download', '-d', dataset_name, '-p', str(download_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Download failed: {result.stderr}")
                return False
            
            print("Download completed successfully!")
            
            # Find and extract zip file
            zip_files = list(download_dir.glob("*.zip"))
            if not zip_files:
                print("No zip file found in download")
                return False
            
            zip_file = zip_files[0]
            extract_dir = self.data_dir / "wlasl_extracted"
            
            print(f"Extracting {zip_file.name}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            print(f"Extracted to: {extract_dir}")
            
            # Clean up zip file
            zip_file.unlink()
            
            return extract_dir
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False
    
    def organize_dataset(self, extract_dir):
        """Organize the dataset to ensure 25 samples per word."""
        print("\nOrganizing dataset with 25 samples per word...")
        
        # Find all video/frame files
        video_files = []
        for ext in ['*.npy', '*.mp4', '*.avi']:
            video_files.extend(extract_dir.rglob(ext))
        
        print(f"Found {len(video_files)} files")
        
        # Group by word/class
        word_files = defaultdict(list)
        
        for file_path in video_files:
            # Extract word from filename (assuming format: word_id.ext)
            filename = file_path.stem
            if '_' in filename:
                word = filename.split('_')[0]
            else:
                word = filename
            
            word_files[word].append(file_path)
        
        print(f"Found {len(word_files)} unique words")
        
        # Create organized dataset
        organized_dir = self.data_dir / "WLASL"
        frames_dir = organized_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        processed_data = []
        word_to_label = {}
        current_label = 0
        
        for word_idx, (word, files) in enumerate(word_files.items()):
            print(f"[{word_idx+1}/{len(word_files)}] Processing word: '{word}'")
            print(f"  Available files: {len(files)}")
            
            # Assign label
            word_to_label[word] = current_label
            current_label += 1
            
            # Take up to 25 samples
            selected_files = files[:25]
            actual_samples = 0
            
            for file_idx, file_path in enumerate(selected_files):
                try:
                    # Copy/process file
                    new_filename = f"{word}_{file_idx:03d}.npy"
                    new_path = frames_dir / new_filename
                    
                    if file_path.suffix == '.npy':
                        # Copy numpy file directly
                        shutil.copy2(file_path, new_path)
                    else:
                        # Convert video to frames (if needed)
                        # For now, skip non-numpy files
                        continue
                    
                    # Add to processed data
                    processed_data.append({
                        'gloss': word,
                        'label': word_to_label[word],
                        'video_id': f"{word}_{file_idx:03d}",
                        'frames_path': f"WLASL/frames/{new_filename}",
                        'instance_id': file_idx,
                        'signer_id': 0
                    })
                    
                    actual_samples += 1
                    
                except Exception as e:
                    print(f"    Error processing {file_path.name}: {e}")
                    continue
            
            print(f"  Successfully processed: {actual_samples} samples")
        
        # Create vocabulary
        vocabulary = {
            'num_classes': len(word_to_label),
            'word_to_label': word_to_label,
            'label_to_word': {v: k for k, v in word_to_label.items()},
            'total_samples': len(processed_data)
        }
        
        # Save metadata
        metadata_path = organized_dir / "processed_metadata.json"
        vocab_path = organized_dir / "vocabulary.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        with open(vocab_path, 'w') as f:
            json.dump(vocabulary, f, indent=2)
        
        print(f"\nDataset organization complete!")
        print(f"Total words: {vocabulary['num_classes']}")
        print(f"Total samples: {vocabulary['total_samples']}")
        print(f"Average samples per word: {vocabulary['total_samples']/vocabulary['num_classes']:.1f}")
        print(f"Metadata saved: {metadata_path}")
        print(f"Vocabulary saved: {vocab_path}")
        
        return {
            'metadata_path': str(metadata_path),
            'vocabulary_path': str(vocab_path),
            'frames_dir': str(frames_dir),
            'num_classes': vocabulary['num_classes'],
            'total_samples': vocabulary['total_samples']
        }

def main():
    """Main function."""
    print("Pre-processed WLASL Dataset Setup")
    print("=" * 50)
    
    downloader = PreprocessedWLASLDownloader()
    
    # Check Kaggle setup
    if not downloader.check_kaggle_setup():
        if not downloader.install_kaggle():
            print("Failed to install Kaggle API")
            return 1
    
    # Setup credentials if needed
    if not downloader.setup_kaggle_credentials():
        print("Kaggle credentials not set up properly")
        return 1
    
    # Download dataset
    extract_dir = downloader.download_dataset()
    if not extract_dir:
        print("Failed to download dataset")
        return 1
    
    # Organize dataset
    result = downloader.organize_dataset(extract_dir)
    if result:
        print("\nSUCCESS!")
        print(f"Dataset ready for training with {result['total_samples']} samples")
        print(f"Classes: {result['num_classes']}")
        return 0
    else:
        print("Failed to organize dataset")
        return 1

if __name__ == "__main__":
    exit(main())
