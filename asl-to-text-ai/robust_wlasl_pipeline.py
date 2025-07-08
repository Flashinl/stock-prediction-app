#!/usr/bin/env python3
"""
Robust WLASL Pipeline with Better Error Handling
Handles download and extraction issues more gracefully.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import subprocess
import sys
import zipfile
import shutil
from collections import defaultdict
import logging
import time
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class RobustWLASLPipeline:
    """Robust WLASL pipeline with better error handling."""
    
    def __init__(self, data_dir="data/real_asl"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        print("ROBUST WLASL PIPELINE - 25 SAMPLES PER WORD")
        print("=" * 60)
        print("With improved error handling and data augmentation")
        print("=" * 60)
    
    def setup_kaggle_credentials(self):
        """Setup Kaggle credentials from Downloads folder."""
        print("\nStep 1: Setting up Kaggle credentials...")
        
        # Check if kaggle.json exists in Downloads
        downloads_kaggle = Path.home() / "Downloads" / "kaggle.json"
        kaggle_dir = Path.home() / ".kaggle"
        target_kaggle = kaggle_dir / "kaggle.json"
        
        if downloads_kaggle.exists() and not target_kaggle.exists():
            print("Moving kaggle.json from Downloads to .kaggle directory...")
            kaggle_dir.mkdir(exist_ok=True)
            shutil.copy2(downloads_kaggle, target_kaggle)
            print("Kaggle credentials set up successfully")
        
        # Install kaggle if needed
        try:
            import kaggle
            print("Kaggle API already installed")
        except ImportError:
            print("Installing Kaggle API...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kaggle'])
        
        # Test credentials
        try:
            result = subprocess.run(['kaggle', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"Kaggle API working: {result.stdout.strip()}")
                return True
            else:
                print("Kaggle credentials test failed")
                return False
        except Exception as e:
            print(f"Kaggle setup error: {e}")
            return False
    
    def download_with_retry(self, max_retries=3):
        """Download dataset with retry logic."""
        print("\nStep 2: Downloading WLASL dataset...")
        
        dataset_name = "risangbaskoro/wlasl-processed"
        download_dir = self.data_dir / "kaggle_download"
        extract_dir = self.data_dir / "wlasl_extracted"
        
        # Check if already extracted
        if extract_dir.exists() and (extract_dir / "WLASL_v0.3.json").exists():
            print("Dataset already downloaded and extracted")
            return extract_dir
        
        # Clean up any partial downloads
        if download_dir.exists():
            shutil.rmtree(download_dir)
        download_dir.mkdir(exist_ok=True)
        
        for attempt in range(max_retries):
            try:
                print(f"Download attempt {attempt + 1}/{max_retries}...")
                
                # Download with timeout
                cmd = ['kaggle', 'datasets', 'download', '-d', dataset_name, '-p', str(download_dir)]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    print(f"Download failed: {result.stderr}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    else:
                        return None
                
                # Find zip file
                zip_files = list(download_dir.glob("*.zip"))
                if not zip_files:
                    print("No zip file found after download")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    else:
                        return None
                
                zip_file = zip_files[0]
                print(f"Found zip file: {zip_file.name} ({zip_file.stat().st_size / 1024 / 1024:.1f} MB)")
                
                # Extract with better error handling
                if extract_dir.exists():
                    shutil.rmtree(extract_dir)
                extract_dir.mkdir(exist_ok=True)
                
                print("Extracting dataset...")
                try:
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        # Extract files one by one to handle errors better
                        for member in zip_ref.namelist():
                            try:
                                zip_ref.extract(member, extract_dir)
                            except Exception as e:
                                print(f"Warning: Failed to extract {member}: {e}")
                                continue
                    
                    # Verify extraction
                    if (extract_dir / "WLASL_v0.3.json").exists():
                        print("Extraction successful")
                        zip_file.unlink()  # Clean up zip file
                        return extract_dir
                    else:
                        print("Extraction incomplete - missing key files")
                        if attempt < max_retries - 1:
                            time.sleep(5)
                            continue
                        else:
                            return None
                
                except zipfile.BadZipFile:
                    print("Corrupted zip file")
                    zip_file.unlink()
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    else:
                        return None
                
            except subprocess.TimeoutExpired:
                print("Download timed out")
                if attempt < max_retries - 1:
                    time.sleep(10)
                    continue
                else:
                    return None
            except Exception as e:
                print(f"Download error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                else:
                    return None
        
        return None
    
    def load_mappings_safely(self, extract_dir):
        """Load mappings with error handling."""
        print("\nStep 3: Loading word mappings...")
        
        try:
            wlasl_path = extract_dir / "WLASL_v0.3.json"
            nslt_2000_path = extract_dir / "nslt_2000.json"
            
            if not wlasl_path.exists():
                print(f"Error: {wlasl_path} not found")
                return None, None, None
            
            if not nslt_2000_path.exists():
                print(f"Error: {nslt_2000_path} not found")
                return None, None, None
            
            # Load WLASL data
            with open(wlasl_path, 'r', encoding='utf-8') as f:
                wlasl_data = json.load(f)
            
            # Load video mappings
            with open(nslt_2000_path, 'r', encoding='utf-8') as f:
                video_mappings = json.load(f)
            
            print(f"Loaded WLASL data: {len(wlasl_data)} words")
            print(f"Loaded video mappings: {len(video_mappings)} videos")
            
            # Create mappings
            action_to_word = {}
            for word_data in wlasl_data:
                word = word_data['gloss']
                for instance in word_data['instances']:
                    video_id = instance['video_id']
                    if video_id in video_mappings:
                        action_index = video_mappings[video_id]['action'][0]
                        action_to_word[action_index] = word
            
            # Create video to word mapping
            video_to_word = {}
            for video_id, data in video_mappings.items():
                action_index = data['action'][0]
                if action_index in action_to_word:
                    video_to_word[video_id] = action_to_word[action_index]
            
            # Create word to label mapping
            unique_words = sorted(set(video_to_word.values()))
            word_to_label = {word: idx for idx, word in enumerate(unique_words)}
            
            print(f"Successfully mapped {len(video_to_word)} videos to {len(unique_words)} words")
            
            return video_to_word, word_to_label, unique_words
            
        except Exception as e:
            print(f"Error loading mappings: {e}")
            return None, None, None

    def extract_video_frames(self, video_path, target_frames=30):
        """Extract frames from video with error handling."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return None

            # Calculate frame indices
            if total_frames <= target_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)

            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret and frame is not None:
                    frame = cv2.resize(frame, (224, 224))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(frame)

            cap.release()

            # Pad if needed
            while len(frames) < target_frames:
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.float32))

            return np.array(frames[:target_frames])

        except Exception as e:
            logger.debug(f"Error processing video: {e}")
            return None

    def augment_frames(self, frames, augmentation_type):
        """Apply data augmentation to frames."""
        augmented = frames.copy()

        if augmentation_type == 'horizontal_flip':
            augmented = np.flip(augmented, axis=2)

        elif augmentation_type == 'brightness':
            factor = np.random.uniform(0.7, 1.3)
            augmented = np.clip(augmented * factor, 0, 1)

        elif augmentation_type == 'contrast':
            factor = np.random.uniform(0.8, 1.2)
            mean = np.mean(augmented)
            augmented = np.clip((augmented - mean) * factor + mean, 0, 1)

        elif augmentation_type == 'rotation':
            angle = np.random.uniform(-10, 10)
            h, w = augmented.shape[1:3]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            for i in range(augmented.shape[0]):
                for c in range(3):
                    augmented[i, :, :, c] = cv2.warpAffine(
                        augmented[i, :, :, c], M, (w, h),
                        borderMode=cv2.BORDER_REFLECT
                    )

        elif augmentation_type == 'temporal_shift':
            shift = np.random.randint(1, min(5, len(augmented)))
            augmented = np.roll(augmented, shift, axis=0)

        elif augmentation_type == 'noise':
            noise = np.random.normal(0, 0.02, augmented.shape)
            augmented = np.clip(augmented + noise, 0, 1)

        return augmented
    
    def process_with_augmentation(self, extract_dir, video_to_word, word_to_label, unique_words):
        """Process videos with augmentation to ensure 25 samples per word."""
        print(f"\nStep 4: Processing with augmentation...")
        print(f"Target: {len(unique_words)} words × 25 samples = {len(unique_words) * 25} total")
        
        # Create output directories
        wlasl_dir = self.data_dir / "WLASL"
        frames_dir = wlasl_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Group videos by word
        word_videos = defaultdict(list)
        videos_dir = extract_dir / "videos"
        
        if not videos_dir.exists():
            print(f"Error: Videos directory not found: {videos_dir}")
            return False
        
        print("Grouping videos by word...")
        for video_id, word in video_to_word.items():
            video_file = videos_dir / f"{video_id}.mp4"
            if video_file.exists():
                word_videos[word].append((video_id, video_file))
        
        print(f"Found videos for {len(word_videos)} words")
        
        # Process each word (simplified for now - just count available videos)
        processed_data = []
        total_samples = 0
        
        for word_idx, word in enumerate(unique_words):  # Process ALL 2000 words
            print(f"\n[{word_idx+1}/{len(unique_words)}] Processing word: '{word}'")
            
            if word not in word_videos:
                print(f"  No videos found - SKIPPING word '{word}'")
                continue

            available_videos = word_videos[word]
            print(f"  Found {len(available_videos)} real videos")

            # Process real videos first
            real_frames_list = []
            for video_id, video_path in available_videos:
                frames = self.extract_video_frames(video_path)
                if frames is not None:
                    real_frames_list.append((video_id, frames))

            print(f"  Successfully processed {len(real_frames_list)} real videos")

            if len(real_frames_list) == 0:
                print(f"  No valid videos for word '{word}' - SKIPPING")
                continue

            # Create exactly 25 samples using real videos + augmentation
            augmentation_types = ['horizontal_flip', 'brightness', 'contrast', 'rotation', 'temporal_shift', 'noise']

            for sample_idx in range(25):
                if sample_idx < len(real_frames_list):
                    # Use real video
                    video_id, frames = real_frames_list[sample_idx]
                    sample_type = "real"
                else:
                    # Use augmented version of a real video
                    base_idx = sample_idx % len(real_frames_list)
                    video_id, base_frames = real_frames_list[base_idx]

                    # Apply random augmentation
                    aug_type = augmentation_types[(sample_idx - len(real_frames_list)) % len(augmentation_types)]
                    frames = self.augment_frames(base_frames, aug_type)
                    video_id = f"{video_id}_aug_{aug_type}_{sample_idx}"
                    sample_type = f"augmented_{aug_type}"

                # Save frames
                frames_filename = f"{word}_{video_id}.npy"
                frames_path = frames_dir / frames_filename
                np.save(frames_path, frames)

                # Add to dataset
                processed_data.append({
                    'gloss': word,
                    'label': word_to_label[word],
                    'video_id': video_id,
                    'frames_path': f"WLASL/frames/{frames_filename}",
                    'instance_id': sample_idx,
                    'signer_id': 0,
                    'sample_type': sample_type
                })

                total_samples += 1

            real_count = len(real_frames_list)
            aug_count = 25 - real_count
            print(f"  ✅ Word '{word}' COMPLETE: 25/25 samples ({real_count} real + {aug_count} augmented)")
        
        # Save metadata
        vocabulary = {
            'num_classes': len(unique_words),
            'word_to_label': word_to_label,
            'label_to_word': {v: k for k, v in word_to_label.items()},
            'total_samples': total_samples
        }
        
        metadata_path = wlasl_dir / "processed_metadata.json"
        vocab_path = wlasl_dir / "vocabulary.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        with open(vocab_path, 'w') as f:
            json.dump(vocabulary, f, indent=2)
        
        print(f"\nProcessing complete!")
        print(f"Total samples: {total_samples}")
        print(f"Metadata saved: {metadata_path}")
        
        return total_samples > 0

def main():
    """Main function with better error handling."""
    pipeline = RobustWLASLPipeline()
    
    try:
        # Step 1: Setup Kaggle
        if not pipeline.setup_kaggle_credentials():
            print("Failed to setup Kaggle credentials")
            return 1
        
        # Step 2: Download dataset
        extract_dir = pipeline.download_with_retry()
        if not extract_dir:
            print("Failed to download dataset")
            return 1
        
        # Step 3: Load mappings
        video_to_word, word_to_label, unique_words = pipeline.load_mappings_safely(extract_dir)
        if not video_to_word:
            print("Failed to load mappings")
            return 1
        
        # Step 4: Process with augmentation
        success = pipeline.process_with_augmentation(extract_dir, video_to_word, word_to_label, unique_words)
        if not success:
            print("Failed to process videos")
            return 1
        
        print("\nPipeline completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Pipeline error: {e}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        input("\nPress Enter to exit...")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nPipeline interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        input("Press Enter to exit...")
        sys.exit(1)
