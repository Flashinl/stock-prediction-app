#!/usr/bin/env python3
"""
Real ASL Dataset Processor
Downloads and processes REAL ASL datasets from Kaggle and other sources.
NO MORE SYNTHETIC DATA!
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import requests
import zipfile
from tqdm import tqdm
import logging
import urllib.request
import urllib.parse
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealASLDataProcessor:
    """Process real ASL datasets - NO synthetic data!"""
    
    def __init__(self, data_dir: str = "data/real_asl"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        print("🎯 REAL ASL Dataset Processor")
        print("=" * 50)
        print("✅ NO MORE SYNTHETIC DATA!")
        print("✅ Using REAL sign language videos")
        print("=" * 50)
    
    def download_wlasl_dataset(self):
        """
        Download the WLASL (Word-Level ASL) dataset.
        This is one of the largest real ASL datasets with 2000+ words.
        """
        print("\n📥 Downloading WLASL Dataset...")

        wlasl_dir = self.data_dir / "WLASL"
        wlasl_dir.mkdir(exist_ok=True)

        # WLASL metadata URL
        metadata_url = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"

        try:
            print("📋 Downloading WLASL metadata...")
            response = requests.get(metadata_url)
            response.raise_for_status()

            metadata_path = wlasl_dir / "WLASL_v0.3.json"
            with open(metadata_path, 'w') as f:
                json.dump(response.json(), f, indent=2)

            print(f"✅ Downloaded metadata to: {metadata_path}")

            # Load and analyze metadata
            with open(metadata_path, 'r') as f:
                wlasl_data = json.load(f)

            print(f"📊 WLASL Dataset Info:")
            print(f"   Total words: {len(wlasl_data)}")

            # Count total videos
            total_videos = sum(len(word_data['instances']) for word_data in wlasl_data)
            print(f"   Total videos: {total_videos}")

            # Show sample words
            sample_words = [word_data['gloss'] for word_data in wlasl_data[:10]]
            print(f"   Sample words: {sample_words}")

            return wlasl_data

        except Exception as e:
            print(f"❌ Error downloading WLASL: {e}")
            return None

    def download_video_with_retry(self, url, output_path, max_retries=3):
        """Download a video with retry logic."""
        for attempt in range(max_retries):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }

                response = requests.get(url, headers=headers, stream=True, timeout=30)
                response.raise_for_status()

                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                return True

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to download {url} after {max_retries} attempts")
                    return False

        return False

    def extract_video_frames(self, video_path, output_dir, target_frames=30):
        """Extract frames from video and resize to standard size."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                return None

            # Calculate frame indices to extract
            if total_frames <= target_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)

            frames = []
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret:
                    # Resize to 224x224 for model input
                    frame = cv2.resize(frame, (224, 224))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)

            cap.release()

            # Pad with last frame if needed
            while len(frames) < target_frames:
                if frames:
                    frames.append(frames[-1])
                else:
                    # Create black frame if no frames extracted
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

            # Convert to numpy array
            frames_array = np.array(frames[:target_frames])

            # Save frames as numpy array
            output_path = output_dir / f"{video_path.stem}.npy"
            np.save(output_path, frames_array)

            return str(output_path.relative_to(self.data_dir))

        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            return None

    def process_wlasl_videos(self, max_words=100, max_videos_per_word=5):
        """
        Process WLASL videos by downloading and extracting frames.
        Limited to a subset for manageable training.
        """
        print(f"\n🎬 Processing WLASL Videos (max {max_words} words, {max_videos_per_word} videos per word)")

        wlasl_dir = self.data_dir / "WLASL"
        metadata_path = wlasl_dir / "WLASL_v0.3.json"

        if not metadata_path.exists():
            print("❌ WLASL metadata not found. Run download_wlasl_dataset() first.")
            return None

        # Load metadata
        with open(metadata_path, 'r') as f:
            wlasl_data = json.load(f)

        # Create directories
        videos_dir = wlasl_dir / "videos"
        frames_dir = wlasl_dir / "frames"
        videos_dir.mkdir(exist_ok=True)
        frames_dir.mkdir(exist_ok=True)

        # Process subset of words
        processed_data = []
        word_to_label = {}
        current_label = 0

        print(f"📊 Processing {min(max_words, len(wlasl_data))} words...")

        for word_idx, word_data in enumerate(tqdm(wlasl_data[:max_words], desc="Processing words")):
            gloss = word_data['gloss']
            instances = word_data['instances']

            # Filter for training split
            train_instances = [inst for inst in instances if inst.get('split') == 'train']

            if not train_instances:
                continue

            # Assign label
            if gloss not in word_to_label:
                word_to_label[gloss] = current_label
                current_label += 1

            label = word_to_label[gloss]

            # Process limited number of videos per word
            videos_processed = 0
            for instance in train_instances[:max_videos_per_word]:
                if videos_processed >= max_videos_per_word:
                    break

                video_url = instance['url']
                video_id = instance['video_id']

                # Create unique filename
                video_filename = f"{gloss}_{video_id}.mp4"
                video_path = videos_dir / video_filename

                # Download video if not exists
                if not video_path.exists():
                    success = self.download_video_with_retry(video_url, video_path)
                    if not success:
                        continue

                # Extract frames
                frames_path = self.extract_video_frames(video_path, frames_dir)
                if frames_path:
                    processed_data.append({
                        'gloss': gloss,
                        'label': label,
                        'video_id': video_id,
                        'frames_path': frames_path,
                        'instance_id': instance['instance_id'],
                        'signer_id': instance['signer_id']
                    })
                    videos_processed += 1

                # Clean up video file to save space
                if video_path.exists():
                    video_path.unlink()

        # Create vocabulary
        vocabulary = {
            'num_classes': len(word_to_label),
            'word_to_label': word_to_label,
            'label_to_word': {v: k for k, v in word_to_label.items()},
            'total_samples': len(processed_data)
        }

        # Save processed data and vocabulary
        processed_metadata_path = wlasl_dir / "processed_metadata.json"
        vocabulary_path = wlasl_dir / "vocabulary.json"

        with open(processed_metadata_path, 'w') as f:
            json.dump(processed_data, f, indent=2)

        with open(vocabulary_path, 'w') as f:
            json.dump(vocabulary, f, indent=2)

        print(f"\n✅ WLASL Processing Complete!")
        print(f"📊 Processed {len(processed_data)} videos")
        print(f"📊 {vocabulary['num_classes']} unique words/classes")
        print(f"💾 Metadata saved to: {processed_metadata_path}")
        print(f"💾 Vocabulary saved to: {vocabulary_path}")

        return {
            'type': 'WLASL_Processed',
            'classes': list(word_to_label.keys()),
            'total_samples': len(processed_data),
            'num_classes': vocabulary['num_classes'],
            'metadata_path': str(processed_metadata_path),
            'vocabulary_path': str(vocabulary_path),
            'frames_dir': str(frames_dir)
        }
    
    def download_asl_alphabet_dataset(self):
        """
        Download ASL Alphabet dataset from Kaggle.
        """
        print("\n📥 Downloading ASL Alphabet Dataset...")
        
        alphabet_dir = self.data_dir / "ASL_Alphabet"
        alphabet_dir.mkdir(exist_ok=True)
        
        print("📋 ASL Alphabet Dataset Instructions:")
        print("1. Go to: https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        print("2. Download the dataset")
        print("3. Extract to:", alphabet_dir)
        print("4. Run this script again to process")
        
        # Check if already downloaded
        if (alphabet_dir / "asl_alphabet_train").exists():
            print("✅ ASL Alphabet dataset found!")
            return self.process_asl_alphabet(alphabet_dir)
        else:
            print("⏳ Please download the dataset manually and run again")
            return None
    
    def process_asl_alphabet(self, alphabet_dir):
        """Process the ASL Alphabet dataset."""
        train_dir = alphabet_dir / "asl_alphabet_train"
        test_dir = alphabet_dir / "asl_alphabet_test"
        
        if not train_dir.exists():
            print(f"❌ Training directory not found: {train_dir}")
            return None
        
        print("🔄 Processing ASL Alphabet dataset...")
        
        # Get all classes (letters)
        classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
        classes.sort()
        
        print(f"📊 Found {len(classes)} classes: {classes}")
        
        # Count images per class
        dataset_info = {}
        total_images = 0
        
        for class_name in classes:
            class_dir = train_dir / class_name
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            dataset_info[class_name] = len(images)
            total_images += len(images)
        
        print(f"📊 Total images: {total_images}")
        print(f"📊 Average per class: {total_images / len(classes):.1f}")
        
        return {
            'type': 'ASL_Alphabet',
            'classes': classes,
            'total_images': total_images,
            'dataset_info': dataset_info,
            'train_dir': str(train_dir),
            'test_dir': str(test_dir) if test_dir.exists() else None
        }
    
    def download_asl_signs_dataset(self):
        """
        Download ASL Signs dataset (larger vocabulary).
        """
        print("\n📥 Downloading ASL Signs Dataset...")
        
        signs_dir = self.data_dir / "ASL_Signs"
        signs_dir.mkdir(exist_ok=True)
        
        print("📋 ASL Signs Dataset Instructions:")
        print("1. Go to: https://www.kaggle.com/datasets/ayuraj/asl-dataset")
        print("2. Download the dataset")
        print("3. Extract to:", signs_dir)
        print("4. Run this script again to process")
        
        # Check if already downloaded
        if any(signs_dir.iterdir()):
            print("✅ ASL Signs dataset found!")
            return self.process_asl_signs(signs_dir)
        else:
            print("⏳ Please download the dataset manually and run again")
            return None
    
    def process_asl_signs(self, signs_dir):
        """Process the ASL Signs dataset."""
        print("🔄 Processing ASL Signs dataset...")
        
        # Look for common dataset structures
        possible_dirs = [
            signs_dir / "train",
            signs_dir / "Train",
            signs_dir / "training",
            signs_dir / "data",
            signs_dir
        ]
        
        train_dir = None
        for dir_path in possible_dirs:
            if dir_path.exists() and any(d.is_dir() for d in dir_path.iterdir()):
                train_dir = dir_path
                break
        
        if not train_dir:
            print(f"❌ Could not find training directory in: {signs_dir}")
            return None
        
        print(f"📁 Using training directory: {train_dir}")
        
        # Get all classes
        classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
        classes.sort()
        
        print(f"📊 Found {len(classes)} classes")
        print(f"📊 Sample classes: {classes[:10]}")
        
        # Count images per class
        dataset_info = {}
        total_images = 0
        
        for class_name in tqdm(classes, desc="Analyzing classes"):
            class_dir = train_dir / class_name
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
            dataset_info[class_name] = len(images)
            total_images += len(images)
        
        print(f"📊 Total images: {total_images}")
        print(f"📊 Average per class: {total_images / len(classes):.1f}")
        
        return {
            'type': 'ASL_Signs',
            'classes': classes,
            'total_images': total_images,
            'dataset_info': dataset_info,
            'train_dir': str(train_dir)
        }
    
    def create_unified_dataset(self, datasets):
        """Create a unified dataset from multiple real ASL datasets."""
        print("\n🔄 Creating unified real ASL dataset...")
        
        unified_dir = self.data_dir / "unified"
        unified_dir.mkdir(exist_ok=True)
        
        all_classes = set()
        total_samples = 0
        
        # Collect all unique classes
        for dataset in datasets:
            if dataset:
                all_classes.update(dataset['classes'])
                total_samples += dataset['total_images']
        
        all_classes = sorted(list(all_classes))
        
        print(f"📊 Unified Dataset:")
        print(f"   Total classes: {len(all_classes)}")
        print(f"   Total samples: {total_samples}")
        
        # Create metadata
        metadata = {
            'dataset_name': 'Real_ASL_Unified',
            'total_classes': len(all_classes),
            'total_samples': total_samples,
            'classes': all_classes,
            'source_datasets': [d['type'] for d in datasets if d],
            'created_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save metadata
        metadata_path = unified_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved metadata to: {metadata_path}")
        
        return metadata
    
    def process_all_datasets(self):
        """Process all available real ASL datasets."""
        print("🚀 Processing ALL Real ASL Datasets")
        print("=" * 50)

        datasets = []

        # Try to download/process each dataset
        print("\n1️⃣ WLASL Dataset (2000+ words)")
        wlasl_data = self.download_wlasl_dataset()
        if wlasl_data:
            # Process WLASL videos
            print("\n🎬 Processing WLASL videos...")
            wlasl_processed = self.process_wlasl_videos(max_words=100, max_videos_per_word=3)
            if wlasl_processed:
                datasets.append(wlasl_processed)

        print("\n2️⃣ ASL Alphabet Dataset")
        alphabet_data = self.download_asl_alphabet_dataset()
        if alphabet_data:
            datasets.append(alphabet_data)

        print("\n3️⃣ ASL Signs Dataset")
        signs_data = self.download_asl_signs_dataset()
        if signs_data:
            datasets.append(signs_data)

        # Create unified dataset
        if datasets:
            unified_metadata = self.create_unified_dataset(datasets)

            print("\n🎉 REAL ASL DATASET PROCESSING COMPLETE!")
            print("=" * 50)
            print(f"✅ Processed {len(datasets)} real datasets")
            print(f"✅ Total classes: {unified_metadata['total_classes']}")
            print(f"✅ Total samples: {unified_metadata['total_samples']}")
            print("✅ NO synthetic data - all REAL sign language!")

            return unified_metadata
        else:
            print("\n❌ No datasets were successfully processed")
            print("Please download the datasets manually and run again")
            return None

    def process_wlasl_only(self, max_words=100, max_videos_per_word=3):
        """Process only WLASL dataset for focused training."""
        print("🎯 Processing WLASL Dataset Only")
        print("=" * 50)

        # Download metadata if needed
        wlasl_data = self.download_wlasl_dataset()
        if not wlasl_data:
            print("❌ Failed to download WLASL metadata")
            return None

        # Process videos
        result = self.process_wlasl_videos(max_words=max_words, max_videos_per_word=max_videos_per_word)

        if result:
            print("\n🎉 WLASL PROCESSING COMPLETE!")
            print("=" * 50)
            print(f"✅ Classes: {result['num_classes']}")
            print(f"✅ Total samples: {result['total_samples']}")
            print(f"✅ Ready for training!")

        return result

def main():
    """Main function to process real ASL data."""

    print("🎯 REAL ASL Data Processor")
    print("🚫 NO MORE SYNTHETIC DATA!")
    print("✅ Processing REAL sign language datasets")
    print("=" * 60)

    processor = RealASLDataProcessor()

    # Process WLASL dataset only for focused training
    result = processor.process_wlasl_only(max_words=100, max_videos_per_word=3)

    if result:
        print(f"\n✅ SUCCESS! Real ASL dataset ready for training")
        print(f"📁 Location: {processor.data_dir}")
        print(f"📁 Frames: {result['frames_dir']}")
        print(f"📁 Metadata: {result['metadata_path']}")
        print(f"📁 Vocabulary: {result['vocabulary_path']}")
    else:
        print(f"\n❌ Failed to process WLASL dataset")

    return 0 if result else 1

if __name__ == "__main__":
    exit(main())
