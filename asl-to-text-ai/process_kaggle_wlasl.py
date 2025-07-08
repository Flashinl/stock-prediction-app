#!/usr/bin/env python3
"""
Process Kaggle WLASL Dataset
Properly processes the downloaded Kaggle WLASL dataset structure.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import shutil

class KaggleWLASLProcessor:
    """Process the Kaggle WLASL dataset structure."""
    
    def __init__(self, data_dir="data/real_asl"):
        self.data_dir = Path(data_dir)
        self.extract_dir = self.data_dir / "wlasl_extracted"
        
        print("Kaggle WLASL Dataset Processor")
        print("=" * 50)
        print("Processing downloaded Kaggle WLASL dataset")
        print("=" * 50)
    
    def load_word_mappings(self):
        """Load the word mappings from JSON files."""
        print("Loading word mappings...")

        # Load WLASL metadata to get word labels
        wlasl_path = self.extract_dir / "WLASL_v0.3.json"
        nslt_2000_path = self.extract_dir / "nslt_2000.json"

        if not wlasl_path.exists() or not nslt_2000_path.exists():
            print(f"Error: Required files not found")
            return None, None

        # Load WLASL data (contains word labels)
        with open(wlasl_path, 'r') as f:
            wlasl_data = json.load(f)

        # Load video mappings (maps video_id to action index)
        with open(nslt_2000_path, 'r') as f:
            video_mappings = json.load(f)

        print(f"Loaded WLASL data: {len(wlasl_data)} words")
        print(f"Loaded video mappings: {len(video_mappings)} videos")

        # Create action_index to word mapping
        action_to_word = {}
        for word_data in wlasl_data:
            word = word_data['gloss']
            for instance in word_data['instances']:
                video_id = instance['video_id']
                if video_id in video_mappings:
                    action_index = video_mappings[video_id]['action'][0]  # First element is action index
                    action_to_word[action_index] = word

        # Create video_id to word mapping
        video_to_word = {}
        for video_id, data in video_mappings.items():
            action_index = data['action'][0]
            if action_index in action_to_word:
                video_to_word[video_id] = action_to_word[action_index]

        print(f"Successfully mapped {len(video_to_word)} videos to words")

        # Create word to label mapping
        unique_words = sorted(set(video_to_word.values()))
        word_to_label = {word: idx for idx, word in enumerate(unique_words)}

        print(f"Found {len(unique_words)} unique words")
        print(f"Sample words: {unique_words[:10]}")

        return video_to_word, word_to_label
    
    def extract_video_frames(self, video_path, target_frames=30):
        """Extract frames from a video file."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return None
            
            # Calculate frame indices to extract
            if total_frames <= target_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
            
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    # Resize to 224x224 and normalize
                    frame = cv2.resize(frame, (224, 224))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(frame)
            
            cap.release()
            
            # Pad with last frame if needed
            while len(frames) < target_frames:
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.float32))
            
            # Convert to numpy array
            frames_array = np.array(frames[:target_frames])
            return frames_array
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            return None
    
    def process_videos(self, max_samples_per_word=25):
        """Process videos and extract frames."""
        print(f"\nProcessing videos with max {max_samples_per_word} samples per word...")
        
        # Load mappings
        video_to_word, word_to_label = self.load_word_mappings()
        if not video_to_word:
            return None
        
        # Create output directories
        wlasl_dir = self.data_dir / "WLASL"
        frames_dir = wlasl_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Group videos by word
        word_videos = defaultdict(list)
        videos_dir = self.extract_dir / "videos"
        
        print("Grouping videos by word...")
        for video_id, word in video_to_word.items():
            video_file = videos_dir / f"{video_id}.mp4"
            if video_file.exists():
                word_videos[word].append((video_id, video_file))
        
        print(f"Found videos for {len(word_videos)} words")
        
        # Process each word
        processed_data = []
        successful_words = 0
        total_successful_samples = 0
        
        for word_idx, (word, video_list) in enumerate(word_videos.items()):
            print(f"\n[{word_idx+1}/{len(word_videos)}] Processing word: '{word}'")
            print(f"  Available videos: {len(video_list)}")
            
            label = word_to_label[word]
            samples_processed = 0
            
            # Process up to max_samples_per_word videos for this word
            for video_idx, (video_id, video_path) in enumerate(video_list[:max_samples_per_word]):
                print(f"    Sample {samples_processed+1}/{max_samples_per_word}: {video_id}")
                
                # Extract frames
                frames = self.extract_video_frames(video_path)
                if frames is not None:
                    # Save frames
                    frames_filename = f"{word}_{video_id}.npy"
                    frames_path = frames_dir / frames_filename
                    np.save(frames_path, frames)
                    
                    # Add to processed data
                    processed_data.append({
                        'gloss': word,
                        'label': label,
                        'video_id': video_id,
                        'frames_path': f"WLASL/frames/{frames_filename}",
                        'instance_id': video_idx,
                        'signer_id': 0
                    })
                    
                    samples_processed += 1
                    total_successful_samples += 1
                    print(f"      SUCCESS: {samples_processed}/{max_samples_per_word}")
                else:
                    print(f"      FAILED: Could not extract frames")
            
            if samples_processed > 0:
                successful_words += 1
            
            print(f"  Word '{word}' completed: {samples_processed} successful samples")
            
            # Progress update
            avg_samples = total_successful_samples / successful_words if successful_words > 0 else 0
            print(f"  Progress: {successful_words} words, {total_successful_samples} total samples")
            print(f"  Average samples per word: {avg_samples:.1f}")
        
        # Create vocabulary
        vocabulary = {
            'num_classes': len(word_to_label),
            'word_to_label': word_to_label,
            'label_to_word': {v: k for k, v in word_to_label.items()},
            'total_samples': len(processed_data)
        }
        
        # Save metadata
        metadata_path = wlasl_dir / "processed_metadata.json"
        vocab_path = wlasl_dir / "vocabulary.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        with open(vocab_path, 'w') as f:
            json.dump(vocabulary, f, indent=2)
        
        print(f"\n" + "="*60)
        print(f"Processing Complete!")
        print(f"="*60)
        print(f"Total words processed: {successful_words}")
        print(f"Total samples: {vocabulary['total_samples']}")
        print(f"Average samples per word: {vocabulary['total_samples']/successful_words:.1f}")
        print(f"Vocabulary size: {vocabulary['num_classes']}")
        print(f"")
        print(f"Files saved:")
        print(f"  Metadata: {metadata_path}")
        print(f"  Vocabulary: {vocab_path}")
        print(f"  Frames: {frames_dir}")
        print(f"="*60)
        
        return {
            'metadata_path': str(metadata_path),
            'vocabulary_path': str(vocab_path),
            'frames_dir': str(frames_dir),
            'num_classes': vocabulary['num_classes'],
            'total_samples': vocabulary['total_samples']
        }

def main():
    """Main function."""
    print("Kaggle WLASL Dataset Processing")
    print("=" * 50)
    
    processor = KaggleWLASLProcessor()
    
    # Check if extracted data exists
    if not processor.extract_dir.exists():
        print(f"Error: Extracted data not found at {processor.extract_dir}")
        print("Please run download_preprocessed_wlasl.py first")
        return 1
    
    # Process videos
    result = processor.process_videos(max_samples_per_word=25)
    
    if result and result['total_samples'] > 0:
        print(f"\nSUCCESS!")
        print(f"Dataset ready for training with {result['total_samples']} samples")
        return 0
    else:
        print(f"\nFAILED!")
        print("No samples were successfully processed")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        input("\nPress Enter to exit...")
        exit(exit_code)
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...")
        exit(1)
