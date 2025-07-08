#!/usr/bin/env python3
"""
Robust WLASL Dataset Processor
Downloads and processes real WLASL videos with error handling and fallbacks.
"""

import json
import cv2
import numpy as np
from pathlib import Path
import requests
from tqdm import tqdm
import logging
import urllib.request
import ssl
import time
import warnings

# Suppress SSL warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
try:
    import urllib3
    urllib3_logger = logging.getLogger('urllib3')
    urllib3_logger.setLevel(logging.CRITICAL)
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WLASLProcessor:
    """Process WLASL dataset with robust error handling."""
    
    def __init__(self, data_dir="data/real_asl"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create SSL context that ignores certificate errors
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
        print("WLASL Dataset Processor")
        print("=" * 50)
        print("Processing REAL ASL videos from WLASL")
        print("Robust error handling for network issues")
        print("=" * 50)
    
    def download_metadata(self):
        """Download WLASL metadata."""
        print("\nDownloading WLASL metadata...")

        wlasl_dir = self.data_dir / "WLASL"
        wlasl_dir.mkdir(exist_ok=True)

        metadata_url = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
        metadata_path = wlasl_dir / "WLASL_v0.3.json"

        if metadata_path.exists():
            print(f"Metadata already exists: {metadata_path}")
        else:
            try:
                response = requests.get(metadata_url, verify=False, timeout=30)
                response.raise_for_status()

                with open(metadata_path, 'w') as f:
                    json.dump(response.json(), f, indent=2)

                print(f"Downloaded metadata to: {metadata_path}")
            except Exception as e:
                print(f"Error downloading metadata: {e}")
                return None

        # Load and analyze
        with open(metadata_path, 'r') as f:
            wlasl_data = json.load(f)

        print(f"WLASL Dataset Info:")
        print(f"   Total words: {len(wlasl_data)}")
        total_videos = sum(len(word_data['instances']) for word_data in wlasl_data)
        print(f"   Total videos: {total_videos}")

        return wlasl_data
    
    def download_video_safe(self, url, output_path, max_retries=2):
        """Download video with comprehensive error handling."""
        for attempt in range(max_retries):
            try:
                # Try different approaches
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                # First try with requests
                try:
                    response = requests.get(url, headers=headers, stream=True, timeout=15, verify=False)
                    response.raise_for_status()
                    
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    # Verify the file is valid and has proper video structure
                    if output_path.stat().st_size > 10000:  # At least 10KB
                        # Quick test if video can be opened
                        try:
                            test_cap = cv2.VideoCapture(str(output_path))
                            frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            test_cap.release()

                            if frame_count > 0:
                                return True
                            else:
                                output_path.unlink(missing_ok=True)
                        except:
                            output_path.unlink(missing_ok=True)
                    else:
                        output_path.unlink(missing_ok=True)
                        
                except Exception:
                    # Try with urllib as fallback
                    try:
                        req = urllib.request.Request(url, headers=headers)
                        with urllib.request.urlopen(req, context=self.ssl_context, timeout=15) as response:
                            with open(output_path, 'wb') as f:
                                f.write(response.read())
                        
                        if output_path.stat().st_size > 1000:
                            return True
                        else:
                            output_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    logger.debug(f"Failed to download {url}: {e}")
        
        return False
    
    def extract_frames_robust(self, video_path, output_dir, target_frames=30):
        """Extract frames with robust error handling."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.debug(f"Could not open video: {video_path}")
                return None

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if total_frames <= 0 or fps <= 0:
                cap.release()
                logger.debug(f"Invalid video properties: frames={total_frames}, fps={fps}")
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
                    # Resize and normalize
                    frame = cv2.resize(frame, (224, 224))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame.astype(np.float32) / 255.0  # Normalize to [0,1]
                    frames.append(frame)
            
            cap.release()
            
            if len(frames) == 0:
                return None
            
            # Pad with last frame if needed
            while len(frames) < target_frames:
                frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.float32))
            
            # Convert to numpy array
            frames_array = np.array(frames[:target_frames])
            
            # Save frames
            output_path = output_dir / f"{video_path.stem}.npy"
            np.save(output_path, frames_array)
            
            return str(output_path.relative_to(self.data_dir))
            
        except Exception as e:
            logger.debug(f"Error extracting frames from {video_path}: {e}")
            return None
    
    def filter_good_urls(self, wlasl_data, max_words=2000):
        """Filter for URLs that are more likely to work."""
        print(f"\nFiltering for reliable video sources...")

        # Accept more sources to get 25 videos per word
        good_sources = ['aslbrick', 'youtube', 'vimeo', 'aslsignbank', 'aslpro']
        filtered_data = []

        for word_data in wlasl_data:  # Check all words
            gloss = word_data['gloss']
            instances = word_data['instances']

            # Filter for training split - be more permissive to get more videos
            good_instances = []
            for inst in instances:
                if inst.get('split') == 'train':
                    url = inst.get('url', '')
                    source = inst.get('source', '')

                    # Accept more URL patterns to get 25 videos per word
                    if (source in good_sources or
                        'youtube' in url.lower() or
                        'vimeo' in url.lower() or
                        'aslbrick' in url.lower() or
                        'aslsignbank' in url.lower() or
                        url.endswith('.mp4') or
                        url.endswith('.swf') or
                        'asl' in url.lower()):
                        good_instances.append(inst)

            if good_instances:
                filtered_data.append({
                    'gloss': gloss,
                    'instances': good_instances[:25]  # Max 25 per word
                })

            if len(filtered_data) >= max_words:
                break

        print(f"Filtered to {len(filtered_data)} words with reliable sources")
        return filtered_data
    
    def process_videos(self, max_words=2000, max_videos_per_word=25):
        """Process WLASL videos with robust error handling."""
        print(f"\nProcessing WLASL Videos")
        print(f"Target: {max_words} words, {max_videos_per_word} videos per word")
        print(f"Expected total samples: {max_words * max_videos_per_word}")
        
        # Download metadata
        wlasl_data = self.download_metadata()
        if not wlasl_data:
            return None
        
        # Filter for good URLs
        filtered_data = self.filter_good_urls(wlasl_data, max_words)
        
        # Create directories
        wlasl_dir = self.data_dir / "WLASL"
        videos_dir = wlasl_dir / "videos"
        frames_dir = wlasl_dir / "frames"
        videos_dir.mkdir(exist_ok=True)
        frames_dir.mkdir(exist_ok=True)
        
        # Process videos
        processed_data = []
        word_to_label = {}
        current_label = 0
        total_attempted = 0
        total_successful = 0

        print(f"\nProcessing {len(filtered_data)} words...")
        print(f"Target: {max_videos_per_word} successful samples per word")

        for word_idx, word_data in enumerate(filtered_data):
            gloss = word_data['gloss']
            instances = word_data['instances']

            print(f"\n[{word_idx+1}/{len(filtered_data)}] Processing word: '{gloss}'")
            print(f"  Available videos: {len(instances)}")

            # Assign label
            if gloss not in word_to_label:
                word_to_label[gloss] = current_label
                current_label += 1

            label = word_to_label[gloss]
            videos_processed = 0
            videos_attempted = 0

            # Keep trying until we get 25 successful samples or run out of videos
            for instance in instances:
                if videos_processed >= max_videos_per_word:
                    break

                videos_attempted += 1
                video_url = instance['url']
                video_id = instance['video_id']

                print(f"    Sample {videos_processed+1}/{max_videos_per_word} (attempt {videos_attempted}): {video_id}")

                # Create unique filename
                video_filename = f"{gloss}_{video_id}.mp4"
                video_path = videos_dir / video_filename

                # Download video
                total_attempted += 1
                print(f"      Downloading from: {video_url[:60]}...")
                success = self.download_video_safe(video_url, video_path)
                if not success:
                    print(f"      FAILED: Download failed")
                    continue

                # Extract frames
                print(f"      Extracting frames...")
                frames_path = self.extract_frames_robust(video_path, frames_dir)
                if frames_path:
                    total_successful += 1
                    processed_data.append({
                        'gloss': gloss,
                        'label': label,
                        'video_id': video_id,
                        'frames_path': frames_path,
                        'instance_id': instance['instance_id'],
                        'signer_id': instance.get('signer_id', 0)
                    })
                    videos_processed += 1
                    print(f"      SUCCESS: Sample {videos_processed}/{max_videos_per_word} completed")
                else:
                    print(f"      FAILED: Frame extraction failed")

                # Clean up video file to save space
                if video_path.exists():
                    video_path.unlink()

            print(f"  Word '{gloss}' completed: {videos_processed}/{max_videos_per_word} successful samples")
            if videos_processed < max_videos_per_word:
                print(f"  WARNING: Only got {videos_processed} samples (needed {max_videos_per_word})")

            # Progress summary
            total_words_processed = word_idx + 1
            avg_samples_per_word = len(processed_data) / total_words_processed if total_words_processed > 0 else 0
            print(f"  Overall progress: {total_words_processed}/{len(filtered_data)} words, {len(processed_data)} total samples")
            print(f"  Average samples per word so far: {avg_samples_per_word:.1f}")
        
        # Create vocabulary and save results
        vocabulary = {
            'num_classes': len(word_to_label),
            'word_to_label': word_to_label,
            'label_to_word': {v: k for k, v in word_to_label.items()},
            'total_samples': len(processed_data)
        }
        
        # Save processed data
        processed_metadata_path = wlasl_dir / "processed_metadata.json"
        vocabulary_path = wlasl_dir / "vocabulary.json"
        
        with open(processed_metadata_path, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        with open(vocabulary_path, 'w') as f:
            json.dump(vocabulary, f, indent=2)
        
        print(f"\n" + "="*60)
        print(f"WLASL Processing Complete!")
        print(f"="*60)
        print(f"Target: {max_videos_per_word} samples per word x {len(filtered_data)} words = {max_videos_per_word * len(filtered_data)} total")
        print(f"Achieved: {len(processed_data)} successful samples")
        print(f"Success rate: {len(processed_data)/total_attempted*100:.1f}% ({len(processed_data)}/{total_attempted})")
        print(f"Words with data: {vocabulary['num_classes']}")
        print(f"Average samples per word: {len(processed_data)/vocabulary['num_classes']:.1f}")
        print(f"")
        print(f"Files saved:")
        print(f"  Metadata: {processed_metadata_path}")
        print(f"  Vocabulary: {vocabulary_path}")
        print(f"  Frames: {frames_dir}")
        print(f"="*60)
        
        return {
            'metadata_path': str(processed_metadata_path),
            'vocabulary_path': str(vocabulary_path),
            'frames_dir': str(frames_dir),
            'num_classes': vocabulary['num_classes'],
            'total_samples': len(processed_data)
        }

def main():
    """Main function."""
    print("WLASL Dataset Processor")
    print("NO MORE SYNTHETIC DATA!")
    print("Processing REAL ASL videos")
    print("=" * 60)

    processor = WLASLProcessor()
    result = processor.process_videos(max_words=2000, max_videos_per_word=25)

    if result and result['total_samples'] > 0:
        print(f"\nSUCCESS!")
        print(f"Processed {result['total_samples']} real ASL videos")
        print(f"{result['num_classes']} classes ready for training")
        print(f"Data location: {processor.data_dir}")
        return 0
    else:
        print(f"\nFailed to process sufficient videos")
        return 1

if __name__ == "__main__":
    exit(main())
