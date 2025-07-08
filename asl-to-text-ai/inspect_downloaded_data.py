#!/usr/bin/env python3
"""
Inspect Downloaded WLASL Data
Check what was actually downloaded and its structure.
"""

import os
import json
from pathlib import Path
import numpy as np

def inspect_data():
    """Inspect the downloaded data structure."""
    
    print("Inspecting Downloaded WLASL Data")
    print("=" * 50)
    
    # Check extraction directory
    extract_dir = Path("data/real_asl/wlasl_extracted")
    
    if not extract_dir.exists():
        print(f"Extract directory not found: {extract_dir}")
        return
    
    print(f"Extract directory: {extract_dir}")
    print(f"Directory exists: {extract_dir.exists()}")
    
    # List all files and directories
    print("\nDirectory structure:")
    for root, dirs, files in os.walk(extract_dir):
        level = root.replace(str(extract_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        # Show first 10 files in each directory
        subindent = ' ' * 2 * (level + 1)
        for i, file in enumerate(files[:10]):
            print(f"{subindent}{file}")
        
        if len(files) > 10:
            print(f"{subindent}... and {len(files) - 10} more files")
    
    # Look for different file types
    print("\nFile type analysis:")
    file_types = {}
    total_files = 0
    
    for file_path in extract_dir.rglob("*"):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1
            total_files += 1
    
    print(f"Total files: {total_files}")
    for ext, count in sorted(file_types.items()):
        print(f"  {ext or '(no extension)'}: {count} files")
    
    # Look for specific patterns
    print("\nLooking for data files...")
    
    # Check for numpy files
    npy_files = list(extract_dir.rglob("*.npy"))
    print(f"NumPy files (.npy): {len(npy_files)}")
    if npy_files:
        print("Sample .npy files:")
        for file in npy_files[:5]:
            try:
                data = np.load(file)
                print(f"  {file.name}: shape {data.shape}, dtype {data.dtype}")
            except Exception as e:
                print(f"  {file.name}: Error loading - {e}")
    
    # Check for video files
    video_exts = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_exts:
        video_files.extend(extract_dir.rglob(f"*{ext}"))
    
    print(f"Video files: {len(video_files)}")
    if video_files:
        print("Sample video files:")
        for file in video_files[:5]:
            print(f"  {file.name}: {file.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Check for JSON/CSV metadata
    json_files = list(extract_dir.rglob("*.json"))
    csv_files = list(extract_dir.rglob("*.csv"))
    
    print(f"JSON files: {len(json_files)}")
    for file in json_files:
        print(f"  {file.name}")
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                print(f"    List with {len(data)} items")
                if data:
                    print(f"    Sample item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}")
            elif isinstance(data, dict):
                print(f"    Dict with keys: {list(data.keys())}")
        except Exception as e:
            print(f"    Error reading: {e}")
    
    print(f"CSV files: {len(csv_files)}")
    for file in csv_files:
        print(f"  {file.name}")
    
    # Look for directory patterns that might indicate word organization
    print("\nDirectory patterns:")
    subdirs = [d for d in extract_dir.iterdir() if d.is_dir()]
    print(f"Top-level subdirectories: {len(subdirs)}")
    for subdir in subdirs[:10]:
        file_count = len(list(subdir.rglob("*")))
        print(f"  {subdir.name}: {file_count} files")
    
    if len(subdirs) > 10:
        print(f"  ... and {len(subdirs) - 10} more directories")

def main():
    inspect_data()
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
