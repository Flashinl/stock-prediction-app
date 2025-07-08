#!/usr/bin/env python3
"""
Deep Cleanup Script
Finds and deletes ALL large files taking up space in the project.
"""

import os
import shutil
from pathlib import Path

def get_size_mb(path):
    """Get size of file or directory in MB."""
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    elif path.is_dir():
        total = 0
        try:
            for item in path.rglob('*'):
                if item.is_file():
                    total += item.stat().st_size
        except:
            pass
        return total / (1024 * 1024)
    return 0

def find_large_files_and_dirs():
    """Find all large files and directories."""
    print("Scanning for large files and directories...")
    print("=" * 60)
    
    # Start from project root
    project_root = Path("..").resolve()  # Go up one level from asl-to-text-ai
    
    large_items = []
    
    # Scan the entire project
    for root, dirs, files in os.walk(project_root):
        root_path = Path(root)
        
        # Skip system directories
        if any(skip in str(root_path) for skip in ['.git', '__pycache__', '.vscode']):
            continue
        
        # Check directories
        for dir_name in dirs:
            dir_path = root_path / dir_name
            if dir_path.exists():
                size_mb = get_size_mb(dir_path)
                if size_mb > 10:  # Directories larger than 10MB
                    large_items.append(('DIR', dir_path, size_mb))
        
        # Check files
        for file_name in files:
            file_path = root_path / file_name
            if file_path.exists():
                size_mb = get_size_mb(file_path)
                if size_mb > 5:  # Files larger than 5MB
                    large_items.append(('FILE', file_path, size_mb))
    
    # Sort by size (largest first)
    large_items.sort(key=lambda x: x[2], reverse=True)
    
    return large_items

def show_large_items(large_items):
    """Display large items with sizes."""
    print(f"\nFound {len(large_items)} large items:")
    print("-" * 80)
    print(f"{'Type':<6} {'Size (MB)':<12} {'Path'}")
    print("-" * 80)
    
    total_size = 0
    for item_type, path, size_mb in large_items:
        print(f"{item_type:<6} {size_mb:>8.1f} MB   {path}")
        total_size += size_mb
    
    print("-" * 80)
    print(f"Total size: {total_size:.1f} MB ({total_size/1024:.1f} GB)")
    print("-" * 80)

def cleanup_specific_items():
    """Clean up known large items."""
    print("\nCleaning up known large items...")
    
    # Known large directories/files to clean
    cleanup_targets = [
        # ASL data
        "data/real_asl",
        "data/mega_asl", 
        
        # Stock prediction data
        "../data",
        "../datasets", 
        "../models",
        "../results",
        "../instance",
        
        # Virtual environments
        "../venv_stocktrek",
        "venv",
        
        # Cache directories
        "__pycache__",
        ".pytest_cache",
        
        # Large model files
        "models",
        
        # Temporary files
        "temp",
        "tmp",
        
        # Downloaded files
        "downloads",
        "kaggle_download"
    ]
    
    total_freed = 0
    
    for target in cleanup_targets:
        target_path = Path(target)
        if target_path.exists():
            size_mb = get_size_mb(target_path)
            print(f"\nDeleting: {target_path} ({size_mb:.1f} MB)")
            
            try:
                if target_path.is_dir():
                    shutil.rmtree(target_path)
                else:
                    target_path.unlink()
                print(f"  SUCCESS: Freed {size_mb:.1f} MB")
                total_freed += size_mb
            except Exception as e:
                print(f"  ERROR: {e}")
    
    print(f"\nTotal space freed: {total_freed:.1f} MB ({total_freed/1024:.1f} GB)")

def cleanup_by_extension():
    """Clean up files by extension."""
    print("\nCleaning up large files by extension...")
    
    # File extensions that can be large and are safe to delete
    extensions_to_clean = [
        '.npy',     # NumPy arrays
        '.pkl',     # Pickle files  
        '.joblib',  # Joblib files
        '.h5',      # HDF5 files
        '.hdf5',    # HDF5 files
        '.mp4',     # Video files
        '.avi',     # Video files
        '.mov',     # Video files
        '.zip',     # Zip files
        '.tar',     # Tar files
        '.gz',      # Gzip files
        '.db',      # Database files
        '.sqlite',  # SQLite files
        '.log',     # Log files
    ]
    
    project_root = Path("..").resolve()
    total_freed = 0
    
    for ext in extensions_to_clean:
        print(f"\nLooking for {ext} files...")
        files_found = list(project_root.rglob(f"*{ext}"))
        
        if files_found:
            ext_size = sum(get_size_mb(f) for f in files_found)
            print(f"  Found {len(files_found)} {ext} files ({ext_size:.1f} MB)")
            
            if ext_size > 1:  # Only delete if more than 1MB total
                confirm = input(f"  Delete all {ext} files? (y/n): ")
                if confirm.lower() == 'y':
                    for file_path in files_found:
                        try:
                            file_path.unlink()
                        except:
                            pass
                    print(f"  Deleted {ext} files, freed {ext_size:.1f} MB")
                    total_freed += ext_size
    
    print(f"\nTotal freed by extension cleanup: {total_freed:.1f} MB")

def main():
    """Main cleanup function."""
    print("DEEP PROJECT CLEANUP")
    print("=" * 60)
    print("This will find and help you delete large files taking up space")
    print("=" * 60)
    
    # Find large items
    large_items = find_large_files_and_dirs()
    show_large_items(large_items)
    
    print("\nCleanup Options:")
    print("1. Auto-cleanup known large items")
    print("2. Cleanup by file extension") 
    print("3. Manual selection (show list again)")
    print("4. Exit")
    
    choice = input("\nChoose option (1-4): ")
    
    if choice == '1':
        cleanup_specific_items()
    elif choice == '2':
        cleanup_by_extension()
    elif choice == '3':
        show_large_items(large_items)
        print("\nManually delete items you don't need from the list above")
    else:
        print("Cleanup cancelled")
    
    print("\nCleanup complete!")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
