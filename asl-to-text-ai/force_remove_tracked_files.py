#!/usr/bin/env python3
"""
Force Remove Tracked Files
Specifically targets the 10k files still being tracked by Git.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run command and show output."""
    try:
        print(f"\n{description}")
        print(f"Command: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            print(f"Output ({len(lines)} lines):")
            for line in lines[:10]:  # Show first 10 lines
                print(f"  {line}")
            if len(lines) > 10:
                print(f"  ... and {len(lines) - 10} more lines")
        
        if result.stderr:
            print(f"Errors: {result.stderr}")
        
        return result.returncode == 0, result.stdout
    except Exception as e:
        print(f"Error: {e}")
        return False, ""

def force_remove_tracked():
    """Force remove all tracked large files."""
    
    print("FORCE REMOVE TRACKED FILES")
    print("=" * 50)
    
    # Navigate to project root
    project_root = Path("..").resolve()
    os.chdir(project_root)
    print(f"Working in: {project_root}")
    
    # Get list of all tracked files
    success, output = run_command("git ls-files", "Getting list of tracked files")
    
    if not success:
        print("Failed to get tracked files")
        return
    
    tracked_files = [f.strip() for f in output.split('\n') if f.strip()]
    print(f"\nCurrently tracking {len(tracked_files)} files")
    
    # Analyze tracked files
    large_files = []
    data_files = []
    video_files = []
    
    for file_path in tracked_files:
        try:
            path = Path(file_path)
            
            # Check if file exists and get size
            if path.exists():
                size = path.stat().st_size
                if size > 50000:  # Files larger than 50KB
                    large_files.append(file_path)
            
            # Check for data-related paths
            if any(part in file_path.lower() for part in ['data/', 'dataset', 'model', 'video', 'frame']):
                data_files.append(file_path)
            
            # Check for video files
            if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append(file_path)
                
        except:
            continue
    
    print(f"Large files (>50KB): {len(large_files)}")
    print(f"Data-related files: {len(data_files)}")
    print(f"Video files: {len(video_files)}")
    
    # Show some examples
    if large_files:
        print(f"\nSample large files:")
        for f in large_files[:5]:
            print(f"  {f}")
    
    if data_files:
        print(f"\nSample data files:")
        for f in data_files[:5]:
            print(f"  {f}")
    
    # Force remove all problematic files
    files_to_remove = set(large_files + data_files + video_files)
    
    if files_to_remove:
        print(f"\nRemoving {len(files_to_remove)} problematic files from Git...")
        
        # Remove in batches to avoid command line length limits
        batch_size = 100
        files_list = list(files_to_remove)
        
        for i in range(0, len(files_list), batch_size):
            batch = files_list[i:i+batch_size]
            files_str = '" "'.join(batch)
            cmd = f'git rm --cached "{files_str}"'
            
            success, _ = run_command(cmd, f"Removing batch {i//batch_size + 1}")
            if success:
                print(f"  ✅ Removed batch {i//batch_size + 1} ({len(batch)} files)")
            else:
                print(f"  ❌ Failed batch {i//batch_size + 1}")
    
    # Also try removing entire directories
    directories_to_remove = [
        "data",
        "datasets", 
        "models",
        "instance",
        "results",
        "asl-to-text-ai/data",
        "temp",
        "cache"
    ]
    
    print(f"\nRemoving entire directories from Git...")
    for directory in directories_to_remove:
        run_command(f'git rm -r --cached "{directory}"', f"Remove directory {directory}")
    
    # Create comprehensive .gitignore
    print(f"\nCreating comprehensive .gitignore...")
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
venv_*/
ENV/
env/

# Data and models - NEVER COMMIT
data/
datasets/
models/
instance/
results/
temp/
tmp/
cache/
downloads/

# ASL specific
asl-to-text-ai/data/
asl-to-text-ai/models/

# Large files
*.npy
*.npz
*.pkl
*.pickle
*.joblib
*.h5
*.hdf5
*.pt
*.pth
*.ckpt
*.pb
*.tflite
*.onnx
*.db
*.sqlite
*.sqlite3

# Media files
*.mp4
*.avi
*.mov
*.mkv
*.wmv
*.flv
*.webm
*.jpg
*.jpeg
*.png
*.gif
*.bmp
*.mp3
*.wav

# Archives
*.zip
*.tar
*.gz
*.bz2
*.rar
*.7z

# Large text files
*.csv
*.tsv
*.json
*.xml
*.yaml
*.yml

# Logs
*.log
logs/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Kaggle
kaggle.json
.kaggle/

# Jupyter
.ipynb_checkpoints/

# Specific problematic directories
wlasl_extracted/
kaggle_download/
frames/
videos/
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content.strip())
    
    # Add .gitignore and commit
    run_command("git add .gitignore", "Add .gitignore")
    run_command('git commit -m "Force remove all large files and add comprehensive .gitignore"', "Commit cleanup")
    
    # Aggressive cleanup
    print(f"\nRunning Git maintenance...")
    run_command("git gc --aggressive --prune=now", "Garbage collection")
    run_command("git repack -ad", "Repack repository")
    
    # Final check
    success, output = run_command("git ls-files | wc -l", "Count remaining tracked files")
    if success:
        remaining_count = output.strip()
        print(f"\n🎯 FINAL RESULT: Now tracking {remaining_count} files")
    
    print(f"\n" + "="*50)
    print(f"FORCE CLEANUP COMPLETE!")
    print(f"="*50)

def main():
    """Main function."""
    try:
        force_remove_tracked()
        input("\nPress Enter to exit...")
    except KeyboardInterrupt:
        print("\nCleanup interrupted")
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
