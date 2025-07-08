#!/usr/bin/env python3
"""
Final Git Cleanup - Remove Remaining 10k Files
More aggressive cleanup to remove all remaining large files from Git.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_git_command(cmd, description, ignore_errors=True):
    """Run a git command with error handling."""
    try:
        print(f"Running: {description}")
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print(f"  ✅ SUCCESS")
            if result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines[:5]:  # Show first 5 lines
                    print(f"     {line}")
                if len(lines) > 5:
                    print(f"     ... and {len(lines) - 5} more lines")
        else:
            if ignore_errors:
                print(f"  ⚠️  WARNING: {result.stderr.strip()}")
            else:
                print(f"  ❌ ERROR: {result.stderr.strip()}")
        return result.returncode == 0
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False

def get_tracked_files():
    """Get list of all tracked files."""
    try:
        result = subprocess.run(['git', 'ls-files'], capture_output=True, text=True)
        if result.returncode == 0:
            files = result.stdout.strip().split('\n')
            return [f for f in files if f.strip()]
        return []
    except:
        return []

def final_cleanup():
    """Final aggressive cleanup."""
    
    print("FINAL GIT CLEANUP - REMOVE REMAINING 10K FILES")
    print("=" * 60)
    
    # Navigate to project root
    project_root = Path("..").resolve()
    os.chdir(project_root)
    print(f"Working in: {project_root}")
    
    # Get current tracked files
    tracked_files = get_tracked_files()
    print(f"\nCurrently tracking {len(tracked_files)} files")
    
    # Analyze what's being tracked
    file_types = {}
    large_files = []
    
    for file_path in tracked_files:
        try:
            path = Path(file_path)
            if path.exists():
                size = path.stat().st_size
                ext = path.suffix.lower()
                
                file_types[ext] = file_types.get(ext, 0) + 1
                
                if size > 100000:  # Files larger than 100KB
                    large_files.append((file_path, size))
        except:
            continue
    
    print(f"\nFile types being tracked:")
    for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
        if count > 10:  # Only show types with many files
            print(f"  {ext or '(no ext)'}: {count} files")
    
    print(f"\nLarge files (>100KB): {len(large_files)}")
    for file_path, size in sorted(large_files, key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {file_path}: {size/1024:.1f} KB")
    
    # Remove specific problematic patterns
    print(f"\nRemoving problematic file patterns...")
    
    patterns_to_remove = [
        # Video files
        "*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv", "*.flv",
        
        # Data files
        "*.npy", "*.npz", "*.pkl", "*.pickle", "*.h5", "*.hdf5",
        
        # Database files
        "*.db", "*.sqlite", "*.sqlite3",
        
        # Archive files
        "*.zip", "*.tar", "*.gz", "*.bz2", "*.rar", "*.7z",
        
        # Large text files
        "*.csv", "*.tsv", "*.json", "*.xml",
        
        # Log files
        "*.log",
        
        # Directories
        "data/", "datasets/", "models/", "instance/", "results/",
        "temp/", "tmp/", "cache/", "downloads/",
        "asl-to-text-ai/data/",
        
        # Specific problematic paths
        "asl-to-text-ai/data/real_asl/",
        "asl-to-text-ai/data/real_asl/wlasl_extracted/",
        "asl-to-text-ai/data/real_asl/wlasl_extracted/videos/",
        
        # Cache and temp
        "__pycache__/", ".pytest_cache/", ".ipynb_checkpoints/",
        
        # Virtual environments
        "venv/", "venv_*/", "ENV/", "env/",
        
        # Kaggle
        "kaggle.json", ".kaggle/"
    ]
    
    for pattern in patterns_to_remove:
        run_git_command(f'git rm -r --cached "{pattern}"', f"Remove {pattern}")
    
    # Force remove any remaining large files
    print(f"\nForce removing remaining large files...")
    for file_path, size in large_files:
        if size > 500000:  # Files larger than 500KB
            run_git_command(f'git rm --cached "{file_path}"', f"Remove large file {file_path}")
    
    # Remove files by extension if many exist
    for ext, count in file_types.items():
        if count > 100 and ext in ['.mp4', '.npy', '.pkl', '.db', '.zip', '.csv', '.json']:
            run_git_command(f'git rm --cached "**/*{ext}"', f"Remove all {ext} files")
    
    # Update .gitignore with even more patterns
    print(f"\nUpdating .gitignore with comprehensive patterns...")
    
    gitignore_content = """
# Comprehensive .gitignore for ML/AI projects

# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class
*.so

# Distribution / packaging
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

# PyInstaller
*.manifest
*.spec

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
venv_*/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# DATA AND MODELS - NEVER COMMIT
data/
datasets/
models/
instance/
results/
temp/
tmp/
cache/
downloads/

# ASL specific data
asl-to-text-ai/data/
asl-to-text-ai/models/

# Machine Learning files
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

# Database files
*.db
*.sqlite
*.sqlite3

# Video and media files
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
*.tiff
*.svg
*.mp3
*.wav
*.flac

# Archive files
*.zip
*.tar
*.gz
*.bz2
*.xz
*.rar
*.7z

# Large data files
*.csv
*.tsv
*.json
*.xml
*.yaml
*.yml

# Logs
*.log
logs/

# Kaggle
kaggle.json
.kaggle/

# Specific large directories
wlasl_extracted/
kaggle_download/
frames/
videos/
"""
    
    gitignore_path = project_root / ".gitignore"
    with open(gitignore_path, 'w') as f:
        f.write(gitignore_content.strip())
    
    # Add .gitignore
    run_git_command("git add .gitignore", "Add updated .gitignore")
    
    # Commit changes
    run_git_command('git commit -m "Final cleanup: remove all large files and update .gitignore"', "Commit final cleanup")
    
    # Aggressive cleanup
    print(f"\nRunning aggressive Git maintenance...")
    run_git_command("git gc --aggressive --prune=now", "Aggressive garbage collection")
    run_git_command("git repack -ad", "Repack repository")
    run_git_command("git prune", "Prune unreachable objects")
    
    # Final status
    print(f"\nFinal status check...")
    final_tracked = get_tracked_files()
    print(f"Files now tracked: {len(final_tracked)}")
    
    # Show what's still being tracked
    if len(final_tracked) > 200:
        print(f"Still tracking {len(final_tracked)} files - analyzing...")
        
        remaining_types = {}
        for file_path in final_tracked:
            try:
                path = Path(file_path)
                ext = path.suffix.lower()
                remaining_types[ext] = remaining_types.get(ext, 0) + 1
            except:
                continue
        
        print("Remaining file types:")
        for ext, count in sorted(remaining_types.items(), key=lambda x: x[1], reverse=True):
            if count > 5:
                print(f"  {ext or '(no ext)'}: {count} files")
    
    # Show repository size
    git_dir = project_root / ".git"
    if git_dir.exists():
        try:
            size_mb = sum(f.stat().st_size for f in git_dir.rglob('*') if f.is_file()) / (1024 * 1024)
            print(f"\nGit repository size: {size_mb:.1f} MB")
        except:
            print("\nCould not calculate repository size")
    
    print(f"\n" + "="*60)
    print(f"FINAL CLEANUP COMPLETE!")
    print(f"Reduced from 20k → 10k → {len(final_tracked)} files")
    print(f"="*60)

def main():
    """Main function."""
    try:
        final_cleanup()
        input("\nPress Enter to exit...")
    except KeyboardInterrupt:
        print("\nCleanup interrupted")
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
