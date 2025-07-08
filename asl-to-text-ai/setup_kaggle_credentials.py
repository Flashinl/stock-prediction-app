#!/usr/bin/env python3
"""
Setup Kaggle Credentials
Moves kaggle.json from Downloads to the correct location.
"""

import os
import shutil
from pathlib import Path

def setup_kaggle_credentials():
    """Move kaggle.json to the correct location."""
    
    print("Setting up Kaggle credentials...")
    
    # Source file (in Downloads)
    downloads_path = Path.home() / "Downloads" / "kaggle.json"
    
    # Target directory
    kaggle_dir = Path.home() / ".kaggle"
    target_path = kaggle_dir / "kaggle.json"
    
    print(f"Looking for: {downloads_path}")
    print(f"Target location: {target_path}")
    
    # Check if source file exists
    if not downloads_path.exists():
        print(f"ERROR: kaggle.json not found in Downloads folder")
        print(f"Please make sure the file is at: {downloads_path}")
        return False
    
    # Create .kaggle directory if it doesn't exist
    kaggle_dir.mkdir(exist_ok=True)
    print(f"Created directory: {kaggle_dir}")
    
    # Copy the file
    try:
        shutil.copy2(downloads_path, target_path)
        print(f"SUCCESS: Copied kaggle.json to {target_path}")
        
        # Set proper permissions (important for security)
        if os.name != 'nt':  # Not Windows
            os.chmod(target_path, 0o600)
            print("Set file permissions to 600")
        
        # Test if it works
        print("\nTesting Kaggle API...")
        import subprocess
        result = subprocess.run(['kaggle', '--version'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("SUCCESS: Kaggle API is working!")
            print(f"Version: {result.stdout.strip()}")
            return True
        else:
            print("WARNING: Kaggle API test failed")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"ERROR: Failed to copy file: {e}")
        return False

def main():
    print("Kaggle Credentials Setup")
    print("=" * 40)
    
    success = setup_kaggle_credentials()
    
    if success:
        print("\n" + "="*40)
        print("SETUP COMPLETE!")
        print("You can now run: python run_preprocessed_wlasl.py")
        print("="*40)
    else:
        print("\n" + "="*40)
        print("SETUP FAILED!")
        print("Please check the error messages above")
        print("="*40)
    
    return 0 if success else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        input("\nPress Enter to exit...")
        exit(exit_code)
    except Exception as e:
        print(f"Unexpected error: {e}")
        input("Press Enter to exit...")
        exit(1)
