#!/usr/bin/env python3
"""
Simple WLASL Training Script
Fixed for Windows Command Prompt - no Unicode characters.
"""

import sys
import os
from pathlib import Path

def main():
    print("WLASL Real ASL Dataset Training - 2000 WORDS x 25 SAMPLES")
    print("=" * 65)
    print("Target: 50,000 total samples (2000 classes x 25 samples each)")
    print("Step 1: Processing dataset (this will take several hours)...")
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Run dataset processing
    try:
        import subprocess
        result = subprocess.run([sys.executable, 'download_and_process_wlasl.py'], 
                              capture_output=False, text=True)
        
        if result.returncode != 0:
            print("Dataset processing failed!")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    print("\nStep 2: Training model...")
    
    # Run training
    try:
        result = subprocess.run([sys.executable, 'train_wlasl_model.py'], 
                              capture_output=False, text=True)
        
        if result.returncode != 0:
            print("Training failed!")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    print("\nTraining complete!")
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        input("Press Enter to exit...")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...")
        sys.exit(1)
