#!/usr/bin/env python3
"""
Stop any running processes and start optimized training
"""

import os
import sys
import subprocess
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def stop_python_processes():
    """Stop any running Python processes that might be causing issues"""
    try:
        # Try to find and stop Python processes
        if os.name == 'nt':  # Windows
            try:
                # Kill any Python processes that might be running the problematic script
                subprocess.run(['taskkill', '/F', '/IM', 'python.exe'], 
                             capture_output=True, check=False)
                subprocess.run(['taskkill', '/F', '/IM', 'python3.exe'], 
                             capture_output=True, check=False)
                logger.info("Stopped any running Python processes")
            except Exception as e:
                logger.warning(f"Could not stop processes: {e}")
        else:  # Unix-like
            try:
                subprocess.run(['pkill', '-f', 'python'], check=False)
                logger.info("Stopped any running Python processes")
            except Exception as e:
                logger.warning(f"Could not stop processes: {e}")
                
        # Wait a moment for processes to stop
        time.sleep(2)
        
    except Exception as e:
        logger.warning(f"Error stopping processes: {e}")

def install_dependencies():
    """Install any missing dependencies"""
    try:
        logger.info("Installing missing dependencies...")
        
        dependencies = [
            'lxml',
            'beautifulsoup4', 
            'requests-ratelimiter',
            'pandas',
            'numpy',
            'scikit-learn',
            'torch',
            'yfinance'
        ]
        
        for dep in dependencies:
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                             check=True, capture_output=True)
                logger.info(f"Installed {dep}")
            except subprocess.CalledProcessError:
                logger.warning(f"Could not install {dep}, might already be installed")
                
    except Exception as e:
        logger.error(f"Error installing dependencies: {e}")

def run_optimized_training():
    """Run the optimized training script"""
    try:
        logger.info("Starting optimized Kaggle training...")
        
        # Run the optimized training script
        result = subprocess.run([sys.executable, 'train_kaggle_optimized.py'], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            logger.info("Training completed successfully!")
            return True
        else:
            logger.error(f"Training failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"Error running training: {e}")
        return False

def main():
    """Main function"""
    logger.info("Starting stop and train process...")
    
    # Stop any running processes
    stop_python_processes()
    
    # Install dependencies
    install_dependencies()
    
    # Run optimized training
    success = run_optimized_training()
    
    if success:
        logger.info("All processes completed successfully!")
        sys.exit(0)
    else:
        logger.error("Training process failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
