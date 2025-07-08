@echo off
echo ========================================
echo WLASL Real ASL Dataset Training
echo ========================================
echo.

echo Step 1: Installing required packages...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python numpy tqdm matplotlib requests urllib3

echo.
echo Step 2: Downloading and processing WLASL dataset...
python download_and_process_wlasl.py

echo.
echo Step 3: Training model on real WLASL data...
python train_wlasl_model.py

echo.
echo ========================================
echo Training Complete!
echo ========================================
pause
