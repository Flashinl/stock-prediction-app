@echo off
echo ========================================
echo Installing StockTrek Training Requirements
echo ========================================
echo.

echo Installing core machine learning packages...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn>=1.1.0
pip install numpy>=1.21.0
pip install pandas>=1.4.0
pip install joblib>=1.1.0

echo.
echo Installing financial data packages...
pip install yfinance>=0.1.87
pip install alpha-vantage>=2.3.1

echo.
echo Installing Flask and database packages...
pip install Flask>=2.2.0
pip install Flask-SQLAlchemy>=2.5.1

echo.
echo Installing utility packages...
pip install python-dotenv>=0.19.0
pip install requests>=2.28.0

echo.
echo ========================================
echo Requirements Installation Complete!
echo ========================================
echo.
echo You can now run the training script:
echo train_stock_model_high_accuracy.bat
echo.
pause
