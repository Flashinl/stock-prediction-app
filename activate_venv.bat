@echo off
echo Activating StockTrek virtual environment...
call venv_stocktrek\Scripts\activate.bat
echo.
echo Virtual environment activated!
echo Python version:
python --version
echo.
echo To run the application:
echo   python app.py
echo.
echo To run tests:
echo   python test_deployment.py
echo   python test_neural_network.py
echo.
cmd /k
