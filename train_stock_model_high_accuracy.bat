@echo off
echo ========================================
echo StockTrek Neural Network Training Script
echo Training Until High Accuracy Achieved
echo ========================================
echo.

:: Set target accuracy (90%+)
set TARGET_ACCURACY=90.0
set MAX_ITERATIONS=10
set CURRENT_ITERATION=1

echo Target Accuracy: %TARGET_ACCURACY%%%
echo Maximum Training Iterations: %MAX_ITERATIONS%
echo.

:: Create models directory if it doesn't exist
if not exist "models" mkdir models

:: Create logs directory for training logs
if not exist "logs" mkdir logs

:: Set log file with simple timestamp
set LOG_FILE=logs\training_log_%RANDOM%.txt

echo Training started at %date% %time% > %LOG_FILE%
echo Target accuracy: %TARGET_ACCURACY%%% >> %LOG_FILE%
echo. >> %LOG_FILE%

:TRAINING_LOOP
echo ========================================
echo Training Iteration %CURRENT_ITERATION% of %MAX_ITERATIONS%
echo ========================================
echo.

echo [%time%] Starting training iteration %CURRENT_ITERATION%... >> %LOG_FILE%

:: Run the training script
echo Running neural network training...
python train_stock_neural_network.py --target-accuracy %TARGET_ACCURACY% --iteration %CURRENT_ITERATION% --log-file %LOG_FILE%

:: Check if training was successful
if %ERRORLEVEL% EQU 0 (
    echo [%time%] Training iteration %CURRENT_ITERATION% completed successfully >> %LOG_FILE%
    echo ✅ Training iteration %CURRENT_ITERATION% completed successfully!
    echo.
    
    :: Check if we achieved target accuracy
    python check_model_accuracy.py --target %TARGET_ACCURACY%
    if %ERRORLEVEL% EQU 0 (
        echo [%time%] 🎉 TARGET ACCURACY ACHIEVED! >> %LOG_FILE%
        echo ========================================
        echo 🎉 SUCCESS! TARGET ACCURACY ACHIEVED!
        echo ========================================
        echo.
        echo Model training completed successfully!
        echo Check the models/ directory for your trained model.
        echo Training log saved to: %LOG_FILE%
        echo.
        goto :SUCCESS
    ) else (
        echo [%time%] Target accuracy not yet achieved, continuing training... >> %LOG_FILE%
        echo Target accuracy not yet achieved. Continuing training...
        echo.
    )
) else (
    echo [%time%] ERROR: Training iteration %CURRENT_ITERATION% failed >> %LOG_FILE%
    echo ❌ Training iteration %CURRENT_ITERATION% failed!
    echo Check the log file for details: %LOG_FILE%
    echo.
)

:: Increment iteration counter
set /a CURRENT_ITERATION+=1

:: Check if we've reached max iterations
if %CURRENT_ITERATION% LEQ %MAX_ITERATIONS% (
    echo Preparing for next training iteration...
    echo.
    timeout /t 5 /nobreak > nul
    goto :TRAINING_LOOP
) else (
    echo [%time%] Maximum iterations reached without achieving target accuracy >> %LOG_FILE%
    echo ========================================
    echo Maximum iterations reached!
    echo ========================================
    echo.
    echo Completed %MAX_ITERATIONS% training iterations.
    echo Check the models/ directory for the best model achieved.
    echo Training log saved to: %LOG_FILE%
    echo.
    goto :END
)

:SUCCESS
echo Final model validation...
python validate_final_model.py
echo.
echo Training log saved to: %LOG_FILE%
echo.

:END
echo ========================================
echo Training Script Complete
echo ========================================
echo.
echo Check these files:
echo - models/stock_nn_model.pth (trained model)
echo - models/stock_scaler.joblib (feature scaler)
echo - models/stock_label_encoder.joblib (label encoder)
echo - %LOG_FILE% (training log)
echo.
pause
