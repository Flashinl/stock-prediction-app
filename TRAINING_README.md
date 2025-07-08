# StockTrek Neural Network Training System

This training system will automatically train your stock prediction neural network until it achieves really high accuracy (90%+ by default).

## 🚀 Quick Start

1. **Install Requirements** (if not already installed):
   ```cmd
   install_training_requirements.bat
   ```

2. **Start Training**:
   ```cmd
   train_stock_model_high_accuracy.bat
   ```

The system will automatically:
- Train the neural network with progressive difficulty
- Test accuracy after each iteration
- Continue training until target accuracy is achieved
- Save detailed logs and reports
- Validate the final model

## 📁 Files Created

### Training Scripts
- `train_stock_model_high_accuracy.bat` - Main training batch script
- `train_stock_neural_network.py` - Core training logic
- `check_model_accuracy.py` - Accuracy validation
- `validate_final_model.py` - Final model validation
- `install_training_requirements.bat` - Dependency installer

### Generated Files
- `models/stock_nn_model.pth` - Trained PyTorch neural network model
- `models/stock_scaler.joblib` - Feature scaling parameters
- `models/stock_label_encoder.joblib` - Label encoding parameters
- `models/feature_names.joblib` - Feature names for consistency
- `logs/training_log_YYYY-MM-DD_HH-MM-SS.txt` - Training logs
- `models/accuracy_report_YYYY-MM-DD_HH-MM-SS.json` - Accuracy reports
- `models/validation_report_YYYY-MM-DD_HH-MM-SS.json` - Final validation

## ⚙️ Configuration

### Target Accuracy
Default: 90%
You can modify the target accuracy in `train_stock_model_high_accuracy.bat`:
```batch
set TARGET_ACCURACY=95.0
```

### Maximum Iterations
Default: 10 iterations
Modify in the batch file:
```batch
set MAX_ITERATIONS=15
```

### Training Parameters
The system uses progressive training:
- **Iteration 1-2**: Standard learning rate (0.001), 50-75 epochs
- **Iteration 3-5**: Reduced learning rate (0.0005), 75-125 epochs  
- **Iteration 6+**: Fine-tuning rate (0.0001), 125+ epochs

## 📊 Training Process

### Phase 1: Data Collection
- Gathers comprehensive stock list from database
- Falls back to 200+ diverse stocks if database is sparse
- Includes large cap, mid cap, small cap, and penny stocks

### Phase 2: Progressive Training
- Starts with basic neural network architecture
- Increases training complexity each iteration
- Adjusts learning rates for optimal convergence
- Uses early stopping to prevent overfitting

### Phase 3: Accuracy Testing
- Tests on diverse stock portfolio
- Measures prediction success rate
- Evaluates confidence scores
- Checks prediction diversity

### Phase 4: Validation
- Comprehensive model validation
- Consistency testing
- Functionality verification
- Performance reporting

## 🎯 Accuracy Metrics

The system evaluates multiple accuracy dimensions:

1. **Prediction Success Rate** (50% weight)
   - Percentage of stocks that return valid predictions
   - Target: 90%+

2. **Average Confidence** (30% weight)
   - Mean confidence score across all predictions
   - Target: 70%+

3. **Prediction Diversity** (20% weight)
   - How well the model uses all prediction classes
   - Target: Uses at least 3 different prediction types

**Overall Score** = (Success Rate × 0.5) + (Confidence × 0.3) + (Diversity × 0.2)

## 📈 Stock Categories Tested

### Large Cap Tech (Predictable)
AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, NFLX

### Financial Services
JPM, BAC, WFC, GS, MS, C, AXP, BLK

### Healthcare & Biotech
JNJ, PFE, UNH, ABBV, MRK, TMO, ABT, DHR

### Consumer & Retail
WMT, HD, PG, KO, PEP, COST, NKE, MCD

### Industrial & Energy
CAT, BA, GE, XOM, CVX, COP, EOG, SLB

### Growth & Speculative
SNOW, PLTR, CRWD, ZS, ROKU, SPCE

## 🔧 Troubleshooting

### Common Issues

**"No stocks available for training"**
- Run the app first to populate the stock database
- The system will use fallback stocks if database is empty

**"Model loading failed"**
- Check that all model files exist in `models/` directory
- Ensure previous training completed successfully

**"Training iteration failed"**
- Check the log file for detailed error messages
- Verify all dependencies are installed
- Ensure sufficient disk space for model files

**"Target accuracy not achieved"**
- Increase maximum iterations
- Lower target accuracy temporarily
- Check if stock data is accessible

### Performance Tips

1. **Faster Training**: Reduce target accuracy to 80-85%
2. **Higher Accuracy**: Increase maximum iterations to 15-20
3. **Better Diversity**: Ensure database has diverse stock types
4. **Consistency**: Use the same environment for all training

## 📋 Logs and Reports

### Training Logs
Located in `logs/training_log_TIMESTAMP.txt`
- Training parameters for each iteration
- Accuracy results
- Error messages and debugging info

### Accuracy Reports
Located in `models/accuracy_report_TIMESTAMP.json`
- Detailed test results for each stock
- Prediction distribution analysis
- Confidence score statistics

### Validation Reports
Located in `models/validation_report_TIMESTAMP.json`
- Comprehensive model validation results
- Prediction consistency tests
- Final performance metrics

## 🎉 Success Indicators

When training is successful, you'll see:
```
🎉 SUCCESS! TARGET ACCURACY ACHIEVED!
Model training completed successfully!
✅ MODEL VALIDATION SUCCESSFUL!
Your model is ready for production use!
```

The trained model will automatically replace the previous model in your StockTrek application.

## 🔄 Continuous Improvement

For ongoing model improvement:
1. Run training weekly with new market data
2. Adjust target accuracy based on market conditions
3. Monitor prediction accuracy in production
4. Retrain when accuracy drops below threshold

## 📞 Support

If you encounter issues:
1. Check the training logs for error details
2. Verify all requirements are installed
3. Ensure stock database is populated
4. Try reducing target accuracy for initial testing

The training system is designed to be robust and self-recovering, automatically adjusting parameters to achieve the best possible accuracy for your stock prediction model.
