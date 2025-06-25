# LLM Stock Price Prediction System

A comprehensive system that leverages Large Language Models (LLMs) to predict stock prices by analyzing financial data, news sentiment, and market indicators.

## 🎯 Project Overview

This project combines the power of LLMs with traditional financial analysis to create a sophisticated stock prediction system that considers:

- **Historical Price Data**: OHLCV data, technical indicators
- **News Sentiment**: Financial news analysis and sentiment scoring
- **Market Context**: Economic indicators, earnings reports, market events
- **Text-based Features**: Company filings, analyst reports, social media sentiment

## 🏗️ Architecture

```
├── data/                   # Data storage and management
│   ├── raw/               # Raw data files
│   ├── processed/         # Cleaned and processed data
│   └── features/          # Engineered features
├── src/                   # Source code
│   ├── data_collection/   # Data gathering modules
│   ├── preprocessing/     # Data cleaning and feature engineering
│   ├── models/           # Model architectures and training
│   ├── evaluation/       # Evaluation and backtesting
│   └── utils/            # Utility functions
├── notebooks/            # Jupyter notebooks for exploration
├── configs/              # Configuration files
├── tests/               # Unit tests
└── scripts/             # Training and inference scripts
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## 📊 Features

- **Multi-modal Data Integration**: Combines numerical and textual data
- **Advanced Feature Engineering**: Technical indicators, sentiment scores, market regime detection
- **Flexible Model Architecture**: Support for various LLM architectures and hybrid approaches
- **Comprehensive Evaluation**: Financial metrics, risk assessment, backtesting
- **Real-time Inference**: API for live predictions

## 🔧 Configuration

The system uses configuration files to manage different aspects:
- `configs/data_config.yaml`: Data collection settings
- `configs/model_config.yaml`: Model architecture and training parameters
- `configs/evaluation_config.yaml`: Evaluation and backtesting settings

## 📈 Usage

### Data Collection
```python
from src.data_collection import StockDataCollector
collector = StockDataCollector()
data = collector.collect_stock_data(['AAPL', 'GOOGL'], start_date='2020-01-01')
```

### Model Training
```python
from src.models import StockPredictionLLM
model = StockPredictionLLM()
model.train(train_data, validation_data)
```

### Prediction
```python
prediction = model.predict('AAPL', context_data)
```

## 🧪 Testing

Run tests with:
```bash
python -m pytest tests/
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This system is for educational and research purposes only. Stock market predictions are inherently uncertain, and this tool should not be used as the sole basis for investment decisions. Always consult with financial professionals and conduct your own research before making investment decisions.
