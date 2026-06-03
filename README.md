
## 🚀 Project Overview
This project builds a deep learning-based trading system using an LSTM model to predict stock price direction (up/down) using technical indicators and historical price data.  
The system also includes a backtesting module to evaluate trading strategy performance against the market.

---

## 🧠 Problem Statement
Financial markets are noisy and non-linear. The goal is not exact price prediction, but **directional movement forecasting** to simulate a trading strategy.

---

## 📊 Dataset
- Source: Yahoo Finance (AAPL stock)
- Time period: 2010 - Present
- Features: OHLCV (Open, High, Low, Close, Volume)

---

## 🔧 Feature Engineering
Technical indicators used:

- EMA (10, 20, 50, 100)
- MACD
- RSI
- ATR (volatility)
- Bollinger Band Width
- Momentum
- ROC (Rate of Change)
- Log Returns
- Volume Change
- Volatility measures

---

## 🏗️ Model Architecture
- LSTM (2 layers)
- Hidden size: 128
- Dropout: 0.3
- Fully connected layers
- Binary classification output (Up/Down)

---

## ⚙️ Training Setup
- Loss: BCEWithLogitsLoss (with class weighting)
- Optimizer: Adam
- Sequence length: 60–90 timesteps (experimented)
- Batch size: 64

---

## 📈 Evaluation Metrics
- Accuracy (~58–65%)
- Precision / Recall / F1-score
- Confusion Matrix
- Probability distribution analysis

---

## 💰 Backtesting Strategy
- If model predicts UP → long position
- Otherwise → no position
- Strategy returns compared with market returns

---

## 📊 Results
- Directional Accuracy: ~59%
- Strategy Return: ~1.8x–2.0x
- Market Return: ~2.0x+

---

## ⚠️ Limitations
- Financial data is highly noisy and non-stationary
- Model does not guarantee profitability
- Performance varies across time periods

---

## 🚀 Future Improvements
- Transformer-based models for time series
- Portfolio-level prediction
- Risk-adjusted returns (Sharpe ratio)

---

## 🧑‍💻 Tech Stack
- Python
- PyTorch
- Pandas / NumPy
- Scikit-learn
- Matplotlib
- yfinance
