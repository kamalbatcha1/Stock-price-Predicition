import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import numpy as np

import config

from data_loader import load_data
from features import add_features
from dataset import create_sequences, StockDataset
from model import LSTMModel
from train import train
from evaluate import evaluate
from backtest import backtest


df = load_data()
df = add_features(df)

features = [
    "Close",
    "Volume",
    "EMA10",
    "EMA20",
    "EMA50",
    "EMA100",
    "EMA_DIFF",
    "MACD",
    "RSI",
    "ATR",
    "Momentum",
    "High_Low",
    "Open_Close",
    "BB_WIDTH",
    "Return_3",
    "Return_5",
    "ROC10",
    "ROC20",
    "Volume_Change",
    "Volatility10",
    "Volatility20"
]

X, y = create_sequences(df[features].values, df["Target"].values, config.SEQ_LEN)

split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=config.BATCH_SIZE)
test_loader = DataLoader(StockDataset(X_test, y_test), batch_size=config.BATCH_SIZE)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = LSTMModel(input_size=len(features)).to(device)

pos = np.sum(y_train == 1)
neg = np.sum(y_train == 0)

pos_weight = neg / pos

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

train(model, train_loader, criterion, optimizer, device)

preds = evaluate(model, test_loader, device)

backtest(df, preds)