import numpy as np
import pandas as pd


def add_features(df):

    df = df.copy()
# hera I add return
    df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
#  hera I add Moving Average
    df["EMA10"] = df["Close"].ewm(span=10).mean()
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA_DIFF"] = df["EMA10"] - df["EMA20"]

    
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26

  
    mean = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()

    df["BB_WIDTH"] = (mean + 2*std) - (mean - 2*std)

   
    high_low = df["High"] - df["Low"]
    high_close = abs(df["High"] - df["Close"].shift(1))
    low_close = abs(df["Low"] - df["Close"].shift(1))

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    #momentum
    df["Momentum"] = df["Close"] - df["Close"].shift(10)
    #price action
    df["High_Low"] = df["High"] - df["Low"]
    df["Open_Close"] = df["Open"] - df["Close"]


    #rsi
    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)

    df["RSI"] = 100 - (100 / (1 + rs))
    #target
    future_return = (
    df["Close"].shift(-5) - df["Close"]) / df["Close"]
    df["Target"] = (future_return > 0.02).astype(int)

    df["Return_3"] = df["Close"].pct_change(3)
    df["Return_5"] = df["Close"].pct_change(5)

    # Long-term Trend
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["EMA100"] = df["Close"].ewm(span=100).mean()

    # Rate of Change
    df["ROC10"] = df["Close"].pct_change(10)
    df["ROC20"] = df["Close"].pct_change(20)

    # Volume Features
    df["Volume_Change"] = df["Volume"].pct_change()

    # Volatility
    df["Volatility10"] = df["Return"].rolling(10).std()
    df["Volatility20"] = df["Return"].rolling(20).std()
    df["Volatility_Regime"] = df["Volatility20"] > df["Volatility20"].rolling(50).mean()


    df.dropna(inplace=True)

    return df