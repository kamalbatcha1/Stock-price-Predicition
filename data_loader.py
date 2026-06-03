import yfinance as yf
import config


def load_data():

    df = yf.download(
        config.TICKER,
        start=config.START_DATE,
        auto_adjust=True
    )

    df = df[["Open", "High", "Low", "Close", "Volume"]]

    return df