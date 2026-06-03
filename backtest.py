import matplotlib.pyplot as plt


def backtest(df, preds):

    test = df.iloc[-len(preds):].copy()

    test["Signal"] = preds

    test["Market_Return"] = test["Close"].pct_change().fillna(0)

    test["Strategy_Return"] = test["Signal"].shift(1) * test["Market_Return"]

    test["Cumulative_Market"] = (1 + test["Market_Return"]).cumprod()
    test["Cumulative_Strategy"] = (1 + test["Strategy_Return"]).cumprod()

    print("\nMarket:", test["Cumulative_Market"].iloc[-1])
    print("Strategy:", test["Cumulative_Strategy"].iloc[-1])

    plt.plot(test["Cumulative_Market"], label="Market")
    plt.plot(test["Cumulative_Strategy"], label="Strategy")
    plt.legend()
    plt.show()