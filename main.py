import ccxt
import pandas as pd
import time

# Exchange connection (public market data only)
exchange = ccxt.binance({
    "enableRateLimit": True
})

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
CANDLE_LIMIT = 100


def get_market_data():
    candles = exchange.fetch_ohlcv(
        SYMBOL,
        timeframe=TIMEFRAME,
        limit=CANDLE_LIMIT
    )

    df = pd.DataFrame(
        candles,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    return df


def analyze_market(df):
    df["EMA_20"] = df["close"].ewm(span=20).mean()
    df["EMA_50"] = df["close"].ewm(span=50).mean()

    latest = df.iloc[-1]

    if latest["EMA_20"] > latest["EMA_50"]:
        trend = "BULLISH"
    elif latest["EMA_20"] < latest["EMA_50"]:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    return trend, latest["close"]


def main():
    print("Crypto AI Agent Started")

    try:
        df = get_market_data()
        trend, price = analyze_market(df)

        print(f"Symbol: {SYMBOL}")
        print(f"Price: {price}")
        print(f"Market Trend: {trend}")

    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
