
import os
import json
import requests
import pandas as pd
from datetime import datetime, timezone

# =========================
# CRYPTO AI AGENT - V1
# =========================

COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]

TIMEFRAME = "1h"
CANDLE_LIMIT = 200

BASE_URL = "https://api.binance.com"


def get_market_data(symbol):
    url = f"{BASE_URL}/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": TIMEFRAME,
        "limit": CANDLE_LIMIT
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    data = response.json()

    columns = [
        "time", "open", "high", "low", "close",
        "volume", "close_time", "quote_volume",
        "trades", "taker_buy_base", "taker_buy_quote", "ignore"
    ]

    df = pd.DataFrame(data, columns=columns)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])

    return df


def calculate_indicators(df):
    close = df["close"]

    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))

    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["ATR"] = (
        pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs()
        ], axis=1).max(axis=1)
    ).rolling(14).mean()

    return df


def analyze_market(symbol):
    df = get_market_data(symbol)
    df = calculate_indicators(df)

    last = df.iloc[-2]
    previous = df.iloc[-3]

    price = float(last["close"])
    ema20 = float(last["EMA20"])
    ema50 = float(last["EMA50"])
    rsi = float(last["RSI"])
    macd = float(last["MACD"])
    signal = float(last["SIGNAL"])
    atr = float(last["ATR"])

    score = 0
    reasons = []

    if price > ema20 > ema50:
        score += 2
        reasons.append("Bullish EMA trend")

    elif price < ema20 < ema50:
        score -= 2
        reasons.append("Bearish EMA trend")

    if rsi > 50 and rsi < 70:
        score += 1
        reasons.append("Positive RSI momentum")

    elif rsi < 50 and rsi > 30:
        score -= 1
        reasons.append("Negative RSI momentum")

    if macd > signal:
        score += 1
        reasons.append("MACD bullish")

    else:
        score -= 1
        reasons.append("MACD bearish")

    if score >= 3:
        decision = "BUY BIAS"
    elif score <= -3:
        decision = "SELL BIAS"
    else:
        decision = "NEUTRAL"

    return {
        "symbol": symbol,
        "price": round(price, 4),
        "rsi": round(rsi, 2),
        "ema20": round(ema20, 4),
        "ema50": round(ema50, 4),
        "macd": round(macd, 6),
        "atr": round(atr, 6),
        "score": score,
        "decision": decision,
        "reasons": reasons,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


def main():
    print("=" * 50)
    print("CRYPTO AI AGENT - MARKET ANALYSIS")
    print("=" * 50)

    results = []

    for coin in COINS:
        try:
            result = analyze_market(coin)
            results.append(result)

            print(f"\n{coin}")
            print(f"Price: {result['price']}")
            print(f"RSI: {result['rsi']}")
            print(f"Score: {result['score']}")
            print(f"Signal: {result['decision']}")

            for reason in result["reasons"]:
                print(f"- {reason}")

        except Exception as error:
            print(f"\nError analyzing {coin}: {error}")

    os.makedirs("reports", exist_ok=True)

    filename = datetime.now(timezone.utc).strftime(
        "reports/market_%Y%m%d_%H%M%S.json"
    )

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print("\nAnalysis completed.")
    print(f"Report saved: {filename}")


if __name__ == "__main__":
    main()
