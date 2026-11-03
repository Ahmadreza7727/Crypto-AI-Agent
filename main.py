
import os
import json
import requests
import pandas as pd

from datetime import datetime, timezone, timedelta

# =====================================
# CRYPTO AI AGENT V2
# LBANK SPOT + WEEX USDT FUTURES
# =====================================

COINS = ["BTC", "ETH", "BNB", "SOL", "XRP"]

TIMEFRAME = "1h"
CANDLE_LIMIT = 200

LBANK_URL = "https://api.lbank.info/v2/kline.do"
WEEX_URL = "https://api-contract.weex.com/capi/v3/market/klines"

session = requests.Session()
session.headers.update({
    "User-Agent": "Crypto-AI-Agent/2.0"
})


def fetch_lbank(symbol):
    params = {
        "symbol": f"{symbol.lower()}_usdt",
        "size": CANDLE_LIMIT,
        "type": "hour1",
        "time": int(
            (datetime.now(timezone.utc) -
             timedelta(hours=CANDLE_LIMIT + 2)).timestamp()
        )
    }

    response = session.get(
        LBANK_URL, params=params, timeout=25
    )
    response.raise_for_status()

    result = response.json()

    if result.get("result") != "true":
        raise ValueError(f"LBank API error: {result}")

    rows = result.get("data", [])

    if not rows:
        raise ValueError("LBank returned empty candles")

    df = pd.DataFrame(
        rows,
        columns=["time", "open", "high", "low", "close", "volume"]
    )

    df["time"] = pd.to_datetime(
        pd.to_numeric(df["time"]), unit="s", utc=True
    )

    return prepare_dataframe(df)


def fetch_weex(symbol):
    params = {
        "symbol": f"{symbol}USDT",
        "interval": "1h",
        "limit": CANDLE_LIMIT
    }

    response = session.get(
        WEEX_URL, params=params, timeout=25
    )
    response.raise_for_status()

    rows = response.json()

    if not isinstance(rows, list) or not rows:
        raise ValueError(f"WEEX returned invalid candles: {rows}")

    df = pd.DataFrame(
        rows,
        columns=[
            "time", "open", "high", "low", "close",
            "volume", "close_time", "quote_volume",
            "trades", "taker_buy_base", "taker_buy_quote"
        ]
    )

    df["time"] = pd.to_datetime(
        pd.to_numeric(df["time"]), unit="ms", utc=True
    )

    return prepare_dataframe(df)


def prepare_dataframe(df):
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["time", "open", "high", "low", "close", "volume"]
    )

    df = df.sort_values("time").drop_duplicates("time")
    df = df.reset_index(drop=True)

    if len(df) < 60:
        raise ValueError(
            f"Not enough candles: received {len(df)}, need at least 60"
        )

    return df


def calculate_indicators(df):
    df = df.copy()

    close = df["close"]

    df["EMA20"] = close.ewm(
        span=20, adjust=False
    ).mean()

    df["EMA50"] = close.ewm(
        span=50, adjust=False
    ).mean()

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))

    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(
        span=12, adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26, adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["SIGNAL"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = true_range.rolling(14).mean()

    return df


def analyze_market(exchange, symbol, fetch_function):
    df = fetch_function(symbol)
    df = calculate_indicators(df)

    # Use the last fully closed candle
    last = df.iloc[-2]

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

    if 50 < rsi < 70:
        score += 1
        reasons.append("Positive RSI momentum")

    elif 30 < rsi < 50:
        score -= 1
        reasons.append("Negative RSI momentum")

    elif rsi >= 70:
        reasons.append("RSI overbought")

    elif rsi <= 30:
        reasons.append("RSI oversold")

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
        "exchange": exchange,
        "market": (
            "SPOT" if exchange == "LBank"
            else "USDT_FUTURES"
        ),
        "symbol": f"{symbol}/USDT",
        "price": round(price, 8),
        "rsi": round(rsi, 2),
        "ema20": round(ema20, 8),
        "ema50": round(ema50, 8),
        "macd": round(macd, 8),
        "atr": round(atr, 8),
        "score": score,
        "decision": decision,
        "reasons": reasons,
        "candle_time": last["time"].isoformat(),
        "analysis_time": datetime.now(
            timezone.utc
        ).isoformat()
    }


def main():
    print("=" * 55)
    print("CRYPTO AI AGENT V2")
    print("LBANK SPOT + WEEX USDT FUTURES")
    print("=" * 55)

    results = []
    errors = []

    exchanges = [
        ("LBank", fetch_lbank),
        ("WEEX", fetch_weex)
    ]

    for exchange, fetch_function in exchanges:
        print(f"\n{'=' * 20} {exchange} {'=' * 20}")

        for coin in COINS:
            try:
                result = analyze_market(
                    exchange, coin, fetch_function
                )

                results.append(result)

                print(f"\n{coin}/USDT")
                print(f"Price: {result['price']}")
                print(f"RSI: {result['rsi']}")
                print(f"EMA20: {result['ema20']}")
                print(f"EMA50: {result['ema50']}")
                print(f"MACD: {result['macd']}")
                print(f"Score: {result['score']}")
                print(f"Signal: {result['decision']}")

                for reason in result["reasons"]:
                    print(f"- {reason}")

            except Exception as error:
                message = (
                    f"{exchange} {coin}/USDT: {error}"
                )

                errors.append(message)
                print(f"ERROR: {message}")

    os.makedirs("reports", exist_ok=True)

    report = {
        "version": "2.0",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "successful_analyses": len(results),
        "failed_analyses": len(errors),
        "results": results,
        "errors": errors
    }

    filename = datetime.now(
        timezone.utc
    ).strftime("reports/market_%Y%m%d_%H%M%S.json")

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 55)
    print(f"Successful analyses: {len(results)}")
    print(f"Failed analyses: {len(errors)}")
    print(f"Report saved: {filename}")

    if not results:
        raise SystemExit(
            "FATAL: No valid market data received from either exchange."
        )

    if errors:
        print("WARNING: Some market data requests failed.")

    print("Analysis completed.")


if __name__ == "__main__":
    main()
