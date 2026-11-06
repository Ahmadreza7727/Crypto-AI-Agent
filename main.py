
import os
import json
import requests
import pandas as pd

from datetime import datetime, timezone

# ==========================================
# CRYPTO AI AGENT V3 - WEEX EDITION
# SPOT + USDT FUTURES
# ==========================================

COINS = ["BTC", "ETH", "BNB", "SOL", "XRP"]

TIMEFRAME = "1h"
CANDLE_LIMIT = 200

CAPITAL_USDT = 250.0
RISK_PER_TRADE_PCT = 1.0

# Temporary planning cap.
# Actual leverage will be configured after
# contract specifications are integrated.
MAX_LEVERAGE = 1.0

ATR_STOP_MULTIPLIER = 1.5
RISK_REWARD_RATIO = 2.0

SPOT_URL = "https://api-spot.weex.com"
FUTURES_URL = "https://api-contract.weex.com"

session = requests.Session()
session.headers.update({
    "User-Agent": "Crypto-AI-Agent/3.0"
})


# ==========================================
# API REQUEST
# ==========================================

def api_get(url, params):

    response = session.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    if isinstance(data, dict):
        raise ValueError(f"WEEX API error: {data}")

    if not isinstance(data, list) or not data:
        raise ValueError("WEEX returned empty market data")

    return data


# ==========================================
# PREPARE CANDLES
# ==========================================

def prepare_dataframe(rows):

    if not rows:
        raise ValueError("No candle data received")

    clean_rows = []

    for row in rows:

        if not isinstance(row, (list, tuple)) or len(row) < 6:
            continue

        clean_rows.append(row)

    if not clean_rows:
        raise ValueError("Invalid candle format")

    df = pd.DataFrame(
        clean_rows,
        columns=None
    )

    df = df.iloc[:, :min(len(df.columns), 11)]

    df = df.iloc[:, :11]

    names = [
        "time", "open", "high", "low", "close",
        "volume", "close_time", "quote_volume",
        "trades", "taker_buy_base", "taker_buy_quote"
    ]

    df.columns = names[:len(df.columns)]

    for col in [
        "open", "high", "low", "close", "volume"
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df["time"] = pd.to_datetime(
        pd.to_numeric(df["time"], errors="coerce"),
        unit="ms",
        utc=True
    )

    if "close_time" in df.columns:

        df["close_time"] = pd.to_numeric(
            df["close_time"],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "time", "open", "high",
            "low", "close", "volume"
        ]
    )

    df = (
        df.sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )

    # Keep only fully closed candles.
    now_ms = int(
        datetime.now(timezone.utc).timestamp() * 1000
    )

    if "close_time" in df.columns:

        df = df[
            df["close_time"].notna()
            & (df["close_time"] < now_ms)
        ]

    else:
        df = df.iloc[:-1]

    df = df.reset_index(drop=True)

    if len(df) < 60:
        raise ValueError(
            f"Not enough closed candles: {len(df)}"
        )

    return df


# ==========================================
# WEEX SPOT DATA
# ==========================================

def fetch_weex_spot(symbol):

    pair = f"{symbol}USDT"

    endpoint = (
        f"{SPOT_URL}/api/v3/market/historyKlines"
    )

    hour_ms = 60 * 60 * 1000

    current_hour = (
        int(datetime.now(timezone.utc).timestamp() * 1000)
        // hour_ms
    ) * hour_ms

    start_time = current_hour - (
        CANDLE_LIMIT * hour_ms
    )

    all_rows = []

    # Spot historical endpoint supports
    # up to 100 candles per request.
    for page in range(2):

        page_start = (
            start_time + page * 100 * hour_ms
        )

        page_end = (
            page_start + 100 * hour_ms - 1
        )

        rows = api_get(
            endpoint,
            {
                "symbol": pair,
                "interval": TIMEFRAME,
                "startTime": page_start,
                "endTime": page_end,
                "limit": 100,
                "priceType": "LAST"
            }
        )

        all_rows.extend(rows)

    return prepare_dataframe(all_rows)


# ==========================================
# WEEX FUTURES DATA
# ==========================================

def fetch_weex_futures(symbol):

    pair = f"{symbol}USDT"

    endpoint = (
        f"{FUTURES_URL}/capi/v3/market/klines"
    )

    rows = api_get(
        endpoint,
        {
            "symbol": pair,
            "interval": TIMEFRAME,
            "limit": CANDLE_LIMIT
        }
    )

    return prepare_dataframe(rows)


# ==========================================
# TECHNICAL INDICATORS
# ==========================================

def calculate_indicators(df):

    df = df.copy()

    close = df["close"]

    # EMA
    df["EMA20"] = close.ewm(
        span=20,
        adjust=False
    ).mean()

    df["EMA50"] = close.ewm(
        span=50,
        adjust=False
    ).mean()

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0, float("nan")
    )

    df["RSI"] = 100 - (
        100 / (1 + rs)
    )

    df.loc[
        (avg_loss == 0) & (avg_gain > 0),
        "RSI"
    ] = 100

    df.loc[
        (avg_loss == 0) & (avg_gain == 0),
        "RSI"
    ] = 50

    # MACD
    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["SIGNAL"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    # ATR
    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = true_range.rolling(
        14
    ).mean()

    return df


# ==========================================
# MARKET ANALYSIS
# ==========================================

def analyze_market(market, symbol, fetch_function):

    df = fetch_function(symbol)

    df = calculate_indicators(df)

    # Last fully closed candle
    last = df.iloc[-1]

    required = [
        "close", "EMA20", "EMA50",
        "RSI", "MACD", "SIGNAL", "ATR"
    ]

    if last[required].isna().any():
        raise ValueError("Invalid indicator values")

    price = float(last["close"])
    ema20 = float(last["EMA20"])
    ema50 = float(last["EMA50"])
    rsi = float(last["RSI"])
    macd = float(last["MACD"])
    macd_signal = float(last["SIGNAL"])
    atr = float(last["ATR"])

    if price <= 0 or atr <= 0:
        raise ValueError("Invalid price or ATR")

    score = 0
    reasons = []

    # EMA trend
    if price > ema20 > ema50:

        score += 2
        reasons.append("Bullish EMA trend")

    elif price < ema20 < ema50:

        score -= 2
        reasons.append("Bearish EMA trend")

    else:

        reasons.append("EMA trend not aligned")

    # RSI momentum
    if 52 < rsi < 68:

        score += 1
        reasons.append("Bullish RSI momentum")

    elif 32 < rsi < 48:

        score -= 1
        reasons.append("Bearish RSI momentum")

    elif rsi >= 70:

        reasons.append("RSI overbought")

    elif rsi <= 30:

        reasons.append("RSI oversold")

    else:

        reasons.append("RSI neutral")

    # MACD
    if macd > macd_signal:

        score += 1
        reasons.append("MACD bullish")

    elif macd < macd_signal:

        score -= 1
        reasons.append("MACD bearish")

    # Direction
    if score >= 3 and price > ema20:

        direction = "BUY"

    elif score <= -3 and price < ema20:

        direction = "SELL"

    else:

        direction = "WAIT"

    # Spot cannot open short positions.
    if market == "SPOT" and direction == "SELL":

        signal = "EXIT_BIAS"

    else:

        signal = direction

    # ==================================
    # RISK MANAGEMENT
    # ==================================

    entry = price
    stop_loss = None
    take_profit = None
    risk_reward = None

    risk_amount = (
        CAPITAL_USDT * RISK_PER_TRADE_PCT / 100
    )

    quantity = 0.0
    position_value = 0.0

    can_trade = (
        direction == "BUY"
        or (
            direction == "SELL"
            and market == "FUTURES"
        )
    )

    if can_trade:

        stop_distance = (
            atr * ATR_STOP_MULTIPLIER
        )

        if direction == "BUY":

            stop_loss = entry - stop_distance

            take_profit = (
                entry
                + stop_distance * RISK_REWARD_RATIO
            )

        else:

            stop_loss = entry + stop_distance

            take_profit = (
                entry
                - stop_distance * RISK_REWARD_RATIO
            )

        if stop_loss <= 0 or take_profit <= 0:
            raise ValueError("Invalid trade levels")

        risk_reward = RISK_REWARD_RATIO

        risk_quantity = (
            risk_amount / stop_distance
        )

        max_notional = (
            CAPITAL_USDT * MAX_LEVERAGE
        )

        max_quantity = (
            max_notional / entry
        )

        quantity = min(
            risk_quantity,
            max_quantity
        )

        position_value = (
            quantity * entry
        )

    # ==================================
    # MARKET ACTION
    # ==================================

    if market == "SPOT" and direction == "SELL":

        action_note = (
            "Spot sell bias. No short position. "
            "Exit decision requires existing holdings."
        )

    elif market == "FUTURES" and direction == "SELL":

        action_note = (
            "Potential futures SHORT setup. "
            "Manual confirmation required."
        )

    elif direction == "BUY":

        action_note = (
            "Potential LONG entry. "
            "Manual confirmation required."
        )

    else:

        action_note = "WAIT: No confirmed trade setup."

    return {

        "exchange": "WEEX",

        "market": market,

        "symbol": f"{symbol}/USDT",

        "price": round(price, 8),

        "timeframe": TIMEFRAME,

        "indicators": {

            "rsi": round(rsi, 2),

            "ema20": round(ema20, 8),

            "ema50": round(ema50, 8),

            "macd": round(macd, 8),

            "macd_signal": round(macd_signal, 8),

            "atr": round(atr, 8)

        },

        "score": score,

        "direction": direction,

        "signal": signal,

        "trade_plan": {

            "entry": (
                round(entry, 8)
                if can_trade else None
            ),

            "stop_loss": (
                round(stop_loss, 8)
                if stop_loss is not None else None
            ),

            "take_profit": (
                round(take_profit, 8)
                if take_profit is not None else None
            ),

            "risk_reward": risk_reward,

            "risk_amount_usdt": round(
                risk_amount, 2
            ),

            "estimated_base_quantity": round(
                quantity, 8
            ),

            "estimated_position_value_usdt": round(
                position_value, 2
            )

        },

        "reasons": reasons,

        "action_note": action_note,

        "candle_time": last["time"].isoformat(),

        "analysis_time": datetime.now(
            timezone.utc
        ).isoformat()

    }


# ==========================================
# MAIN ENGINE
# ==========================================

def main():

    print("=" * 60)

    print("CRYPTO AI AGENT V3")

    print("WEEX SPOT + WEEX USDT FUTURES")

    print("=" * 60)

    print(f"Capital: {CAPITAL_USDT} USDT")

    print(f"Risk: {RISK_PER_TRADE_PCT}%")

    results = []
    errors = []

    markets = [

        ("SPOT", fetch_weex_spot),

        ("FUTURES", fetch_weex_futures)

    ]

    for market, fetch_function in markets:

        print("\n" + "=" * 20)

        print(f"WEEX {market}")

        print("=" * 20)

        for coin in COINS:

            try:

                result = analyze_market(
                    market,
                    coin,
                    fetch_function
                )

                results.append(result)

                plan = result["trade_plan"]

                print(f"\n{coin}/USDT")

                print(f"Price: {result['price']}")

                print(f"Score: {result['score']}")

                print(f"Signal: {result['signal']}")

                if plan["entry"] is not None:

                    print(f"Entry: {plan['entry']}")

                    print(
                        f"Stop Loss: {plan['stop_loss']}"
                    )

                    print(
                        f"Take Profit: {plan['take_profit']}"
                    )

                    print(
                        f"Risk/Reward: {plan['risk_reward']}"
                    )

                    print(
                        f"Quantity: "
                        f"{plan['estimated_base_quantity']}"
                    )

                    print(
                        f"Position Value: "
                        f"{plan['estimated_position_value_usdt']} USDT"
                    )

                for reason in result["reasons"]:

                    print(f"- {reason}")

                print(result["action_note"])

            except Exception as error:

                message = (
                    f"{market} {coin}/USDT: {error}"
                )

                errors.append(message)

                print(f"ERROR: {message}")

    # ==================================
    # SAVE REPORT
    # ==================================

    os.makedirs(
        "reports",
        exist_ok=True
    )

    report = {

        "version": "3.0",

        "exchange": "WEEX",

        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "capital_usdt": CAPITAL_USDT,

        "risk_per_trade_pct": RISK_PER_TRADE_PCT,

        "successful_analyses": len(results),

        "failed_analyses": len(errors),

        "results": results,

        "errors": errors

    }

    filename = datetime.now(
        timezone.utc
    ).strftime(
        "reports/weex_market_%Y%m%d_%H%M%S.json"
    )

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)

    print("WEEX V3 ANALYSIS COMPLETED")

    print(f"Successful: {len(results)}")

    print(f"Failed: {len(errors)}")

    print(f"Report: {filename}")

    if not results:

        raise SystemExit(
            "FATAL: No valid WEEX market data received."
        )

    if errors:

        print("WARNING: Some analyses failed.")

    print("No real orders were placed.")

    print("=" * 60)


if __name__ == "__main__":

    main()
