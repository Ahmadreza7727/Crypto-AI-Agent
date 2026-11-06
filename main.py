import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timezone

# ============================================================
# CRYPTO AI AGENT V4
# WEEX SPOT + USDT FUTURES | CMC TOP-50 / CUSTOM UNIVERSE
# Multi-timeframe strategy scanner. No order execution.
# ============================================================

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT, "config.json")
REPORT_DIR = os.path.join(ROOT, "reports")

SPOT_URL = "https://api-spot.weex.com"
FUTURES_URL = "https://api-contract.weex.com"
CMC_KEYLESS_URL = (
    "https://pro-api.coinmarketcap.com/public-api/"
    "v3/cryptocurrency/listings/latest"
)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Crypto-AI-Agent/4.0"})


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg.setdefault("universe", {})
    cfg.setdefault("markets", ["SPOT", "FUTURES"])
    cfg.setdefault("timeframes", ["5m", "15m", "1h", "4h"])
    cfg.setdefault("strategies", {})
    cfg.setdefault("risk", {})
    cfg.setdefault("scanner", {})

    return cfg


def get_json(url, params=None):
    response = SESSION.get(url, params=params, timeout=25)
    response.raise_for_status()
    payload = response.json()

    if isinstance(payload, dict) and payload.get("code") not in (None, "00000", 0, "0"):
        raise ValueError(f"API returned an error: {payload}")

    return payload


def get_weex_symbols(market):
    if market == "SPOT":
        url = f"{SPOT_URL}/api/v3/apiTradingSymbols"
    else:
        url = f"{FUTURES_URL}/capi/v3/market/apiTradingSymbols"

    payload = get_json(url)
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected WEEX symbol response for {market}")

    return {
        str(item).upper()
        for item in payload
        if isinstance(item, str) and str(item).upper().endswith("USDT")
    }


def get_cmc_top_symbols(limit):
    payload = get_json(
        CMC_KEYLESS_URL,
        params={"start": 1, "limit": int(limit), "convert": "USD"}
    )

    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("Unexpected CoinMarketCap listings response")

    excluded = {
        "USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "USDS",
        "USDD", "PYUSD", "USDP", "EURC", "USTC", "UST"
    }
    symbols = []
    for coin in payload["data"]:
        symbol = str(coin.get("symbol", "")).upper().strip()
        if symbol and symbol not in excluded and symbol not in symbols:
            symbols.append(symbol)

    if not symbols:
        raise ValueError("CoinMarketCap returned no usable symbols")

    return symbols


def build_universe(cfg, available_by_market):
    universe_cfg = cfg["universe"]
    mode = str(universe_cfg.get("mode", "top_cmc")).lower()

    if mode == "custom":
        candidates = [
            str(s).upper().replace("/USDT", "").replace("USDT", "").strip()
            for s in universe_cfg.get("symbols", [])
        ]
    else:
        limit = int(universe_cfg.get("limit", 50))
        candidates = get_cmc_top_symbols(limit)

    excluded = {
        str(s).upper()
        for s in universe_cfg.get("exclude_symbols", [])
    }
    candidates = [s for s in candidates if s and s not in excluded]

    matched = {}
    for market in cfg["markets"]:
        available = available_by_market[market]
        matched[market] = [
            symbol for symbol in candidates
            if f"{symbol}USDT" in available
        ]

    return candidates, matched


def timeframe_ms(timeframe):
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    value = int(timeframe[:-1])
    unit = timeframe[-1]
    if unit not in units:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return value * units[unit]


def fetch_candles(market, symbol, timeframe, limit):
    pair = f"{symbol}USDT"

    if market == "SPOT":
        url = f"{SPOT_URL}/api/v3/market/historyKlines"
        interval_ms = timeframe_ms(timeframe)
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        current_open = (now_ms // interval_ms) * interval_ms
        start_ms = current_open - (limit * interval_ms)

        # WEEX spot history endpoint allows up to 100 candles per request.
        all_rows = []
        remaining = limit
        page_start = start_ms
        while remaining > 0:
            page_size = min(100, remaining)
            page_end = page_start + page_size * interval_ms - 1
            rows = get_json(
                url,
                params={
                    "symbol": pair,
                    "interval": timeframe,
                    "startTime": page_start,
                    "endTime": page_end,
                    "limit": page_size,
                    "priceType": "LAST"
                }
            )
            if not isinstance(rows, list):
                raise ValueError(f"Unexpected spot candles for {pair} {timeframe}")
            all_rows.extend(rows)
            page_start += page_size * interval_ms
            remaining -= page_size
            time.sleep(0.08)
        return prepare_dataframe(all_rows)

    url = f"{FUTURES_URL}/capi/v3/market/klines"
    rows = get_json(
        url,
        params={"symbol": pair, "interval": timeframe, "limit": limit}
    )
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected futures candles for {pair} {timeframe}")
    return prepare_dataframe(rows)


def prepare_dataframe(rows):
    clean = [r for r in rows if isinstance(r, (list, tuple)) and len(r) >= 6]
    if not clean:
        raise ValueError("No valid candle rows received")

    df = pd.DataFrame(clean)
    names = [
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote"
    ]
    df = df.iloc[:, :min(len(df.columns), len(names))]
    df.columns = names[:len(df.columns)]

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["time"] = pd.to_datetime(
        pd.to_numeric(df["time"], errors="coerce"),
        unit="ms", utc=True, errors="coerce"
    )

    if "close_time" in df.columns:
        df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce")

    df = df.dropna(subset=["time", "open", "high", "low", "close", "volume"])
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    if "close_time" in df.columns and df["close_time"].notna().any():
        df = df[df["close_time"].notna() & (df["close_time"] < now_ms)]
    else:
        # If close_time is absent, conservatively drop the latest candle.
        df = df.iloc[:-1]

    df = df.reset_index(drop=True)
    if len(df) < 60:
        raise ValueError(f"Not enough closed candles: {len(df)}")

    return df


def calculate_indicators(df):
    df = df.copy()
    close = df["close"]

    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["RSI"] = 100 - (100 / (1 + rs))
    df.loc[(avg_loss == 0) & (avg_gain > 0), "RSI"] = 100
    df.loc[(avg_loss == 0) & (avg_gain == 0), "RSI"] = 50

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

    prev_close = close.shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol_sum = df["volume"].rolling(20).sum()
    df["VWAP20"] = (typical * df["volume"]).rolling(20).sum() / vol_sum.replace(0, float("nan"))

    df["VOL_SMA20"] = df["volume"].rolling(20).mean()
    df["BB_MID"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_UPPER"] = df["BB_MID"] + 2 * bb_std
    df["BB_LOWER"] = df["BB_MID"] - 2 * bb_std

    df["PREV_HIGH20"] = df["high"].shift(1).rolling(20).max()
    df["PREV_LOW20"] = df["low"].shift(1).rolling(20).min()

    return df


def strategy_vote(name, row, prev, settings):
    if not settings.get("enabled", True):
        return 0, "disabled"

    price = float(row["close"])
    rsi = float(row["RSI"])
    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])
    macd = float(row["MACD"])
    macd_signal = float(row["MACD_SIGNAL"])
    volume = float(row["volume"])
    vol_sma = float(row["VOL_SMA20"]) if pd.notna(row["VOL_SMA20"]) else 0

    if name == "ema_rsi_macd":
        if price > ema20 > ema50 and rsi >= settings.get("long_rsi_min", 52) and macd > macd_signal:
            return 1, "EMA trend + RSI + MACD bullish"
        if price < ema20 < ema50 and rsi <= settings.get("short_rsi_max", 48) and macd < macd_signal:
            return -1, "EMA trend + RSI + MACD bearish"
        return 0, "EMA/RSI/MACD conditions not aligned"

    if name == "breakout_volume":
        high = row["PREV_HIGH20"]
        low = row["PREV_LOW20"]
        multiplier = float(settings.get("volume_multiplier", 1.5))
        if pd.notna(high) and price > float(high) and vol_sma > 0 and volume >= vol_sma * multiplier:
            return 1, "20-candle upside breakout with volume"
        if pd.notna(low) and price < float(low) and vol_sma > 0 and volume >= vol_sma * multiplier:
            return -1, "20-candle downside breakdown with volume"
        return 0, "No confirmed volume breakout"

    if name == "bollinger_rsi":
        if price < float(row["BB_LOWER"]) and rsi <= settings.get("oversold_rsi", 30):
            return 1, "Below lower Bollinger Band with oversold RSI"
        if price > float(row["BB_UPPER"]) and rsi >= settings.get("overbought_rsi", 70):
            return -1, "Above upper Bollinger Band with overbought RSI"
        return 0, "No Bollinger/RSI reversal condition"

    if name == "vwap_momentum":
        vwap = row["VWAP20"]
        if pd.isna(vwap):
            return 0, "VWAP unavailable"
        if price > float(vwap) and price > ema20 and rsi > 50 and ema20 > float(prev["EMA20"]):
            return 1, "Price above VWAP with positive momentum"
        if price < float(vwap) and price < ema20 and rsi < 50 and ema20 < float(prev["EMA20"]):
            return -1, "Price below VWAP with negative momentum"
        return 0, "VWAP momentum conditions not aligned"

    return 0, f"Unknown strategy: {name}"


def build_trade_plan(direction, market, price, atr, cfg):
    risk_cfg = cfg["risk"]
    capital = float(risk_cfg.get("capital_usdt", 250))
    risk_pct = float(risk_cfg.get("risk_per_trade_pct", 1.0))
    atr_mult = float(risk_cfg.get("atr_stop_multiplier", 1.5))
    rr = float(risk_cfg.get("risk_reward_ratio", 2.0))
    max_leverage = float(risk_cfg.get("planning_max_leverage", 1.0))

    if direction == "WAIT" or (market == "SPOT" and direction == "SHORT"):
        return {
            "entry": None, "stop_loss": None, "take_profit": None,
            "risk_reward": None, "risk_amount_usdt": round(capital * risk_pct / 100, 2),
            "estimated_base_quantity": 0, "estimated_position_value_usdt": 0
        }

    stop_distance = atr * atr_mult
    if stop_distance <= 0 or price <= 0:
        raise ValueError("Invalid price or ATR for risk calculation")

    if direction == "LONG":
        stop = price - stop_distance
        target = price + stop_distance * rr
    else:
        stop = price + stop_distance
        target = price - stop_distance * rr

    if stop <= 0 or target <= 0:
        raise ValueError("Calculated stop/target is not positive")

    risk_amount = capital * risk_pct / 100
    risk_qty = risk_amount / stop_distance
    max_qty = (capital * max_leverage) / price
    qty = min(risk_qty, max_qty)

    return {
        "entry": round(price, 10),
        "stop_loss": round(stop, 10),
        "take_profit": round(target, 10),
        "risk_reward": rr,
        "risk_amount_usdt": round(risk_amount, 2),
        "estimated_base_quantity": round(qty, 10),
        "estimated_position_value_usdt": round(qty * price, 2)
    }


def analyze_one(market, symbol, timeframe, cfg):
    df = calculate_indicators(
        fetch_candles(
            market, symbol, timeframe,
            int(cfg["scanner"].get("candle_limit", 100))
        )
    )

    row = df.iloc[-1]
    prev = df.iloc[-2]
    needed = [
        "close", "EMA20", "EMA50", "RSI", "MACD",
        "MACD_SIGNAL", "ATR", "BB_UPPER", "BB_LOWER"
    ]
    if row[needed].isna().any():
        raise ValueError("Indicator values are incomplete")

    votes = []
    reasons = []
    strategy_cfg = cfg["strategies"]

    for strategy_name, settings in strategy_cfg.items():
        if not settings.get("enabled", False):
            continue
        vote, reason = strategy_vote(strategy_name, row, prev, settings)
        votes.append(vote)
        reasons.append({"strategy": strategy_name, "vote": vote, "reason": reason})

    bullish = sum(1 for v in votes if v > 0)
    bearish = sum(1 for v in votes if v < 0)
    min_votes = int(cfg["scanner"].get("min_strategy_votes", 2))

    if bullish >= min_votes and bullish > bearish:
        direction = "LONG"
    elif bearish >= min_votes and bearish > bullish:
        direction = "SHORT"
    else:
        direction = "WAIT"

    if market == "SPOT" and direction == "SHORT":
        signal = "EXIT_BIAS"
        plan_direction = "WAIT"
    elif direction == "LONG":
        signal = "BUY"
        plan_direction = "LONG"
    elif direction == "SHORT":
        signal = "SHORT"
        plan_direction = "SHORT"
    else:
        signal = "WAIT"
        plan_direction = "WAIT"

    price = float(row["close"])
    atr = float(row["ATR"])
    plan = build_trade_plan(plan_direction, market, price, atr, cfg)

    return {
        "exchange": "WEEX",
        "market": market,
        "symbol": f"{symbol}/USDT",
        "timeframe": timeframe,
        "price": round(price, 10),
        "candle_time": row["time"].isoformat(),
        "indicators": {
            "rsi": round(float(row["RSI"]), 2),
            "ema20": round(float(row["EMA20"]), 10),
            "ema50": round(float(row["EMA50"]), 10),
            "macd": round(float(row["MACD"]), 10),
            "macd_signal": round(float(row["MACD_SIGNAL"]), 10),
            "atr": round(atr, 10),
            "vwap20": round(float(row["VWAP20"]), 10) if pd.notna(row["VWAP20"]) else None
        },
        "strategy_votes": {"bullish": bullish, "bearish": bearish, "minimum_required": min_votes},
        "direction": direction,
        "signal": signal,
        "trade_plan": plan,
        "strategy_details": reasons,
        "note": "Analysis only. No live order is sent."
    }


def main():
    cfg = load_config()
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=" * 68)
    print("CRYPTO AI AGENT V4 | WEEX SPOT + USDT FUTURES")
    print("=" * 68)
    print("Universe:", cfg["universe"].get("mode", "top_cmc"))
    print("Timeframes:", ", ".join(cfg["timeframes"]))
    print("Markets:", ", ".join(cfg["markets"]))
    print("Enabled strategies:", ", ".join(
        k for k, v in cfg["strategies"].items() if v.get("enabled", False)
    ))

    available = {}
    errors = []
    results = []

    for market in cfg["markets"]:
        try:
            available[market] = get_weex_symbols(market)
            print(f"{market}: {len(available[market])} WEEX API symbols")
        except Exception as exc:
            available[market] = set()
            errors.append(f"{market} symbol list: {exc}")

    try:
        candidates, matched = build_universe(cfg, available)
    except Exception as exc:
        raise SystemExit(f"Could not build scan universe: {exc}")

    print(f"Candidate assets: {len(candidates)}")
    for market in cfg["markets"]:
        print(f"{market} matched assets: {len(matched.get(market, []))}")

    for market in cfg["markets"]:
        for symbol in matched.get(market, []):
            for timeframe in cfg["timeframes"]:
                try:
                    result = analyze_one(market, symbol, timeframe, cfg)
                    results.append(result)
                    print(
                        f"{market:7} {symbol:12} {timeframe:4} "
                        f"{result['signal']:9} price={result['price']}"
                    )
                    if result["signal"] in ("BUY", "SHORT"):
                        print("  PLAN:", result["trade_plan"])
                except Exception as exc:
                    msg = f"{market} {symbol} {timeframe}: {exc}"
                    errors.append(msg)
                    print("ERROR:", msg)
                time.sleep(float(cfg["scanner"].get("request_pause_seconds", 0.08)))

    timestamp = datetime.now(timezone.utc)
    report = {
        "version": "4.0",
        "exchange": "WEEX",
        "generated_at": timestamp.isoformat(),
        "universe_mode": cfg["universe"].get("mode", "top_cmc"),
        "candidate_assets": candidates,
        "matched_assets": matched,
        "markets": cfg["markets"],
        "timeframes": cfg["timeframes"],
        "successful_analyses": len(results),
        "failed_analyses": len(errors),
        "results": results,
        "errors": errors,
        "safety": {
            "live_trading_enabled": False,
            "orders_placed": 0,
            "message": "Signals and risk plans only. Manual review and execution remain with the user."
        }
    }

    filename = timestamp.strftime("weex_v4_market_%Y%m%d_%H%M%S.json")
    report_path = os.path.join(REPORT_DIR, filename)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    buy_count = sum(1 for x in results if x["signal"] == "BUY")
    short_count = sum(1 for x in results if x["signal"] == "SHORT")
    wait_count = sum(1 for x in results if x["signal"] in ("WAIT", "EXIT_BIAS"))

    print("\n" + "=" * 68)
    print("V4 SCAN COMPLETED")
    print(f"Successful analyses: {len(results)}")
    print(f"Failed analyses: {len(errors)}")
    print(f"BUY setups: {buy_count}")
    print(f"SHORT setups: {short_count}")
    print(f"WAIT / EXIT_BIAS: {wait_count}")
    print(f"Report: {report_path}")
    print("No real orders were placed.")
    print("=" * 68)

    if not results:
        raise SystemExit("No valid WEEX market analyses were produced.")


if __name__ == "__main__":
    main()
