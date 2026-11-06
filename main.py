import os
import json
import time
import logging
from datetime import datetime, timezone

import requests
import pandas as pd

# ============================================================
# CRYPTO AI AGENT V4.1
# WEEX SPOT + USDT FUTURES | CMC TOP-N / COINGECKO FALLBACK
# Multi-timeframe analysis only. NO live order execution.
# ============================================================

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT, "config.json")
REPORT_DIR = os.path.join(ROOT, "reports")

SPOT_URL = "https://api-spot.weex.com"
FUTURES_URL = "https://api-contract.weex.com"
CMC_URL = "https://pro-api.coinmarketcap.com/public-api/v3/cryptocurrency/listings/latest"
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Crypto-AI-Agent/4.1"})
UNIVERSE_SOURCE = "unknown"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("crypto-agent")


def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError("config.json was not found next to main.py")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg.setdefault("universe", {})
    cfg["universe"].setdefault("mode", "top_cmc")
    cfg["universe"].setdefault("limit", 50)
    cfg["universe"].setdefault("symbols", [])
    cfg["universe"].setdefault("exclude_symbols", [])

    cfg.setdefault("markets", ["SPOT", "FUTURES"])
    cfg.setdefault("timeframes", ["5m", "15m", "1h", "4h"])
    cfg.setdefault("strategies", {})
    cfg.setdefault("risk", {})
    cfg.setdefault("scanner", {})

    cfg["scanner"].setdefault("candle_limit", 100)
    cfg["scanner"].setdefault("request_pause_seconds", 0.15)
    cfg["scanner"].setdefault("min_strategy_votes", 1)
    cfg["scanner"].setdefault("api_retries", 3)
    cfg["scanner"].setdefault("retry_base_seconds", 1.5)

    cfg["risk"].setdefault("capital_usdt", 250)
    cfg["risk"].setdefault("risk_per_trade_pct", 1.0)
    cfg["risk"].setdefault("atr_stop_multiplier", 1.5)
    cfg["risk"].setdefault("risk_reward_ratio", 2.0)
    cfg["risk"].setdefault("planning_max_leverage", 1.0)

    # Validate core configuration early.
    cfg["markets"] = [str(x).upper() for x in cfg["markets"]]
    invalid_markets = set(cfg["markets"]) - {"SPOT", "FUTURES"}
    if invalid_markets:
        raise ValueError(f"Unsupported markets in config: {sorted(invalid_markets)}")
    if not cfg["markets"]:
        raise ValueError("At least one market must be enabled")
    if not cfg["timeframes"]:
        raise ValueError("At least one timeframe must be enabled")
    if int(cfg["scanner"]["candle_limit"]) < 60:
        raise ValueError("scanner.candle_limit must be at least 60")

    return cfg


def request_json(url, params=None, headers=None, retries=3, timeout=25):
    """GET JSON with bounded retries for transient errors and rate limits."""
    last_error = None
    for attempt in range(max(1, int(retries))):
        try:
            response = SESSION.get(
                url, params=params, headers=headers, timeout=timeout
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                try:
                    wait = float(retry_after) if retry_after else 10 * (attempt + 1)
                except (TypeError, ValueError):
                    wait = 10 * (attempt + 1)
                last_error = RuntimeError(
                    f"HTTP 429 rate limit from {url}; retry after {wait:.0f}s"
                )
                if attempt < retries - 1:
                    log.warning(str(last_error))
                    time.sleep(min(wait, 45))
                    continue
                raise last_error

            if response.status_code in (500, 502, 503, 504):
                last_error = RuntimeError(f"Temporary HTTP {response.status_code} from {url}")
                if attempt < retries - 1:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise last_error

            response.raise_for_status()
            payload = response.json()

            # WEEX APIs commonly use code=00000 for success.
            if isinstance(payload, dict) and "code" in payload:
                code = str(payload.get("code"))
                if code not in ("00000", "0", "None"):
                    raise RuntimeError(
                        f"API returned code={code}: {payload.get('msg', payload)}"
                    )
            return payload

        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
        except requests.HTTPError:
            raise
        except ValueError:
            # Invalid JSON should not be retried as if it were a rate limit.
            raise

    raise RuntimeError(f"Request failed: {last_error}")


def get_weex_symbols(market, cfg):
    if market == "SPOT":
        url = f"{SPOT_URL}/api/v3/apiTradingSymbols"
    else:
        url = f"{FUTURES_URL}/capi/v3/market/apiTradingSymbols"

    payload = request_json(
        url,
        retries=int(cfg["scanner"]["api_retries"])
    )

    # Accommodate either a plain list or common data-wrapped responses.
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("symbols"))

    if not isinstance(payload, list):
        raise ValueError(f"Unexpected WEEX symbol response for {market}: {type(payload).__name__}")

    symbols = set()
    for item in payload:
        if isinstance(item, str):
            raw = item
        elif isinstance(item, dict):
            raw = item.get("symbol", "")
            status = str(item.get("status", "")).lower()
            if status and status not in ("online", "trading", "1", "true"):
                continue
        else:
            continue

        raw = str(raw).upper().replace("/", "").replace("-", "").strip()
        if raw.endswith("USDT"):
            symbols.add(raw)

    if not symbols:
        raise ValueError(f"WEEX returned zero USDT symbols for {market}")
    return symbols


def _clean_ranked_symbols(coins, limit):
    excluded = {
        "USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "USDS",
        "USDD", "PYUSD", "USDP", "EURC", "USTC", "UST"
    }
    symbols = []
    for coin in coins:
        symbol = str(coin.get("symbol", "")).upper().strip()
        if symbol and symbol not in excluded and symbol not in symbols:
            symbols.append(symbol)
        if len(symbols) >= int(limit):
            break
    if not symbols:
        raise ValueError("Ranking provider returned no usable symbols")
    return symbols


def get_cmc_top_symbols(limit, cfg):
    payload = request_json(
        CMC_URL,
        params={"start": 1, "limit": int(limit), "convert": "USD"},
        retries=int(cfg["scanner"]["api_retries"])
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("Unexpected CoinMarketCap listings response")
    return _clean_ranked_symbols(payload["data"], limit)


def get_coingecko_top_symbols(limit, cfg):
    limit = max(1, min(int(limit), 250))
    headers = {}
    api_key = os.getenv("COINGECKO_DEMO_API_KEY", "").strip()
    if api_key:
        headers["x-cg-demo-api-key"] = api_key

    payload = request_json(
        COINGECKO_URL,
        params={
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1,
            "sparkline": "false"
        },
        headers=headers or None,
        retries=int(cfg["scanner"]["api_retries"])
    )
    if not isinstance(payload, list):
        raise ValueError("Unexpected CoinGecko markets response")
    return _clean_ranked_symbols(payload, limit)


def build_universe(cfg, available_by_market):
    global UNIVERSE_SOURCE
    universe_cfg = cfg["universe"]
    mode = str(universe_cfg.get("mode", "top_cmc")).lower()

    if mode == "custom":
        candidates = [
            str(s).upper().replace("/USDT", "").replace("-", "").replace("USDT", "").strip()
            for s in universe_cfg.get("symbols", [])
        ]
        UNIVERSE_SOURCE = "CUSTOM"
    elif mode in ("top_cmc", "top_market_cap", "top"):
        limit = int(universe_cfg.get("limit", 50))
        if not 1 <= limit <= 250:
            raise ValueError("universe.limit must be between 1 and 250")
        try:
            candidates = get_cmc_top_symbols(limit, cfg)
            UNIVERSE_SOURCE = "COINMARKETCAP"
            log.info("Universe source: CoinMarketCap")
        except Exception as cmc_error:
            log.warning(f"CoinMarketCap unavailable: {cmc_error}")
            log.info("Trying CoinGecko fallback...")
            try:
                candidates = get_coingecko_top_symbols(limit, cfg)
                UNIVERSE_SOURCE = "COINGECKO_FALLBACK"
                log.info("Universe source: CoinGecko fallback")
            except Exception as cg_error:
                raise RuntimeError(
                    "Both ranking providers failed. "
                    f"CMC: {cmc_error} | CoinGecko: {cg_error}"
                )
    else:
        raise ValueError(f"Unknown universe.mode: {mode}")

    excluded = {str(s).upper().strip() for s in universe_cfg.get("exclude_symbols", [])}
    candidates = list(dict.fromkeys(s for s in candidates if s and s not in excluded))

    matched = {}
    for market in cfg["markets"]:
        available = available_by_market.get(market, set())
        matched[market] = [symbol for symbol in candidates if f"{symbol}USDT" in available]

    return candidates, matched


def timeframe_ms(timeframe):
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    value = int(timeframe[:-1])
    unit = timeframe[-1]
    if value <= 0 or unit not in units:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return value * units[unit]


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
        # Conservative fallback: do not use the newest possibly-open candle.
        df = df.iloc[:-1]

    df = df.reset_index(drop=True)
    if len(df) < 60:
        raise ValueError(f"Not enough closed candles: {len(df)}")
    return df


def fetch_candles(market, symbol, timeframe, limit, cfg):
    pair = f"{symbol}USDT"
    retries = int(cfg["scanner"]["api_retries"])

    if market == "SPOT":
        url = f"{SPOT_URL}/api/v3/market/historyKlines"
        interval_ms = timeframe_ms(timeframe)
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        current_open = (now_ms // interval_ms) * interval_ms
        start_ms = current_open - (limit * interval_ms)

        all_rows = []
        remaining = limit
        page_start = start_ms
        while remaining > 0:
            page_size = min(100, remaining)
            page_end = page_start + page_size * interval_ms - 1
            rows = request_json(
                url,
                params={
                    "symbol": pair,
                    "interval": timeframe,
                    "startTime": page_start,
                    "endTime": page_end,
                    "limit": page_size,
                    "priceType": "LAST"
                },
                retries=retries
            )
            if not isinstance(rows, list):
                raise ValueError(f"Unexpected spot candles for {pair} {timeframe}")
            all_rows.extend(rows)
            page_start += page_size * interval_ms
            remaining -= page_size
            time.sleep(0.05)
        return prepare_dataframe(all_rows)

    url = f"{FUTURES_URL}/capi/v3/market/klines"
    rows = request_json(
        url,
        params={"symbol": pair, "interval": timeframe, "limit": limit},
        retries=retries
    )
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected futures candles for {pair} {timeframe}")
    return prepare_dataframe(rows)


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
        if price > ema20 > ema50 and rsi >= float(settings.get("long_rsi_min", 52)) and macd > macd_signal:
            return 1, "EMA trend + RSI + MACD bullish"
        if price < ema20 < ema50 and rsi <= float(settings.get("short_rsi_max", 48)) and macd < macd_signal:
            return -1, "EMA trend + RSI + MACD bearish"
        return 0, "EMA/RSI/MACD conditions not aligned"

    if name == "breakout_volume":
        high, low = row["PREV_HIGH20"], row["PREV_LOW20"]
        multiplier = float(settings.get("volume_multiplier", 1.5))
        if pd.notna(high) and price > float(high) and vol_sma > 0 and volume >= vol_sma * multiplier:
            return 1, "20-candle upside breakout with volume"
        if pd.notna(low) and price < float(low) and vol_sma > 0 and volume >= vol_sma * multiplier:
            return -1, "20-candle downside breakdown with volume"
        return 0, "No confirmed volume breakout"

    if name == "bollinger_rsi":
        if price < float(row["BB_LOWER"]) and rsi <= float(settings.get("oversold_rsi", 30)):
            return 1, "Below lower Bollinger Band with oversold RSI"
        if price > float(row["BB_UPPER"]) and rsi >= float(settings.get("overbought_rsi", 70)):
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

    risk_amount = capital * risk_pct / 100
    empty = {
        "entry": None, "stop_loss": None, "take_profit": None,
        "risk_reward": None, "risk_amount_usdt": round(risk_amount, 2),
        "estimated_base_quantity": 0,
        "estimated_position_value_usdt": 0,
        "calculation_only": True
    }

    if direction == "WAIT" or (market == "SPOT" and direction == "SHORT"):
        return empty

    if capital <= 0 or not 0 < risk_pct <= 100 or atr_mult <= 0 or rr <= 0 or max_leverage <= 0:
        raise ValueError("Invalid risk configuration")
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
        "estimated_position_value_usdt": round(qty * price, 2),
        "calculation_only": True,
        "exchange_filters_checked": False
    }


def analyze_one(market, symbol, timeframe, cfg):
    raw = fetch_candles(
        market, symbol, timeframe,
        int(cfg["scanner"]["candle_limit"]), cfg
    )
    df = calculate_indicators(raw)
    row, prev = df.iloc[-1], df.iloc[-2]

    needed = [
        "close", "EMA20", "EMA50", "RSI", "MACD",
        "MACD_SIGNAL", "ATR", "BB_UPPER", "BB_LOWER"
    ]
    if row[needed].isna().any():
        raise ValueError("Indicator values are incomplete")

    votes, reasons = [], []
    for name, settings in cfg["strategies"].items():
        if not settings.get("enabled", False):
            continue
        vote, reason = strategy_vote(name, row, prev, settings)
        votes.append(vote)
        reasons.append({"strategy": name, "vote": vote, "reason": reason})

    if not votes:
        raise ValueError("No strategies are enabled")

    bullish = sum(v > 0 for v in votes)
    bearish = sum(v < 0 for v in votes)
    min_votes = max(1, int(cfg["scanner"].get("min_strategy_votes", 1)))

    if bullish >= min_votes and bullish > bearish:
        direction = "LONG"
    elif bearish >= min_votes and bearish > bullish:
        direction = "SHORT"
    else:
        direction = "WAIT"

    if market == "SPOT" and direction == "SHORT":
        signal, plan_direction = "EXIT_BIAS", "WAIT"
    elif direction == "LONG":
        signal, plan_direction = "BUY", "LONG"
    elif direction == "SHORT":
        signal, plan_direction = "SHORT", "SHORT"
    else:
        signal, plan_direction = "WAIT", "WAIT"

    price, atr = float(row["close"]), float(row["ATR"])
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
        "strategy_votes": {
            "bullish": int(bullish), "bearish": int(bearish),
            "minimum_required": min_votes
        },
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
    print("CRYPTO AI AGENT V4.1 | WEEX SPOT + USDT FUTURES")
    print("=" * 68)
    print("Universe:", cfg["universe"].get("mode", "top_cmc"))
    print("Timeframes:", ", ".join(cfg["timeframes"]))
    print("Markets:", ", ".join(cfg["markets"]))
    print("Enabled strategies:", ", ".join(
        k for k, v in cfg["strategies"].items() if v.get("enabled", False)
    ))

    available, errors, results = {}, [], []

    for market in cfg["markets"]:
        try:
            available[market] = get_weex_symbols(market, cfg)
            print(f"{market}: {len(available[market])} WEEX USDT symbols")
        except Exception as exc:
            available[market] = set()
            errors.append(f"{market} symbol list: {exc}")
            print(f"ERROR loading {market} symbols: {exc}")

    if not any(available.values()):
        raise SystemExit("Could not retrieve any WEEX symbols. No scan performed.")

    try:
        candidates, matched = build_universe(cfg, available)
    except Exception as exc:
        raise SystemExit(f"Could not build scan universe: {exc}")

    print("Universe source:", UNIVERSE_SOURCE)
    print(f"Candidate assets: {len(candidates)}")
    for market in cfg["markets"]:
        print(f"{market} matched assets: {len(matched.get(market, []))}")

    pause = max(0.0, float(cfg["scanner"].get("request_pause_seconds", 0.15)))
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
                except Exception as exc:
                    msg = f"{market} {symbol} {timeframe}: {exc}"
                    errors.append(msg)
                    print("ERROR:", msg)
                time.sleep(pause)

    timestamp = datetime.now(timezone.utc)
    report = {
        "version": "4.1",
        "exchange": "WEEX",
        "generated_at": timestamp.isoformat(),
        "universe_mode": cfg["universe"].get("mode", "top_cmc"),
        "universe_source": UNIVERSE_SOURCE,
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
            "message": "Analysis and estimated risk plans only. No live orders are sent."
        }
    }

    filename = timestamp.strftime("weex_v41_market_%Y%m%d_%H%M%S.json")
    report_path = os.path.join(REPORT_DIR, filename)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    buy_count = sum(x["signal"] == "BUY" for x in results)
    short_count = sum(x["signal"] == "SHORT" for x in results)
    wait_count = sum(x["signal"] in ("WAIT", "EXIT_BIAS") for x in results)

    print("\n" + "=" * 68)
    print("V4.1 SCAN COMPLETED")
    print(f"Universe source: {UNIVERSE_SOURCE}")
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
    if errors:
        log.warning(f"Completed with {len(errors)} analysis/API errors. Check report.")


if __name__ == "__main__":
    main()
