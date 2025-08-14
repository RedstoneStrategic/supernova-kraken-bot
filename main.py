#!/usr/bin/env python3
import os, json, time, math, sys, traceback, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import ccxt
from dotenv import load_dotenv

# -------------------------
# Utility
# -------------------------
def now_utc():
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)

def ts():
    return now_utc().strftime("%Y-%m-%d %H:%M:%S")

def bps(x):  # basis points to fraction
    return x / 10000.0

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# -------------------------
# Indicators
# -------------------------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    down = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = up / (down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bbands(series, period=20, std=2.0):
    basis = series.rolling(period).mean()
    dev = series.rolling(period).std(ddof=0)
    upper = basis + std * dev
    lower = basis - std * dev
    return lower, basis, upper

def donchian(high, low, period=20):
    upper = high.rolling(period).max()
    lower = low.rolling(period).min()
    return lower, upper

def atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# -------------------------
# Exchange / Broker
# -------------------------
class Broker:
    def __init__(self, cfg, env_path="secrets.env"):
        load_dotenv(env_path)
        self.cfg = cfg
        self.paper = bool(cfg.get("paper", False))
        self.exchange = ccxt.kraken({
            "apiKey": os.getenv("KRAKEN_API_KEY", ""),
            "secret": os.getenv("KRAKEN_API_SECRET", ""),
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        # Fees estimate (adjust if you have better tier)
        self.fee_rate = 0.0026  # 0.26%
        self.min_order_usd = 5.0

    def fetch_ohlcv(self, symbol, timeframe="1m", limit=250):
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def fetch_ohlcv_tf(self, symbol, timeframe="5m", limit=250):
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def ticker(self, symbol):
        return self.exchange.fetch_ticker(symbol)

    def price(self, symbol):
        return self.ticker(symbol)["last"]

    def balance(self, code="USDT"):
        bal = self.exchange.fetch_balance()
        free = bal.get(code, {}).get("free", 0.0)
        total = bal.get(code, {}).get("total", 0.0)
        return free, total

    def create_market_order(self, symbol, side, amount):
        if self.paper:
            price = self.price(symbol)
            return {"id": f"paper_{int(time.time()*1000)}", "symbol": symbol, "side": side, "amount": amount, "price": price, "status": "closed"}
        return self.exchange.create_order(symbol, "market", side, amount, None, {"reduce_only": False})

    def create_reduce_only_limit(self, symbol, side, amount, price):
        if self.paper:
            return {"id": f"paper_reduce_{int(time.time()*1000)}", "symbol": symbol, "side": side, "amount": amount, "price": price, "status": "open"}
        params = {"reduce_only": True}
        return self.exchange.create_order(symbol, "limit", side, amount, price, params)

    def create_reduce_only_stoploss(self, symbol, side, amount, stop_price):
        if self.paper:
            return {"id": f"paper_sl_{int(time.time()*1000)}", "symbol": symbol, "side": side, "amount": amount, "price": stop_price, "status": "open"}
        params = {"reduce_only": True, "stopPrice": stop_price, "type": "stop"}
        return self.exchange.create_order(symbol, "stop", side, amount, None, params)

    def cancel(self, id, symbol):
        if self.paper:
            return {"id": id, "status": "canceled"}
        return self.exchange.cancel_order(id, symbol)

# -------------------------
# State / Journal
# -------------------------
class Store:
    def __init__(self, path="state.json"):
        self.path = Path(path)
        if not self.path.exists():
            self
