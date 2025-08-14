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
            self.data = {"open_trades": [], "trade_log": [], "profit_stash": 0.0, "equity_base": 100.0, "daily": {}}
            self.save()
        else:
            self.data = json.loads(self.path.read_text())

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)

    def open_trades(self):
        return self.data["open_trades"]

    def log_trade(self, rec):
        self.data["trade_log"].append(rec)

# -------------------------
# Strategy & Risk
# -------------------------
def calc_qty(usdt_equity, price, stop_pct, risk_perc):
    risk_usdt = usdt_equity * (risk_perc/100.0)
    stop_amount = price * (stop_pct/100.0)
    if stop_amount <= 0:
        return 0.0
    qty = risk_usdt / stop_amount
    return max(0.0, qty)

def spread_bps(ask, bid):
    if ask <= 0 or bid <= 0: return 1e9
    return (ask - bid) / ((ask + bid)/2.0) * 10000.0

def in_session(session_block_utc):
    now = now_utc()
    hhmm = now.strftime("%H:%M")
    for block in session_block_utc:
        start, end = block.split("-")
        if start <= hhmm <= end:
            return False
    return True

def atr_bps(atr_val, price):
    return (atr_val / price) * 10000.0  # bps

def build_df(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def entry_signal_A(df1m, df5m):
    # Trend pullback
    if len(df1m) < 60 or len(df5m) < 60: return False
    c1 = df1m["close"]
    c5 = df5m["close"]
    ema50 = ema(c5, 50)
    ema200 = ema(c5, 200)
    uptrend = ema50.iloc[-1] > ema200.iloc[-1]
    macd_line, sig, hist = macd(c1)
    r = rsi(c1)
    cond_rsi = (r.iloc[-1] > 50) and (r.iloc[-1] < 65)
    ema21 = ema(c1, 21)
    pullback = c1.iloc[-2] >= ema21.iloc[-2] and c1.iloc[-1] >= ema21.iloc[-1]
    return bool(uptrend and cond_rsi and hist.iloc[-1] > 0 and pullback)

def entry_signal_B(df1m):
    # Breakout micro-range
    if len(df1m) < 80: return False
    low_don, up_don = donchian(df1m["high"], df1m["low"], 30)
    last = df1m.iloc[-1]
    prev = df1m.iloc[-2]
    broke = last["close"] > up_don.iloc[-2] and prev["close"] <= up_don.iloc[-3]
    retest = last["low"] <= up_don.iloc[-2] * 1.001
    return bool(broke and retest)

def entry_signal_C(df1m):
    # Mean reversion snap
    if len(df1m) < 60: return False
    lower, basis, upper = bbands(df1m["close"], 20, 2.0)
    r = rsi(df1m["close"], 2)
    w2 = df1m.iloc[-2]
    w1 = df1m.iloc[-1]
    pierced = (w2["low"] < lower.iloc[-2]) and (w1["close"] > lower.iloc[-1])
    return bool(pierced and r.iloc[-1] < 15)

def choose_entry(df1m, df5m):
    if entry_signal_A(df1m, df5m): return "A"
    if entry_signal_B(df1m): return "B"
    if entry_signal_C(df1m): return "C"
    return None

# -------------------------
# Bot
# -------------------------
class Bot:
    def __init__(self, config_path="config.json"):
        self.cfg = json.loads(Path(config_path).read_text())
        self.broker = Broker(self.cfg, env_path="secrets.env")
        self.store = Store("state.json")
        self.max_positions = int(self.cfg.get("max_positions", 3))
        self.loop_seconds = int(self.cfg.get("loop_seconds", 5))
        self.halt = False
        self.loss_streak = 0
        self.today = now_utc().date()

    def daily_guard(self):
        # Reset per UTC day
        if now_utc().date() != self.today:
            self.today = now_utc().date()
            self.store.data["daily"] = {}
            self.loss_streak = 0
        dd = self.store.data["daily"].get("dd_pct", 0.0)
        if dd <= -abs(self.cfg["daily_dd_halt_pct"]):
            return False
        # Session block
        if not in_session(self.cfg.get("session_block_utc", [])):
            return False
        return True

    def equity_usdt(self):
        free, total = self.broker.balance(self.cfg.get("account_currency", "USDT"))
        if self.cfg.get("paper", False):
            return float(self.store.data.get("equity_base", 100.0))
        return float(total if total else free)

    def update_dd(self, pnl_usd):
        d = self.store.data["daily"]
        d.setdefault("pnl", 0.0)
        d["pnl"] += pnl_usd
        eq0 = max(1.0, self.store.data.get("equity_base", 100.0))
        d["dd_pct"] = min(0.0, (d["pnl"]/eq0)*100.0)

    def add_stash_if_needed(self, equity):
        for step in self.cfg.get("profit_stash_steps", []):
            if not step: continue
            if equity >= step["equity"] and not step.get("done"):
                self.store.data["profit_stash"] += step["stash"]
                step["done"] = True

    def manage_open_trades(self):
        # Manage trailing + exits
        to_close = []
        for tr in self.store.open_trades():
            symbol = tr["symbol"]
            side = tr["side"]
            entry_price = tr["entry_price"]
            qty = tr["qty"]
            price = self.broker.price(symbol)

            tr["max_price"] = max(tr.get("max_price", entry_price), price)
            tr["min_price"] = min(tr.get("min_price", entry_price), price)

            tgt = self.cfg["target_pct"]/100.0
            trail_start = self.cfg["trail_start_pct"]/100.0
            trail_gap = self.cfg["trail_gap_pct"]/100.0

            up = (price - entry_price)/entry_price
            if up >= trail_start:
                trail_stop = tr["max_price"] * (1 - trail_gap)
                tr["trail_stop"] = max(tr.get("trail_stop", 0.0), trail_stop)

            stop_price = tr.get("stop_price", entry_price*(1 - self.cfg["stop_pct_default"]/100.0))
            if tr.get("trail_stop") and tr["trail_stop"] > stop_price:
                stop_price = tr["trail_stop"]

            if price <= stop_price:
                to_close.append(tr)

            if up >= tgt:
                to_close.append(tr)

        for tr in to_close:
            self.close_trade(tr)

    def close_trade(self, tr):
        symbol, qty = tr["symbol"], tr["qty"]
        side = "sell" if tr["side"] == "buy" else "buy"
        try:
            self.broker.create_market_order(symbol, side, qty)
        except Exception as e:
            print(ts(), "close_trade error", e)
        price = self.broker.price(symbol)
        pnl = (price - tr["entry_price"]) * qty * (1 if tr["side"]=="buy" else -1)
        fees = abs(price*qty)*self.broker.fee_rate + abs(tr["entry_price"]*qty)*self.broker.fee_rate
        pnl -= fees
        self.update_dd(pnl)
        self.store.log_trade({
            "time": ts(),
            "symbol": symbol,
            "side": tr["side"],
            "entry": tr["entry_price"],
            "exit": price,
            "qty": qty,
            "pnl": pnl
        })
        self.store.data["open_trades"] = [x for x in self.store.data["open_trades"] if x.get("id") != tr.get("id")]
        if self.cfg.get("paper", False):
            self.store.data["equity_base"] = self.store.data.get("equity_base", 100.0) + pnl
        self.store.save()
        print(ts(), f"Closed {symbol} PnL {pnl:.4f} USDT")

    def can_open_more(self):
        return len(self.store.open_trades()) < self.max_positions

    def open_trade(self, symbol, price, stop_pct):
        equity = self.equity_usdt()
        qty = calc_qty(equity, price, stop_pct, self.cfg["risk_per_trade_pct"])
        if price*qty < self.broker.min_order_usd:
            print(ts(), f"Skip {symbol}: size too small ({price*qty:.2f} USD)")
            return
        try:
            side = "buy"
            self.broker.create_market_order(symbol, side, qty)
            rec = {
                "id": f"{symbol}_{int(time.time()*1000)}",
                "time": ts(),
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "entry_price": price,
                "stop_price": price*(1 - stop_pct/100.0),
                "max_price": price,
                "min_price": price
            }
            self.store.data["open_trades"].append(rec)
            self.store.save()
            print(ts(), f"Opened {symbol} qty={qty:.6f} at {price:.6f}")
        except Exception as e:
            print(ts(), "open_trade error", e)

    def run_once(self):
        if not self.daily_guard():
            time.sleep(self.loop_seconds)
            return

        try:
            self.manage_open_trades()
        except Exception as e:
            print(ts(), "manage_open_trades error", e)

        if not self.can_open_more():
            time.sleep(self.loop_seconds)
            return

        for symbol in self.cfg["coins"]:
            if not self.can_open_more():
                break
            try:
                ohlcv1 = self.broker.fetch_ohlcv(symbol, "1m", 250)
                ohlcv5 = self.broker.fetch_ohlcv_tf(symbol, "5m", 250)
            except Exception as e:
                print(ts(), f"{symbol} fetch ohlcv error {e}")
                continue

            df1 = build_df(ohlcv1)
            df5 = build_df(ohlcv5)
            price = float(df1["close"].iloc[-1])

            a = float(atr(df1["high"], df1["low"], df1["close"]).iloc[-1])
            a_bps = atr_bps(a, price)
            low_bps, high_bps = self.cfg["atr_vol_gate_bps"]
            if a_bps < low_bps or a_bps > high_bps:
                continue

            tkr = self.broker.ticker(symbol)
            sp_bps = spread_bps(tkr.get("ask", price), tkr.get("bid", price))
            if sp_bps < self.cfg.get("min_spread_bps", 0) or sp_bps > self.cfg.get("max_spread_bps", 10):
                continue

            sig = choose_entry(df1, df5)
            if not sig:
                continue

            if not self.can_open_more():
                break

            stop_pct = self.cfg["stop_pct_default"]
            atrp = a_bps/100.0  # percent
            stop_pct = max(stop_pct, atrp * self.cfg.get("atr_mult_stop", 1.5))

            self.open_trade(symbol, price, stop_pct)

        time.sleep(self.loop_seconds)

    def run(self):
        print(ts(), "SUPERNOVA Kraken Scalper starting...")
        print(ts(), f"Paper mode: {self.cfg.get('paper', False)}  Max positions: {self.max_positions}")
        while True:
            try:
                self.run_once()
            except KeyboardInterrupt:
                print("Stopping...")
                break
            except Exception as e:
                print(ts(), "Top-level error:", e)
                traceback.print_exc()
                time.sleep(2)

if __name__ == "__main__":
    Bot("config.json").run()
