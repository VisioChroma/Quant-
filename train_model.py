"""
train_model.py
--------------
Downloads 5 years of historical data for Gold, Silver, Oil.
Engineers technical features.
Trains two models per commodity:
  1. Volatility classifier  →  LOW / MEDIUM / HIGH
  2. Direction classifier   →  UP / DOWN
Saves models + scaler + feature list to models/

Run once:
    python train_model.py

Requirements:
    pip install yfinance scikit-learn pandas numpy joblib ta
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# ── Config ────────────────────────────────────────────────────────────────────

COMMODITIES = {
    "gold":   "GC=F",
    "silver": "SI=F",
    "oil":    "CL=F",
}

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Technical Indicator Functions ────────────────────────────────────────────
# We implement manually so you have zero hidden dependencies

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast, adjust=False).mean()
    ema_slow   = series.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_bollinger(series, period=20, std_dev=2):
    sma    = series.rolling(period).mean()
    std    = series.rolling(period).std()
    upper  = sma + std_dev * std
    lower  = sma - std_dev * std
    # %B: position within bands (0=lower, 1=upper, 0.5=middle)
    pct_b  = (series - lower) / (upper - lower + 1e-9)
    width  = (upper - lower) / sma
    return pct_b, width

def compute_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def compute_stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low   = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)
    d = k.rolling(d_period).mean()
    return k, d

# ── Feature Engineering ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw OHLCV DataFrame, returns feature DataFrame.
    All features are based only on past data (no lookahead).
    """
    f = pd.DataFrame(index=df.index)
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # ── Returns ──────────────────────────────────────────────────────────────
    f["return_1d"]  = close.pct_change(1)
    f["return_3d"]  = close.pct_change(3)
    f["return_5d"]  = close.pct_change(5)
    f["return_10d"] = close.pct_change(10)
    f["return_20d"] = close.pct_change(20)

    # ── Volatility ────────────────────────────────────────────────────────────
    f["volatility_5"]  = f["return_1d"].rolling(5).std()
    f["volatility_10"] = f["return_1d"].rolling(10).std()
    f["volatility_20"] = f["return_1d"].rolling(20).std()
    f["volatility_30"] = f["return_1d"].rolling(30).std()

    # Volatility ratio: short vs long (regime indicator)
    f["vol_ratio"]     = f["volatility_5"] / (f["volatility_20"] + 1e-9)

    # ── Moving Averages ───────────────────────────────────────────────────────
    for p in [5, 10, 12, 20, 26, 50, 200]:
        f[f"sma_{p}"]  = close.rolling(p).mean()
        f[f"ema_{p}"]  = close.ewm(span=p, adjust=False).mean()

    # Price relative to MAs (normalized)
    for p in [5, 10, 20, 50, 200]:
        f[f"price_vs_sma{p}"] = (close - f[f"sma_{p}"]) / (f[f"sma_{p}"] + 1e-9)

    # MA crossover signals
    f["sma_5_above_20"]  = (f["sma_5"]  > f["sma_20"]).astype(int)
    f["sma_10_above_50"] = (f["sma_10"] > f["sma_50"]).astype(int)
    f["ema_12_above_26"] = (f["ema_12"] > f["ema_26"]).astype(int)

    # ── RSI ───────────────────────────────────────────────────────────────────
    f["rsi_14"] = compute_rsi(close, 14)
    f["rsi_7"]  = compute_rsi(close, 7)

    # Overbought / oversold flags
    f["rsi_overbought"] = (f["rsi_14"] > 70).astype(int)
    f["rsi_oversold"]   = (f["rsi_14"] < 30).astype(int)

    # ── MACD ──────────────────────────────────────────────────────────────────
    macd, macd_signal, macd_hist = compute_macd(close)
    f["macd"]           = macd
    f["macd_signal"]    = macd_signal
    f["macd_hist"]      = macd_hist
    f["macd_above_sig"] = (macd > macd_signal).astype(int)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    f["bb_pct_b"], f["bb_width"] = compute_bollinger(close)

    # ── ATR (Average True Range) ──────────────────────────────────────────────
    f["atr_14"] = compute_atr(high, low, close, 14)
    # Normalized ATR
    f["atr_pct"] = f["atr_14"] / (close + 1e-9)

    # ── Stochastic ────────────────────────────────────────────────────────────
    f["stoch_k"], f["stoch_d"] = compute_stochastic(high, low, close)

    # ── Volume Features ───────────────────────────────────────────────────────
    if volume.sum() > 0:
        f["volume_change"]   = volume.pct_change()
        f["volume_sma_10"]   = volume.rolling(10).mean()
        f["volume_ratio"]    = volume / (f["volume_sma_10"] + 1e-9)
        f["volume_spike"]    = (f["volume_ratio"] > 1.5).astype(int)
    else:
        # Futures sometimes have zero volume in fast_info; safe fallback
        f["volume_change"]   = 0
        f["volume_sma_10"]   = 1
        f["volume_ratio"]    = 1
        f["volume_spike"]    = 0

    # ── Price Position ────────────────────────────────────────────────────────
    roll52w_high = high.rolling(252).max()
    roll52w_low  = low.rolling(252).min()
    f["pct_from_52w_high"] = (close - roll52w_high) / (roll52w_high + 1e-9)
    f["pct_from_52w_low"]  = (close - roll52w_low)  / (roll52w_low  + 1e-9)

    # High/Low range of the day normalized
    f["daily_range_pct"] = (high - low) / (close + 1e-9)

    # ── Candle patterns (simple) ──────────────────────────────────────────────
    f["is_green"]      = (close > df["Open"]).astype(int)
    f["body_size"]     = (close - df["Open"]).abs() / (close + 1e-9)
    f["upper_wick"]    = (high - close.clip(lower=df["Open"])) / (close + 1e-9)
    f["lower_wick"]    = (close.clip(upper=df["Open"]) - low) / (close + 1e-9)

    # ── Lag features (past N days of return) ──────────────────────────────────
    for lag in [1, 2, 3, 5, 10]:
        f[f"lag_return_{lag}"] = f["return_1d"].shift(lag)

    # Drop raw MAs (we keep only price_vs_sma signals which are normalized)
    drop_cols = [c for c in f.columns if c.startswith("sma_") or c.startswith("ema_")]
    f = f.drop(columns=drop_cols)

    return f

# ── Target Engineering ────────────────────────────────────────────────────────

def build_targets(df: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """
    Two targets:
      1. vol_label   : tomorrow's volatility class  LOW=0 / MEDIUM=1 / HIGH=2
      2. direction   : tomorrow's price direction   DOWN=0 / UP=1
    """
    close = df["Close"]

    # Future 5-day volatility (what we're predicting)
    future_ret = close.pct_change()
    future_vol = future_ret.rolling(5).std().shift(-5)   # next 5 days vol

    # Bin into LOW / MEDIUM / HIGH using training-set quantiles
    # (We'll recompute thresholds per commodity during training)
    targets = pd.DataFrame(index=df.index)
    targets["future_vol"]    = future_vol
    targets["future_return"] = close.pct_change(5).shift(-5)   # 5-day forward return

    # Direction: 1 = price higher in 1 day, 0 = lower
    targets["direction_1d"]  = (close.shift(-1) > close).astype(int)

    return targets

# ── Train one commodity ───────────────────────────────────────────────────────

def train_commodity(name: str, ticker: str):
    print(f"\n{'='*55}")
    print(f"  Training: {name.upper()} ({ticker})")
    print(f"{'='*55}")

    # ── 1. Download data ──────────────────────────────────────────────────────
    print("  Downloading 5 years of data...")
    raw = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)

    if raw.empty:
        print(f"  ERROR: No data for {ticker}. Skipping.")
        return

    # yfinance sometimes returns MultiIndex columns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.dropna()
    print(f"  Downloaded {len(raw)} trading days")

    # ── 2. Features ───────────────────────────────────────────────────────────
    print("  Engineering features...")
    feat    = build_features(raw)
    feat.replace([np.inf, -np.inf], np.nan, inplace=True)   # ← add this
    targets = build_targets(raw, feat)
    # Align
    data = pd.concat([feat, targets], axis=1).dropna()
    print(f"  Clean rows after feature engineering: {len(data)}")

    # ── 3. Volatility label thresholds ────────────────────────────────────────
    vol_33  = data["future_vol"].quantile(0.33)
    vol_66  = data["future_vol"].quantile(0.66)
    print(f"  Volatility thresholds → LOW<{vol_33:.4f} | MEDIUM<{vol_66:.4f} | HIGH>{vol_66:.4f}")

    def vol_label(v):
        if v < vol_33:  return 0   # LOW
        if v < vol_66:  return 1   # MEDIUM
        return 2                   # HIGH

    data["vol_class"] = data["future_vol"].apply(vol_label)

    # ── 4. Feature matrix ─────────────────────────────────────────────────────
    feature_cols = [c for c in feat.columns]
    X = data[feature_cols].values
    y_vol = data["vol_class"].values
    y_dir = data["direction_1d"].values

    # ── 5. Time-series split (no data leakage) ────────────────────────────────
    # Use last 20% as holdout test set — never shuffle time series!
    split_idx = int(len(X) * 0.80)
    X_train, X_test   = X[:split_idx], X[split_idx:]
    yv_train, yv_test = y_vol[:split_idx], y_vol[split_idx:]
    yd_train, yd_test = y_dir[:split_idx], y_dir[split_idx:]

    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # ── 6. Train Volatility Classifier ───────────────────────────────────────
    print("\n  [A] Training Volatility Classifier (LOW/MEDIUM/HIGH)...")

    vol_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
        ))
    ])
    vol_model.fit(X_train, yv_train)
    vol_pred = vol_model.predict(X_test)
    vol_acc  = accuracy_score(yv_test, vol_pred)
    print(f"  Volatility accuracy: {vol_acc*100:.1f}%")
    print(classification_report(yv_test, vol_pred,
                                target_names=["LOW","MEDIUM","HIGH"],
                                zero_division=0))

    # ── 7. Train Direction Classifier ─────────────────────────────────────────
    print("  [B] Training Direction Classifier (UP/DOWN)...")

    dir_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=15,
            random_state=42,
        ))
    ])
    dir_model.fit(X_train, yd_train)
    dir_pred = dir_model.predict(X_test)
    dir_acc  = accuracy_score(yd_test, dir_pred)
    print(f"  Direction accuracy: {dir_acc*100:.1f}%")
    print(classification_report(yd_test, dir_pred,
                                target_names=["DOWN","UP"],
                                zero_division=0))

    # ── 8. Feature importance (top 10) ───────────────────────────────────────
    fi = pd.Series(
        vol_model.named_steps["clf"].feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)
    print(f"\n  Top 10 features (volatility model):")
    for feat_name, imp in fi.head(10).items():
        print(f"    {feat_name:<30} {imp:.4f}")

    # ── 9. Save everything ────────────────────────────────────────────────────
    bundle = {
        "vol_model":     vol_model,
        "dir_model":     dir_model,
        "feature_cols":  feature_cols,
        "vol_thresholds": {"low": vol_33, "high": vol_66},
        "vol_accuracy":  vol_acc,
        "dir_accuracy":  dir_acc,
        "ticker":        ticker,
        "name":          name,
    }

    path = os.path.join(MODELS_DIR, f"{name}_model.pkl")
    joblib.dump(bundle, path)
    print(f"\n  ✓ Saved → {path}")
    print(f"  Vol accuracy: {vol_acc*100:.1f}% | Dir accuracy: {dir_acc*100:.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nQuant – Model Training Pipeline")
    print("================================\n")

    results = {}
    for name, ticker in COMMODITIES.items():
        try:
            train_commodity(name, ticker)
            results[name] = "✓ OK"
        except Exception as e:
            results[name] = f"✗ FAILED: {e}"
            print(f"  ERROR training {name}: {e}")

    print("\n\nSummary")
    print("-------")
    for name, status in results.items():
        print(f"  {name:<10} {status}")

    print("\nDone. Models saved in ./models/")
    print("Next step: python server.py  →  then open predictions.html")