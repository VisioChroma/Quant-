"""
server.py  —  AWS Version
--------------------------
Changes from local version:
  1. Models load from S3 (quant-data-siddu-bucket) instead of local models/ folder
  2. Runs on 0.0.0.0:5050 so EC2 can receive external traffic
  3. debug=False for production

Routes:
  GET /api/commodity?name=gold&period=1d&interval=5m
  GET /api/prediction?name=gold
  GET /api/news?name=gold
  GET /api/og-image?url=<article_url>

Run on EC2:
  python3 server.py
"""

import os
import re
import io
import json
import math
import warnings
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
warnings.filterwarnings("ignore")

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
import boto3

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# ── CONFIG ────────────────────────────────────────────────────────────────────

NEWSAPI_KEY = "Your_NewsAPI_Key_Here"  # Get a free key from https://newsapi.org/

# ── S3 CONFIG (changed from local) ───────────────────────────────────────────
S3_BUCKET  = "quant-data-siddu-bucket"   # your bucket name
S3_REGION  = "ap-south-1"               # Mumbai
s3_client  = boto3.client("s3", region_name=S3_REGION)

TICKERS = {
    "gold":   "GC=F",
    "silver": "SI=F",
    "oil":    "CL=F",
}

NEWS_KEYWORDS = {
    "gold":   "gold+commodity+price",
    "silver": "silver+commodity+price",
    "oil":    "crude+oil+price",
}

NEWSAPI_QUERIES = {
    "gold":   "gold price OR gold market OR gold rally OR gold demand",
    "silver": "silver price OR silver market OR silver demand",
    "oil":    "crude oil price OR oil market OR OPEC OR oil supply",
}

_model_cache     = {}
_sentiment_cache = {}




def load_model(name: str):
    """
    Load model from S3 bucket instead of local models/ folder.
    Caches in memory so S3 is only called once per commodity per server restart.
    """
    if name not in _model_cache:
        s3_key = f"models/{name}_model.pkl"
        print(f"[S3] Loading {s3_key} from bucket {S3_BUCKET} ...")
        try:
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
            model_bytes = obj["Body"].read()
            _model_cache[name] = joblib.load(io.BytesIO(model_bytes))
            print(f"[S3] Loaded {name} model OK")
        except Exception as e:
            print(f"[S3] ERROR loading {name} model: {e}")
            return None
    return _model_cache[name]




SENTIMENT_KEYWORDS = {
    "gold": {
        "bullish": {
            "safe haven":          0.9,
            "flight to safety":    0.9,
            "gold rally":          0.85,
            "gold surge":          0.85,
            "gold hits record":    0.95,
            "gold all-time high":  0.95,
            "gold demand":         0.7,
            "central bank buying": 0.85,
            "gold etf inflow":     0.75,
            "inflation hedge":     0.8,
            "dollar weakness":     0.75,
            "dollar falls":        0.7,
            "rate cut":            0.8,
            "dovish":              0.7,
            "geopolitical tension":0.8,
            "war":                 0.75,
            "conflict":            0.65,
            "sanctions":           0.6,
            "gold bullish":        0.85,
            "precious metals rise":0.75,
            "recession fear":      0.7,
            "economic slowdown":   0.65,
            "market uncertainty":  0.6,
            "gold prices up":      0.85,
            "gold gains":          0.8,
            "gold climbs":         0.8,
        },
        "bearish": {
            "gold falls":          0.85,
            "gold drops":          0.85,
            "gold slides":         0.8,
            "gold decline":        0.8,
            "gold selloff":        0.85,
            "gold sell-off":       0.85,
            "rate hike":           0.8,
            "hawkish":             0.7,
            "dollar strengthens":  0.75,
            "dollar rises":        0.7,
            "dollar surges":       0.75,
            "risk-on":             0.65,
            "equities rally":      0.6,
            "stock market gains":  0.6,
            "inflation cooling":   0.7,
            "gold etf outflow":    0.75,
            "gold bearish":        0.85,
            "gold loses":          0.8,
            "gold weakens":        0.75,
            "gold pressure":       0.65,
        },
    },
    "silver": {
        "bullish": {
            "silver rally":        0.85,
            "silver surge":        0.85,
            "silver demand":       0.7,
            "industrial demand":   0.75,
            "solar panel":         0.7,
            "green energy":        0.65,
            "silver etf inflow":   0.75,
            "safe haven":          0.8,
            "silver hits":         0.8,
            "silver gains":        0.8,
            "silver climbs":       0.75,
            "precious metals":     0.65,
            "silver bullish":      0.85,
            "rate cut":            0.75,
            "inflation hedge":     0.7,
            "silver prices up":    0.85,
            "dollar weakness":     0.7,
        },
        "bearish": {
            "silver falls":        0.85,
            "silver drops":        0.85,
            "silver slides":       0.8,
            "silver decline":      0.8,
            "industrial slowdown": 0.7,
            "manufacturing slump": 0.65,
            "silver selloff":      0.85,
            "silver sell-off":     0.85,
            "rate hike":           0.75,
            "dollar strengthens":  0.7,
            "silver bearish":      0.85,
            "silver loses":        0.8,
            "silver weakens":      0.75,
        },
    },
    "oil": {
        "bullish": {
            "opec cut":            0.95,
            "opec+ cut":           0.95,
            "production cut":      0.9,
            "supply cut":          0.85,
            "oil rally":           0.85,
            "oil surge":           0.85,
            "oil demand":          0.7,
            "oil hits":            0.8,
            "crude gains":         0.8,
            "oil climbs":          0.75,
            "energy crisis":       0.8,
            "supply disruption":   0.85,
            "pipeline attack":     0.9,
            "middle east tension": 0.85,
            "iran sanctions":      0.8,
            "russia oil":          0.75,
            "oil embargo":         0.85,
            "oil bullish":         0.85,
            "brent rises":         0.8,
            "wti rises":           0.8,
            "hurricane":           0.7,
            "refinery outage":     0.75,
            "oil prices up":       0.85,
            "crude oil up":        0.85,
        },
        "bearish": {
            "opec increase":       0.9,
            "production increase": 0.85,
            "supply increase":     0.8,
            "oil falls":           0.85,
            "oil drops":           0.85,
            "oil slides":          0.8,
            "oil decline":         0.8,
            "oil selloff":         0.85,
            "oil sell-off":        0.85,
            "recession":           0.75,
            "demand slowdown":     0.8,
            "china slowdown":      0.75,
            "ev adoption":         0.6,
            "oil bearish":         0.85,
            "crude falls":         0.85,
            "brent drops":         0.8,
            "wti drops":           0.8,
            "oil glut":            0.85,
            "oil surplus":         0.8,
            "oil prices down":     0.85,
        },
    },
}

GEO_RISK_KEYWORDS = [
    "war", "conflict", "attack", "missile", "troops", "invasion",
    "sanction", "nuclear", "terror", "coup", "crisis", "explosion",
    "military", "nato", "iran", "russia", "ukraine", "israel", "hamas",
    "hezbollah", "red sea", "strait of hormuz", "blockade",
]

MACRO_BULLISH_FOR_METALS = [
    "rate cut", "rate cuts", "dovish", "inflation rises", "inflation surge",
    "cpi higher", "fed pauses", "fed holds", "quantitative easing", "qe",
    "recession", "economic contraction", "gdp falls", "gdp shrinks",
    "banking crisis", "bank failure", "debt crisis", "deficit",
]
MACRO_BEARISH_FOR_METALS = [
    "rate hike", "rate hikes", "hawkish", "inflation cools", "inflation falls",
    "cpi lower", "fed raises", "tightening", "strong jobs", "strong economy",
    "gdp growth", "economic growth", "dollar index rises",
]
MACRO_BEARISH_FOR_OIL = [
    "recession", "economic contraction", "gdp falls", "china slowdown",
    "demand destruction", "manufacturing pmi falls",
]
MACRO_BULLISH_FOR_OIL = [
    "economic growth", "gdp growth", "strong manufacturing",
    "china recovery", "travel demand", "aviation demand",
]


def _score_text(text: str, commodity: str) -> dict:
    t  = text.lower()
    kw = SENTIMENT_KEYWORDS.get(commodity, SENTIMENT_KEYWORDS["gold"])

    bull_score = 0.0
    bear_score = 0.0

    for phrase, weight in kw["bullish"].items():
        if phrase in t:
            bull_score += weight

    for phrase, weight in kw["bearish"].items():
        if phrase in t:
            bear_score += weight

    geo_hits = sum(1 for kw_g in GEO_RISK_KEYWORDS if kw_g in t)
    if geo_hits > 0:
        geo_intensity = min(geo_hits * 0.15, 0.6)
        if commodity in ("gold", "silver"):
            bull_score += geo_intensity
        else:
            bull_score += geo_intensity * 0.8

    if commodity in ("gold", "silver"):
        for phrase in MACRO_BULLISH_FOR_METALS:
            if phrase in t:
                bull_score += 0.35
        for phrase in MACRO_BEARISH_FOR_METALS:
            if phrase in t:
                bear_score += 0.35
    else:
        for phrase in MACRO_BULLISH_FOR_OIL:
            if phrase in t:
                bull_score += 0.3
        for phrase in MACRO_BEARISH_FOR_OIL:
            if phrase in t:
                bear_score += 0.35

    return {"bull": bull_score, "bear": bear_score, "geo_hits": geo_hits}


def _recency_weight(published_str: str) -> float:
    try:
        from email.utils import parsedate_to_datetime
        try:
            dt = parsedate_to_datetime(published_str)
        except Exception:
            dt = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        age_hours = (now - dt).total_seconds() / 3600
        if age_hours < 2:   return 1.0
        if age_hours < 6:   return 0.85
        if age_hours < 12:  return 0.7
        if age_hours < 24:  return 0.55
        if age_hours < 48:  return 0.35
        return 0.15
    except Exception:
        return 0.5


def fetch_google_news_articles(commodity: str) -> list:
    keyword = NEWS_KEYWORDS.get(commodity, commodity + "+commodity")
    url = f"https://news.google.com/rss/search?q={keyword}&hl=en-US&gl=US&ceid=US:en"
    articles = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            xml_bytes = resp.read()
        root  = ET.fromstring(xml_bytes)
        items = root.findall(".//item")[:15]
        for item in items:
            def _t(tag):
                el = item.find(tag)
                return el.text.strip() if el is not None and el.text else ""
            desc = re.sub(r'<[^>]+>', '', _t("description")).strip()
            articles.append({
                "title":       _t("title"),
                "description": desc[:400],
                "published":   _t("pubDate"),
                "source":      "google_rss",
            })
    except Exception:
        pass
    return articles


def fetch_newsapi_articles(commodity: str) -> list:
    if not NEWSAPI_KEY:
        return []
    query = NEWSAPI_QUERIES.get(commodity, commodity + " price")
    url = (
        f"https://newsapi.org/v2/everything"
        f"?q={urllib.parse.quote(query)}"
        f"&language=en"
        f"&sortBy=publishedAt"
        f"&pageSize=20"
        f"&apiKey={NEWSAPI_KEY}"
    )
    articles = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        for a in data.get("articles", []):
            articles.append({
                "title":       a.get("title", ""),
                "description": (a.get("description") or "")[:400],
                "published":   a.get("publishedAt", ""),
                "source":      "newsapi",
            })
    except Exception:
        pass
    return articles


def analyze_news_sentiment(commodity: str) -> dict:
    cache_key = commodity
    if cache_key in _sentiment_cache:
        cached_time, cached_result = _sentiment_cache[cache_key]
        age_min = (datetime.now(timezone.utc) - cached_time).total_seconds() / 60
        if age_min < 15:
            return cached_result

    articles = fetch_google_news_articles(commodity)
    articles += fetch_newsapi_articles(commodity)

    if not articles:
        result = {
            "score": 0.0, "label": "NEUTRAL", "strength": 0.0,
            "article_count": 0, "bull_articles": 0, "bear_articles": 0,
            "geo_risk": False, "summary": "No news data available",
        }
        _sentiment_cache[cache_key] = (datetime.now(timezone.utc), result)
        return result

    total_bull   = 0.0
    total_bear   = 0.0
    total_weight = 0.0
    bull_count   = 0
    bear_count   = 0
    geo_total    = 0

    for a in articles:
        text = f"{a.get('title', '')} {a.get('description', '')}"
        if not text.strip():
            continue
        scores  = _score_text(text, commodity)
        recency = _recency_weight(a.get("published", ""))

        bull_w = scores["bull"] * recency
        bear_w = scores["bear"] * recency
        geo_total += scores["geo_hits"]

        total_bull   += bull_w
        total_bear   += bear_w
        total_weight += recency

        if bull_w > bear_w:
            bull_count += 1
        elif bear_w > bull_w:
            bear_count += 1

    if total_weight == 0:
        net_score = 0.0
    else:
        raw_net   = (total_bull - total_bear) / total_weight
        net_score = math.tanh(raw_net * 0.6)

    total_directional = bull_count + bear_count
    if total_directional > 0:
        dominant = max(bull_count, bear_count)
        strength = (dominant / total_directional) * min(1.0, total_directional / 8.0)
    else:
        strength = 0.0

    if net_score > 0.45:    label = "STRONG_BULLISH"
    elif net_score > 0.15:  label = "BULLISH"
    elif net_score < -0.45: label = "STRONG_BEARISH"
    elif net_score < -0.15: label = "BEARISH"
    else:                   label = "NEUTRAL"

    geo_risk       = geo_total >= 3
    direction_word = "bullish" if net_score > 0.1 else ("bearish" if net_score < -0.1 else "neutral")
    geo_note       = " with elevated geopolitical risk" if geo_risk else ""
    summary = (
        f"{len(articles)} articles scanned — sentiment {direction_word}{geo_note}. "
        f"{bull_count} bullish vs {bear_count} bearish articles."
    )

    result = {
        "score":         round(net_score, 4),
        "label":         label,
        "strength":      round(strength, 3),
        "article_count": len(articles),
        "bull_articles": bull_count,
        "bear_articles": bear_count,
        "geo_risk":      geo_risk,
        "summary":       summary,
    }
    _sentiment_cache[cache_key] = (datetime.now(timezone.utc), result)
    return result




ML_WEIGHT   = 0.60
NEWS_WEIGHT = 0.40

def blend_prediction(ml_dir_proba: list, ml_vol_proba: list, sentiment: dict) -> dict:
    ml_up   = ml_dir_proba[1]
    ml_down = ml_dir_proba[0]

    news_score    = sentiment["score"]
    news_up       = (news_score + 1) / 2
    news_down     = 1 - news_up
    news_strength = sentiment["strength"]
    news_up_adj   = 0.5 + (news_up   - 0.5) * news_strength
    news_down_adj = 0.5 + (news_down - 0.5) * news_strength

    blended_up   = ML_WEIGHT * ml_up   + NEWS_WEIGHT * news_up_adj
    blended_down = ML_WEIGHT * ml_down + NEWS_WEIGHT * news_down_adj
    total        = blended_up + blended_down
    blended_up  /= total
    blended_down /= total

    direction  = "UP" if blended_up >= blended_down else "DOWN"
    confidence = round(max(blended_up, blended_down) * 100, 1)

    vol_low, vol_med, vol_high = ml_vol_proba
    if sentiment["geo_risk"]:
        shift    = 0.05
        vol_high = min(1.0, vol_high + shift)
        vol_med  = max(0.0, vol_med  - shift * 0.5)
        vol_low  = max(0.0, vol_low  - shift * 0.5)
        s        = vol_low + vol_med + vol_high
        vol_low /= s; vol_med /= s; vol_high /= s

    vol_class  = int(np.argmax([vol_low, vol_med, vol_high]))
    vol_labels = ["LOW", "MEDIUM", "HIGH"]

    return {
        "direction":  direction,
        "confidence": confidence,
        "dir_proba":  {"up": round(blended_up*100,1), "down": round(blended_down*100,1)},
        "vol_label":  vol_labels[vol_class],
        "vol_proba":  {
            "low":    round(vol_low*100,  1),
            "medium": round(vol_med*100,  1),
            "high":   round(vol_high*100, 1),
        },
        "ml_only_dir": {"up": round(ml_up*100,1), "down": round(ml_down*100,1)},
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE BUILDER  (identical to train_model.py — must stay in sync)
# ══════════════════════════════════════════════════════════════════════════════

def compute_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast    = series.ewm(span=fast, adjust=False).mean()
    ema_slow    = series.ewm(span=slow, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def compute_bollinger(series, period=20, std_dev=2):
    sma   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    pct_b = (series - lower) / (upper - lower + 1e-9)
    width = (upper - lower) / (sma + 1e-9)
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

def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    f      = pd.DataFrame(index=raw.index)
    close  = raw["Close"]
    high   = raw["High"]
    low    = raw["Low"]
    volume = raw["Volume"]

    f["return_1d"]  = close.pct_change(1)
    f["return_3d"]  = close.pct_change(3)
    f["return_5d"]  = close.pct_change(5)
    f["return_10d"] = close.pct_change(10)
    f["return_20d"] = close.pct_change(20)

    f["volatility_5"]  = f["return_1d"].rolling(5).std()
    f["volatility_10"] = f["return_1d"].rolling(10).std()
    f["volatility_20"] = f["return_1d"].rolling(20).std()
    f["volatility_30"] = f["return_1d"].rolling(30).std()
    f["vol_ratio"]     = f["volatility_5"] / (f["volatility_20"] + 1e-9)

    for p in [5, 10, 12, 20, 26, 50, 200]:
        f[f"sma_{p}"] = close.rolling(p).mean()
        f[f"ema_{p}"] = close.ewm(span=p, adjust=False).mean()

    for p in [5, 10, 20, 50, 200]:
        f[f"price_vs_sma{p}"] = (close - f[f"sma_{p}"]) / (f[f"sma_{p}"] + 1e-9)

    f["sma_5_above_20"]  = (f["sma_5"]  > f["sma_20"]).astype(int)
    f["sma_10_above_50"] = (f["sma_10"] > f["sma_50"]).astype(int)
    f["ema_12_above_26"] = (f["ema_12"] > f["ema_26"]).astype(int)

    f["rsi_14"] = compute_rsi(close, 14)
    f["rsi_7"]  = compute_rsi(close, 7)
    f["rsi_overbought"] = (f["rsi_14"] > 70).astype(int)
    f["rsi_oversold"]   = (f["rsi_14"] < 30).astype(int)

    macd, macd_sig, macd_hist = compute_macd(close)
    f["macd"]           = macd
    f["macd_signal"]    = macd_sig
    f["macd_hist"]      = macd_hist
    f["macd_above_sig"] = (macd > macd_sig).astype(int)

    f["bb_pct_b"], f["bb_width"] = compute_bollinger(close)

    f["atr_14"] = compute_atr(high, low, close, 14)
    f["atr_pct"] = f["atr_14"] / (close + 1e-9)

    f["stoch_k"], f["stoch_d"] = compute_stochastic(high, low, close)

    if volume.sum() > 0:
        f["volume_change"] = volume.pct_change()
        f["volume_sma_10"] = volume.rolling(10).mean()
        f["volume_ratio"]  = volume / (f["volume_sma_10"] + 1e-9)
        f["volume_spike"]  = (f["volume_ratio"] > 1.5).astype(int)
    else:
        f["volume_change"] = 0
        f["volume_sma_10"] = 1
        f["volume_ratio"]  = 1
        f["volume_spike"]  = 0

    roll52w_high = high.rolling(252).max()
    roll52w_low  = low.rolling(252).min()
    f["pct_from_52w_high"] = (close - roll52w_high) / (roll52w_high + 1e-9)
    f["pct_from_52w_low"]  = (close - roll52w_low)  / (roll52w_low  + 1e-9)
    f["daily_range_pct"]   = (high - low) / (close + 1e-9)

    f["is_green"]   = (close > raw["Open"]).astype(int)
    f["body_size"]  = (close - raw["Open"]).abs() / (close + 1e-9)
    f["upper_wick"] = (high - close.clip(lower=raw["Open"])) / (close + 1e-9)
    f["lower_wick"] = (close.clip(upper=raw["Open"]) - low) / (close + 1e-9)

    for lag in [1, 2, 3, 5, 10]:
        f[f"lag_return_{lag}"] = f["return_1d"].shift(lag)

    drop_cols = [c for c in f.columns if c.startswith("sma_") or c.startswith("ema_")]
    f = f.drop(columns=drop_cols)
    return f


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return send_from_directory('.', 'predictions.html')

@app.route('/predictions.html')
def predictions_page():
    return send_from_directory('.', 'predictions.html')

@app.route('/dashboard.html')
def dashboard():
    return send_from_directory('.', 'dashboard.html')


# ── /api/commodity ────────────────────────────────────────────────────────────

@app.route("/api/commodity")
def commodity():
    name     = request.args.get("name", "gold").lower()
    period   = request.args.get("period", "1d")
    interval = request.args.get("interval", "5m")

    ticker_sym = TICKERS.get(name)
    if not ticker_sym:
        return jsonify({"error": "Unknown commodity"}), 400

    ticker = yf.Ticker(ticker_sym)
    info   = ticker.fast_info
    hist   = ticker.history(period=period, interval=interval)

    chart_data = []
    for ts, row in hist.iterrows():
        chart_data.append({"time": str(ts), "close": round(float(row["Close"]), 2)})

    return jsonify({
        "name":     name,
        "ticker":   ticker_sym,
        "price":    info.last_price,
        "open":     info.open,
        "day_high": info.day_high,
        "day_low":  info.day_low,
        "volume":   info.three_month_average_volume,
        "chart":    chart_data,
    })


# ── /api/news ─────────────────────────────────────────────────────────────────

@app.route("/api/news")
def news():
    name    = request.args.get("name", "gold").lower()
    keyword = NEWS_KEYWORDS.get(name, name + "+commodity")
    url     = f"https://news.google.com/rss/search?q={keyword}&hl=en-US&gl=US&ceid=US:en"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            xml_bytes = resp.read()

        root     = ET.fromstring(xml_bytes)
        items    = root.findall(".//item")[:10]
        articles = []

        for item in items:
            def _t(tag):
                el = item.find(tag)
                return el.text.strip() if el is not None and el.text else ""
            source_el = item.find("source")
            source    = source_el.text.strip() if source_el is not None and source_el.text else "Google News"
            desc      = re.sub(r'<[^>]+>', '', _t("description")).strip()
            articles.append({
                "title":       _t("title"),
                "link":        _t("link"),
                "published":   _t("pubDate"),
                "source":      source,
                "description": desc[:300] if desc else "",
            })

        return jsonify(articles)
    except Exception as e:
        return jsonify({"error": f"News fetch failed: {str(e)}"}), 502


# ── /api/og-image ─────────────────────────────────────────────────────────────

@app.route("/api/og-image")
def og_image():
    article_url = request.args.get("url", "")
    if not article_url:
        return jsonify({"image": None})
    try:
        req = urllib.request.Request(article_url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
            "Accept":     "text/html,application/xhtml+xml",
        })
        with urllib.request.urlopen(req, timeout=6) as resp:
            html = resp.read(50000).decode("utf-8", errors="ignore")

        for pattern in [
            r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
            r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
        ]:
            m = re.search(pattern, html, re.IGNORECASE)
            if m:
                img = m.group(1).strip()
                if img.startswith("http"):
                    return jsonify({"image": img})

        return jsonify({"image": None})
    except Exception:
        return jsonify({"image": None})


# ── /api/prediction ───────────────────────────────────────────────────────────

@app.route("/api/prediction")
def prediction():
    name = request.args.get("name", "gold").lower()

    if name not in TICKERS:
        return jsonify({"error": "Unknown commodity"}), 400

    bundle = load_model(name)
    if bundle is None:
        return jsonify({"error": f"Model not found for {name} in S3. Did you upload the .pkl files?"}), 404

    vol_model    = bundle["vol_model"]
    dir_model    = bundle["dir_model"]
    feature_cols = bundle["feature_cols"]

    ticker_sym = TICKERS[name]
    raw = yf.download(ticker_sym, period="2y", interval="1d",
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.dropna()

    if len(raw) < 60:
        return jsonify({"error": "Not enough data to generate prediction"}), 500

    feat     = build_features(raw)[feature_cols]
    last_row = feat.dropna().iloc[[-1]]
    if last_row.empty:
        return jsonify({"error": "Feature generation failed (NaN in last row)"}), 500

    X = last_row.values

    ml_vol_proba = vol_model.predict_proba(X)[0].tolist()
    ml_dir_proba = dir_model.predict_proba(X)[0].tolist()

    sentiment = analyze_news_sentiment(name)
    blended   = blend_prediction(ml_dir_proba, ml_vol_proba, sentiment)

    last_feat = feat.dropna().iloc[-1]
    rsi_val   = float(last_feat.get("rsi_14",   50))
    macd_hist = float(last_feat.get("macd_hist",  0))
    bb_pct_b  = float(last_feat.get("bb_pct_b", 0.5))
    stoch_k   = float(last_feat.get("stoch_k",  50))
    atr_val   = float(last_feat.get("atr_pct",   0))

    t_bull = 0; t_bear = 0
    if rsi_val  < 40:  t_bull += 1
    if rsi_val  > 60:  t_bear += 1
    if macd_hist > 0:  t_bull += 1
    if macd_hist < 0:  t_bear += 1
    if bb_pct_b < 0.2: t_bull += 1
    if bb_pct_b > 0.8: t_bear += 1
    if stoch_k  < 20:  t_bull += 1
    if stoch_k  > 80:  t_bear += 1

    news_bull    = sentiment["bull_articles"]
    news_bear    = sentiment["bear_articles"]
    total_bull   = t_bull + (news_bull * 0.5)
    total_bear   = t_bear + (news_bear * 0.5)
    combined_bias = (
        "BULLISH" if total_bull > total_bear else
        "BEARISH" if total_bear > total_bull else
        "NEUTRAL"
    )

    vol_accuracy = round(bundle.get("vol_accuracy", 0) * 100, 1)
    dir_accuracy = round(bundle.get("dir_accuracy", 0) * 100, 1)

    return jsonify({
        "commodity": name,
        "ticker":    ticker_sym,
        "volatility": {
            "label":          blended["vol_label"],
            "confidence":     round(max(blended["vol_proba"].values()), 1),
            "proba":          blended["vol_proba"],
            "model_accuracy": vol_accuracy,
        },
        "direction": {
            "label":          blended["direction"],
            "confidence":     blended["confidence"],
            "proba":          blended["dir_proba"],
            "model_accuracy": dir_accuracy,
            "ml_only":        blended["ml_only_dir"],
        },
        "expected_move_pct": round(atr_val * 100, 2),
        "signals": {
            "rsi":         round(rsi_val,   1),
            "macd_hist":   round(macd_hist, 4),
            "bb_position": round(bb_pct_b,  2),
            "stoch_k":     round(stoch_k,   1),
            "bias":        combined_bias,
            "bullish":     t_bull,
            "bearish":     t_bear,
        },
        "news_sentiment": {
            "score":         sentiment["score"],
            "label":         sentiment["label"],
            "strength":      sentiment["strength"],
            "article_count": sentiment["article_count"],
            "bull_articles": sentiment["bull_articles"],
            "bear_articles": sentiment["bear_articles"],
            "geo_risk":      sentiment["geo_risk"],
            "summary":       sentiment["summary"],
        },
    })


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n Quant Server — AWS Version")
    print("============================")
    print(f" S3 Bucket   : {S3_BUCKET}")
    print(f" S3 Region   : {S3_REGION}")
    print(f" ML weight   : {ML_WEIGHT*100:.0f}%")
    print(f" News weight : {NEWS_WEIGHT*100:.0f}%")
    print(f" NewsAPI key : {'✓ set' if NEWSAPI_KEY else '✗ not set (using Google RSS only)'}")
    print(" Starting on http://0.0.0.0:5050\n")
    app.run(host="0.0.0.0", port=5050, debug=False)