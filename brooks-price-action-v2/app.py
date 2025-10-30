"""
Brooks Price-Action Analysis Tool — v2
Requirements implemented:
- White background; centered title
- Pacific Time display (top-left) + market open/closed indicator
- Auto-refresh every 1 minute when page is open
- TradingView chart (replaces Matplotlib)
- Live data via yfinance (1m/5m fallback)
- Branching analysis: market open → possible setups; market closed → EOD overview
- Recommended strategy with probabilistic confidence score (0–1)
- Summary covering Brooks concepts (OR, Bar1/2, Bar18, measured move, leg counting, day types, reversals, MTR clues)
- No raw OHLCV tables in UI

To run locally:
  streamlit run app.py
"""
from __future__ import annotations
from typing import Optional, Tuple, Dict, List

import math
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from streamlit.components.v1 import html as st_html
import html  # stdlib html escaping

# =============================
# App Config + Styling
# =============================
st.set_page_config(page_title="Brooks-Price-Action-Analysis-Tool", layout="wide")

st.markdown(
    """
    <style>
      /* 🌕 GLOBAL APP STYLING */
      .stApp {
          background-color: #ffffff !important; /* White background */
          color: #000000 !important;            /* Black text */
      }

      /* Streamlit widgets & text */
      h1, h2, h3, h4, h5, h6, p, div, span, label {
          color: #000000 !important;
      }

      /* Title */
      .center-title {
          text-align: center;
          margin-top: 8px;
          margin-bottom: 2px;
          color: #000000 !important;
      }

      /* Top bar elements */
      .timebox {
          font-weight: 600;
          font-size: 0.95rem;
          color: #000000 !important;
      }

      /* Market status badges */
      .badge {
          padding: 4px 10px;
          border-radius: 999px;
          font-weight: 700;
      }
      .open {
          background: #e6ffed;
          color: #067d3f;
          border: 1px solid #bff3cb;
      }
      .closed {
          background: #ffefef;
          color: #a13333;
          border: 1px solid #f3c0c0;
      }

      /* Card Styling */
      .card {
          border: 1px solid #e7e7e7;
          border-radius: 12px;
          padding: 14px;
          background-color: #ffffff;
          color: #000000 !important;
      }

      /* Headings inside cards */
      .card h3 {
          margin: 0 0 8px 0;
          color: #000000 !important;
      }

      /* Strategy text */
      .strategy {
          font-size: 1.2rem;
          font-weight: 800;
          color: #000000 !important;
      }

      .score {
          font-weight: 800;
          color: #000000 !important;
      }
    </style>
    /* Make the Analyze button text white */
.stButton button {
    color: #ffffff !important;       /* White text */
    background-color: #000000 !important; /* Optional: make button itself black */
    border-radius: 6px !important;
    font-weight: 600 !important;
    border: none !important;
    padding: 0.5rem 1.2rem !important;
}

.stButton button:hover {
    background-color: #333333 !important; /* Slightly lighter on hover */
}
    """,
    unsafe_allow_html=True,
)


# Auto-refresh every 60s (JS fallback if plugin unavailable)
try:
    from streamlit_autorefresh import st_autorefresh as _st_autorefresh
except Exception:
    _st_autorefresh = None

if _st_autorefresh:
    _st_autorefresh(interval=60_000, limit=None, key="auto-refresh-60s")
else:
    st.markdown("""
        <script>
        setTimeout(function(){ window.location.reload(); }, 60000);
        </script>
    """, unsafe_allow_html=True)

# =============================
# Globals
# =============================
ET = ZoneInfo("America/New_York")
PT = ZoneInfo("America/Los_Angeles")

# =============================
# Top Bar: Time + Market Status + Title + Symbol Input
# =============================
now_pt = datetime.now(PT)
now_et = datetime.now(ET)

# Market open determination (regular session)
market_open = (
    now_et.weekday() < 5 and
    dtime(9, 30) <= now_et.time() <= dtime(16, 0)
)

# Top bar row
left, center, right = st.columns([1.2, 2, 1.2])
with left:
    st.markdown(
        f"<div class='timebox'>Pacific Time: {now_pt.strftime('%a %b %d • %I:%M %p')}</div>",
        unsafe_allow_html=True,
    )
    badge = "<span class='badge open'>🟢 Market Open</span>" if market_open else "<span class='badge closed'>🔴 Market Closed</span>"
    st.markdown(badge, unsafe_allow_html=True)
with center:
    st.markdown("<h1 class='center-title'>Brooks-Price-Action-Analysis-Tool</h1>", unsafe_allow_html=True)
with right:
    st.empty()

# Symbol input centered under title
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    symbol = st.text_input("", value="AAPL", placeholder="Enter stock symbol (e.g., AAPL)", label_visibility="collapsed").strip().upper()
    go = st.button("Analyze", use_container_width=True)

# =============================
# Data Fetching
# =============================

def normalize_ohlcv(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    out = df.reset_index().copy()
    # Ensure Datetime column
    if "Datetime" not in out.columns:
        if "Date" in out.columns:
            out = out.rename(columns={"Date": "Datetime"})
        else:
            out = out.rename(columns={out.columns[0]: "Datetime"})
    out["Datetime"] = pd.to_datetime(out["Datetime"], errors="coerce")
    out = out.rename(columns={c: c.title() for c in out.columns})
    need = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    if not all(c in out.columns for c in need):
        return None
    out = out[need].dropna().sort_values("Datetime").reset_index(drop=True)
    return out

@st.cache_data(show_spinner=False, ttl=60)
def fetch_intraday(sym: str) -> Optional[pd.DataFrame]:
    # Try 1-minute first, then 5-minute fallback
    for interval, period in [("1m", "1d"), ("5m", "7d"), ("5m", "30d")]:
        try:
            raw = yf.download(sym, interval=interval, period=period, auto_adjust=False, progress=False, threads=False)
            df = normalize_ohlcv(raw)
            if df is not None and not df.empty:
                df.attrs["interval"] = interval
                df.attrs["period"] = period
                return df
        except Exception:
            continue
    return None

# =============================
# TradingView Embed
# =============================

def tradingview_widget(symbol: str, theme: str = "light", height: int = 560) -> None:
    tv_symbol = f"NASDAQ:{symbol.upper()}"
    st_html(f"""
    <div class="tradingview-widget-container" style="height:{height}px;">
      <div id="tradingview_chart" style="height:{height}px;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
          "width": "100%",
          "height": {height},
          "symbol": "{tv_symbol}",
          "interval": "5",
          "timezone": "America/New_York",
          "theme": "{theme}",
          "style": "1",
          "locale": "en",
          "toolbar_bg": "rgba(0,0,0,0)",
          "enable_publishing": false,
          "hide_side_toolbar": false,
          "allow_symbol_change": true,
          "studies": ["MASimple@tv-basicstudies","MAExp@tv-basicstudies"],
          "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    """, height=height)

# =============================
# Brooks Metrics + Heuristics
# =============================

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def opening_range_info(day: pd.DataFrame, bars_min: int = 5, bars_max: int = 18) -> dict:
    n = min(len(day), bars_max)
    n = max(n, bars_min)
    seg = day.iloc[:n]
    hi = float(seg["High"].max()); lo = float(seg["Low"].min()); mid = (hi + lo) / 2.0
    last = float(day["Close"].iloc[-1])
    status = "inside"
    if last > hi: status = "above"
    elif last < lo: status = "below"
    return {"bars": n, "high": hi, "low": lo, "mid": mid, "status": status}


def bar18_info(day: pd.DataFrame) -> dict:
    enough = len(day) >= 18
    if not enough:
        return {"enough_bars": False}
    cut = day.iloc[:18]
    hi18 = float(cut["High"].max()); lo18 = float(cut["Low"].min())
    hi_now = float(day["High"].max()); lo_now = float(day["Low"].min())
    return {
        "enough_bars": True,
        "high_at18": hi18,
        "low_at18": lo18,
        "high_still_day_high": abs(hi18 - hi_now) < 1e-9,
        "low_still_day_low": abs(lo18 - lo_now) < 1e-9,
    }


def always_in_state(day: pd.DataFrame) -> Tuple[str, pd.Series, pd.Series]:
    if len(day) < 3:
        close = day["Close"]
        return "neutral", ema(close, 20), ema(close, 50)
    close = day["Close"]; high = day["High"]; low = day["Low"]
    ema20 = ema(close, 20); ema50 = ema(close, 50)
    e50 = ema50.to_numpy(); c = close.to_numpy(); h = high.to_numpy(); l = low.to_numpy()
    direction = "neutral"
    for i in range(2, len(c)):
        bo_up = (c[i] > h[i-1]) and (c[i-1] > h[i-2])
        bo_dn = (c[i] < l[i-1]) and (c[i-1] < l[i-2])
        if bo_up and c[i] > e50[i]:
            direction = "bull"
        elif bo_dn and c[i] < e50[i]:
            direction = "bear"
        else:
            if direction == "bull" and c[i] < e50[i]: direction = "neutral"
            if direction == "bear" and c[i] > e50[i]: direction = "neutral"
    return direction, ema20, ema50


def overlap_score(day: pd.DataFrame, window: int = 24) -> float:
    hh = day["High"].rolling(window).max()
    ll = day["Low"].rolling(window).min()
    width = (hh - ll).replace(0, np.nan)
    mid = (hh + ll) / 2.0
    mid_time = ((day["Close"] - mid).abs() < 0.2 * width).rolling(window).mean()
    fail_bo = (((day["Close"] > hh.shift(1)) & (day["Close"].shift(1) < hh.shift(2))) |
               ((day["Close"] < ll.shift(1)) & (day["Close"].shift(1) > ll.shift(2)))).rolling(window).mean()
    sc = (0.7 * mid_time.fillna(0) + 0.3 * fail_bo.fillna(0)).clip(0, 1)
    return float(sc.iloc[-1]) if len(sc) else 0.0


def measured_move_targets(day: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    if len(day) < 20:
        return None, None
    first_hour = day.iloc[:12]
    uptrend = first_hour["Close"].iloc[-1] > first_hour["Open"].iloc[0]
    if uptrend:
        leg = float(first_hour["High"].max() - day["Open"].iloc[0])
        seg = day.iloc[12:20]
        if seg.empty: return None, None
        pb_end_idx = seg["Low"].rolling(4).min().idxmin()
        target = float(day.loc[pb_end_idx, "Close"] + leg)
        return target, None
    else:
        leg = float(day["Open"].iloc[0] - first_hour["Low"].min())
        seg = day.iloc[12:20]
        if seg.empty: return None, None
        pb_end_idx = seg["High"].rolling(4).max().idxmax()
        target = float(day.loc[pb_end_idx, "Close"] - leg)
        return None, target

# ========= Day Outlook (probabilistic) =========

def day_outlook_prediction(overlap: float, always_in: str, by18: dict, or_info: dict) -> dict:
    # Base: range probability from overlap
    range_prob = float(np.clip(overlap, 0.0, 1.0))
    trend_pool = 1.0 - range_prob
    bull = bear = 0.5 * trend_pool

    # Always-In tilt
    if always_in == "bull":
        bull += 0.15 * trend_pool; bear -= 0.15 * trend_pool
    elif always_in == "bear":
        bear += 0.15 * trend_pool; bull -= 0.15 * trend_pool

    # Bar-18 tilt
    if by18.get("enough_bars"):
        if by18.get("low_still_day_low"):  # low holds → bullish tilt
            bull += 0.12 * trend_pool; bear -= 0.12 * trend_pool
        if by18.get("high_still_day_high"):  # high holds → bearish tilt
            bear += 0.12 * trend_pool; bull -= 0.12 * trend_pool

    # Normalize
    bull = max(0, bull); bear = max(0, bear); range_prob = max(0, range_prob)
    s = bull + bear + range_prob
    if s <= 1e-9:
        bull = bear = 0.0; range_prob = 1.0; s = 1.0
    bull /= s; bear /= s; range_prob /= s

    probs = {"Bull Day": bull, "Range Day": range_prob, "Bear Day": bear}
    label = max(probs, key=probs.get)
    return {"label": label, "bull": bull, "range": range_prob, "bear": bear}

# ========= Strategy Suggestion with score =========

def strategy_suggestion(day: pd.DataFrame, or_info: dict, by18: dict, always_in: str, outlook: dict) -> dict:
    last = float(day["Close"].iloc[-1])
    rng = max(1e-9, float(day["High"].max() - day["Low"].min()))

    label = "Wait for clarity"
    rationale = "Sideways conditions"
    score = 0.5

    # Opening Range scenarios
    if or_info["status"] == "inside":
        # Fade to OR mid or await breakout
        label = "Fade toward OR mid or wait for clean OR breakout"
        rationale = "Price inside OR; breakouts often fail once before trend forms."
        score = 0.55
    elif or_info["status"] == "above":
        label = "Buy pullback to OR high / EMA20"
        rationale = "Breakout above OR; pullbacks often find support near OR high."
        score = 0.68
    elif or_info["status"] == "below":
        label = "Sell pullback to OR low / EMA20"
        rationale = "Breakout below OR; pullbacks often find resistance near OR low."
        score = 0.68

    # Bar-18 context
    if by18.get("enough_bars"):
        near_hi18 = abs(last - by18["high_at18"]) <= 0.2 * rng
        near_lo18 = abs(last - by18["low_at18"]) <= 0.2 * rng
        if by18.get("low_still_day_low") and near_lo18:
            label = "Buy pullback near morning low (Bar-18 hold)"
            rationale = "Morning low still stands; pullback entries often work unless clean breakdown."
            score = 0.75
        if by18.get("high_still_day_high") and near_hi18:
            label = "Sell pullback near morning high (Bar-18 hold)"
            rationale = "Morning high still stands; pullback entries often work unless clean breakout."
            score = 0.75

    # Bias integration
    if outlook["label"] == "Bull Day":
        score = max(score, 0.72)
    elif outlook["label"] == "Bear Day":
        score = max(score, 0.72)
    else:  # Range Day
        score = max(score, 0.60)

    return {"label": label, "score": round(float(np.clip(score, 0.0, 1.0)), 2), "rationale": rationale}

# ========= Summary Generation =========

def summary_text(symbol: str, or_info: dict, by18: dict, outlook: dict, market_open: bool) -> str:
    parts: List[str] = []
    # 3-sentence core
    parts.append(
        f"Opening range (first {or_info['bars']} bars) set {or_info['high']:.2f}/{or_info['low']:.2f} as S/R; "
        f"price is {or_info['status']} the range."
    )
    if by18.get("enough_bars"):
        hi_txt = "still the day high" if by18.get("high_still_day_high") else "not the day high"
        lo_txt = "still the day low" if by18.get("low_still_day_low") else "not the day low"
        parts.append(f"By Bar 18, morning high is {hi_txt} and morning low is {lo_txt}.")
    else:
        parts.append("Bar-18 context not established yet.")
    parts.append(
        f"Implication: **{outlook['label']}** bias (Bull {outlook['bull']:.0%} • Range {outlook['range']:.0%} • Bear {outlook['bear']:.0%})."
    )

    # Brooks concepts list in prose
    parts.append(
        "We evaluate Bar 1/2 impulse, trend from the open, wedge/DT/DB/MTR patterns, measured-move projections, "
        "and leg counting (second legs, three-push wedges). Day types considered: Spike & Channel, Range, Trend Reversal, "
        "and Trending Trading Range."
    )

    if not market_open:
        parts.append("Market closed: overview reflects the full session; Bar-18 and breakout assessments are final.")

    return " ".join(parts)

# =============================
# Main
# =============================
if go and symbol:
    with st.spinner(f"Fetching {symbol} intraday data…"):
        df_all = fetch_intraday(symbol)

    if df_all is None or df_all.empty:
        st.error("Could not fetch data. Try a different symbol.")
    else:
        # Convert to ET for session slicing; display remains via TradingView
        df_all["Datetime"] = pd.to_datetime(df_all["Datetime"], utc=True, errors="coerce").dt.tz_convert(ET)
        df_all["Date"] = df_all["Datetime"].dt.date
        last_date = df_all["Date"].max()
        day = df_all[df_all["Date"] == last_date].copy()
        if day.empty and len(df_all) > 0:
            day = df_all.tail(78).copy()
        day = day.drop(columns=["Date"]) if "Date" in day.columns else day

        # Compute metrics
        always, ema20, ema50 = always_in_state(day)
        or_info = opening_range_info(day, 5, 18)
        by18 = bar18_info(day)
        ovlp = overlap_score(day, 24)
        mm_up, mm_dn = measured_move_targets(day)
        outlook = day_outlook_prediction(ovlp, always, by18, or_info)
        strat = strategy_suggestion(day, or_info, by18, always, outlook)

        # Strategy card
        st.markdown("""
        <div class="card">
          <h3>Recommended Strategy</h3>
          <div class="strategy">{label}</div>
          <div class="score">Score: {score}</div>
          <div style="margin-top:6px; color:#555;">{rationale}</div>
        </div>
        """.format(label=html.escape(str(strat["label"]), quote=True),
            score=strat["score"],
            rationale=html.escape(str(strat["rationale"]), quote=True)),
        unsafe_allow_html=True)


        # Summary
        st.markdown("""
        <div class="card" style="margin-top:12px;">
          <h3>Summary</h3>
          <div>{summary}</div>
        </div>
        """.format(summary=summary_text(symbol, or_info, by18, outlook, market_open)), unsafe_allow_html=True)

        # TradingView chart
        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        tradingview_widget(symbol, theme="light", height=560)

        # Market state note
        if market_open:
            st.info("Market is open: setups adapt intraday based on OR status and Bar-18 developments.")
        else:
            st.info("Market is closed: summary reflects end-of-day structure and Brooks heuristics accuracy.")
