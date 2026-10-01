"""
Deterministic technical + fundamental + sentiment scoring, each 0-100.
This is the ONLY thing that decides which stocks are candidates — the LLM
(llm.py) never picks; it only explains whatever this module has already
shortlisted, using the numbers this module computed.
"""
import numpy as np
import pandas as pd

HORIZON_WEIGHTS = {
    # (technical, fundamental, sentiment)
    "daily": (0.70, 0.05, 0.25),
    "short_term": (0.50, 0.25, 0.25),   # ~5-10 trading days
    "monthly": (0.30, 0.50, 0.20),      # ~20-40 trading days
}
MIN_SCORE_FOR_PICK = 60  # below this -> "no validated pick" for that horizon


def _sma(s, n): return s.rolling(n, min_periods=n).mean()
def _ema(s, n): return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _rsi(close, n=14):
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd_hist(close):
    macd = _ema(close, 12) - _ema(close, 26)
    return macd - _ema(macd, 9)


def _atr(df, n=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(),
                     (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()


def compute_indicators(df: pd.DataFrame, nifty_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """df: OHLCV indexed by date, ascending. Returns full history of indicators (for backtesting)."""
    out = pd.DataFrame(index=df.index)
    out["close"] = df["close"]
    out["sma_20"], out["sma_50"], out["sma_200"] = _sma(df.close, 20), _sma(df.close, 50), _sma(df.close, 200)
    out["rsi_14"] = _rsi(df.close)
    out["macd_hist"] = _macd_hist(df.close)
    out["atr_pct"] = _atr(df) / df.close * 100
    out["vol_ratio"] = df.volume / _sma(df.volume, 20)
    if nifty_df is not None and not nifty_df.empty:
        nifty_close = nifty_df["close"].reindex(df.index).ffill()
        out["rel_strength_20d"] = (df.close / df.close.shift(20) - 1) - (nifty_close / nifty_close.shift(20) - 1)
    else:
        out["rel_strength_20d"] = np.nan
    return out


def technical_score_series(ind: pd.DataFrame) -> pd.Series:
    """0-100 per row. Pure function of indicators up to that row — safe for backtesting."""
    score = pd.Series(50.0, index=ind.index)  # neutral baseline

    uptrend = (ind.close > ind.sma_50) & (ind.sma_50 > ind.sma_200)
    downtrend = (ind.close < ind.sma_50) & (ind.sma_50 < ind.sma_200)
    score += np.where(uptrend, 20, np.where(downtrend, -20, 0))

    rsi_component = ((ind.rsi_14 - 50) / 50 * 15).clip(-15, 15)
    rsi_component = pd.Series(
        np.where(ind.rsi_14 > 75, rsi_component - 10, rsi_component), index=ind.index
    )  # overbought penalty
    score += rsi_component.fillna(0)

    score += pd.Series(np.where(ind.macd_hist > 0, 10, np.where(ind.macd_hist < 0, -10, 0)), index=ind.index)
    score += (ind.rel_strength_20d.clip(-0.15, 0.15) / 0.15 * 15).fillna(0)
    score += pd.Series(np.where(ind.vol_ratio > 1.5, 5, 0), index=ind.index)  # volume expansion confirms the move
    score -= pd.Series(np.where(ind.atr_pct > 5, 10, 0), index=ind.index)      # very high daily volatility -> risk penalty

    return score.clip(0, 100)


def fundamental_score(fund_rows: list[dict]) -> tuple[float | None, dict]:
    """fund_rows: ascending by period_end. Returns (score, detail) or (None, {}) if too little data."""
    if len(fund_rows) < 2:
        return None, {}
    curr, prev = fund_rows[-1], fund_rows[-2]
    detail = {}
    score = 50.0

    if curr["revenue"] and prev["revenue"]:
        rev_growth = (curr["revenue"] - prev["revenue"]) / abs(prev["revenue"]) * 100
        detail["revenue_qoq_growth_pct"] = round(rev_growth, 1)
        score += np.clip(rev_growth, -20, 20)

    if curr["pat"] and prev["pat"]:
        pat_growth = (curr["pat"] - prev["pat"]) / abs(prev["pat"]) * 100
        detail["pat_qoq_growth_pct"] = round(pat_growth, 1)
        score += np.clip(pat_growth / 2, -20, 20)

    if curr.get("total_debt") is not None and prev.get("total_debt") is not None and prev["total_debt"]:
        debt_change = (curr["total_debt"] - prev["total_debt"]) / abs(prev["total_debt"]) * 100
        detail["debt_change_pct"] = round(debt_change, 1)
        score -= np.clip(debt_change / 2, -10, 10)

    return float(np.clip(score, 0, 100)), detail


def sentiment_score(events: list[dict]) -> tuple[float | None, dict]:
    """events: recent news_events rows for one stock. None if no classified news."""
    if not events:
        return None, {}
    avg = sum(e["event_score"] for e in events) / len(events)
    high_risk = any(e["direction"] < 0 and e["materiality"] in ("high", "medium") and e["confidence"] >= 0.6
                     for e in events)
    score = 50 + avg * 50
    if high_risk:
        score = min(score, 25)  # a real negative material event caps the score regardless of average
    return float(np.clip(score, 0, 100)), {
        "n_events": len(events), "avg_event_score": round(avg, 3), "high_risk_event": high_risk,
    }


def composite_score(technical, fundamental, sentiment, horizon: str) -> float:
    w_t, w_f, w_s = HORIZON_WEIGHTS[horizon]
    parts, weights = [], []
    for val, w in ((technical, w_t), (fundamental, w_f), (sentiment, w_s)):
        if val is not None:
            parts.append(val * w)
            weights.append(w)
    if not weights:
        return 0.0
    return round(sum(parts) / sum(weights), 2)  # renormalized when a component is missing, never guessed
