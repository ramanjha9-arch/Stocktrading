"""
Simplified backtest: quartile-return analysis of the TECHNICAL score only.

Honesty about scope (surfaced in the UI, not hidden): this tests whether
the technical score has historically been associated with better forward
returns. It does NOT replay historical fundamentals or historical news
(point-in-time fundamental/news history isn't collected — that's the full
walk-forward engine, a bigger build). Treat this as a directional sanity
check on the technical half of the composite score, not a validation of
the whole system.
"""
import numpy as np
import pandas as pd
from scoring import compute_indicators, technical_score_series

HORIZON_DAYS = {"daily": 1, "short_term": 7, "monthly": 25}


def backtest_technical_score(price_history: dict[str, pd.DataFrame], nifty_df: pd.DataFrame) -> dict:
    """
    price_history: {symbol: OHLCV df indexed by date}. Returns, per horizon,
    quartile buckets of historical score -> mean/median forward return + win rate,
    plus the sample size so the reader can judge how much to trust it.
    """
    rows = []  # one row per (symbol, date): score + forward returns per horizon
    for symbol, df in price_history.items():
        if len(df) < 210:  # need 200d for SMA200 to exist, plus room for a forward window
            continue
        ind = compute_indicators(df, nifty_df)
        score = technical_score_series(ind)
        for horizon, days in HORIZON_DAYS.items():
            fwd_ret = df["close"].shift(-days) / df["close"] - 1
            valid = score.notna() & fwd_ret.notna()
            for dt_idx in df.index[valid]:
                rows.append({"symbol": symbol, "date": dt_idx, "horizon": horizon,
                             "score": score.loc[dt_idx], "fwd_return": fwd_ret.loc[dt_idx]})

    if not rows:
        return {"status": "insufficient_data", "message": "Not enough price history to backtest yet."}

    all_df = pd.DataFrame(rows)
    results = {}
    for horizon in HORIZON_DAYS:
        sub = all_df[all_df.horizon == horizon].copy()
        if len(sub) < 100:
            results[horizon] = {"status": "insufficient_data", "n": len(sub)}
            continue
        sub["quartile"] = pd.qcut(sub.score, 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
        agg = sub.groupby("quartile", observed=True).fwd_return.agg(
            mean_return="mean", median_return="median",
            win_rate=lambda x: (x > 0).mean(), n="count",
        )
        results[horizon] = {
            "n_total": int(len(sub)),
            "by_quartile": {
                str(q): {"mean_return_pct": round(r["mean_return"] * 100, 2),
                          "median_return_pct": round(r["median_return"] * 100, 2),
                          "win_rate_pct": round(r["win_rate"] * 100, 1),
                          "n": int(r["n"])}
                for q, r in agg.iterrows()
            },
        }
    return {"status": "ok", "scope": "technical_score_only", "results": results}
