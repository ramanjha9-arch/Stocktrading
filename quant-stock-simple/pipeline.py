"""Orchestrates a full refresh: ingest data -> score every tracked stock -> shortlist per horizon -> LLM reasoning."""
import datetime as dt
import pandas as pd
from db import get_conn, init_db
from ingestion import refresh_prices, refresh_fundamentals, refresh_news, NIFTY_SYMBOL
from scoring import compute_indicators, technical_score_series, fundamental_score, sentiment_score, composite_score, MIN_SCORE_FOR_PICK
from llm import generate_reasoning


def _load_price_df(conn, stock_id) -> pd.DataFrame:
    rows = conn.execute(
        "SELECT trade_date, open, high, low, close, volume FROM prices WHERE stock_id=? ORDER BY trade_date",
        (stock_id,)).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["trade_date", "open", "high", "low", "close", "volume"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")


def run_full_refresh(symbols: list[str]) -> dict:
    init_db()
    summary = {"prices": refresh_prices(symbols), "fundamentals": refresh_fundamentals(symbols),
               "news": refresh_news(symbols)}
    summary["scoring"] = run_scoring(symbols)
    with get_conn() as conn:
        conn.execute("INSERT INTO refresh_log (ran_at, summary) VALUES (?, ?)",
                     (dt.datetime.utcnow().isoformat(), str(summary)))
    return summary


def run_scoring(symbols: list[str]) -> dict:
    with get_conn() as conn:
        nifty_row = conn.execute("SELECT id FROM stocks WHERE symbol=?", (NIFTY_SYMBOL,)).fetchone()
        nifty_df = _load_price_df(conn, nifty_row["id"]) if nifty_row else pd.DataFrame()

        candidates = []  # per-stock computed scores, before shortlisting
        for symbol in symbols:
            stock = conn.execute("SELECT id, name FROM stocks WHERE symbol=?", (symbol,)).fetchone()
            if not stock:
                continue
            price_df = _load_price_df(conn, stock["id"])
            if len(price_df) < 60:
                continue
            ind = compute_indicators(price_df, nifty_df)
            tech = round(float(technical_score_series(ind).iloc[-1]), 1)
            tech_detail = {k: (round(float(v), 2) if pd.notna(v) else None)
                            for k, v in ind.iloc[-1][["rsi_14", "macd_hist", "rel_strength_20d"]].items()}

            fund_rows = [dict(r) for r in conn.execute(
                "SELECT * FROM fundamentals WHERE stock_id=? ORDER BY period_end", (stock["id"],)).fetchall()]
            fund, fund_detail = fundamental_score(fund_rows)

            since = (dt.datetime.utcnow() - dt.timedelta(days=7)).isoformat()
            news_rows = [dict(r) for r in conn.execute(
                "SELECT * FROM news_events WHERE stock_id=? AND published_at >= ? ORDER BY published_at DESC",
                (stock["id"], since)).fetchall()]
            sent, sent_detail = sentiment_score(news_rows)
            headlines = [n["headline"] for n in news_rows[:5]]

            candidates.append({
                "symbol": symbol, "technical": tech, "tech_detail": tech_detail,
                "fundamental": fund, "fund_detail": fund_detail,
                "sentiment": sent, "sent_detail": sent_detail, "headlines": headlines,
            })

        stats = {"candidates_scored": len(candidates), "picks_by_horizon": {}}
        for horizon in ("daily", "short_term", "monthly"):
            ranked = sorted(
                ({**c, "composite": composite_score(c["technical"], c["fundamental"], c["sentiment"], horizon)}
                 for c in candidates),
                key=lambda c: c["composite"], reverse=True)
            top = [c for c in ranked if c["composite"] >= MIN_SCORE_FOR_PICK][:3]
            stats["picks_by_horizon"][horizon] = len(top)

            conn.execute("DELETE FROM picks_cache WHERE horizon=?", (horizon,))
            for c in top:
                reasoning = generate_reasoning(
                    c["symbol"], horizon, c["composite"], c["technical"], c["tech_detail"],
                    c["fundamental"], c["fund_detail"], c["sentiment"], c["sent_detail"], c["headlines"])
                conn.execute(
                    "INSERT INTO picks_cache (horizon, symbol, composite_score, technical_score, "
                    "fundamental_score, sentiment_score, reasoning, generated_at) VALUES (?,?,?,?,?,?,?,?)",
                    (horizon, c["symbol"], c["composite"], c["technical"], c["fundamental"], c["sentiment"],
                     reasoning, dt.datetime.utcnow().isoformat()))
    return stats
