import os
import datetime as dt
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler

from db import init_db, get_conn
from pipeline import run_full_refresh, _load_price_df
from backtest import backtest_technical_score
from ingestion import NIFTY_SYMBOL

WATCHLIST = [s.strip().upper() for s in os.getenv("WATCHLIST", "").split(",") if s.strip()]

app = FastAPI(title="Simple Quant Stock Picker")


@app.on_event("startup")
def startup():
    init_db()
    scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
    # Once daily after market close (15:45 IST) — set to your host's local time if different.
    scheduler.add_job(lambda: run_full_refresh(WATCHLIST), "cron", hour=15, minute=45)
    scheduler.start()


class HoldingCreate(BaseModel):
    symbol: str
    quantity: float
    buy_price: float
    buy_date: str
    notes: str | None = None


@app.get("/api/health")
def health():
    return {"status": "ok", "watchlist_size": len(WATCHLIST), "gemini_configured": bool(os.getenv("GEMINI_API_KEY"))}


@app.post("/api/refresh")
def manual_refresh():
    return run_full_refresh(WATCHLIST)


@app.get("/api/picks")
def get_picks(horizon: str = "short_term"):
    if horizon not in ("daily", "short_term", "monthly"):
        raise HTTPException(400, "horizon must be daily, short_term, or monthly")
    with get_conn() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM picks_cache WHERE horizon=? ORDER BY composite_score DESC", (horizon,)).fetchall()]
    if not rows:
        return {"horizon": horizon, "picks": [], "message": (
            "NO VALIDATED PICK for this horizon right now — either no stock cleared the score "
            "threshold, or a refresh hasn't run yet (POST /api/refresh)."
        )}
    return {"horizon": horizon, "picks": rows}


@app.get("/api/features/{symbol}")
def get_features(symbol: str):
    symbol = symbol.upper()
    with get_conn() as conn:
        stock = conn.execute("SELECT * FROM stocks WHERE symbol=?", (symbol,)).fetchone()
        if not stock:
            raise HTTPException(404, f"{symbol} not tracked — add it to WATCHLIST and refresh")
        price_df = _load_price_df(conn, stock["id"])
        news = [dict(r) for r in conn.execute(
            "SELECT * FROM news_events WHERE stock_id=? ORDER BY published_at DESC LIMIT 10",
            (stock["id"],)).fetchall()]
        fundamentals = [dict(r) for r in conn.execute(
            "SELECT * FROM fundamentals WHERE stock_id=? ORDER BY period_end DESC LIMIT 4",
            (stock["id"],)).fetchall()]
    return {
        "symbol": symbol,
        "price_history": [{"date": d.date().isoformat(), "close": r.close}
                           for d, r in price_df.tail(180).iterrows()] if not price_df.empty else [],
        "recent_news": news,
        "fundamentals": fundamentals,
    }


@app.get("/api/backtest")
def get_backtest():
    with get_conn() as conn:
        nifty = conn.execute("SELECT id FROM stocks WHERE symbol=?", (NIFTY_SYMBOL,)).fetchone()
        nifty_df = _load_price_df(conn, nifty["id"]) if nifty else None
        price_history = {}
        for symbol in WATCHLIST:
            stock = conn.execute("SELECT id FROM stocks WHERE symbol=?", (symbol,)).fetchone()
            if stock:
                df = _load_price_df(conn, stock["id"])
                if not df.empty:
                    price_history[symbol] = df
    if not price_history:
        return {"status": "insufficient_data", "message": "Run a refresh first to collect price history."}
    return backtest_technical_score(price_history, nifty_df if nifty_df is not None else None)


@app.get("/api/holdings")
def list_holdings():
    with get_conn() as conn:
        holdings = [dict(r) for r in conn.execute("SELECT * FROM holdings").fetchall()]
        for h in holdings:
            stock = conn.execute("SELECT id FROM stocks WHERE symbol=?", (h["symbol"],)).fetchone()
            price_df = _load_price_df(conn, stock["id"]) if stock else None
            h["current_price"] = float(price_df["close"].iloc[-1]) if price_df is not None and not price_df.empty else None
            if h["current_price"]:
                h["current_value"] = h["current_price"] * h["quantity"]
                h["unrealized_pnl"] = h["current_value"] - h["buy_price"] * h["quantity"]
                h["unrealized_pnl_pct"] = h["unrealized_pnl"] / (h["buy_price"] * h["quantity"]) * 100
    return holdings


@app.post("/api/holdings")
def add_holding(holding: HoldingCreate):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO holdings (symbol, quantity, buy_price, buy_date, notes) VALUES (?,?,?,?,?)",
            (holding.symbol.upper(), holding.quantity, holding.buy_price, holding.buy_date, holding.notes))
    return {"status": "created"}


@app.delete("/api/holdings/{holding_id}")
def delete_holding(holding_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM holdings WHERE id=?", (holding_id,))
    return {"status": "deleted"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
