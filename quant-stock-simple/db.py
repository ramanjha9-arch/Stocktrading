"""SQLite storage — single file, zero external services, zero setup."""
import sqlite3
from contextlib import contextmanager

DB_PATH = "data.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS stocks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT UNIQUE NOT NULL,
    yahoo_symbol TEXT UNIQUE NOT NULL,
    name TEXT,
    sector TEXT
);

CREATE TABLE IF NOT EXISTS prices (
    stock_id INTEGER NOT NULL,
    trade_date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (stock_id, trade_date)
);

CREATE TABLE IF NOT EXISTS fundamentals (
    stock_id INTEGER NOT NULL,
    period_end TEXT NOT NULL,
    revenue REAL, pat REAL, eps REAL, total_debt REAL, market_cap REAL,
    PRIMARY KEY (stock_id, period_end)
);

CREATE TABLE IF NOT EXISTS news_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_id INTEGER,
    headline TEXT NOT NULL,
    url TEXT UNIQUE,
    event_type TEXT,
    direction INTEGER,
    materiality TEXT,
    confidence REAL,
    event_score REAL,
    published_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS holdings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    quantity REAL NOT NULL,
    buy_price REAL NOT NULL,
    buy_date TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS picks_cache (
    horizon TEXT NOT NULL,
    symbol TEXT NOT NULL,
    composite_score REAL,
    technical_score REAL,
    fundamental_score REAL,
    sentiment_score REAL,
    reasoning TEXT,
    generated_at TEXT NOT NULL,
    PRIMARY KEY (horizon, symbol)
);

CREATE TABLE IF NOT EXISTS data_quality_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at TEXT NOT NULL,
    issue_type TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS refresh_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ran_at TEXT NOT NULL,
    summary TEXT
);
"""


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_or_create_stock(conn, symbol: str, yahoo_symbol: str | None = None) -> int:
    row = conn.execute("SELECT id FROM stocks WHERE symbol = ?", (symbol,)).fetchone()
    if row:
        return row["id"]
    ys = yahoo_symbol or f"{symbol}.NS"
    cur = conn.execute("INSERT INTO stocks (symbol, yahoo_symbol) VALUES (?, ?)", (symbol, ys))
    return cur.lastrowid


def log_issue(conn, issue_type: str, description: str):
    import datetime as dt
    conn.execute(
        "INSERT INTO data_quality_log (logged_at, issue_type, description) VALUES (?, ?, ?)",
        (dt.datetime.utcnow().isoformat(), issue_type, description),
    )
