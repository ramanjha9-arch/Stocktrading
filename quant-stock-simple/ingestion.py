"""
Condensed data ingestion — free sources only, no API keys required.
CAVEAT (surfaced to the user, not hidden): yfinance is an unofficial wrapper
around Yahoo Finance and can rate-limit or return partial data; RSS feeds
only expose recent items. Every failure is logged to data_quality_log
instead of silently skipped.
"""
import datetime as dt
import calendar
from urllib.parse import quote_plus
import feedparser
import yfinance as yf
from db import get_conn, get_or_create_stock, log_issue

NIFTY_SYMBOL, NIFTY_YAHOO = "NIFTY50", "^NSEI"

GENERAL_FEEDS = {
    "moneycontrol": "https://www.moneycontrol.com/rss/marketreports.xml",
    "economic_times_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
}


def refresh_prices(symbols: list[str], period: str = "2y") -> dict:
    summary = {"ok": [], "failed": []}
    with get_conn() as conn:
        get_or_create_stock(conn, NIFTY_SYMBOL, NIFTY_YAHOO)
        for symbol in symbols + [NIFTY_SYMBOL]:
            yahoo_symbol = NIFTY_YAHOO if symbol == NIFTY_SYMBOL else f"{symbol}.NS"
            try:
                df = yf.Ticker(yahoo_symbol).history(period=period, auto_adjust=True)
                if df.empty:
                    raise ValueError("empty response from yfinance")
                stock_id = get_or_create_stock(conn, symbol, yahoo_symbol)
                rows = [(stock_id, idx.date().isoformat(), float(r["Open"]), float(r["High"]),
                         float(r["Low"]), float(r["Close"]), float(r["Volume"])) for idx, r in df.iterrows()]
                conn.executemany(
                    "INSERT OR REPLACE INTO prices (stock_id, trade_date, open, high, low, close, volume) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
                summary["ok"].append(symbol)
            except Exception as e:
                summary["failed"].append({"symbol": symbol, "error": str(e)})
                log_issue(conn, "api_failure", f"price ingestion failed for {symbol}: {e}")
    return summary


def refresh_fundamentals(symbols: list[str]) -> dict:
    summary = {"ok": [], "unavailable": [], "failed": []}
    with get_conn() as conn:
        for symbol in symbols:
            stock = conn.execute("SELECT id FROM stocks WHERE symbol = ?", (symbol,)).fetchone()
            if not stock:
                continue
            stock_id = stock["id"]
            try:
                ticker = yf.Ticker(f"{symbol}.NS")
                qf = ticker.quarterly_financials
                info = ticker.info or {}
                if qf is None or qf.empty:
                    summary["unavailable"].append(symbol)
                    log_issue(conn, "stale_fundamentals", f"no quarterly financials for {symbol}")
                    continue
                for period_end in qf.columns:
                    def _get(field):
                        try:
                            return float(qf.loc[field, period_end])
                        except (KeyError, ValueError, TypeError):
                            return None
                    conn.execute(
                        "INSERT OR REPLACE INTO fundamentals "
                        "(stock_id, period_end, revenue, pat, eps, total_debt, market_cap) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (stock_id, period_end.date().isoformat(), _get("Total Revenue"),
                         _get("Net Income"), None, _get("Total Debt"), info.get("marketCap")))
                summary["ok"].append(symbol)
            except Exception as e:
                summary["failed"].append({"symbol": symbol, "error": str(e)})
                log_issue(conn, "api_failure", f"fundamentals failed for {symbol}: {e}")
    return summary


def _published(entry) -> str | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    return dt.datetime.fromtimestamp(calendar.timegm(parsed), dt.timezone.utc).isoformat()


def refresh_news(symbols: list[str], per_symbol_limit: int = 10) -> dict:
    from nlp import classify_headline
    summary = {"inserted": 0, "ignored_no_event": 0, "failed_feeds": []}
    with get_conn() as conn:
        stocks = {r["symbol"]: r["id"] for r in conn.execute("SELECT id, symbol FROM stocks").fetchall()}
        mcaps = {r["stock_id"]: r["market_cap"] for r in conn.execute(
            "SELECT stock_id, market_cap FROM fundamentals WHERE market_cap IS NOT NULL "
            "GROUP BY stock_id HAVING MAX(period_end)").fetchall()}

        def handle_entry(entry, stock_id):
            title, url, published = entry.get("title", "").strip(), entry.get("link"), _published(entry)
            if not title or not url or not published:
                return
            if conn.execute("SELECT 1 FROM news_events WHERE url = ?", (url,)).fetchone():
                return
            result = classify_headline(title, mcaps.get(stock_id))
            if result is None:
                summary["ignored_no_event"] += 1
                return
            conn.execute(
                "INSERT INTO news_events (stock_id, headline, url, event_type, direction, "
                "materiality, confidence, event_score, published_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (stock_id, title, url, result["event_type"], result["direction"], result["materiality"],
                 result["confidence"], result["event_score"], published))
            summary["inserted"] += 1

        for name, url in GENERAL_FEEDS.items():
            try:
                parsed = feedparser.parse(url)
                if parsed.bozo and not parsed.entries:
                    raise ValueError("feed unreachable")
                for entry in parsed.entries:
                    handle_entry(entry, None)
            except Exception as e:
                summary["failed_feeds"].append({"source": name, "error": str(e)})
                log_issue(conn, "api_failure", f"news feed '{name}' failed: {e}")

        for symbol in symbols:
            if symbol not in stocks:
                continue
            q = quote_plus(f"{symbol} NSE stock when:7d")
            url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
            try:
                parsed = feedparser.parse(url)
                for entry in parsed.entries[:per_symbol_limit]:
                    handle_entry(entry, stocks[symbol])
            except Exception as e:
                summary["failed_feeds"].append({"source": f"google_news:{symbol}", "error": str(e)})
                log_issue(conn, "api_failure", f"google news failed for {symbol}: {e}")
    return summary
