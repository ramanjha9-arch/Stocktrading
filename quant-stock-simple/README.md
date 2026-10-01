# Simple Quant Stock Picker

One service, one SQLite file, one dashboard. No login, no external database,
no Docker Compose. Deploy it as a single Render/Railway web service and
you're done.

## What it does

- Scores every stock in your `WATCHLIST` on three deterministic, auditable
  factors — **technical** (trend/momentum/relative-strength via plain
  pandas), **fundamental** (quarter-over-quarter revenue/profit/debt trend),
  **sentiment** (rule-based news event classification, no black-box model).
- Shortlists the top-scoring stocks per horizon (**daily**, **short-term
  5–10 days**, **monthly 20–40 days**) — only stocks scoring ≥60/100 make
  the list; otherwise you get an honest "no validated pick" rather than a
  forced recommendation.
- Sends **only the shortlisted stock's own numbers** to Gemini, which writes
  the plain-English reasoning paragraph. Gemini never chooses the stock and
  is explicitly instructed not to invent any figure it wasn't given.
- Includes a **backtest** tab: a quartile-return analysis showing whether
  the technical score has historically preceded better forward returns.
  Scope-limited — see the caveat below.
- **Portfolio tracking**: add holdings, see live P&L. No login (single-user,
  per your call — everyone hitting this URL shares the same data).

## Why this differs from the full 12-phase platform

Your original spec explicitly says LLMs should never generate probabilities
or predict prices directly (section 41). This build honors that split:
a rule-based scoring engine does 100% of the picking; the LLM only explains
what already got picked, grounded in the same numbers you can see on screen.
It's a deliberately smaller, faster-to-ship version — not a replacement for
the full walk-forward/ML/calibration platform, if you want that later.

## Setup

```
cp .env.example .env        # add your Gemini key from https://aistudio.google.com/apikey
                             # edit WATCHLIST if you want different stocks
pip install -r requirements.txt
uvicorn app:app --reload
```
Open http://localhost:8000 — click **Refresh data** first (takes a few
minutes: it pulls prices, fundamentals and news for every watchlist stock,
then scores everything and generates reasoning for the shortlist).

## Deploy (Render)

1. Push this folder to a GitHub repo.
2. Render → New → Blueprint → point at the repo (uses `render.yaml`), or
   New → Web Service manually with build command `pip install -r requirements.txt`
   and start command `uvicorn app:app --host 0.0.0.0 --port $PORT`.
3. Set `GEMINI_API_KEY` in the Render dashboard's environment variables —
   never commit it to the repo.
4. **Persistent disk**: Render's free web services have an ephemeral
   filesystem — `data.db` resets on every redeploy/restart unless you attach
   a persistent disk (see `render.yaml`, or Railway's volumes, which are
   simpler to attach).

The daily auto-refresh (APScheduler, 15:45 IST) runs inside the same
process — no separate cron service needed. You can also trigger it manually
any time via the "Refresh data" button or `POST /api/refresh`.

## Known limitations (read before trusting a pick)

- **Free data sources**: `yfinance` can rate-limit or return partial data;
  fundamentals coverage for mid/small-caps is inconsistent; RSS/Google News
  only surface recent headlines (no deep news history).
- **Backtest scope**: tests the technical score only, using each date's
  actual price history — it does NOT replay historical fundamentals or
  historical news (that needs point-in-time feature snapshots, a bigger
  build). Treat it as a sanity check on the technical half, not full
  validation of the composite score.
- **No walk-forward/ML/calibration**: this is rule-based scoring, not the
  LightGBM/XGBoost ensemble + probability calibration from your original
  spec's Phase 6–7. Composite scores are a 0–100 weighted blend, not a
  calibrated P(+10%) type probability.
- **Single-user, no auth**: fine for personal use; don't put sensitive data
  behind this if you ever share the URL.
- **Gemini model names change often** — if `GEMINI_MODEL` in `.env` 404s,
  check https://ai.google.dev/gemini-api/docs/models for the current name.
