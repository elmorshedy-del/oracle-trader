# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Polymarket Algo Trader — a single-service Python FastAPI app that runs an autonomous trading pipeline for Polymarket prediction markets with a real-time web dashboard on port 8000. No database, no Docker, no separate frontend build step. All state is file-based (JSON/JSONL in `logs/` and `/data/state.json`).

### Quick reference

- **Install deps:** `pip install -r requirements.txt`
- **Configure:** `cp .env.example .env` (paper mode needs no API keys)
- **Run:** `python3 main.py` (serves dashboard at `http://localhost:8000`)
- **Health check:** `curl http://localhost:8000/api/health`
- **Deployment:** Railway via Nixpacks (see `railway.json`, `Procfile`)

### Known issues

- `data/models.py` and `config.py` contain duplicate definitions (duplicate enum members in `SignalSource`, duplicate `CryptoArbConfig`/`WeatherForecastConfig` classes, duplicate `PipelineConfig` fields). These were introduced in the crypto temporal arb + weather forecast feature commits. On **Python 3.11+**, duplicate enum members raise `TypeError` and crash the app on import. They must be removed before the app can run locally or on Railway with Python 3.11+.

### Gotchas

- The system `python` command may not exist; always use `python3`.
- pip installs go to `~/.local/bin` — ensure this is on `PATH`.
- No linter, formatter, or test framework is configured in this repo. There are no automated tests.
- The app makes outbound HTTP calls to Polymarket APIs, Binance, CoinGecko, NOAA, and NYT RSS feeds on startup. Network access is required.

### Railway deployment

Railway deployment requires a `RAILWAY_TOKEN` environment variable for CLI-based deploys (`railway up`). Alternatively, if the repo is connected to Railway for auto-deploy, pushing to `main` triggers a deployment. The Railway CLI can be installed via `npm install -g @railway/cli`.
