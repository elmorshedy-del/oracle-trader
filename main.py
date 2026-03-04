"""
Polymarket Algo Trading — FastAPI Server
========================================
Serves the dashboard and API endpoints.
Runs the algo pipeline in background tasks.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import PipelineConfig
from engine.pipeline import Pipeline

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline: Pipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start pipeline on startup, stop on shutdown."""
    global pipeline
    config = PipelineConfig()
    pipeline = Pipeline(config)

    # Start pipeline in background
    task = asyncio.create_task(pipeline.start())
    logger.info("Pipeline background task started")

    yield

    # Shutdown
    if pipeline:
        await pipeline.stop()
    task.cancel()


app = FastAPI(
    title="Polymarket Algo Trader",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/state")
async def get_state():
    """Get full dashboard state."""
    if pipeline is None:
        return {"mode":"paper","uptime_human":"starting...","scan_count":0,"active_markets":0,"portfolio":{"total_value":0,"cash":0,"positions_value":0,"total_pnl":0,"total_pnl_pct":0,"total_trades":0,"win_rate":0,"max_drawdown":0,"total_fees":0,"positions":[]},"signals":[],"trades":[],"strategies":{},"whale_wallets":[],"recent_news":[],"performance":{"by_strategy":{}},"errors":["Initializing..."],"markets_sample":[]}
    try:
        return pipeline.get_state()
    except Exception as e:
        return {"mode":"paper","uptime_human":"error","scan_count":0,"active_markets":0,"portfolio":{"total_value":0,"cash":0,"positions_value":0,"total_pnl":0,"total_pnl_pct":0,"total_trades":0,"win_rate":0,"max_drawdown":0,"total_fees":0,"positions":[]},"signals":[],"trades":[],"strategies":{},"whale_wallets":[],"recent_news":[],"performance":{"by_strategy":{}},"errors":[str(e)],"markets_sample":[]}


@app.get("/api/portfolio")
async def get_portfolio():
    """Get portfolio details."""
    if pipeline is None:
        return JSONResponse({"error": "Pipeline not initialized"}, status_code=503)
    state = pipeline.get_state()
    return state.get("portfolio", {})


@app.get("/api/signals")
async def get_signals():
    """Get recent signals."""
    if pipeline is None:
        return JSONResponse({"error": "Pipeline not initialized"}, status_code=503)
    state = pipeline.get_state()
    return state.get("signals", [])


@app.get("/api/trades")
async def get_trades():
    """Get trade history."""
    if pipeline is None:
        return JSONResponse({"error": "Pipeline not initialized"}, status_code=503)
    state = pipeline.get_state()
    return state.get("trades", [])


@app.get("/api/performance")
async def get_performance():
    """Get performance report for tuning."""
    if pipeline is None:
        return JSONResponse({"error": "Pipeline not initialized"}, status_code=503)
    state = pipeline.get_state()
    return state.get("performance", {})


@app.get("/api/markets")
async def get_markets():
    """Get top active markets."""
    if pipeline is None:
        return JSONResponse({"error": "Pipeline not initialized"}, status_code=503)
    state = pipeline.get_state()
    return state.get("markets_sample", [])


@app.get("/api/whales")
async def get_whales():
    """Get tracked whale wallets."""
    if pipeline is None:
        return JSONResponse({"error": "Pipeline not initialized"}, status_code=503)
    state = pipeline.get_state()
    return state.get("whale_wallets", [])


@app.get("/api/health")
async def health():
    """Health check for Railway."""
    return {"status": "ok", "mode": pipeline.config.mode if pipeline else "unknown"}


# ---------------------------------------------------------------------------
# Dashboard — serves the React build or inline HTML
# ---------------------------------------------------------------------------

DASHBOARD_HTML = Path(__file__).parent / "dashboard" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard."""
    if DASHBOARD_HTML.exists():
        return HTMLResponse(DASHBOARD_HTML.read_text())
    return HTMLResponse("<h1>Polymarket Algo Trader</h1><p>Dashboard loading...</p>")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
