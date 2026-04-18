"""
Polymarket Algo Trading — FastAPI Server
========================================
Serves the dashboard and API endpoints.
Runs the algo pipeline in background tasks.
"""

import asyncio
import json
import logging
import os
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import io
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


def _record_count(payload: bytes) -> int:
    """Count newline-delimited records without parsing the whole file."""
    if not payload:
        return 0
    line_count = payload.count(b"\n")
    if payload.endswith(b"\n"):
        return line_count
    return line_count + 1


def _build_export_manifest(log_dir: Path, runtime_state_path: Path) -> tuple[dict, list[tuple[str, bytes]]]:
    """Collect export files and a lightweight manifest for the zip bundle."""
    export_files: list[tuple[str, bytes]] = []
    file_entries: list[dict] = []

    for log_file in sorted(log_dir.glob("*")):
        if not log_file.is_file() or log_file.suffix not in {".jsonl", ".json"}:
            continue
        payload = log_file.read_bytes()
        archive_path = f"logs/{log_file.name}"
        export_files.append((archive_path, payload))
        entry = {
            "path": archive_path,
            "kind": "jsonl" if log_file.suffix == ".jsonl" else "json",
            "size_bytes": len(payload),
        }
        if log_file.suffix == ".jsonl":
            entry["records"] = _record_count(payload)
        file_entries.append(entry)

    if runtime_state_path.exists():
        state_payload = runtime_state_path.read_bytes()
        export_files.append(("state/runtime_state.json", state_payload))
        file_entries.append({
            "path": "state/runtime_state.json",
            "kind": "json",
            "size_bytes": len(state_payload),
        })

    manifest = {
        "format": "oracle-trader-log-export",
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": file_entries,
    }

    if pipeline is not None:
        manifest["runtime"] = {
            "mode": pipeline.config.mode,
            "scan_count": pipeline._scan_count,
            "active_markets": len(pipeline._markets),
            "open_positions": len(pipeline.trader.portfolio.positions),
            "portfolio_total_value": round(pipeline.trader.portfolio.total_value, 2),
            "portfolio_cash": round(pipeline.trader.portfolio.cash, 2),
            "total_trades": pipeline.trader.portfolio.total_trades,
        }

    return manifest, export_files


def _build_export_readme() -> str:
    """Human-readable guide bundled with the exported archive."""
    return "\n".join([
        "Oracle Trader export layout",
        "",
        "summary/manifest.json",
        "  Inventory of exported files, sizes, and JSONL record counts.",
        "",
        "state/live_state.json",
        "  Current API/dashboard snapshot optimized for quick inspection.",
        "",
        "state/runtime_state.json",
        "  Runtime checkpoint from disk. This stays lightweight for recovery.",
        "",
        "logs/*.jsonl",
        "  Full append-only histories split by data type:",
        "  - signals.jsonl: signal lifecycle events with portfolio snapshots",
        "  - trades.jsonl: trade executions with signal context and execution details",
        "  - scans.jsonl: per-scan summaries and counts",
        "  - health.jsonl / slippage.jsonl / ab_tests.jsonl: subsystem logs",
        "",
        "JSONL files are the source of truth for full historical detail.",
    ])


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


@app.get("/api/logs/download")
async def download_logs():
    """Download a compressed multi-file export of logs and state."""
    log_dir = Path("logs")
    runtime_state_path = pipeline.trader.state_path if pipeline else Path("/data/state.json")
    manifest, export_files = _build_export_manifest(log_dir, runtime_state_path)

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary/manifest.json", json.dumps(manifest, indent=2, default=str))
        zf.writestr("summary/README.txt", _build_export_readme())
        if pipeline:
            zf.writestr(
                "state/live_state.json",
                json.dumps(pipeline.get_state(), indent=2, default=str),
            )
        for archive_path, payload in export_files:
            zf.writestr(archive_path, payload)

    archive.seek(0)
    filename = f"oracle-trader-export-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.zip"
    return StreamingResponse(
        archive,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/api/logs/manifest")
async def logs_manifest():
    """Preview the export manifest without downloading the full archive."""
    log_dir = Path("logs")
    runtime_state_path = pipeline.trader.state_path if pipeline else Path("/data/state.json")
    manifest, _ = _build_export_manifest(log_dir, runtime_state_path)
    return manifest


@app.get("/api/reset")
async def reset_portfolio():
    """Reset portfolio to fresh start with current config capital."""
    import os
    state_path = "/data/state.json"
    if os.path.exists(state_path):
        os.remove(state_path)
    return {"status": "reset", "message": "Restart the service to apply"}

@app.get("/api/health/detail")
async def health_detail():
    """Get detailed health report."""
    if pipeline is None:
        return {"overall_status": "unknown", "apis": {}, "strategies": {}}
    try:
        return pipeline.health.get_health_report()
    except Exception as e:
        return {"overall_status": "error", "error": str(e)}


@app.get("/api/ab-report")
async def ab_report():
    """Get A/B testing report."""
    if pipeline is None:
        return {}
    try:
        return pipeline.ab_tester.get_report()
    except Exception:
        return {}


@app.get("/api/slippage")
async def slippage_stats():
    """Get slippage model calibration stats."""
    if pipeline is None:
        return {}
    try:
        return pipeline.slippage.get_stats()
    except Exception:
        return {}


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
