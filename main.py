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
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import io
from fastapi.staticfiles import StaticFiles

from config import PipelineConfig
from engine.pipeline import Pipeline
from engine.multiagent import (
    MultiagentRuntime,
    OrchestratorConfig,
    consult_multiagent_logs,
    dataclass_to_dict,
)
from runtime_paths import LOG_DIR, STATE_PATH

# Logging
stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler(LOG_DIR / "app.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[stream_handler, file_handler],
    force=True,
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# Global pipeline instance
pipeline: Pipeline | None = None
multiagent_runtime: MultiagentRuntime | None = None


class MultiagentConsultRequest(BaseModel):
    question: str
    provider: str | None = "auto"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start pipeline on startup, stop on shutdown."""
    global pipeline, multiagent_runtime
    config = PipelineConfig()
    pipeline = Pipeline(config)
    multiagent_runtime = MultiagentRuntime(pipeline_config=config)

    # Start pipeline in background
    task = asyncio.create_task(pipeline.start())
    multiagent_task = asyncio.create_task(multiagent_runtime.start())
    logger.info("Pipeline background task started")
    logger.info("Multi-agent runtime background task started")

    yield

    # Shutdown
    if pipeline:
        await pipeline.stop()
    if multiagent_runtime:
        await multiagent_runtime.stop()
    task.cancel()
    multiagent_task.cancel()


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
    """Download all logs as a JSON bundle."""
    import json as _json
    bundle = {}
    for log_file in sorted(LOG_DIR.iterdir()):
        if log_file.suffix not in {".jsonl", ".json", ".log"}:
            continue
        try:
            content = log_file.read_text(errors="replace").strip()
            if log_file.suffix == ".jsonl":
                lines = []
                for line in content.split("\n"):
                    if line:
                        try:
                            lines.append(_json.loads(line))
                        except _json.JSONDecodeError:
                            lines.append({"raw": line})
                bundle[log_file.name] = lines
            elif log_file.suffix == ".json":
                bundle[log_file.name] = _json.loads(content) if content else {}
            else:
                bundle[log_file.name] = content
        except Exception:
            bundle[log_file.name] = {"error": "failed to read log file"}
    # Add current state
    if pipeline:
        try:
            bundle["current_state"] = pipeline.get_state()
        except Exception:
            pass
    bundle["log_dir"] = str(LOG_DIR)
    bundle["state_path"] = str(STATE_PATH)
    content = _json.dumps(bundle, indent=2, default=str)
    return StreamingResponse(
        io.BytesIO(content.encode()),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=oracle-trader-logs.json"}
    )


@app.get("/api/reset")
async def reset_portfolio():
    """Reset portfolio to fresh start with current config capital."""
    if pipeline is not None:
        await pipeline.reset_state()
        return {
            "status": "reset",
            "message": "Portfolio and strategy caches reset in-memory and on disk",
            "state_path": str(STATE_PATH),
            "log_dir": str(LOG_DIR),
        }

    STATE_PATH.unlink(missing_ok=True)
    return {
        "status": "reset",
        "message": "State file deleted",
        "state_path": str(STATE_PATH),
        "log_dir": str(LOG_DIR),
    }

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


@app.get("/api/multiagent/defaults")
async def multiagent_defaults():
    """Expose the current recommended multi-agent defaults."""
    config = OrchestratorConfig()
    return JSONResponse(dataclass_to_dict(config))


@app.get("/api/multiagent/status")
async def multiagent_status():
    """Expose the isolated Opus runtime status."""
    if multiagent_runtime is None:
        config = OrchestratorConfig()
        return JSONResponse(
            {
                "bridge": {
                    "mode": "isolated_runtime",
                    "state": "booting",
                    "next_step": "wait_for_multiagent_startup",
                },
                "summary": {
                    "scan_count": 0,
                    "active_markets": 0,
                    "open_positions": 0,
                    "top_blocker": "Multi-agent runtime is still starting",
                },
                "defaults": dataclass_to_dict(config),
                "portfolio": {},
                "health": {},
                "diagnostics": {},
                "market_mix": {},
                "market_preview": [],
                "module_cards": [],
                "strategy_cards": [],
                "comparison_views": [],
                "blockers": [],
            }
        )
    return JSONResponse(multiagent_runtime.get_status())


@app.get("/api/multiagent/logs/export")
async def multiagent_logs_export():
    """Export compact Opus runtime logs, metrics, and snapshots."""
    if multiagent_runtime is None:
        return JSONResponse(
            {
                "ok": False,
                "error": "multiagent runtime not initialized",
            },
            status_code=503,
        )

    bundle = io.BytesIO()
    llm_context = multiagent_runtime.llm_context()
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "status.json",
            json.dumps(multiagent_runtime.get_status(), indent=2, default=str),
        )
        archive.writestr(
            "llm_context.json",
            json.dumps(llm_context, indent=2, default=str),
        )
        archive.writestr(
            "README.txt",
            "\n".join(
                [
                    "Oracle Opus runtime export",
                    "",
                    "Contents:",
                    "- status.json: current runtime payload used by /multiagent",
                    "- llm_context.json: compact diagnostic context for on-request LLM review",
                    "- logs/multiagent_metrics.jsonl: compact per-cycle metrics log",
                    "- logs/multiagent_runtime.sqlite: persisted compact runtime database",
                    "- logs/multiagent_runtime_state.json: persisted Opus portfolio state",
                    "- logs/multiagent_runtime_meta.json: persisted Opus scan counters/totals",
                    "- snapshots/: recent scan-cycle snapshots",
                ]
            ),
        )

        for key in ("metrics_log_path", "metrics_db_path", "state_path", "runtime_meta_path"):
            raw_path = llm_context.get(key)
            if not raw_path:
                continue
            path = Path(raw_path)
            if path.exists():
                archive.write(path, arcname=f"logs/{path.name}")

        for snapshot in sorted(
            multiagent_runtime.snapshot_store.config.snapshot_dir.glob("cycle_*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )[:50]:
            archive.write(snapshot, arcname=f"snapshots/{snapshot.name}")

    bundle.seek(0)
    return StreamingResponse(
        bundle,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=oracle-opus-runtime-export.zip"},
    )


@app.post("/api/multiagent/consult")
async def multiagent_consult(payload: MultiagentConsultRequest):
    """Consult an LLM against compact Opus runtime diagnostics."""
    if multiagent_runtime is None:
        return JSONResponse(
            {
                "ok": False,
                "answer": "The isolated Opus runtime is not running yet.",
                "model": None,
            },
            status_code=503,
        )
    if not payload.question.strip():
        return JSONResponse(
            {
                "ok": False,
                "answer": "Ask a non-empty diagnostic question.",
                "model": None,
            },
            status_code=400,
        )

    result = await consult_multiagent_logs(
        question=payload.question.strip(),
        context=multiagent_runtime.llm_context(),
        preferred_provider=payload.provider,
    )
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Dashboard — serves the React build or inline HTML
# ---------------------------------------------------------------------------

DASHBOARD_HTML = Path(__file__).parent / "dashboard" / "index.html"
MULTIAGENT_DIR = Path(__file__).parent / "dashboard" / "multiagent"
MULTIAGENT_HTML = MULTIAGENT_DIR / "index.html"

app.mount("/multiagent-assets", StaticFiles(directory=MULTIAGENT_DIR), name="multiagent-assets")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard."""
    if DASHBOARD_HTML.exists():
        return HTMLResponse(DASHBOARD_HTML.read_text())
    return HTMLResponse("<h1>Polymarket Algo Trader</h1><p>Dashboard loading...</p>")


@app.get("/multiagent", response_class=HTMLResponse)
async def multiagent_dashboard():
    """Serve the separate multi-agent section."""
    if MULTIAGENT_HTML.exists():
        return HTMLResponse(MULTIAGENT_HTML.read_text())
    return HTMLResponse("<h1>Oracle Multi-Agent Lab</h1><p>Multi-agent dashboard unavailable.</p>")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
