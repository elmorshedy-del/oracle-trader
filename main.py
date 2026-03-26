"""
Polymarket Algo Trading — FastAPI Server
========================================
Serves the dashboard and API endpoints.
Runs the algo pipeline in background tasks.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import threading
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
import io
from fastapi.staticfiles import StaticFiles

from config import PipelineConfig
from engine.pipeline import Pipeline
from engine.multiagent import (
    MultiagentRuntime,
    OrchestratorConfig,
    consult_legacy_logs,
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
runtime_loop: asyncio.AbstractEventLoop | None = None
runtime_thread: threading.Thread | None = None
pipeline_future = None
multiagent_future = None
OPUS_RUNTIME_ENABLED = os.getenv("OPUS_RUNTIME_ENABLED", "1").lower() in {"1", "true", "yes", "on"}


class MultiagentConsultRequest(BaseModel):
    question: str
    provider: str | None = "auto"
    history: list[dict[str, str]] | None = None


def _iter_legacy_export_files() -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    allowed_suffixes = {".jsonl", ".json", ".log", ".sqlite"}

    for path in sorted(LOG_DIR.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in allowed_suffixes:
            continue
        relative = path.relative_to(LOG_DIR)
        if relative.name.startswith("multiagent_"):
            continue
        if any(part.startswith("multiagent_") for part in relative.parts):
            continue
        files.append((path, f"logs/{relative.as_posix()}"))

    for path in sorted(STATE_PATH.parent.glob("state*.json")):
        if path.is_file():
            files.append((path, f"state/{path.name}"))

    weather_state = STATE_PATH.with_name("weather_state.json")
    if weather_state.exists():
        files.append((weather_state, f"state/{weather_state.name}"))

    return files


def _build_legacy_export_response():
    if pipeline is None:
        return JSONResponse(
            {"ok": False, "error": "legacy pipeline not initialized"},
            status_code=503,
        )

    bundle = io.BytesIO()
    state = pipeline.get_state()
    llm_context = pipeline.llm_context()
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "status.json",
            json.dumps(state, indent=2, default=str),
        )
        archive.writestr(
            "llm_context.json",
            json.dumps(llm_context, indent=2, default=str),
        )
        archive.writestr(
            "README.txt",
            "\n".join(
                [
                    "Oracle legacy-engine export",
                    "",
                    "Contents:",
                    "- status.json: current legacy dashboard payload used by /",
                    "- llm_context.json: compact legacy-engine diagnostic context for on-request LLM review",
                    "- logs/*: legacy engine logs and compact diagnostics only",
                    "- state/*: legacy engine state files and comparison-book state files",
                    "",
                    "This export intentionally excludes Opus / multiagent artifacts.",
                ]
            ),
        )

        for path, archive_name in _iter_legacy_export_files():
            try:
                archive.write(path, archive_name)
            except OSError as exc:
                archive.writestr(
                    f"errors/{Path(archive_name).name}.txt",
                    f"Failed to include {path}: {exc}",
                )

    bundle.seek(0)
    return StreamingResponse(
        bundle,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=oracle-legacy-engine-export.zip"},
    )


def _run_runtime_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Own a separate event loop for long-running trading runtimes."""
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _log_runtime_future(name: str, future) -> None:
    try:
        future.result()
        logger.info("%s background task exited cleanly", name)
    except (asyncio.CancelledError, concurrent.futures.CancelledError):
        logger.info("%s background task cancelled", name)
    except Exception:
        logger.exception("%s background task crashed", name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start pipeline on startup, stop on shutdown."""
    global pipeline, multiagent_runtime, runtime_loop, runtime_thread, pipeline_future, multiagent_future
    config = PipelineConfig()
    pipeline = Pipeline(config)
    multiagent_runtime = MultiagentRuntime(pipeline_config=config) if OPUS_RUNTIME_ENABLED else None

    runtime_loop = asyncio.new_event_loop()
    runtime_thread = threading.Thread(
        target=_run_runtime_loop,
        args=(runtime_loop,),
        name="oracle-runtime-loop",
        daemon=True,
    )
    runtime_thread.start()

    pipeline_future = asyncio.run_coroutine_threadsafe(pipeline.start(), runtime_loop)
    pipeline_future.add_done_callback(lambda future: _log_runtime_future("Pipeline", future))
    logger.info("Pipeline background task started on dedicated runtime loop")
    if multiagent_runtime is not None:
        multiagent_future = asyncio.run_coroutine_threadsafe(multiagent_runtime.start(), runtime_loop)
        multiagent_future.add_done_callback(lambda future: _log_runtime_future("Multi-agent runtime", future))
        logger.info("Multi-agent runtime background task started on dedicated runtime loop")
    else:
        multiagent_future = None
        logger.info("Multi-agent runtime disabled by OPUS_RUNTIME_ENABLED=0")

    yield

    # Shutdown
    stop_futures = []
    if pipeline and runtime_loop:
        stop_futures.append(asyncio.run_coroutine_threadsafe(pipeline.stop(), runtime_loop))
    if multiagent_runtime and runtime_loop:
        stop_futures.append(asyncio.run_coroutine_threadsafe(multiagent_runtime.stop(), runtime_loop))

    for future in stop_futures:
        try:
            future.result(timeout=15)
        except Exception:
            logger.exception("Runtime shutdown future failed")

    if pipeline_future:
        pipeline_future.cancel()
    if multiagent_future:
        multiagent_future.cancel()

    if runtime_loop:
        runtime_loop.call_soon_threadsafe(runtime_loop.stop)
    if runtime_thread:
        runtime_thread.join(timeout=10)
    if runtime_loop:
        runtime_loop.close()

    pipeline_future = None
    multiagent_future = None
    runtime_loop = None
    runtime_thread = None


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
app.add_middleware(GZipMiddleware, minimum_size=1024)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/state")
async def get_state():
    """Get full dashboard state."""
    if pipeline is None:
        return {"mode":"paper","uptime_human":"starting...","scan_count":0,"active_markets":0,"portfolio":{"total_value":0,"cash":0,"positions_value":0,"total_pnl":0,"total_pnl_pct":0,"total_trades":0,"win_rate":0,"max_drawdown":0,"total_fees":0,"positions":[]},"signals":[],"trades":[],"strategies":{},"whale_wallets":[],"recent_news":[],"performance":{"by_strategy":{}},"errors":["Initializing..."],"markets_sample":[]}
    try:
        return Response(content=pipeline.get_state_json(), media_type="application/json")
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
    """Legacy-only log export kept for backward compatibility."""
    return _build_legacy_export_response()


@app.get("/api/legacy/logs/export")
async def legacy_logs_export():
    """Export compact legacy-engine logs, diagnostics, and state without mixing Opus data."""
    return _build_legacy_export_response()


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
                    "state": "disabled",
                    "next_step": "set_OPUS_RUNTIME_ENABLED=1_to_resume",
                },
                "summary": {
                    "scan_count": 0,
                    "active_markets": 0,
                    "open_positions": 0,
                    "top_blocker": "Multi-agent runtime is disabled",
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
                "error": "multiagent runtime disabled",
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
                "answer": "The isolated Opus runtime is disabled.",
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
        history=payload.history or [],
    )
    return JSONResponse(result)


@app.post("/api/legacy/consult")
async def legacy_consult(payload: MultiagentConsultRequest):
    """Consult an LLM against compact legacy-engine diagnostics only."""
    if pipeline is None:
        return JSONResponse(
            {
                "ok": False,
                "answer": "Legacy engine is still starting.",
                "provider": None,
                "model": None,
            },
            status_code=503,
        )

    question = (payload.question or "").strip()
    if not question:
        return JSONResponse(
            {
                "ok": False,
                "answer": "Ask a non-empty diagnostic question.",
                "provider": None,
                "model": None,
            },
            status_code=400,
        )

    result = await consult_legacy_logs(
        question=question,
        context=pipeline.llm_context(),
        preferred_provider=payload.provider,
        history=payload.history or [],
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
