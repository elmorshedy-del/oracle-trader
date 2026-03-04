"""
Health Monitor
==============
Tracks system health, API connectivity, errors, and data freshness.
Logs everything to health.jsonl for debugging.
Provides dashboard-friendly status indicators.
"""

import json
import time
import logging
import traceback
import psutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    HEALTHY = "healthy"      # green
    DEGRADED = "degraded"    # yellow
    UNHEALTHY = "unhealthy"  # red
    UNKNOWN = "unknown"      # gray


class HealthMonitor:
    """
    Centralized health tracking for the trading pipeline.

    Monitors:
    - API connectivity (Gamma, CLOB, Data, Anthropic)
    - Data freshness (when was last successful fetch?)
    - Strategy errors
    - System resources (memory, CPU)
    - Rate limit proximity
    - Connection timeouts
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.started_at = datetime.now(timezone.utc)

        # API health tracking
        self._api_status: dict[str, dict] = {
            "gamma": {"status": HealthStatus.UNKNOWN, "last_success": None, "last_error": None, "error_count": 0, "success_count": 0},
            "clob": {"status": HealthStatus.UNKNOWN, "last_success": None, "last_error": None, "error_count": 0, "success_count": 0},
            "data": {"status": HealthStatus.UNKNOWN, "last_success": None, "last_error": None, "error_count": 0, "success_count": 0},
            "anthropic": {"status": HealthStatus.UNKNOWN, "last_success": None, "last_error": None, "error_count": 0, "success_count": 0},
        }

        # Strategy health
        self._strategy_status: dict[str, dict] = {}

        # Error log (rolling buffer)
        self._errors: list[dict] = []
        self._max_errors = 500

        # Heartbeat tracking
        self._last_scan: datetime | None = None
        self._scan_times: list[float] = []  # last 100 scan durations

        # Rate limit tracking
        self._api_calls: dict[str, list[float]] = defaultdict(list)  # api -> [timestamps]

        # Data freshness
        self._data_timestamps: dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # API Health
    # ------------------------------------------------------------------

    def record_api_success(self, api_name: str, response_time_ms: float = 0):
        """Record a successful API call."""
        if api_name not in self._api_status:
            self._api_status[api_name] = {
                "status": HealthStatus.UNKNOWN, "last_success": None,
                "last_error": None, "error_count": 0, "success_count": 0,
            }

        entry = self._api_status[api_name]
        entry["last_success"] = datetime.now(timezone.utc).isoformat()
        entry["success_count"] += 1
        entry["response_time_ms"] = response_time_ms
        entry["status"] = HealthStatus.HEALTHY

        # Track rate
        self._api_calls[api_name].append(time.time())
        # Keep only last hour
        cutoff = time.time() - 3600
        self._api_calls[api_name] = [t for t in self._api_calls[api_name] if t > cutoff]

    def record_api_error(self, api_name: str, error: str, status_code: int = 0):
        """Record a failed API call."""
        if api_name not in self._api_status:
            self._api_status[api_name] = {
                "status": HealthStatus.UNKNOWN, "last_success": None,
                "last_error": None, "error_count": 0, "success_count": 0,
            }

        entry = self._api_status[api_name]
        entry["last_error"] = datetime.now(timezone.utc).isoformat()
        entry["last_error_msg"] = error[:200]
        entry["last_status_code"] = status_code
        entry["error_count"] += 1

        # Determine status based on recent errors
        if entry["error_count"] > 10 and entry["success_count"] == 0:
            entry["status"] = HealthStatus.UNHEALTHY
        elif entry["error_count"] > 3:
            entry["status"] = HealthStatus.DEGRADED
        else:
            entry["status"] = HealthStatus.HEALTHY

        self._log_error("api_error", {
            "api": api_name,
            "error": error[:200],
            "status_code": status_code,
            "total_errors": entry["error_count"],
        })

    # ------------------------------------------------------------------
    # Strategy Health
    # ------------------------------------------------------------------

    def record_strategy_run(self, strategy_name: str, signals: int, duration_ms: float):
        """Record a strategy scan result."""
        if strategy_name not in self._strategy_status:
            self._strategy_status[strategy_name] = {
                "status": HealthStatus.HEALTHY,
                "runs": 0, "errors": 0, "total_signals": 0,
                "last_run": None, "last_error": None,
            }

        entry = self._strategy_status[strategy_name]
        entry["runs"] += 1
        entry["total_signals"] += signals
        entry["last_run"] = datetime.now(timezone.utc).isoformat()
        entry["last_duration_ms"] = round(duration_ms, 1)
        entry["status"] = HealthStatus.HEALTHY

    def record_strategy_error(self, strategy_name: str, error: str):
        """Record a strategy error."""
        if strategy_name not in self._strategy_status:
            self._strategy_status[strategy_name] = {
                "status": HealthStatus.UNKNOWN,
                "runs": 0, "errors": 0, "total_signals": 0,
                "last_run": None, "last_error": None,
            }

        entry = self._strategy_status[strategy_name]
        entry["errors"] += 1
        entry["last_error"] = error[:200]
        entry["last_error_at"] = datetime.now(timezone.utc).isoformat()

        if entry["errors"] > 5 and entry["runs"] > 0:
            error_rate = entry["errors"] / entry["runs"]
            if error_rate > 0.5:
                entry["status"] = HealthStatus.UNHEALTHY
            elif error_rate > 0.2:
                entry["status"] = HealthStatus.DEGRADED

        self._log_error("strategy_error", {
            "strategy": strategy_name,
            "error": error[:200],
            "total_errors": entry["errors"],
        })

    # ------------------------------------------------------------------
    # Scan Health
    # ------------------------------------------------------------------

    def record_scan(self, duration_secs: float, markets: int, signals: int, trades: int):
        """Record a completed scan cycle."""
        self._last_scan = datetime.now(timezone.utc)
        self._scan_times.append(duration_secs)
        if len(self._scan_times) > 100:
            self._scan_times = self._scan_times[-100:]

    def record_data_freshness(self, data_type: str):
        """Record when a data type was last refreshed."""
        self._data_timestamps[data_type] = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # General Error Logging
    # ------------------------------------------------------------------

    def record_error(self, category: str, message: str, details: dict = None):
        """Record any error with context."""
        self._log_error(category, {
            "message": message[:500],
            **(details or {}),
        })

    def record_exception(self, category: str, exc: Exception):
        """Record a full exception with traceback."""
        tb = traceback.format_exc()
        self._log_error(category, {
            "exception": type(exc).__name__,
            "message": str(exc)[:500],
            "traceback": tb[:1000],
        })

    def _log_error(self, category: str, data: dict):
        """Add to error buffer and persist to file."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "category": category,
            **data,
        }
        self._errors.append(entry)
        if len(self._errors) > self._max_errors:
            self._errors = self._errors[-self._max_errors // 2:]

        try:
            with open(self.log_dir / "health.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass  # can't log logging errors

    # ------------------------------------------------------------------
    # System Resources
    # ------------------------------------------------------------------

    def _get_system_stats(self) -> dict:
        """Get current system resource usage."""
        try:
            process = psutil.Process()
            return {
                "memory_mb": round(process.memory_info().rss / 1024 / 1024, 1),
                "cpu_percent": round(process.cpu_percent(), 1),
                "threads": process.num_threads(),
            }
        except Exception:
            return {"memory_mb": 0, "cpu_percent": 0, "threads": 0}

    # ------------------------------------------------------------------
    # Dashboard Report
    # ------------------------------------------------------------------

    def get_health_report(self) -> dict:
        """Full health report for the dashboard."""
        now = datetime.now(timezone.utc)

        # Overall status — worst of all components
        all_statuses = [s["status"] for s in self._api_status.values()]
        all_statuses += [s["status"] for s in self._strategy_status.values()]

        if HealthStatus.UNHEALTHY in all_statuses:
            overall = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in all_statuses:
            overall = HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in all_statuses if s != HealthStatus.UNKNOWN):
            overall = HealthStatus.HEALTHY
        else:
            overall = HealthStatus.UNKNOWN

        # Data freshness check
        stale_data = []
        for dtype, ts in self._data_timestamps.items():
            age_mins = (now - ts).total_seconds() / 60
            if age_mins > 5:
                stale_data.append({"type": dtype, "age_minutes": round(age_mins, 1)})

        # Scan health
        scan_status = HealthStatus.HEALTHY
        if self._last_scan:
            since_last = (now - self._last_scan).total_seconds()
            if since_last > 120:
                scan_status = HealthStatus.UNHEALTHY
            elif since_last > 60:
                scan_status = HealthStatus.DEGRADED

        avg_scan_time = (
            sum(self._scan_times) / len(self._scan_times)
            if self._scan_times else 0
        )

        # Rate limit proximity
        rate_limits = {}
        for api, timestamps in self._api_calls.items():
            calls_per_hour = len(timestamps)
            rate_limits[api] = {
                "calls_last_hour": calls_per_hour,
                "warning": calls_per_hour > 800,
            }

        return {
            "overall_status": overall.value,
            "uptime_seconds": int((now - self.started_at).total_seconds()),
            "apis": {
                name: {
                    "status": info["status"].value if isinstance(info["status"], HealthStatus) else info["status"],
                    "last_success": info["last_success"],
                    "last_error": info.get("last_error_msg"),
                    "error_count": info["error_count"],
                    "success_count": info["success_count"],
                }
                for name, info in self._api_status.items()
            },
            "strategies": {
                name: {
                    "status": info["status"].value if isinstance(info["status"], HealthStatus) else info["status"],
                    "runs": info["runs"],
                    "errors": info["errors"],
                    "total_signals": info["total_signals"],
                    "last_error": info.get("last_error"),
                }
                for name, info in self._strategy_status.items()
            },
            "scan": {
                "status": scan_status.value,
                "last_scan": self._last_scan.isoformat() if self._last_scan else None,
                "avg_duration_secs": round(avg_scan_time, 2),
            },
            "stale_data": stale_data,
            "rate_limits": rate_limits,
            "system": self._get_system_stats(),
            "recent_errors": self._errors[-20:],
            "total_errors": len(self._errors),
        }
