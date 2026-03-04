#!/usr/bin/env python3
"""
Oracle Trader Patch Script
==========================
Applies all new features to existing codebase:
1. Self-calibrating slippage model
2. A/B testing engine
3. Health monitor + bug logging
4. Log download endpoint
5. Dashboard updates (health panel, A/B tab, download button)
"""

import os
import sys

PROJECT = os.path.dirname(os.path.abspath(__file__))
if PROJECT.endswith("engine"):
    PROJECT = os.path.dirname(PROJECT)

def patch_file(filepath, old, new, label=""):
    """Replace old text with new text in a file."""
    full = os.path.join(PROJECT, filepath)
    with open(full) as f:
        text = f.read()
    if old not in text:
        print(f"  SKIP ({label}): pattern not found in {filepath}")
        return False
    text = text.replace(old, new)
    with open(full, "w") as f:
        f.write(text)
    print(f"  OK ({label}): {filepath}")
    return True


def append_to_file(filepath, content, label=""):
    """Append content to a file."""
    full = os.path.join(PROJECT, filepath)
    with open(full, "a") as f:
        f.write(content)
    print(f"  OK ({label}): appended to {filepath}")


def main():
    print("=" * 60)
    print("ORACLE TRADER PATCH — Applying upgrades...")
    print("=" * 60)

    # ---------------------------------------------------------------
    # 1. Patch pipeline.py — add health monitor, slippage, A/B tester
    # ---------------------------------------------------------------
    print("\n[1/4] Patching engine/pipeline.py...")

    # Add imports
    patch_file(
        "engine/pipeline.py",
        "from engine.paper_trader import PaperTrader",
        """from engine.paper_trader import PaperTrader
from engine.slippage import SlippageModel
from engine.ab_tester import ABTester
from engine.health_monitor import HealthMonitor""",
        "imports"
    )

    # Add new instances in __init__
    patch_file(
        "engine/pipeline.py",
        """        # State
        self.dashboard_state = DashboardState(mode=self.config.mode)""",
        """        # Slippage model (self-calibrating)
        self.slippage = SlippageModel(initial_k=0.1, log_dir="logs")

        # A/B tester
        self.ab_tester = ABTester(log_dir="logs")

        # Health monitor
        self.health = HealthMonitor(log_dir="logs")

        # State
        self.dashboard_state = DashboardState(mode=self.config.mode)""",
        "new instances"
    )

    # Add health tracking to scan cycle
    patch_file(
        "engine/pipeline.py",
        """        for name, strategy in self.strategies.items():
            if not strategy.enabled:
                continue
            try:
                signals = await strategy.scan(self._markets, self._events)
                all_signals.extend(signals)
            except Exception as e:
                logger.error(f"Strategy {name} error: {e}")
                strategy._stats["errors"] += 1""",
        """        import time as _time
        for name, strategy in self.strategies.items():
            if not strategy.enabled:
                continue
            try:
                _strat_start = _time.time()
                signals = await strategy.scan(self._markets, self._events)
                _strat_dur = (_time.time() - _strat_start) * 1000
                all_signals.extend(signals)
                self.health.record_strategy_run(name, len(signals), _strat_dur)
            except Exception as e:
                logger.error(f"Strategy {name} error: {e}")
                strategy._stats["errors"] += 1
                self.health.record_strategy_error(name, str(e))""",
        "strategy health tracking"
    )

    # Add health tracking to data refresh
    patch_file(
        "engine/pipeline.py",
        """    async def _refresh_data(self):
        \"\"\"Fetch fresh market and event data.\"\"\"
        try:
            self._markets = await self.collector.get_all_active_markets()
            self._events = await self.collector.get_events(limit=100)
            self.dashboard_state.active_markets = len(self._markets)
        except Exception as e:
            logger.error(f"Data refresh failed: {e}")
            self._errors.append(f"Data refresh: {e}")""",
        """    async def _refresh_data(self):
        \"\"\"Fetch fresh market and event data.\"\"\"
        try:
            self._markets = await self.collector.get_all_active_markets()
            self._events = await self.collector.get_events(limit=100)
            self.dashboard_state.active_markets = len(self._markets)
            self.health.record_api_success("gamma")
            self.health.record_data_freshness("markets")
        except Exception as e:
            logger.error(f"Data refresh failed: {e}")
            self._errors.append(f"Data refresh: {e}")
            self.health.record_api_error("gamma", str(e))""",
        "data refresh health"
    )

    # Add scan tracking
    patch_file(
        "engine/pipeline.py",
        """        cycle_time = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        logger.info(""",
        """        cycle_time = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        self.health.record_scan(cycle_time, len(self._markets), len(all_signals), executed)
        logger.info(""",
        "scan tracking"
    )

    # Add health, slippage, A/B to get_state()
    patch_file(
        "engine/pipeline.py",
        """            \"markets_sample\": [""",
        """            \"health\": self.health.get_health_report(),
            \"slippage\": self.slippage.get_stats(),
            \"ab_tests\": self.ab_tester.get_report(),
            \"markets_sample\": [""",
        "state additions"
    )

    # ---------------------------------------------------------------
    # 2. Patch main.py — add log download + A/B + health endpoints
    # ---------------------------------------------------------------
    print("\n[2/4] Patching main.py...")

    # Add imports
    patch_file(
        "main.py",
        "from fastapi.responses import HTMLResponse, JSONResponse",
        """from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import io""",
        "streaming import"
    )

    # Add new endpoints before the health endpoint
    patch_file(
        "main.py",
        """@app.get("/api/health")""",
        """@app.get("/api/logs/download")
async def download_logs():
    \"\"\"Download all logs as a JSON bundle.\"\"\"
    import json as _json
    from pathlib import Path as _Path
    log_dir = _Path("logs")
    bundle = {}
    for log_file in log_dir.glob("*.jsonl"):
        lines = []
        try:
            for line in log_file.read_text().strip().split("\\n"):
                if line:
                    try:
                        lines.append(_json.loads(line))
                    except _json.JSONDecodeError:
                        lines.append({"raw": line})
        except Exception:
            pass
        bundle[log_file.stem] = lines
    # Add current state
    if pipeline:
        try:
            bundle["current_state"] = pipeline.get_state()
        except Exception:
            pass
    content = _json.dumps(bundle, indent=2, default=str)
    return StreamingResponse(
        io.BytesIO(content.encode()),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=oracle-trader-logs.json"}
    )


@app.get("/api/health/detail")
async def health_detail():
    \"\"\"Get detailed health report.\"\"\"
    if pipeline is None:
        return {"overall_status": "unknown", "apis": {}, "strategies": {}}
    try:
        return pipeline.health.get_health_report()
    except Exception as e:
        return {"overall_status": "error", "error": str(e)}


@app.get("/api/ab-report")
async def ab_report():
    \"\"\"Get A/B testing report.\"\"\"
    if pipeline is None:
        return {}
    try:
        return pipeline.ab_tester.get_report()
    except Exception:
        return {}


@app.get("/api/slippage")
async def slippage_stats():
    \"\"\"Get slippage model calibration stats.\"\"\"
    if pipeline is None:
        return {}
    try:
        return pipeline.slippage.get_stats()
    except Exception:
        return {}


@app.get("/api/health")""",
        "new endpoints"
    )

    # ---------------------------------------------------------------
    # 3. Patch dashboard HTML — add health panel, download, A/B
    # ---------------------------------------------------------------
    print("\n[3/4] Patching dashboard/index.html...")

    # Add download button to header
    patch_file(
        "dashboard/index.html",
        """<span id="last-update" class="dot-pulse">Connecting...</span>""",
        """<a href="/api/logs/download" style="color:var(--blue);text-decoration:none;border:1px solid var(--blue);padding:2px 8px;border-radius:3px;font-size:11px">⬇ LOGS</a>
    <span id="last-update" class="dot-pulse">Connecting...</span>""",
        "download button"
    )

    # Add health panel and A/B section before closing </div> of grid
    patch_file(
        "dashboard/index.html",
        """  <!-- Errors / Logs -->
  <div class="card">
    <div class="card-header">
      <span class="card-title">System Log</span>
    </div>
    <div class="scrollable" id="error-log" style="font-family: var(--mono); font-size: 11px; color: var(--text-dim);">
      <div>Initializing pipeline...</div>
    </div>
  </div>

</div>""",
        """  <!-- Errors / Logs -->
  <div class="card">
    <div class="card-header">
      <span class="card-title">System Log</span>
    </div>
    <div class="scrollable" id="error-log" style="font-family: var(--mono); font-size: 11px; color: var(--text-dim);">
      <div>Initializing pipeline...</div>
    </div>
  </div>

  <!-- System Health -->
  <div class="card">
    <div class="card-header">
      <span class="card-title">System Health</span>
      <span class="card-badge" id="health-badge">—</span>
    </div>
    <div class="scrollable" id="health-panel" style="font-family: var(--mono); font-size: 11px;">
      <div style="color:var(--text-dim)">Loading health data...</div>
    </div>
  </div>

  <!-- Slippage Calibration -->
  <div class="card">
    <div class="card-header">
      <span class="card-title">Slippage Model</span>
    </div>
    <div id="slippage-panel" style="font-family: var(--mono); font-size: 12px;">
      <div style="color:var(--text-dim)">Calibrating...</div>
    </div>
  </div>

  <!-- A/B Tests -->
  <div class="card card-wide">
    <div class="card-header">
      <span class="card-title">A/B Tests</span>
    </div>
    <div class="scrollable" id="ab-panel">
      <div style="color:var(--text-dim);padding:20px;text-align:center;font-family:var(--mono);font-size:12px">No A/B tests configured yet</div>
    </div>
  </div>

</div>""",
        "new dashboard panels"
    )

    # Add render logic for new panels
    patch_file(
        "dashboard/index.html",
        """// Poll every 5 seconds
fetchState();
setInterval(fetchState, 5000);""",
        """function renderHealth(d) {
  const h = d.health;
  if (!h) return;

  // Badge
  const badge = document.getElementById('health-badge');
  const statusColors = {healthy:'var(--green)',degraded:'var(--yellow)',unhealthy:'var(--red)',unknown:'var(--text-dim)'};
  badge.textContent = (h.overall_status || 'unknown').toUpperCase();
  badge.style.background = (statusColors[h.overall_status] || 'var(--text-dim)') + '22';
  badge.style.color = statusColors[h.overall_status] || 'var(--text-dim)';

  // API status
  const apis = h.apis || {};
  let html = '<div style="margin-bottom:8px;font-weight:600;color:var(--text)">APIs</div>';
  for (const [name, info] of Object.entries(apis)) {
    const color = statusColors[info.status] || 'var(--text-dim)';
    const dot = '<span style="color:' + color + '">●</span>';
    html += '<div style="padding:2px 0">' + dot + ' ' + name.toUpperCase() +
      ' <span style="color:var(--text-dim)">ok:' + (info.success_count||0) + ' err:' + (info.error_count||0) + '</span>';
    if (info.last_error) html += '<div style="color:var(--red);font-size:10px;margin-left:14px">' + info.last_error.slice(0,60) + '</div>';
    html += '</div>';
  }

  // Strategy status
  const strats = h.strategies || {};
  if (Object.keys(strats).length > 0) {
    html += '<div style="margin:8px 0 4px;font-weight:600;color:var(--text)">Strategies</div>';
    for (const [name, info] of Object.entries(strats)) {
      const color = statusColors[info.status] || 'var(--text-dim)';
      const dot = '<span style="color:' + color + '">●</span>';
      html += '<div style="padding:2px 0">' + dot + ' ' + name +
        ' <span style="color:var(--text-dim)">runs:' + (info.runs||0) + ' err:' + (info.errors||0) + ' sig:' + (info.total_signals||0) + '</span></div>';
    }
  }

  // System
  const sys = h.system || {};
  if (sys.memory_mb) {
    html += '<div style="margin:8px 0 4px;font-weight:600;color:var(--text)">System</div>';
    html += '<div style="padding:2px 0;color:var(--text-dim)">Memory: ' + sys.memory_mb + 'MB | CPU: ' + sys.cpu_percent + '% | Threads: ' + sys.threads + '</div>';
  }

  document.getElementById('health-panel').innerHTML = html;
}

function renderSlippage(d) {
  const s = d.slippage;
  if (!s) return;
  document.getElementById('slippage-panel').innerHTML =
    '<div style="font-family:var(--mono)">'+
    '<div style="font-size:20px;font-weight:700;color:var(--blue)">k = ' + (s.k || 0.1).toFixed(5) + '</div>' +
    '<div style="color:var(--text-dim);margin-top:4px">EMA Error: ' + (s.ema_error || 0).toFixed(5) + '</div>' +
    '<div style="color:var(--text-dim)">Observations: ' + (s.calibration_count || 0) + '</div>' +
    '</div>';
}

function renderAB(d) {
  const tests = d.ab_tests;
  if (!tests || Object.keys(tests).length === 0) return;

  let html = '<table><thead><tr><th>Test</th><th>Variant A</th><th>Variant B</th><th>Leader</th></tr></thead><tbody>';
  for (const [name, test] of Object.entries(tests)) {
    const a = test.variant_a || {};
    const b = test.variant_b || {};
    const leaderColor = test.leader === a.name ? 'var(--green)' : test.leader === b.name ? 'var(--green)' : 'var(--text-dim)';
    html += '<tr>' +
      '<td>' + name + '</td>' +
      '<td>' + (a.name||'A') + ': ' + (a.total_trades||0) + ' trades, $' + (a.total_pnl||0).toFixed(2) + ' (' + (a.win_rate||0) + '%)</td>' +
      '<td>' + (b.name||'B') + ': ' + (b.total_trades||0) + ' trades, $' + (b.total_pnl||0).toFixed(2) + ' (' + (b.win_rate||0) + '%)</td>' +
      '<td style="color:' + leaderColor + '">' + (test.leader||'—') + '</td>' +
      '</tr>';
  }
  html += '</tbody></table>';
  document.getElementById('ab-panel').innerHTML = html;
}

// Poll every 5 seconds
fetchState();
setInterval(fetchState, 5000);""",
        "render functions"
    )

    # Call new render functions from main render()
    patch_file(
        "dashboard/index.html",
        """  // Errors
  const errors = d.errors || [];""",
        """  // Health, Slippage, A/B
  renderHealth(d);
  renderSlippage(d);
  renderAB(d);

  // Errors
  const errors = d.errors || [];""",
        "render calls"
    )

    # ---------------------------------------------------------------
    # 4. Add psutil to requirements
    # ---------------------------------------------------------------
    print("\n[4/4] Updating requirements.txt...")

    req_path = os.path.join(PROJECT, "requirements.txt")
    with open(req_path) as f:
        reqs = f.read()
    if "psutil" not in reqs:
        with open(req_path, "a") as f:
            f.write("psutil==5.9.8\n")
        print("  OK: added psutil to requirements.txt")
    else:
        print("  SKIP: psutil already in requirements.txt")

    print("\n" + "=" * 60)
    print("PATCH COMPLETE!")
    print("=" * 60)
    print("\nNew files to copy:")
    print("  engine/slippage.py")
    print("  engine/ab_tester.py")
    print("  engine/health_monitor.py")
    print("\nTo deploy:")
    print("  git add . && git commit -m 'feat: slippage model, A/B testing, health monitor, log download'")
    print("  git push && railway up")


if __name__ == "__main__":
    main()
