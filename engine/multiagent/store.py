from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .contracts import ExecutionResult, PortfolioSnapshot, ScanCycleReport


class RuntimeMetricsStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def record_cycle(
        self,
        *,
        scan_id: int,
        report: ScanCycleReport,
        portfolio: PortfolioSnapshot,
        closed_events: list[dict[str, Any]],
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cycles (
                    scan_id, cycle_id, started_at, completed_at, duration_seconds, success,
                    markets_scanned, markets_after_filter, candidates_generated, candidates_validated,
                    candidates_rejected, intents_created, allocation_rejections,
                    executions_attempted, executions_succeeded, executions_failed,
                    capital_total, capital_available, capital_deployed, open_positions,
                    unrealized_pnl, realized_pnl, top_rejection_reason,
                    top_allocation_rejection_reason, zero_trade_explanation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scan_id,
                    report.cycle_id,
                    report.started_at.isoformat(),
                    report.completed_at.isoformat() if report.completed_at else None,
                    report.duration_seconds,
                    1 if report.success else 0,
                    report.markets_scanned,
                    report.markets_after_filter,
                    report.candidates_generated,
                    report.candidates_validated,
                    report.candidates_rejected,
                    report.intents_created,
                    report.allocation_rejections,
                    report.executions_attempted,
                    report.executions_succeeded,
                    report.executions_failed,
                    portfolio.total_capital,
                    portfolio.available_capital,
                    portfolio.deployed_capital,
                    portfolio.position_count,
                    portfolio.total_unrealized_pnl,
                    portfolio.total_realized_pnl,
                    report.top_rejection_reason,
                    report.top_allocation_rejection_reason,
                    report.zero_trade_explanation,
                ),
            )

            conn.execute("DELETE FROM fills WHERE scan_id = ?", (scan_id,))
            for result in report.execution_results_detail:
                conn.execute(
                    """
                    INSERT INTO fills (
                        scan_id, timestamp, market_id, market_question, strategy_name, direction, outcome,
                        fill_price, shares, executed, error, edge_estimate, rationale
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    _execution_row(scan_id, result),
                )

            conn.execute("DELETE FROM strategy_counts WHERE scan_id = ?", (scan_id,))
            for strategy_name, count in report.candidates_per_strategy.items():
                conn.execute(
                    "INSERT INTO strategy_counts (scan_id, strategy_name, candidates) VALUES (?, ?, ?)",
                    (scan_id, strategy_name, count),
                )

            conn.execute("DELETE FROM rejection_counts WHERE scan_id = ?", (scan_id,))
            for reason, count in report.rejection_reasons.items():
                conn.execute(
                    "INSERT INTO rejection_counts (scan_id, reason, count) VALUES (?, ?, ?)",
                    (scan_id, reason, count),
                )
            for reason, count in report.allocation_rejection_reasons.items():
                conn.execute(
                    "INSERT INTO rejection_counts (scan_id, reason, count) VALUES (?, ?, ?)",
                    (scan_id, f"alloc:{reason}", count),
                )

            for event in closed_events:
                conn.execute(
                    """
                    INSERT INTO position_closes (
                        scan_id, closed_at, market_id, market_question, strategy_name,
                        entry_price, exit_price, shares, realized_pnl, hold_hours, close_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        scan_id,
                        event["closed_at"],
                        event["market_id"],
                        event["market_question"],
                        event["strategy_name"],
                        event["entry_price"],
                        event["exit_price"],
                        event["shares"],
                        event["realized_pnl"],
                        event["hold_hours"],
                        event["close_reason"],
                    ),
                )

            conn.commit()

    def llm_summary(self, *, recent_scans: int = 24, recent_closes: int = 20) -> dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            recent_scan_rows = conn.execute(
                """
                SELECT scan_id
                FROM cycles
                ORDER BY scan_id DESC
                LIMIT ?
                """,
                (recent_scans,),
            ).fetchall()
            recent_scan_ids = [int(row["scan_id"]) for row in recent_scan_rows]
            if recent_scan_ids:
                placeholders = ",".join("?" for _ in recent_scan_ids)
            else:
                placeholders = "NULL"

            cycles = conn.execute(
                """
                SELECT scan_id, completed_at, markets_scanned, markets_after_filter,
                       candidates_generated, candidates_validated, executions_succeeded,
                       executions_failed, capital_total, capital_available, open_positions,
                       unrealized_pnl, realized_pnl, top_rejection_reason,
                       top_allocation_rejection_reason, zero_trade_explanation
                FROM cycles
                ORDER BY scan_id DESC
                LIMIT ?
                """,
                (recent_scans,),
            ).fetchall()

            strategy_rollup = []
            rejection_rollup = []
            closes = []
            fills = []
            window_summary = {
                "scan_ids": recent_scan_ids,
                "scan_count": len(recent_scan_ids),
                "first_scan_id": min(recent_scan_ids) if recent_scan_ids else None,
                "last_scan_id": max(recent_scan_ids) if recent_scan_ids else None,
            }

            if recent_scan_ids:
                strategy_rollup = conn.execute(
                    f"""
                    SELECT strategy_name, SUM(candidates) AS total_candidates
                    FROM strategy_counts
                    WHERE scan_id IN ({placeholders})
                    GROUP BY strategy_name
                    ORDER BY total_candidates DESC
                    LIMIT 12
                    """,
                    tuple(recent_scan_ids),
                ).fetchall()

                rejection_rollup = conn.execute(
                    f"""
                    SELECT reason, SUM(count) AS total
                    FROM rejection_counts
                    WHERE scan_id IN ({placeholders})
                    GROUP BY reason
                    ORDER BY total DESC
                    LIMIT 12
                    """,
                    tuple(recent_scan_ids),
                ).fetchall()

                closes = conn.execute(
                    f"""
                    SELECT closed_at, market_question, strategy_name, realized_pnl, hold_hours, close_reason
                    FROM position_closes
                    WHERE scan_id IN ({placeholders})
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (*recent_scan_ids, recent_closes),
                ).fetchall()

                fills = conn.execute(
                    f"""
                    SELECT timestamp, market_id, market_question, strategy_name, direction, outcome,
                           executed, fill_price, shares, edge_estimate, rationale
                    FROM fills
                    WHERE scan_id IN ({placeholders})
                    ORDER BY id DESC
                    LIMIT 60
                    """,
                    tuple(recent_scan_ids),
                ).fetchall()

        return {
            "window": window_summary,
            "recent_cycles": [dict(row) for row in cycles],
            "strategy_rollup": [dict(row) for row in strategy_rollup],
            "rejection_rollup": [dict(row) for row in rejection_rollup],
            "recent_closes": [dict(row) for row in closes],
            "recent_fills": [dict(row) for row in fills],
        }

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS cycles (
                    scan_id INTEGER PRIMARY KEY,
                    cycle_id TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    duration_seconds REAL,
                    success INTEGER,
                    markets_scanned INTEGER,
                    markets_after_filter INTEGER,
                    candidates_generated INTEGER,
                    candidates_validated INTEGER,
                    candidates_rejected INTEGER,
                    intents_created INTEGER,
                    allocation_rejections INTEGER,
                    executions_attempted INTEGER,
                    executions_succeeded INTEGER,
                    executions_failed INTEGER,
                    capital_total REAL,
                    capital_available REAL,
                    capital_deployed REAL,
                    open_positions INTEGER,
                    unrealized_pnl REAL,
                    realized_pnl REAL,
                    top_rejection_reason TEXT,
                    top_allocation_rejection_reason TEXT,
                    zero_trade_explanation TEXT
                );

                CREATE TABLE IF NOT EXISTS fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id INTEGER,
                    timestamp TEXT,
                    market_id TEXT,
                    market_question TEXT,
                    strategy_name TEXT,
                    direction TEXT,
                    outcome TEXT,
                    fill_price REAL,
                    shares REAL,
                    executed INTEGER,
                    error TEXT,
                    edge_estimate REAL,
                    rationale TEXT
                );

                CREATE TABLE IF NOT EXISTS strategy_counts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id INTEGER,
                    strategy_name TEXT,
                    candidates INTEGER
                );

                CREATE TABLE IF NOT EXISTS rejection_counts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id INTEGER,
                    reason TEXT,
                    count INTEGER
                );

                CREATE TABLE IF NOT EXISTS position_closes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id INTEGER,
                    closed_at TEXT,
                    market_id TEXT,
                    market_question TEXT,
                    strategy_name TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    shares REAL,
                    realized_pnl REAL,
                    hold_hours REAL,
                    close_reason TEXT
                );
                """
            )
            fill_columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(fills)").fetchall()
            }
            if "market_question" not in fill_columns:
                conn.execute("ALTER TABLE fills ADD COLUMN market_question TEXT")
            if "direction" not in fill_columns:
                conn.execute("ALTER TABLE fills ADD COLUMN direction TEXT")
            conn.commit()


def _execution_row(scan_id: int, result: ExecutionResult) -> tuple[Any, ...]:
    signal = result.intent.signal.signal if result.intent and result.intent.signal else None
    return (
        scan_id,
        result.executed_at.isoformat(),
        signal.market_id if signal else "",
        signal.market_snapshot.question if signal and signal.market_snapshot else "",
        signal.strategy_name if signal else "",
        signal.direction.value if signal else "",
        signal.outcome if signal else "",
        result.fill_price,
        result.shares_filled,
        1 if result.executed else 0,
        result.error,
        signal.edge_estimate if signal else None,
        signal.reasoning if signal else None,
    )
