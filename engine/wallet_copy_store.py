from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any


class WalletCopyResearchStore:
    def __init__(
        self,
        db_path: Path,
        *,
        schema_version: str,
        collector_version: str,
        target_labeled_buys: int,
    ) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.schema_version = schema_version
        self.collector_version = collector_version
        self.target_labeled_buys = target_labeled_buys
        self._lock = threading.Lock()
        self._init_db()
        self.set_meta("schema_version", schema_version)
        self.set_meta("collector_version", collector_version)
        if self.get_meta("collection_started_at") in {None, "null"}:
            self.set_meta("collection_started_at", None)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS wallet_copy_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tracked_wallets (
                    address TEXT PRIMARY KEY,
                    leaderboard_rank_all INTEGER,
                    leaderboard_rank_30d INTEGER,
                    leaderboard_rank_30d_volume INTEGER,
                    leaderboard_profit REAL,
                    leaderboard_volume REAL,
                    leaderboard_num_trades INTEGER,
                    leaderboard_win_rate REAL,
                    leaderboard_source_flags TEXT,
                    raw_leaderboard_json TEXT,
                    last_refreshed REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tracked_wallet_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_at REAL NOT NULL,
                    address TEXT NOT NULL,
                    leaderboard_rank_all INTEGER,
                    leaderboard_rank_30d INTEGER,
                    leaderboard_rank_30d_volume INTEGER,
                    leaderboard_profit REAL,
                    leaderboard_volume REAL,
                    leaderboard_num_trades INTEGER,
                    leaderboard_win_rate REAL,
                    leaderboard_source_flags TEXT,
                    raw_leaderboard_json TEXT,
                    UNIQUE(snapshot_at, address)
                );

                CREATE TABLE IF NOT EXISTS wallet_position_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    wallet_address TEXT NOT NULL,
                    snapshot_at REAL NOT NULL,
                    open_positions INTEGER,
                    held_condition_ids_json TEXT,
                    held_asset_ids_json TEXT,
                    raw_positions_json TEXT,
                    UNIQUE(wallet_address, snapshot_at)
                );

                CREATE TABLE IF NOT EXISTS wallet_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_key TEXT UNIQUE NOT NULL,
                    tx_hash TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    detected_at REAL NOT NULL,
                    snapshot_taken_at REAL NOT NULL,
                    wallet_address TEXT NOT NULL,
                    market_condition_id TEXT NOT NULL,
                    asset_token_id TEXT NOT NULL,
                    market_title TEXT,
                    market_slug TEXT,
                    side TEXT,
                    outcome TEXT,
                    price REAL,
                    size_shares REAL,
                    size_usd REAL,
                    token_best_bid REAL,
                    token_best_ask REAL,
                    token_best_bid_size REAL,
                    token_best_ask_size REAL,
                    token_depth_within_2pct REAL,
                    market_yes_bid REAL,
                    market_yes_ask REAL,
                    market_no_bid REAL,
                    market_no_ask REAL,
                    market_spread REAL,
                    market_volume_24h REAL,
                    market_volume_total REAL,
                    market_liquidity REAL,
                    market_midpoint REAL,
                    market_reward_pool REAL,
                    market_primary_tag TEXT,
                    market_tags_json TEXT,
                    market_end_date TEXT,
                    market_seconds_to_expiry REAL,
                    market_category TEXT,
                    market_category_version TEXT,
                    market_active INTEGER,
                    market_closed INTEGER,
                    wallet_leaderboard_rank INTEGER,
                    wallet_leaderboard_profit REAL,
                    wallet_leaderboard_win_rate REAL,
                    wallet_open_positions INTEGER,
                    wallet_trade_count_24h INTEGER,
                    is_adding_to_position INTEGER,
                    size_vs_wallet_avg REAL,
                    detection_delay_seconds REAL,
                    hour_of_day REAL,
                    day_of_week INTEGER,
                    btc_price REAL,
                    btc_momentum_60s REAL,
                    wallet_sell_trade_key TEXT,
                    wallet_sell_tx_hash TEXT,
                    wallet_sell_timestamp REAL,
                    wallet_sell_price REAL,
                    wallet_sell_size_shares REAL,
                    wallet_sell_is_partial INTEGER,
                    wallet_sell_return REAL,
                    wallet_sell_match_quality TEXT,
                    is_win_wallet_sell INTEGER,
                    market_resolved INTEGER DEFAULT 0,
                    resolution_timestamp REAL,
                    winning_outcome TEXT,
                    resolution_return REAL,
                    resolution_source TEXT,
                    is_win_resolution INTEGER,
                    price_1min_after REAL,
                    price_5min_after REAL,
                    price_30min_after REAL,
                    raw_activity_json TEXT,
                    raw_market_json TEXT,
                    raw_orderbook_json TEXT,
                    raw_positions_json TEXT,
                    collector_version TEXT,
                    schema_version TEXT
                );

                CREATE TABLE IF NOT EXISTS copy_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_key TEXT UNIQUE NOT NULL,
                    strategy_key TEXT NOT NULL,
                    profile_kind TEXT NOT NULL,
                    wallet_trade_id INTEGER NOT NULL,
                    wallet_trade_key TEXT NOT NULL,
                    tx_hash TEXT NOT NULL,
                    wallet_address TEXT NOT NULL,
                    market_condition_id TEXT NOT NULL,
                    asset_token_id TEXT NOT NULL,
                    market_slug TEXT,
                    source_side TEXT,
                    source_outcome TEXT,
                    score REAL NOT NULL,
                    should_copy INTEGER NOT NULL,
                    suggested_size_usd REAL,
                    reasons_json TEXT,
                    context_json TEXT,
                    paper_executed INTEGER DEFAULT 0,
                    paper_trade_id TEXT,
                    paper_side TEXT,
                    paper_entry_timestamp REAL,
                    paper_entry_price REAL,
                    paper_entry_size_usd REAL,
                    paper_exit_timestamp REAL,
                    paper_exit_price REAL,
                    paper_realized_pnl_usd REAL,
                    paper_close_reason TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY(wallet_trade_id) REFERENCES wallet_trades(id)
                );

                CREATE INDEX IF NOT EXISTS idx_wallet_trades_wallet_time
                ON wallet_trades(wallet_address, timestamp);

                CREATE INDEX IF NOT EXISTS idx_wallet_trades_condition_asset_time
                ON wallet_trades(market_condition_id, asset_token_id, timestamp);

                CREATE INDEX IF NOT EXISTS idx_wallet_trades_buy_labels
                ON wallet_trades(side, wallet_sell_timestamp, market_resolved, timestamp);

                CREATE INDEX IF NOT EXISTS idx_wallet_trades_market_slug
                ON wallet_trades(market_slug, timestamp);

                CREATE INDEX IF NOT EXISTS idx_wallet_position_snapshots_wallet_time
                ON wallet_position_snapshots(wallet_address, snapshot_at);

                CREATE INDEX IF NOT EXISTS idx_copy_decisions_strategy_created
                ON copy_decisions(strategy_key, created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_copy_decisions_trade
                ON copy_decisions(wallet_trade_id, strategy_key);
                """
            )
            conn.commit()

    @staticmethod
    def _json_value(value: Any) -> str:
        return json.dumps(value, sort_keys=True, default=str)

    @staticmethod
    def _ml_ready_buy_predicate(alias: str = "") -> str:
        prefix = f"{alias}." if alias else ""
        return (
            f"{prefix}side = 'BUY'"
            f" AND COALESCE({prefix}price, 0) > 0"
            f" AND COALESCE({prefix}size_usd, 0) > 0"
            f" AND COALESCE({prefix}market_closed, 0) = 0"
            f" AND ({prefix}market_seconds_to_expiry IS NULL OR {prefix}market_seconds_to_expiry > 0)"
        )

    @staticmethod
    def _stale_buy_predicate(alias: str = "") -> str:
        prefix = f"{alias}." if alias else ""
        return (
            f"{prefix}side = 'BUY'"
            f" AND (COALESCE({prefix}market_closed, 0) = 1"
            f" OR ({prefix}market_seconds_to_expiry IS NOT NULL AND {prefix}market_seconds_to_expiry <= 0))"
        )

    def set_meta(self, key: str, value: Any, *, updated_at: float | None = None) -> None:
        updated_at = float(updated_at if updated_at is not None else __import__("time").time())
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO wallet_copy_meta (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, self._json_value(value), updated_at),
            )
            conn.commit()

    def get_meta(self, key: str) -> Any | None:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM wallet_copy_meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return row["value"]

    def mark_heartbeat(self, component: str, timestamp: float) -> None:
        self.set_meta(f"{component}_last_heartbeat", timestamp, updated_at=timestamp)

    def replace_tracked_wallets(self, wallets: list[dict[str, Any]], *, snapshot_at: float) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM tracked_wallets")
            for wallet in wallets:
                source_flags = wallet.get("leaderboard_source_flags") or []
                raw_payload = wallet.get("raw_leaderboard_json") or {}
                conn.execute(
                    """
                    INSERT INTO tracked_wallets (
                        address, leaderboard_rank_all, leaderboard_rank_30d, leaderboard_rank_30d_volume,
                        leaderboard_profit, leaderboard_volume, leaderboard_num_trades, leaderboard_win_rate,
                        leaderboard_source_flags, raw_leaderboard_json, last_refreshed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        wallet["address"],
                        wallet.get("leaderboard_rank_all"),
                        wallet.get("leaderboard_rank_30d"),
                        wallet.get("leaderboard_rank_30d_volume"),
                        wallet.get("leaderboard_profit"),
                        wallet.get("leaderboard_volume"),
                        wallet.get("leaderboard_num_trades"),
                        wallet.get("leaderboard_win_rate"),
                        self._json_value(source_flags),
                        self._json_value(raw_payload),
                        snapshot_at,
                    ),
                )
                conn.execute(
                    """
                    INSERT OR IGNORE INTO tracked_wallet_snapshots (
                        snapshot_at, address, leaderboard_rank_all, leaderboard_rank_30d,
                        leaderboard_rank_30d_volume, leaderboard_profit, leaderboard_volume,
                        leaderboard_num_trades, leaderboard_win_rate, leaderboard_source_flags,
                        raw_leaderboard_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot_at,
                        wallet["address"],
                        wallet.get("leaderboard_rank_all"),
                        wallet.get("leaderboard_rank_30d"),
                        wallet.get("leaderboard_rank_30d_volume"),
                        wallet.get("leaderboard_profit"),
                        wallet.get("leaderboard_volume"),
                        wallet.get("leaderboard_num_trades"),
                        wallet.get("leaderboard_win_rate"),
                        self._json_value(source_flags),
                        self._json_value(raw_payload),
                    ),
                )
            conn.commit()
        self.set_meta("leaderboard_last_refreshed_at", snapshot_at, updated_at=snapshot_at)
        self.set_meta("tracked_wallet_count", len(wallets), updated_at=snapshot_at)

    def record_positions_snapshot(
        self,
        *,
        wallet_address: str,
        snapshot_at: float,
        open_positions: int,
        held_condition_ids: list[str],
        held_asset_ids: list[str],
        raw_positions: list[dict[str, Any]],
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO wallet_position_snapshots (
                    wallet_address, snapshot_at, open_positions, held_condition_ids_json,
                    held_asset_ids_json, raw_positions_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    wallet_address,
                    snapshot_at,
                    open_positions,
                    self._json_value(held_condition_ids),
                    self._json_value(held_asset_ids),
                    self._json_value(raw_positions),
                ),
            )
            conn.commit()

    def latest_position_snapshot(self, wallet_address: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT wallet_address, snapshot_at, open_positions, held_condition_ids_json, held_asset_ids_json, raw_positions_json
                FROM wallet_position_snapshots
                WHERE wallet_address = ?
                ORDER BY snapshot_at DESC
                LIMIT 1
                """,
                (wallet_address,),
            ).fetchone()
        if row is None:
            return None
        return {
            "wallet_address": row["wallet_address"],
            "snapshot_at": row["snapshot_at"],
            "open_positions": row["open_positions"],
            "held_condition_ids": json.loads(row["held_condition_ids_json"] or "[]"),
            "held_asset_ids": json.loads(row["held_asset_ids_json"] or "[]"),
            "raw_positions": json.loads(row["raw_positions_json"] or "[]"),
        }

    def trade_exists(self, trade_key: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM wallet_trades WHERE trade_key = ? LIMIT 1", (trade_key,)).fetchone()
        return row is not None

    def wallet_metrics_before(
        self,
        *,
        wallet_address: str,
        trade_timestamp: float,
        condition_id: str,
        asset_token_id: str,
        trade_size_usd: float,
        latest_positions_snapshot: dict[str, Any] | None,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            trade_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS trade_count_24h,
                    AVG(CASE WHEN size_usd > 0 THEN size_usd END) AS avg_trade_size_usd
                FROM wallet_trades
                WHERE wallet_address = ?
                  AND timestamp < ?
                  AND timestamp >= ?
                """,
                (wallet_address, trade_timestamp, trade_timestamp - 86400.0),
            ).fetchone()
            leaderboard_row = conn.execute(
                """
                SELECT leaderboard_rank_all, leaderboard_rank_30d, leaderboard_profit, leaderboard_win_rate
                FROM tracked_wallets
                WHERE address = ?
                LIMIT 1
                """,
                (wallet_address,),
            ).fetchone()
        avg_trade_size = float(trade_row["avg_trade_size_usd"] or 0.0) if trade_row else 0.0
        held_condition_ids = set((latest_positions_snapshot or {}).get("held_condition_ids") or [])
        held_asset_ids = set((latest_positions_snapshot or {}).get("held_asset_ids") or [])
        is_adding = 1 if (condition_id in held_condition_ids or asset_token_id in held_asset_ids) else 0
        leaderboard_rank = None
        leaderboard_profit = None
        leaderboard_win_rate = None
        if leaderboard_row is not None:
            leaderboard_rank = leaderboard_row["leaderboard_rank_30d"] or leaderboard_row["leaderboard_rank_all"]
            leaderboard_profit = leaderboard_row["leaderboard_profit"]
            leaderboard_win_rate = leaderboard_row["leaderboard_win_rate"]
        return {
            "wallet_leaderboard_rank": leaderboard_rank,
            "wallet_leaderboard_profit": float(leaderboard_profit or 0.0) if leaderboard_profit is not None else None,
            "wallet_leaderboard_win_rate": float(leaderboard_win_rate or 0.0) if leaderboard_win_rate is not None else None,
            "wallet_open_positions": int((latest_positions_snapshot or {}).get("open_positions") or 0),
            "wallet_trade_count_24h": int(trade_row["trade_count_24h"] or 0) if trade_row else 0,
            "is_adding_to_position": is_adding,
            "size_vs_wallet_avg": (trade_size_usd / avg_trade_size) if avg_trade_size > 0 else None,
            "raw_positions_json": self._json_value((latest_positions_snapshot or {}).get("raw_positions") or []),
        }

    def insert_wallet_trade(self, row: dict[str, Any]) -> bool:
        columns = [
            "trade_key",
            "tx_hash",
            "timestamp",
            "detected_at",
            "snapshot_taken_at",
            "wallet_address",
            "market_condition_id",
            "asset_token_id",
            "market_title",
            "market_slug",
            "side",
            "outcome",
            "price",
            "size_shares",
            "size_usd",
            "token_best_bid",
            "token_best_ask",
            "token_best_bid_size",
            "token_best_ask_size",
            "token_depth_within_2pct",
            "market_yes_bid",
            "market_yes_ask",
            "market_no_bid",
            "market_no_ask",
            "market_spread",
            "market_volume_24h",
            "market_volume_total",
            "market_liquidity",
            "market_midpoint",
            "market_reward_pool",
            "market_primary_tag",
            "market_tags_json",
            "market_end_date",
            "market_seconds_to_expiry",
            "market_category",
            "market_category_version",
            "market_active",
            "market_closed",
            "wallet_leaderboard_rank",
            "wallet_leaderboard_profit",
            "wallet_leaderboard_win_rate",
            "wallet_open_positions",
            "wallet_trade_count_24h",
            "is_adding_to_position",
            "size_vs_wallet_avg",
            "detection_delay_seconds",
            "hour_of_day",
            "day_of_week",
            "btc_price",
            "btc_momentum_60s",
            "raw_activity_json",
            "raw_market_json",
            "raw_orderbook_json",
            "raw_positions_json",
            "collector_version",
            "schema_version",
        ]
        placeholders = ", ".join("?" for _ in columns)
        values = [row.get(column) for column in columns]
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                f"""
                INSERT OR IGNORE INTO wallet_trades (
                    {", ".join(columns)}
                ) VALUES ({placeholders})
                """,
                values,
            )
            conn.commit()
            inserted = cursor.rowcount > 0
        if inserted:
            if self.get_meta("collection_started_at") in {None, "null"}:
                self.set_meta("collection_started_at", row["detected_at"], updated_at=row["detected_at"])
            self.set_meta("tracker_last_trade_detected_at", row["detected_at"], updated_at=row["detected_at"])
        return inserted

    def get_unlabeled_buy_trades(self, *, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM wallet_trades
                WHERE side = 'BUY'
                  AND wallet_sell_timestamp IS NULL
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def find_first_later_sell(self, buy_row: dict[str, Any]) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM wallet_trades
                WHERE side = 'SELL'
                  AND wallet_address = ?
                  AND market_condition_id = ?
                  AND asset_token_id = ?
                  AND timestamp > ?
                ORDER BY timestamp ASC
                LIMIT 1
                """,
                (
                    buy_row["wallet_address"],
                    buy_row["market_condition_id"],
                    buy_row["asset_token_id"],
                    buy_row["timestamp"],
                ),
            ).fetchone()
        return dict(row) if row is not None else None

    def apply_wallet_sell_label(self, *, buy_trade_id: int, sell_row: dict[str, Any], match_quality: str) -> None:
        with self._lock, self._connect() as conn:
            buy_row = conn.execute(
                "SELECT size_shares, price FROM wallet_trades WHERE id = ? LIMIT 1",
                (buy_trade_id,),
            ).fetchone()
            if buy_row is None:
                return
            buy_size = float(buy_row["size_shares"] or 0.0)
            buy_price = float(buy_row["price"] or 0.0)
            sell_price = float(sell_row.get("price") or 0.0)
            sell_size = float(sell_row.get("size_shares") or 0.0)
            wallet_sell_return = ((sell_price - buy_price) / buy_price) if buy_price > 0 else None
            conn.execute(
                """
                UPDATE wallet_trades
                SET wallet_sell_trade_key = ?,
                    wallet_sell_tx_hash = ?,
                    wallet_sell_timestamp = ?,
                    wallet_sell_price = ?,
                    wallet_sell_size_shares = ?,
                    wallet_sell_is_partial = ?,
                    wallet_sell_return = ?,
                    wallet_sell_match_quality = ?,
                    is_win_wallet_sell = ?
                WHERE id = ?
                """,
                (
                    sell_row.get("trade_key"),
                    sell_row.get("tx_hash"),
                    sell_row.get("timestamp"),
                    sell_price,
                    sell_size,
                    1 if buy_size > 0 and sell_size < buy_size else 0,
                    wallet_sell_return,
                    match_quality,
                    1 if wallet_sell_return is not None and wallet_sell_return > 0 else 0,
                    buy_trade_id,
                ),
            )
            conn.commit()

    def unresolved_markets(self, *, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT market_condition_id, market_slug
                FROM wallet_trades
                WHERE side = 'BUY'
                  AND market_resolved = 0
                GROUP BY market_condition_id, market_slug
                ORDER BY MIN(timestamp) ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def apply_market_resolution(
        self,
        *,
        condition_id: str,
        resolution_timestamp: float,
        winning_outcome: str | None,
        outcome_prices: dict[str, float],
        resolution_source: str,
    ) -> int:
        updated = 0
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, outcome, price
                FROM wallet_trades
                WHERE side = 'BUY'
                  AND market_condition_id = ?
                  AND market_resolved = 0
                """,
                (condition_id,),
            ).fetchall()
            for row in rows:
                entry_price = float(row["price"] or 0.0)
                settlement_price = outcome_prices.get(str(row["outcome"] or ""), 0.0)
                resolution_return = ((settlement_price - entry_price) / entry_price) if entry_price > 0 else None
                conn.execute(
                    """
                    UPDATE wallet_trades
                    SET market_resolved = 1,
                        resolution_timestamp = ?,
                        winning_outcome = ?,
                        resolution_return = ?,
                        resolution_source = ?,
                        is_win_resolution = ?
                    WHERE id = ?
                    """,
                    (
                        resolution_timestamp,
                        winning_outcome,
                        resolution_return,
                        resolution_source,
                        1 if settlement_price > 0.5 else 0,
                        row["id"],
                    ),
                )
                updated += 1
            conn.commit()
        return updated

    def trades_missing_price_checkpoints(self, *, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, asset_token_id, timestamp
                FROM wallet_trades
                WHERE side = 'BUY'
                  AND (
                    price_1min_after IS NULL
                    OR price_5min_after IS NULL
                    OR price_30min_after IS NULL
                  )
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_price_checkpoints(
        self,
        *,
        trade_id: int,
        price_1min_after: float | None,
        price_5min_after: float | None,
        price_30min_after: float | None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE wallet_trades
                SET price_1min_after = COALESCE(price_1min_after, ?),
                    price_5min_after = COALESCE(price_5min_after, ?),
                    price_30min_after = COALESCE(price_30min_after, ?)
                WHERE id = ?
                """,
                (price_1min_after, price_5min_after, price_30min_after, trade_id),
            )
            conn.commit()

    def list_wallet_trades(
        self,
        *,
        limit: int = 100,
        wallet_address: str | None = None,
        market_slug: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if wallet_address:
            clauses.append("wallet_address = ?")
            params.append(wallet_address)
        if market_slug:
            clauses.append("market_slug = ?")
            params.append(market_slug)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    id, trade_key, tx_hash, timestamp, detected_at, wallet_address, market_condition_id,
                    asset_token_id, market_title, market_slug, side, outcome, price, size_shares, size_usd,
                    market_category, market_spread, market_volume_24h, market_seconds_to_expiry,
                    wallet_leaderboard_rank, wallet_trade_count_24h, is_adding_to_position, size_vs_wallet_avg,
                    btc_price, btc_momentum_60s, wallet_sell_timestamp, wallet_sell_price, wallet_sell_return,
                    market_resolved, resolution_timestamp, winning_outcome, resolution_return,
                    CASE WHEN {self._ml_ready_buy_predicate()} THEN 1 ELSE 0 END AS ml_ready_candidate,
                    CASE WHEN {self._stale_buy_predicate()} THEN 1 ELSE 0 END AS stale_candidate,
                    price_1min_after, price_5min_after, price_30min_after
                FROM wallet_trades
                {where}
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_wallet_trades_after_id(
        self,
        *,
        last_id: int = 0,
        limit: int = 250,
        side: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["id > ?"]
        params: list[Any] = [last_id]
        if side:
            clauses.append("side = ?")
            params.append(side.upper())
        params.append(limit)
        where = " AND ".join(clauses)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM wallet_trades
                WHERE {where}
                ORDER BY id ASC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_wallet_trade(self, *, trade_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM wallet_trades
                WHERE id = ?
                LIMIT 1
                """,
                (trade_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def latest_wallet_trade_id(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(id) AS latest_id FROM wallet_trades"
            ).fetchone()
        return int((row["latest_id"] if row else 0) or 0)

    def observed_wallet_performance_before(
        self,
        *,
        wallet_address: str,
        before_timestamp: float,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS labeled_trades,
                    SUM(
                        CASE
                            WHEN COALESCE(is_win_wallet_sell, is_win_resolution) = 1 THEN 1
                            ELSE 0
                        END
                    ) AS wins,
                    AVG(
                        CASE
                            WHEN COALESCE(is_win_wallet_sell, is_win_resolution) IS NOT NULL
                            THEN COALESCE(is_win_wallet_sell, is_win_resolution)
                            ELSE NULL
                        END
                    ) AS win_rate,
                    AVG(
                        CASE
                            WHEN wallet_sell_timestamp IS NOT NULL THEN (wallet_sell_timestamp - timestamp) / 3600.0
                            WHEN market_resolved = 1 AND resolution_timestamp IS NOT NULL THEN (resolution_timestamp - timestamp) / 3600.0
                            ELSE NULL
                        END
                    ) AS avg_hold_hours
                FROM wallet_trades
                WHERE wallet_address = ?
                  AND side = 'BUY'
                  AND timestamp < ?
                  AND (wallet_sell_timestamp IS NOT NULL OR market_resolved = 1)
                """,
                (wallet_address, before_timestamp),
            ).fetchone()
        labeled_trades = int((row["labeled_trades"] if row else 0) or 0)
        wins = int((row["wins"] if row else 0) or 0)
        win_rate = float(row["win_rate"]) if row and row["win_rate"] is not None else None
        avg_hold_hours = float(row["avg_hold_hours"]) if row and row["avg_hold_hours"] is not None else None
        return {
            "labeled_trades": labeled_trades,
            "wins": wins,
            "losses": max(labeled_trades - wins, 0),
            "win_rate": win_rate,
            "avg_hold_hours": avg_hold_hours,
        }

    def recent_same_side_trades(
        self,
        *,
        condition_id: str,
        asset_token_id: str,
        outcome: str,
        since_timestamp: float,
        exclude_trade_id: int | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses = [
            "side = 'BUY'",
            "market_condition_id = ?",
            "asset_token_id = ?",
            "outcome = ?",
            "timestamp >= ?",
        ]
        params: list[Any] = [condition_id, asset_token_id, outcome, since_timestamp]
        if exclude_trade_id is not None:
            clauses.append("id != ?")
            params.append(exclude_trade_id)
        params.append(limit)
        where = " AND ".join(clauses)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM wallet_trades
                WHERE {where}
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [dict(row) for row in rows]

    def market_first_seen_timestamp(self, *, condition_id: str) -> float | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT MIN(timestamp) AS first_seen
                FROM wallet_trades
                WHERE market_condition_id = ?
                """,
                (condition_id,),
            ).fetchone()
        if row is None or row["first_seen"] is None:
            return None
        return float(row["first_seen"])

    def find_prior_buy(
        self,
        *,
        wallet_address: str,
        condition_id: str,
        asset_token_id: str,
        before_timestamp: float,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM wallet_trades
                WHERE wallet_address = ?
                  AND market_condition_id = ?
                  AND asset_token_id = ?
                  AND side = 'BUY'
                  AND timestamp < ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (wallet_address, condition_id, asset_token_id, before_timestamp),
            ).fetchone()
        return dict(row) if row is not None else None

    def insert_copy_decision(self, row: dict[str, Any]) -> int:
        columns = [
            "decision_key",
            "strategy_key",
            "profile_kind",
            "wallet_trade_id",
            "wallet_trade_key",
            "tx_hash",
            "wallet_address",
            "market_condition_id",
            "asset_token_id",
            "market_slug",
            "source_side",
            "source_outcome",
            "score",
            "should_copy",
            "suggested_size_usd",
            "reasons_json",
            "context_json",
            "paper_executed",
            "paper_trade_id",
            "paper_side",
            "paper_entry_timestamp",
            "paper_entry_price",
            "paper_entry_size_usd",
            "paper_exit_timestamp",
            "paper_exit_price",
            "paper_realized_pnl_usd",
            "paper_close_reason",
            "created_at",
            "updated_at",
        ]
        payload = dict(row)
        if not isinstance(payload.get("reasons_json"), str):
            payload["reasons_json"] = self._json_value(payload.get("reasons_json") or [])
        if not isinstance(payload.get("context_json"), str):
            payload["context_json"] = self._json_value(payload.get("context_json") or {})
        placeholders = ", ".join("?" for _ in columns)
        values = [payload.get(column) for column in columns]
        with self._lock, self._connect() as conn:
            conn.execute(
                f"""
                INSERT OR IGNORE INTO copy_decisions (
                    {", ".join(columns)}
                ) VALUES ({placeholders})
                """,
                values,
            )
            row_id = conn.execute(
                "SELECT id FROM copy_decisions WHERE decision_key = ? LIMIT 1",
                (payload["decision_key"],),
            ).fetchone()
            conn.commit()
        return int(row_id["id"])

    def mark_copy_decision_executed(
        self,
        *,
        decision_id: int,
        paper_trade_id: str,
        paper_side: str,
        entry_timestamp: float,
        entry_price: float,
        entry_size_usd: float,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE copy_decisions
                SET paper_executed = 1,
                    paper_trade_id = ?,
                    paper_side = ?,
                    paper_entry_timestamp = ?,
                    paper_entry_price = ?,
                    paper_entry_size_usd = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    paper_trade_id,
                    paper_side,
                    entry_timestamp,
                    entry_price,
                    entry_size_usd,
                    entry_timestamp,
                    decision_id,
                ),
            )
            conn.commit()

    def close_copy_decision(
        self,
        *,
        decision_id: int,
        exit_timestamp: float,
        exit_price: float,
        realized_pnl_usd: float,
        close_reason: str,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE copy_decisions
                SET paper_exit_timestamp = ?,
                    paper_exit_price = ?,
                    paper_realized_pnl_usd = ?,
                    paper_close_reason = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    exit_timestamp,
                    exit_price,
                    realized_pnl_usd,
                    close_reason,
                    exit_timestamp,
                    decision_id,
                ),
            )
            conn.commit()

    def list_copy_decisions(
        self,
        *,
        limit: int = 100,
        strategy_key: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if strategy_key:
            clauses.append("strategy_key = ?")
            params.append(strategy_key)
        params.append(limit)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM copy_decisions
                {where}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        decisions: list[dict[str, Any]] = []
        for row in rows:
            payload = dict(row)
            for key in ("reasons_json", "context_json"):
                try:
                    payload[key[:-5]] = json.loads(payload[key] or "[]")
                except json.JSONDecodeError:
                    payload[key[:-5]] = [] if key == "reasons_json" else {}
            payload["should_copy"] = bool(payload.get("should_copy"))
            payload["paper_executed"] = bool(payload.get("paper_executed"))
            decisions.append(payload)
        return decisions

    def list_tracked_wallets(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    address, leaderboard_rank_all, leaderboard_rank_30d, leaderboard_rank_30d_volume,
                    leaderboard_profit, leaderboard_volume, leaderboard_num_trades, leaderboard_win_rate,
                    leaderboard_source_flags, last_refreshed
                FROM tracked_wallets
                ORDER BY COALESCE(leaderboard_rank_30d, leaderboard_rank_all, leaderboard_rank_30d_volume, 999999) ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        wallets: list[dict[str, Any]] = []
        for row in rows:
            payload = dict(row)
            try:
                payload["leaderboard_source_flags"] = json.loads(payload["leaderboard_source_flags"] or "[]")
            except json.JSONDecodeError:
                payload["leaderboard_source_flags"] = []
            wallets.append(payload)
        return wallets

    def collection_stats(
        self,
        *,
        now_ts: float,
        tracker_poll_seconds: int,
        labeler_poll_seconds: int,
        leaderboard_refresh_seconds: int,
        positions_refresh_seconds: int,
    ) -> dict[str, Any]:
        ml_ready_buy = self._ml_ready_buy_predicate()
        stale_buy = self._stale_buy_predicate()
        with self._connect() as conn:
            counts = conn.execute(
                f"""
                SELECT
                    COUNT(*) AS total_trades,
                    SUM(CASE WHEN side = 'BUY' THEN 1 ELSE 0 END) AS total_buys,
                    SUM(CASE WHEN side = 'SELL' THEN 1 ELSE 0 END) AS total_sells,
                    SUM(CASE WHEN side = 'BUY' AND wallet_sell_timestamp IS NOT NULL THEN 1 ELSE 0 END) AS wallet_sell_labeled_buys,
                    SUM(CASE WHEN side = 'BUY' AND market_resolved = 1 THEN 1 ELSE 0 END) AS resolution_labeled_buys,
                    SUM(CASE WHEN side = 'BUY' AND (wallet_sell_timestamp IS NOT NULL OR market_resolved = 1) THEN 1 ELSE 0 END) AS labeled_buys,
                    SUM(CASE WHEN {ml_ready_buy} THEN 1 ELSE 0 END) AS ml_ready_buys,
                    SUM(CASE WHEN {ml_ready_buy} AND wallet_sell_timestamp IS NOT NULL THEN 1 ELSE 0 END) AS ml_ready_wallet_sell_labeled_buys,
                    SUM(CASE WHEN {ml_ready_buy} AND market_resolved = 1 THEN 1 ELSE 0 END) AS ml_ready_resolution_labeled_buys,
                    SUM(CASE WHEN {ml_ready_buy} AND (wallet_sell_timestamp IS NOT NULL OR market_resolved = 1) THEN 1 ELSE 0 END) AS ml_ready_labeled_buys,
                    SUM(CASE WHEN {stale_buy} THEN 1 ELSE 0 END) AS stale_buy_candidates,
                    COUNT(DISTINCT wallet_address) AS unique_wallets,
                    COUNT(DISTINCT market_condition_id) AS unique_markets,
                    MIN(timestamp) AS first_trade_timestamp,
                    MIN(detected_at) AS first_trade_detected_at,
                    MAX(detected_at) AS latest_trade_detected_at
                FROM wallet_trades
                """
            ).fetchone()
            tracked_wallet_count_row = conn.execute(
                "SELECT COUNT(*) AS tracked_wallet_count FROM tracked_wallets"
            ).fetchone()
        collection_started_at = self.get_meta("collection_started_at")
        if not isinstance(collection_started_at, (int, float)) or collection_started_at <= 0:
            first_trade_detected_at = float(counts["first_trade_detected_at"] or 0.0)
            collection_started_at = first_trade_detected_at if first_trade_detected_at > 0 else None
        labeled_buys = int(counts["labeled_buys"] or 0)
        ml_ready_labeled_buys = int(counts["ml_ready_labeled_buys"] or 0)
        elapsed_days = None
        labels_per_day = None
        eta_days = None
        if isinstance(collection_started_at, (int, float)) and collection_started_at > 0 and now_ts > collection_started_at:
            elapsed_days = (now_ts - collection_started_at) / 86400.0
            if elapsed_days > 0:
                labels_per_day = ml_ready_labeled_buys / elapsed_days
                remaining = max(self.target_labeled_buys - ml_ready_labeled_buys, 0)
                eta_days = (remaining / labels_per_day) if labels_per_day > 0 and remaining > 0 else 0.0

        tracker_heartbeat = self.get_meta("tracker_last_heartbeat")
        labeler_heartbeat = self.get_meta("labeler_last_heartbeat")
        leaderboard_heartbeat = self.get_meta("leaderboard_last_refreshed_at")
        positions_heartbeat = self.get_meta("positions_last_refreshed_at")
        tracked_wallet_count = int(
            (tracked_wallet_count_row["tracked_wallet_count"] if tracked_wallet_count_row else 0) or 0
        )
        db_exists = self.db_path.exists()
        db_writable = db_exists and os.access(self.db_path, os.W_OK)

        def alive(last_seen: Any, threshold: float) -> bool:
            return isinstance(last_seen, (int, float)) and (now_ts - float(last_seen)) <= threshold

        progress_pct = min(100.0, (ml_ready_labeled_buys / self.target_labeled_buys) * 100.0) if self.target_labeled_buys > 0 else 0.0
        tracker_threshold = max(float(tracker_poll_seconds * 3), 180.0)
        labeler_threshold = max(float(labeler_poll_seconds * 3), 180.0)
        positions_threshold = max(float(positions_refresh_seconds * 3), 900.0)
        return {
            "target_labeled_buys": self.target_labeled_buys,
            "progress_basis": "ml_ready_labeled_buys",
            "progress_pct": round(progress_pct, 2),
            "counts": {
                "total_trades": int(counts["total_trades"] or 0),
                "total_buys": int(counts["total_buys"] or 0),
                "total_sells": int(counts["total_sells"] or 0),
                "wallet_sell_labeled_buys": int(counts["wallet_sell_labeled_buys"] or 0),
                "resolution_labeled_buys": int(counts["resolution_labeled_buys"] or 0),
                "labeled_buys": labeled_buys,
                "ml_ready_buys": int(counts["ml_ready_buys"] or 0),
                "ml_ready_wallet_sell_labeled_buys": int(counts["ml_ready_wallet_sell_labeled_buys"] or 0),
                "ml_ready_resolution_labeled_buys": int(counts["ml_ready_resolution_labeled_buys"] or 0),
                "ml_ready_labeled_buys": ml_ready_labeled_buys,
                "stale_buy_candidates": int(counts["stale_buy_candidates"] or 0),
                "unique_wallets": int(counts["unique_wallets"] or 0),
                "unique_markets": int(counts["unique_markets"] or 0),
                "tracked_wallets": tracked_wallet_count,
            },
            "timing": {
                "collection_started_at": collection_started_at,
                "latest_trade_detected_at": counts["latest_trade_detected_at"],
                "elapsed_days": round(elapsed_days, 3) if elapsed_days is not None else None,
                "labels_per_day": round(labels_per_day, 3) if labels_per_day is not None else None,
                "eta_days_to_target": round(eta_days, 3) if eta_days is not None else None,
            },
            "health": {
                "tracker_alive": alive(tracker_heartbeat, tracker_threshold),
                "labeler_alive": alive(labeler_heartbeat, labeler_threshold),
                "leaderboard_fresh": alive(leaderboard_heartbeat, leaderboard_refresh_seconds * 1.5),
                "positions_fresh": alive(positions_heartbeat, positions_threshold),
                "db_writable": db_writable,
                "last_tracker_heartbeat": tracker_heartbeat,
                "last_labeler_heartbeat": labeler_heartbeat,
                "last_leaderboard_refresh_at": leaderboard_heartbeat,
                "last_positions_refresh_at": positions_heartbeat,
            },
            "schema_version": self.schema_version,
            "collector_version": self.collector_version,
            "db_path": str(self.db_path),
        }
