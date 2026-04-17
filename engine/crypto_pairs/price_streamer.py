"""Real-time Binance spot trade streamer with 1-second OHLCV aggregation."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Callable

import httpx
import websockets

from .config import DEFAULT_BAR_INTERVAL_SECONDS, PriceStreamerConfig


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PriceBar:
    symbol: str
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int


class PriceStreamer:
    """Stream Binance spot trades and emit closed 1-second bars."""

    def __init__(self, symbols: list[str], config: PriceStreamerConfig | None = None):
        self.config = config or PriceStreamerConfig()
        self.symbols = [symbol.upper() for symbol in symbols]
        self.callbacks: list[Callable[[PriceBar], None]] = []
        self.latest_prices: dict[str, float] = {}
        self.latest_volumes: dict[str, float] = {}
        self.stats = {
            "messages": 0,
            "bars_emitted": 0,
            "reconnects": 0,
            "started_at": int(time.time() * 1000),
            "connected_url": None,
            "last_message_at": None,
            "last_error": None,
        }
        self._active_bars: dict[str, dict[str, float | int]] = {}
        self._stop_requested = False

    def on_bar(self, callback: Callable[[PriceBar], None]) -> None:
        self.callbacks.append(callback)

    def stop(self) -> None:
        self._stop_requested = True

    async def run_forever(self, *, runtime_seconds: int | None = None) -> None:
        deadline = None if runtime_seconds is None else time.monotonic() + runtime_seconds
        while not self._stop_requested:
            if deadline is not None and time.monotonic() >= deadline:
                break
            connected = False
            for ws_url in self.config.ws_urls:
                try:
                    await self._connect_once(ws_url=ws_url, deadline=deadline)
                    connected = True
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.stats["reconnects"] += 1
                    self.stats["last_error"] = str(exc)
                    logger.warning("[CRYPTO_PAIRS] stream reconnect after error via %s: %s", ws_url, exc)
                    continue
            if connected:
                continue
            await asyncio.sleep(self.config.reconnect_delay_seconds)
        self._flush_active_bars()

    async def _connect_once(self, *, ws_url: str, deadline: float | None) -> None:
        streams = "/".join(f"{symbol.lower()}@trade" for symbol in self.symbols)
        url = f"{ws_url}?streams={streams}"
        async with websockets.connect(url, ping_interval=20, ping_timeout=20, max_size=None) as ws:
            self.stats["connected_url"] = ws_url
            poller = asyncio.create_task(self._rest_fallback_loop(deadline=deadline), name="crypto-pairs-rest-fallback")
            try:
                async for raw_message in ws:
                    if self._stop_requested:
                        break
                    if deadline is not None and time.monotonic() >= deadline:
                        break
                    self.stats["messages"] += 1
                    self.stats["last_message_at"] = int(time.time() * 1000)
                    payload = json.loads(raw_message)
                    trade = payload["data"]
                    self._handle_trade(
                        symbol=str(trade["s"]).upper(),
                        price=float(trade["p"]),
                        quantity=float(trade["q"]),
                        timestamp_ms=int(trade["T"]),
                    )
            finally:
                poller.cancel()
                await asyncio.gather(poller, return_exceptions=True)

    async def _rest_fallback_loop(self, *, deadline: float | None) -> None:
        async with httpx.AsyncClient(timeout=5.0) as client:
            while not self._stop_requested:
                if deadline is not None and time.monotonic() >= deadline:
                    return
                if not self._should_use_rest_fallback():
                    await asyncio.sleep(self.config.rest_poll_seconds)
                    continue
                for rest_url in self.config.rest_urls:
                    try:
                        await self._poll_rest_prices(client=client, rest_url=rest_url)
                        break
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        self.stats["last_error"] = f"rest:{exc}"
                        logger.warning("[CRYPTO_PAIRS] REST fallback failed via %s: %s", rest_url, exc)
                await asyncio.sleep(self.config.rest_poll_seconds)

    def _should_use_rest_fallback(self) -> bool:
        last_message_at = self.stats.get("last_message_at")
        if last_message_at is None:
            return True
        idle_seconds = (int(time.time() * 1000) - int(last_message_at)) / 1000.0
        return idle_seconds >= self.config.rest_idle_fallback_seconds

    async def _poll_rest_prices(self, *, client: httpx.AsyncClient, rest_url: str) -> None:
        now_ms = int(time.time() * 1000)
        responses = await asyncio.gather(
            *(
                client.get(rest_url, params={"symbol": symbol})
                for symbol in self.symbols
            )
        )
        for response in responses:
            response.raise_for_status()
            payload = response.json()
            symbol = str(payload["symbol"]).upper()
            price = float(payload["price"])
            self._handle_trade(symbol=symbol, price=price, quantity=0.0, timestamp_ms=now_ms)

    def _handle_trade(self, *, symbol: str, price: float, quantity: float, timestamp_ms: int) -> None:
        bucket_ms = self._bucket_timestamp_ms(timestamp_ms)
        current = self._active_bars.get(symbol)
        if current is None or int(current["timestamp_ms"]) != bucket_ms:
            if current is not None:
                self._emit_bar(symbol, current)
            self._active_bars[symbol] = {
                "timestamp_ms": bucket_ms,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": quantity,
                "trade_count": 1,
            }
            return
        current["high"] = max(float(current["high"]), price)
        current["low"] = min(float(current["low"]), price)
        current["close"] = price
        current["volume"] = float(current["volume"]) + quantity
        current["trade_count"] = int(current["trade_count"]) + 1

    def _bucket_timestamp_ms(self, timestamp_ms: int) -> int:
        interval_ms = self.config.bar_interval_seconds * 1000
        return timestamp_ms - (timestamp_ms % interval_ms)

    def _emit_bar(self, symbol: str, current: dict[str, float | int]) -> None:
        bar = PriceBar(
            symbol=symbol,
            timestamp_ms=int(current["timestamp_ms"]),
            open=float(current["open"]),
            high=float(current["high"]),
            low=float(current["low"]),
            close=float(current["close"]),
            volume=float(current["volume"]),
            trade_count=int(current["trade_count"]),
        )
        self.latest_prices[symbol] = bar.close
        self.latest_volumes[symbol] = bar.volume
        self.stats["bars_emitted"] += 1
        for callback in self.callbacks:
            callback(bar)

    def _flush_active_bars(self) -> None:
        for symbol, current in list(self._active_bars.items()):
            self._emit_bar(symbol, current)
        self._active_bars.clear()
