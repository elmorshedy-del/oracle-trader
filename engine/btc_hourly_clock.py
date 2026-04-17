from __future__ import annotations

from datetime import datetime, timedelta, timezone

from zoneinfo import ZoneInfo


ET = ZoneInfo("America/New_York")
UTC = timezone.utc


def current_polymarket_hour_start(now: datetime | None = None) -> datetime:
    current = (now or datetime.now(UTC)).astimezone(ET)
    return current.replace(minute=0, second=0, microsecond=0)


def current_kalshi_hour_label(now: datetime | None = None) -> datetime:
    return current_polymarket_hour_start(now) + timedelta(hours=1)


def polymarket_btc_hourly_slug(now: datetime | None = None) -> str:
    target = current_polymarket_hour_start(now)
    month = target.strftime("%B").lower()
    hour = int(target.strftime("%I"))
    am_pm = target.strftime("%p").lower()
    return f"bitcoin-up-or-down-{month}-{target.day}-{target.year}-{hour}{am_pm}-et"


def kalshi_btc_event_ticker(now: datetime | None = None) -> str:
    target = current_kalshi_hour_label(now)
    return f"KXBTCD-{target.strftime('%y').upper()}{target.strftime('%b').upper()}{target.strftime('%d')}{target.strftime('%H')}"
