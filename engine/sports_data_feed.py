"""
NBA Sports Data Feed
====================
Pulls upcoming NBA games plus sportsbook anchor lines from ESPN's public
scoreboard endpoint. This is intentionally lightweight and defensive so the
legacy sports sleeve can paper trade without coupling strategy logic to raw
scoreboard JSON.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

HTTP_TIMEOUT_SECONDS = 20.0
DEFAULT_SCOREBOARD_DAYS = 2
DEFAULT_USER_AGENT = "Mozilla/5.0 (Oracle Sports Feed)"
TEAM_ABBREV_SPREAD_RE = re.compile(r"\b([A-Z]{2,4})\s*([+-]?\d+(?:\.\d+)?)\b")
FLOAT_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")


@dataclass
class SportsGameSnapshot:
    event_id: str
    league: str
    start_time: datetime
    status: str
    home_team: str
    away_team: str
    home_abbrev: str
    away_abbrev: str
    home_record: str = ""
    away_record: str = ""
    sportsbook: str = "unknown"
    home_moneyline: int | None = None
    away_moneyline: int | None = None
    home_win_prob: float | None = None
    away_win_prob: float | None = None
    home_spread: float | None = None
    away_spread: float | None = None
    open_home_spread: float | None = None
    open_away_spread: float | None = None
    total_line: float | None = None
    open_total_line: float | None = None
    spread_move_points: float = 0.0
    total_move_points: float = 0.0
    details: str = ""
    source_updated_at: str | None = None
    injury_alerts: list[str] = field(default_factory=list)

    @property
    def hours_to_tip(self) -> float:
        return max(0.0, (self.start_time - datetime.now(timezone.utc)).total_seconds() / 3600.0)


class SportsDataFeed:
    """Caches and normalizes ESPN NBA scoreboard + odds data."""

    def __init__(
        self,
        *,
        scoreboard_url: str,
        league: str,
        refresh_seconds: int,
        market_horizon_hours: float,
    ):
        self.scoreboard_url = scoreboard_url
        self.league = league
        self.refresh_seconds = max(30, int(refresh_seconds))
        self.market_horizon_hours = max(6.0, float(market_horizon_hours))
        self.client = httpx.AsyncClient(
            timeout=HTTP_TIMEOUT_SECONDS,
            headers={"User-Agent": DEFAULT_USER_AGENT},
        )
        self._games: list[SportsGameSnapshot] = []
        self._last_refresh_at: datetime | None = None
        self._last_error: str | None = None
        self.stats: dict[str, Any] = {
            "league": league,
            "refresh_seconds": self.refresh_seconds,
            "games_loaded": 0,
            "upcoming_games": 0,
            "line_move_candidates": 0,
            "injury_alerts": 0,
            "last_refresh_at": None,
            "last_error": None,
        }

    async def close(self) -> None:
        await self.client.aclose()

    async def snapshot(self) -> list[SportsGameSnapshot]:
        now = datetime.now(timezone.utc)
        if self._last_refresh_at and (now - self._last_refresh_at).total_seconds() < self.refresh_seconds:
            return list(self._games)
        await self._refresh()
        return list(self._games)

    async def _refresh(self) -> None:
        now = datetime.now(timezone.utc)
        horizon = now + timedelta(hours=self.market_horizon_hours)
        target_days = self._target_dates(now, horizon)
        games: list[SportsGameSnapshot] = []

        for target_day in target_days:
            params = {"dates": target_day.strftime("%Y%m%d")}
            response = await self.client.get(self.scoreboard_url, params=params)
            response.raise_for_status()
            payload = response.json()
            events = payload.get("events") or []
            for event in events:
                game = self._parse_event(event)
                if not game:
                    continue
                if game.start_time > horizon:
                    continue
                games.append(game)

        games.sort(key=lambda item: item.start_time)
        self._games = games
        self._last_refresh_at = now
        self._last_error = None
        self.stats["games_loaded"] = len(games)
        self.stats["upcoming_games"] = sum(1 for game in games if game.hours_to_tip >= 0)
        self.stats["line_move_candidates"] = sum(
            1 for game in games if abs(game.spread_move_points) >= 1.0 or abs(game.total_move_points) >= 1.0
        )
        self.stats["injury_alerts"] = sum(len(game.injury_alerts) for game in games)
        self.stats["last_refresh_at"] = now.isoformat()
        self.stats["last_error"] = None

    @staticmethod
    def _target_dates(start: datetime, end: datetime) -> list[datetime]:
        days = max(DEFAULT_SCOREBOARD_DAYS, int(math.ceil((end - start).total_seconds() / 86400.0)) + 1)
        return [start + timedelta(days=offset) for offset in range(days)]

    def _parse_event(self, payload: dict[str, Any]) -> SportsGameSnapshot | None:
        competitions = payload.get("competitions") or []
        if not competitions:
            return None
        competition = competitions[0]
        competitors = competition.get("competitors") or []
        home = next((team for team in competitors if (team.get("homeAway") or "").lower() == "home"), None)
        away = next((team for team in competitors if (team.get("homeAway") or "").lower() == "away"), None)
        if not home or not away:
            return None

        try:
            start_time = datetime.fromisoformat(str(payload.get("date")).replace("Z", "+00:00"))
        except Exception:
            return None

        odds = (competition.get("odds") or [{}])[0]
        home_abbrev = str(home.get("team", {}).get("abbreviation") or "").upper()
        away_abbrev = str(away.get("team", {}).get("abbreviation") or "").upper()
        details = str(odds.get("details") or "")

        home_spread, away_spread = self._extract_spreads(
            odds=odds,
            details=details,
            home_abbrev=home_abbrev,
            away_abbrev=away_abbrev,
        )
        open_home_spread, open_away_spread = self._extract_spreads(
            odds={"spread": odds.get("open"), "details": str(odds.get("open") or "")},
            details=str(odds.get("open") or ""),
            home_abbrev=home_abbrev,
            away_abbrev=away_abbrev,
        )

        home_moneyline = self._extract_int(
            odds,
            "hometeamodds",
            "homemoneyline",
            "homeodds",
            "moneylinehome",
        )
        away_moneyline = self._extract_int(
            odds,
            "awayteamodds",
            "awaymoneyline",
            "awayodds",
            "moneylineaway",
        )
        home_win_prob, away_win_prob = self._normalize_moneyline_probs(home_moneyline, away_moneyline)
        total_line = self._extract_float(odds, "overunder", "total", "overunderline")
        open_total_line = self._extract_float({"overunder": odds.get("open")}, "overunder")
        spread_move_points = 0.0
        total_move_points = 0.0
        if home_spread is not None and open_home_spread is not None:
            spread_move_points = home_spread - open_home_spread
        if total_line is not None and open_total_line is not None:
            total_move_points = total_line - open_total_line

        return SportsGameSnapshot(
            event_id=str(payload.get("id") or ""),
            league=self.league,
            start_time=start_time,
            status=str((competition.get("status") or {}).get("type", {}).get("description") or ""),
            home_team=str(home.get("team", {}).get("displayName") or ""),
            away_team=str(away.get("team", {}).get("displayName") or ""),
            home_abbrev=home_abbrev,
            away_abbrev=away_abbrev,
            home_record=self._extract_record(home),
            away_record=self._extract_record(away),
            sportsbook=str((odds.get("provider") or {}).get("name") or "ESPN"),
            home_moneyline=home_moneyline,
            away_moneyline=away_moneyline,
            home_win_prob=home_win_prob,
            away_win_prob=away_win_prob,
            home_spread=home_spread,
            away_spread=away_spread,
            open_home_spread=open_home_spread,
            open_away_spread=open_away_spread,
            total_line=total_line,
            open_total_line=open_total_line,
            spread_move_points=spread_move_points,
            total_move_points=total_move_points,
            details=details,
            source_updated_at=str(odds.get("lastUpdated") or payload.get("date") or ""),
        )

    @staticmethod
    def _extract_record(team_payload: dict[str, Any]) -> str:
        records = team_payload.get("records") or []
        for record in records:
            summary = record.get("summary")
            if summary:
                return str(summary)
        return ""

    @classmethod
    def _extract_spreads(
        cls,
        *,
        odds: dict[str, Any],
        details: str,
        home_abbrev: str,
        away_abbrev: str,
    ) -> tuple[float | None, float | None]:
        explicit_home = cls._extract_float(odds, "homespread", "homespreadline")
        explicit_away = cls._extract_float(odds, "awayspread", "awayspreadline")
        if explicit_home is not None and explicit_away is not None:
            return explicit_home, explicit_away

        spread_value = cls._extract_float(odds, "spread", "line")
        if spread_value is not None and details:
            parsed = TEAM_ABBREV_SPREAD_RE.search(details.upper())
            if parsed:
                favorite_abbrev = parsed.group(1)
                if favorite_abbrev == home_abbrev:
                    return spread_value, -spread_value
                if favorite_abbrev == away_abbrev:
                    return -spread_value, spread_value

        parsed = TEAM_ABBREV_SPREAD_RE.search(details.upper())
        if parsed:
            favorite_abbrev = parsed.group(1)
            favorite_spread = float(parsed.group(2))
            if favorite_abbrev == home_abbrev:
                return favorite_spread, -favorite_spread
            if favorite_abbrev == away_abbrev:
                return -favorite_spread, favorite_spread
        return None, None

    @classmethod
    def _extract_float(cls, payload: Any, *keys: str) -> float | None:
        value = cls._lookup_value(payload, *keys)
        if value in (None, "", "null"):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        match = FLOAT_RE.search(str(value))
        return float(match.group(0)) if match else None

    @classmethod
    def _extract_int(cls, payload: Any, *keys: str) -> int | None:
        value = cls._lookup_value(payload, *keys)
        if value in (None, "", "null"):
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return int(value)
        match = FLOAT_RE.search(str(value))
        return int(float(match.group(0))) if match else None

    @classmethod
    def _lookup_value(cls, payload: Any, *keys: str) -> Any:
        targets = {key.lower() for key in keys}
        stack = [payload]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                for key, value in current.items():
                    key_str = str(key).lower()
                    if key_str in targets:
                        return value
                    stack.append(value)
            elif isinstance(current, list):
                stack.extend(current)
        return None

    @staticmethod
    def _normalize_moneyline_probs(home_odds: int | None, away_odds: int | None) -> tuple[float | None, float | None]:
        def american_to_prob(odds: int | None) -> float | None:
            if odds is None or odds == 0:
                return None
            if odds < 0:
                return abs(odds) / (abs(odds) + 100.0)
            return 100.0 / (odds + 100.0)

        home_prob = american_to_prob(home_odds)
        away_prob = american_to_prob(away_odds)
        if home_prob is None or away_prob is None:
            return home_prob, away_prob
        total = home_prob + away_prob
        if total <= 0:
            return None, None
        return home_prob / total, away_prob / total
