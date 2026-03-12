"""
Strategy: Legacy NBA Sports Sleeve
==================================
Standalone comparison-book sleeve that anchors Polymarket NBA markets to
sportsbook lines from ESPN's public scoreboard feed.

Phase 1 focuses on:
- game winner markets
- totals markets
- spread markets with clear team-side parsing

It does not affect the main legacy portfolio.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone

from data.models import Event, Market, Signal, SignalAction, SignalSource
from engine.sports_data_feed import SportsDataFeed, SportsGameSnapshot
from runtime_paths import LOG_DIR
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

TOTAL_PATTERN = re.compile(
    r"(?:over\/under|total(?:\s+points)?|points)(?:[^0-9+-]{0,16})([+-]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
SPREAD_PATTERN = re.compile(
    r"(?:spread|cover)(?:[^0-9+-]{0,16})([+-]?\d+(?:\.\d+)?)|([+-]?\d+(?:\.\d+)?)\s*spread",
    re.IGNORECASE,
)
SPORTS_TRIGGER_WORDS = ("nba", "basketball", "total points", "over/under", "spread", "cover", "win", "beat")
BASE_SIGNAL_CONFIDENCE = 0.56
MAX_SIGNAL_CONFIDENCE = 0.97
EDGE_CONFIDENCE_MULTIPLIER = 1.35
LINE_MOVE_CONFIDENCE_MULTIPLIER = 0.08
TOTAL_MOVE_PROB_MULTIPLIER = 0.18
SPREAD_MOVE_PROB_MULTIPLIER = 0.15
WIN_MOVE_PROB_MULTIPLIER = 0.03
EDGE_SIZE_USD_MULTIPLIER = 240.0
LINE_MOVE_SIZE_MULTIPLIER = 20.0
SIGNAL_LIMIT_MULTIPLIER = 3
FEE_BUFFER_RATE = 0.025

NBA_TEAM_ALIASES = {
    "atlanta hawks": {"atlanta hawks", "hawks", "atlanta", "atl"},
    "boston celtics": {"boston celtics", "celtics", "boston", "bos"},
    "brooklyn nets": {"brooklyn nets", "nets", "brooklyn", "bkn"},
    "charlotte hornets": {"charlotte hornets", "hornets", "charlotte", "cha"},
    "chicago bulls": {"chicago bulls", "bulls", "chicago", "chi"},
    "cleveland cavaliers": {"cleveland cavaliers", "cavaliers", "cavs", "cleveland", "cle"},
    "dallas mavericks": {"dallas mavericks", "mavericks", "mavs", "dallas", "dal"},
    "denver nuggets": {"denver nuggets", "nuggets", "denver", "den"},
    "detroit pistons": {"detroit pistons", "pistons", "detroit", "det"},
    "golden state warriors": {"golden state warriors", "warriors", "golden state", "gsw"},
    "houston rockets": {"houston rockets", "rockets", "houston", "hou"},
    "indiana pacers": {"indiana pacers", "pacers", "indiana", "ind"},
    "la clippers": {"la clippers", "clippers", "los angeles clippers", "lac"},
    "los angeles clippers": {"la clippers", "clippers", "los angeles clippers", "lac"},
    "los angeles lakers": {"los angeles lakers", "lakers", "la lakers", "lal"},
    "memphis grizzlies": {"memphis grizzlies", "grizzlies", "memphis", "mem"},
    "miami heat": {"miami heat", "heat", "miami", "mia"},
    "milwaukee bucks": {"milwaukee bucks", "bucks", "milwaukee", "mil"},
    "minnesota timberwolves": {"minnesota timberwolves", "timberwolves", "wolves", "minnesota", "min"},
    "new orleans pelicans": {"new orleans pelicans", "pelicans", "new orleans", "nop"},
    "new york knicks": {"new york knicks", "knicks", "new york", "nyk"},
    "oklahoma city thunder": {"oklahoma city thunder", "thunder", "oklahoma city", "okc"},
    "orlando magic": {"orlando magic", "magic", "orlando", "orl"},
    "philadelphia 76ers": {"philadelphia 76ers", "76ers", "sixers", "philadelphia", "phi"},
    "phoenix suns": {"phoenix suns", "suns", "phoenix", "phx"},
    "portland trail blazers": {"portland trail blazers", "trail blazers", "blazers", "portland", "por"},
    "sacramento kings": {"sacramento kings", "kings", "sacramento", "sac"},
    "san antonio spurs": {"san antonio spurs", "spurs", "san antonio", "sas"},
    "toronto raptors": {"toronto raptors", "raptors", "toronto", "tor"},
    "utah jazz": {"utah jazz", "jazz", "utah", "uta"},
    "washington wizards": {"washington wizards", "wizards", "washington", "was"},
}
AMBIGUOUS_TEAM_ALIASES = {
    "heat",
    "magic",
    "jazz",
    "kings",
    "spurs",
    "nets",
    "suns",
    "bulls",
    "wolves",
}


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9.+-]+", " ", str(value or "").lower()).strip()


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _alias_in_text(text: str, alias: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text) is not None


def _alias_index(text: str, alias: str) -> int | None:
    match = re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text)
    return match.start() if match else None


class SportsModelStrategy(BaseStrategy):
    name = "sports_model"
    description = "Standalone NBA sportsbook-anchor sleeve (comparison-book only)"

    def __init__(self, config):
        super().__init__(config)
        self.cfg = config.sports_model
        self.enabled = bool(self.cfg.enabled)
        self.feed = SportsDataFeed(
            scoreboard_url=self.cfg.scoreboard_url,
            league=self.cfg.league,
            refresh_seconds=self.cfg.refresh_seconds,
            market_horizon_hours=self.cfg.market_horizon_hours,
        )
        self.log_path = LOG_DIR / "sports_model_sleeve.jsonl"
        self._stats.update(
            {
                "candidate_markets": 0,
                "matched_markets": 0,
                "scored_markets": 0,
                "games_loaded": 0,
                "line_move_candidates": 0,
                "last_scan_at": None,
                "last_signals": 0,
                "last_log_at": None,
                "log_entries": 0,
                "feed_stats": self.feed.stats,
            }
        )

    async def close(self) -> None:
        await self.feed.close()

    @property
    def stats(self) -> dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            **self._stats,
            "log_path": str(self.log_path.resolve()),
        }

    async def scan(self, markets: list[Market], events: list[Event]) -> list[Signal]:
        del events
        self._stats["scans_completed"] += 1
        self._stats["last_scan_at"] = datetime.now(timezone.utc).isoformat()

        if not self.enabled:
            self._stats["last_signals"] = 0
            self._append_scan_log([], games=[])
            return []

        games = await self.feed.snapshot()
        self._stats["games_loaded"] = len(games)
        self._stats["line_move_candidates"] = self.feed.stats.get("line_move_candidates", 0)
        self._stats["feed_stats"] = dict(self.feed.stats)

        candidate_markets = self._candidate_markets(markets)
        self._stats["candidate_markets"] = len(candidate_markets)

        eligible_games = [
            game
            for game in games
            if self.cfg.min_hours_to_tip <= game.hours_to_tip <= self.cfg.max_hours_to_tip
        ]

        signals: list[Signal] = []
        matched_markets = 0
        scored_markets = 0
        for market in candidate_markets:
            game = self._best_game_match(market, eligible_games)
            if game is None:
                continue
            matched_markets += 1
            signal = self._score_market(game, market)
            if signal is None:
                continue
            scored_markets += 1
            signals.append(signal)

        signals.sort(key=lambda item: (item.expected_edge, item.confidence), reverse=True)
        signals = signals[: self.cfg.max_signals_per_scan]
        self._stats["matched_markets"] = matched_markets
        self._stats["scored_markets"] = scored_markets
        self._stats["last_signals"] = len(signals)
        self._stats["signals_generated"] += len(signals)
        self._append_scan_log(signals, games=games)
        return signals

    def _candidate_markets(self, markets: list[Market]) -> list[Market]:
        candidates: list[Market] = []
        for market in markets:
            if market.closed or not market.active:
                continue
            if market.liquidity < self.cfg.min_market_liquidity_usd:
                continue
            if market.volume_24h < self.cfg.min_market_volume_usd:
                continue
            if not self._is_binary_yes_no_market(market):
                continue
            text = self._market_text(market)
            if not any(token in text for token in SPORTS_TRIGGER_WORDS):
                continue
            if not any(
                _alias_in_text(text, alias)
                for alias_set in NBA_TEAM_ALIASES.values()
                for alias in alias_set
            ):
                continue
            candidates.append(market)
        return candidates

    @staticmethod
    def _is_binary_yes_no_market(market: Market) -> bool:
        if len(market.outcomes) != 2:
            return False
        names = {_normalize_text(outcome.name) for outcome in market.outcomes}
        return names == {"yes", "no"}

    @staticmethod
    def _market_text(market: Market) -> str:
        return _normalize_text(
            " ".join(
                [
                    market.question,
                    market.slug,
                    " ".join(market.tags or []),
                ]
            )
        )

    def _market_match_profile(self, market: Market, game: SportsGameSnapshot) -> tuple[int, int]:
        text = self._market_text(market)
        home_aliases = self._aliases_for_team(game.home_team, game.home_abbrev)
        away_aliases = self._aliases_for_team(game.away_team, game.away_abbrev)
        home_hits = sum(1 for alias in home_aliases if _alias_in_text(text, alias))
        away_hits = sum(1 for alias in away_aliases if _alias_in_text(text, alias))
        return home_hits, away_hits

    def _best_game_match(self, market: Market, games: list[SportsGameSnapshot]) -> SportsGameSnapshot | None:
        ranked: list[tuple[int, int, SportsGameSnapshot]] = []
        for game in games:
            home_hits, away_hits = self._market_match_profile(market, game)
            team_sides = int(home_hits > 0) + int(away_hits > 0)
            score = home_hits + away_hits
            if team_sides == 0 or score == 0:
                continue
            ranked.append((team_sides, score, game))

        if not ranked:
            return None

        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_sides, best_score, best_game = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0
        if best_sides >= 2:
            return best_game
        if best_score >= 1 and second_score == 0:
            return best_game
        return None

    def _score_market(self, game: SportsGameSnapshot, market: Market) -> Signal | None:
        text = self._market_text(market)
        market_type = self._market_type(text)
        if market_type == "winner":
            scored = self._score_winner_market(game, market, text)
        elif market_type == "total":
            scored = self._score_total_market(game, market, text)
        elif market_type == "spread":
            scored = self._score_spread_market(game, market, text)
        else:
            return None

        if scored is None:
            return None

        action, modeled_yes, edge, token_id, market_price, line_signal = scored
        if edge < self.cfg.min_edge or market_price > self.cfg.max_entry_price or token_id is None:
            return None

        size_usd = min(
            self.cfg.max_size_usd,
            max(
                self.cfg.min_size_usd,
                self.cfg.min_size_usd + (edge * EDGE_SIZE_USD_MULTIPLIER) + (abs(line_signal) * LINE_MOVE_SIZE_MULTIPLIER),
            ),
        )
        confidence = min(
            MAX_SIGNAL_CONFIDENCE,
            max(
                BASE_SIGNAL_CONFIDENCE,
                BASE_SIGNAL_CONFIDENCE + (edge * EDGE_CONFIDENCE_MULTIPLIER) + min(abs(line_signal), 3.0) * LINE_MOVE_CONFIDENCE_MULTIPLIER,
            ),
        )
        subject = self._subject_team(text, game)
        reasoning = (
            f"SPORTS_NBA: {game.away_team} @ {game.home_team} | {market_type} | "
            f"book={self._book_reasoning(game, market_type, subject)} | "
            f"market_yes={market.outcomes[0].price:.3f} | modeled_yes={modeled_yes:.3f} | "
            f"picked={'YES' if action == SignalAction.BUY_YES else 'NO'} edge={edge:.3f}"
        )
        return Signal(
            source=SignalSource.SPORTS_MODEL,
            action=action,
            market_slug=market.slug,
            condition_id=market.condition_id,
            token_id=token_id,
            confidence=confidence,
            expected_edge=edge * 100.0,
            reasoning=reasoning,
            suggested_size_usd=size_usd,
            group_key=f"sports:nba:{game.event_id}:{market_type}:{subject or 'neutral'}",
        )

    def _score_winner_market(
        self,
        game: SportsGameSnapshot,
        market: Market,
        text: str,
    ) -> tuple[SignalAction, float, float, str | None, float, float] | None:
        subject = self._subject_team(text, game)
        if subject not in {"home", "away"}:
            return None
        modeled_yes = game.home_win_prob if subject == "home" else game.away_win_prob
        if modeled_yes is None:
            return None
        yes_price = market.outcomes[0].price
        no_price = market.outcomes[1].price
        fee_yes = self._fee_buffer(yes_price)
        fee_no = self._fee_buffer(no_price)
        yes_edge = modeled_yes - yes_price - fee_yes
        no_edge = (1.0 - modeled_yes) - no_price - fee_no
        line_signal = game.spread_move_points * (1.0 if subject == "home" else -1.0)
        edge_floor = max(self.cfg.min_edge, self.cfg.win_prob_edge_floor)
        if max(yes_edge, no_edge) < edge_floor:
            return None
        if yes_edge >= no_edge:
            return SignalAction.BUY_YES, modeled_yes, yes_edge, market.outcomes[0].token_id, yes_price, line_signal
        return SignalAction.BUY_NO, modeled_yes, no_edge, market.outcomes[1].token_id, no_price, line_signal

    def _score_total_market(
        self,
        game: SportsGameSnapshot,
        market: Market,
        text: str,
    ) -> tuple[SignalAction, float, float, str | None, float, float] | None:
        market_total = self._extract_total_line(text)
        if market_total is None or game.total_line is None:
            return None
        if abs(game.total_line - market_total) > self.cfg.max_total_line_gap:
            return None
        yes_represents_over = "under" not in text or "over" in text
        line_signal = (game.total_line - market_total) + (game.total_move_points * TOTAL_MOVE_PROB_MULTIPLIER)
        probability_over = _sigmoid(line_signal / max(self.cfg.total_points_scale, 1.0))
        modeled_yes = probability_over if yes_represents_over else 1.0 - probability_over
        yes_price = market.outcomes[0].price
        no_price = market.outcomes[1].price
        fee_yes = self._fee_buffer(yes_price)
        fee_no = self._fee_buffer(no_price)
        yes_edge = modeled_yes - yes_price - fee_yes
        no_edge = (1.0 - modeled_yes) - no_price - fee_no
        edge_floor = max(self.cfg.min_edge, self.cfg.total_prob_edge_floor)
        if max(yes_edge, no_edge) < edge_floor:
            return None
        if yes_edge >= no_edge:
            return SignalAction.BUY_YES, modeled_yes, yes_edge, market.outcomes[0].token_id, yes_price, line_signal
        return SignalAction.BUY_NO, modeled_yes, no_edge, market.outcomes[1].token_id, no_price, line_signal

    def _score_spread_market(
        self,
        game: SportsGameSnapshot,
        market: Market,
        text: str,
    ) -> tuple[SignalAction, float, float, str | None, float, float] | None:
        subject = self._subject_team(text, game)
        market_spread = self._extract_spread_line(text)
        if subject not in {"home", "away"} or market_spread is None:
            return None
        book_spread = game.home_spread if subject == "home" else game.away_spread
        open_spread = game.open_home_spread if subject == "home" else game.open_away_spread
        if book_spread is None:
            return None
        if abs(book_spread - market_spread) > self.cfg.max_spread_line_gap:
            return None
        movement = 0.0
        if open_spread is not None:
            movement = book_spread - open_spread
        line_signal = (market_spread - book_spread) + (movement * SPREAD_MOVE_PROB_MULTIPLIER)
        modeled_yes = _sigmoid(line_signal / max(self.cfg.spread_points_scale, 1.0))
        yes_price = market.outcomes[0].price
        no_price = market.outcomes[1].price
        fee_yes = self._fee_buffer(yes_price)
        fee_no = self._fee_buffer(no_price)
        yes_edge = modeled_yes - yes_price - fee_yes
        no_edge = (1.0 - modeled_yes) - no_price - fee_no
        edge_floor = max(self.cfg.min_edge, self.cfg.spread_prob_edge_floor)
        if max(yes_edge, no_edge) < edge_floor:
            return None
        if yes_edge >= no_edge:
            return SignalAction.BUY_YES, modeled_yes, yes_edge, market.outcomes[0].token_id, yes_price, line_signal
        return SignalAction.BUY_NO, modeled_yes, no_edge, market.outcomes[1].token_id, no_price, line_signal

    @staticmethod
    def _market_type(text: str) -> str | None:
        if "points" in text or "over/under" in text or "total" in text:
            return "total"
        if "spread" in text or "cover" in text:
            return "spread"
        if "win" in text or "beat" in text:
            return "winner"
        return None

    def _subject_team(self, text: str, game: SportsGameSnapshot) -> str | None:
        candidates: list[tuple[int, str]] = []
        for side, aliases in (
            ("home", self._aliases_for_team(game.home_team, game.home_abbrev)),
            ("away", self._aliases_for_team(game.away_team, game.away_abbrev)),
        ):
            indices = [index for alias in aliases if (index := _alias_index(text, alias)) is not None]
            if indices:
                candidates.append((min(indices), side))
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1] if candidates else None

    @staticmethod
    def _extract_total_line(text: str) -> float | None:
        match = TOTAL_PATTERN.search(text)
        return float(match.group(1)) if match else None

    @staticmethod
    def _extract_spread_line(text: str) -> float | None:
        match = SPREAD_PATTERN.search(text)
        if not match:
            return None
        value = match.group(1) or match.group(2)
        return float(value) if value is not None else None

    @staticmethod
    def _aliases_for_team(team_name: str, abbrev: str) -> set[str]:
        normalized_name = _normalize_text(team_name)
        aliases = set(NBA_TEAM_ALIASES.get(normalized_name, {normalized_name}))
        if abbrev and len(abbrev.strip()) >= 4:
            aliases.add(_normalize_text(abbrev))
        return {alias for alias in aliases if alias and len(alias) >= 4 and alias not in AMBIGUOUS_TEAM_ALIASES}

    @staticmethod
    def _fee_buffer(price: float) -> float:
        return max(0.005, float(price) * FEE_BUFFER_RATE)

    @staticmethod
    def _book_reasoning(game: SportsGameSnapshot, market_type: str, subject: str | None) -> str:
        if market_type == "winner":
            probability = game.home_win_prob if subject == "home" else game.away_win_prob
            return f"win_prob={probability:.3f}" if probability is not None else "win_prob=unknown"
        if market_type == "total":
            return f"total={game.total_line:.1f} move={game.total_move_points:+.1f}" if game.total_line is not None else "total=unknown"
        spread = game.home_spread if subject == "home" else game.away_spread
        if spread is None:
            return "spread=unknown"
        return f"spread={spread:+.1f} move={game.spread_move_points:+.1f}"

    def _append_scan_log(self, signals: list[Signal], *, games: list[SportsGameSnapshot]) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": self.name,
            "enabled": self.enabled,
            "games_loaded": len(games),
            "candidate_markets": self._stats.get("candidate_markets", 0),
            "matched_markets": self._stats.get("matched_markets", 0),
            "scored_markets": self._stats.get("scored_markets", 0),
            "signals": len(signals),
            "line_move_candidates": self._stats.get("line_move_candidates", 0),
        }
        try:
            with open(self.log_path, "a") as handle:
                handle.write(json.dumps(entry) + "\n")
            self._stats["log_entries"] += 1
            self._stats["last_log_at"] = entry["timestamp"]
        except Exception as exc:
            logger.warning("[SPORTS_NBA] Failed to write sports sleeve log: %s", exc)
