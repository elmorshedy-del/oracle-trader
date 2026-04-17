from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from engine.https_support import urlopen_with_default_context


logger = logging.getLogger(__name__)

TELEGRAM_API_HOST = "api.telegram.org"
TELEGRAM_SEND_TIMEOUT_SECONDS = 15


@dataclass(slots=True)
class TelegramNotifier:
    bot_token: str = ""
    chat_id: str = ""

    @classmethod
    def from_env(cls) -> "TelegramNotifier":
        return cls(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send_message(self, message: str) -> bool:
        if not self.enabled:
            return False

        payload = urllib.parse.urlencode(
            {
                "chat_id": self.chat_id,
                "text": message,
                "disable_web_page_preview": "true",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"https://{TELEGRAM_API_HOST}/bot{self.bot_token}/sendMessage",
            data=payload,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "User-Agent": "oracle-trader/telegram-notifier",
            },
            method="POST",
        )
        try:
            with urlopen_with_default_context(
                request,
                timeout=TELEGRAM_SEND_TIMEOUT_SECONDS,
            ) as response:
                json.loads(response.read())
            return True
        except urllib.error.URLError as exc:
            logger.warning("[TELEGRAM] Failed to send message: %s", exc)
            return False
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[TELEGRAM] Unexpected send failure: %s", exc)
            return False
