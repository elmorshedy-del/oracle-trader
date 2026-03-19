from __future__ import annotations

import ssl
import urllib.request
from functools import lru_cache


@lru_cache(maxsize=1)
def build_ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def urlopen_with_default_context(
    request: urllib.request.Request,
    *,
    timeout: float,
):
    return urllib.request.urlopen(
        request,
        timeout=timeout,
        context=build_ssl_context(),
    )
