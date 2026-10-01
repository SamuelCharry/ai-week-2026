"""Cliente HTTP cortés: user-agent, rate-limit por host, retries con backoff."""
from __future__ import annotations

import logging
import time
import urllib.parse
from collections import defaultdict
from pathlib import Path

import requests

log = logging.getLogger("http")


class PoliteHttp:
    def __init__(self, user_agent: str, rate_limit: float = 1.0, retries: int = 3,
                 backoff: float = 2.0, timeout: int = 30, cache_dir: Path | None = None):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Language": "es-CO,es;q=0.9"})
        self.rate = rate_limit
        self.retries = retries
        self.backoff = backoff
        self.timeout = timeout
        self._last_hit: dict[str, float] = defaultdict(float)
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _host(self, url: str) -> str:
        return urllib.parse.urlparse(url).netloc

    def get(self, url: str, cache_key: str | None = None) -> requests.Response | None:
        if cache_key and self.cache_dir:
            cache_path = self.cache_dir / cache_key
            if cache_path.is_file():
                r = requests.Response()
                r._content = cache_path.read_bytes()
                r.status_code = 200
                r.url = url
                return r
        host = self._host(url)
        delay = self.rate - (time.monotonic() - self._last_hit[host])
        if delay > 0:
            time.sleep(delay)
        wait = 1.0
        for attempt in range(1, self.retries + 1):
            try:
                r = self.session.get(url, timeout=self.timeout, allow_redirects=True)
                self._last_hit[host] = time.monotonic()
                if r.status_code == 200:
                    if cache_key and self.cache_dir:
                        (self.cache_dir / cache_key).write_bytes(r.content)
                    return r
                if 500 <= r.status_code < 600:
                    log.warning("HTTP %s en %s (intento %s)", r.status_code, url, attempt)
                else:
                    log.warning("HTTP %s definitivo en %s", r.status_code, url)
                    return None
            except requests.RequestException as exc:
                log.warning("Error red %s (intento %s): %s", url, attempt, exc)
            time.sleep(wait)
            wait *= self.backoff
        log.error("Agoté reintentos en %s", url)
        return None
