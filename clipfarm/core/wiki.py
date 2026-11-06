"""Wikipedia API client for grounded true stories (free, permitted, CC BY-SA content).

We never copy article text into videos: the LLM writes an original retelling using only
facts from the extract, and the caption credits the article. Wikimedia asks API clients
to send a descriptive User-Agent with contact info - set WIKI_USER_AGENT in .env.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import requests

from .log import get_logger
from .state import atomic_write_json, read_json

log = get_logger("wiki")
API = "https://en.wikipedia.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"


def _ua() -> str:
    return os.environ.get("WIKI_USER_AGENT") or "Clipfarm/0.1 (personal short-video pipeline; python-requests)"


class _Busy(Exception):
    """Wikipedia answered 429/503 and said how long to wait."""
    def __init__(self, msg: str, wait: float):
        super().__init__(msg)
        self.wait = wait


def _retry_after(r, default: float = 30.0) -> float:
    try:
        return min(120.0, max(5.0, float((getattr(r, "headers", None) or {}).get("retry-after", default))))
    except ValueError:
        return default


def api(params: dict, endpoint: str | None = None) -> dict:
    """The Wikipedia API; when it is busy (429/503) wait as long as it asks, up to about five minutes in all, instead of failing a build after 9 seconds."""
    for k in range(6):
        try:
            r = requests.get(endpoint or API, params={**params, "format": "json", "formatversion": 2}, headers={"User-Agent": _ua()}, timeout=30)
            if getattr(r, "status_code", 200) in (429, 503):
                raise _Busy(f"{r.status_code} from {str(getattr(r, 'url', ''))[:80]}", _retry_after(r))
            r.raise_for_status()
            return r.json()
        except _Busy as e:
            if k == 5:
                raise RuntimeError(f"Wikipedia keeps saying it is busy ({e})") from e
            log.warning(f"wikipedia api busy ({e}); waiting {e.wait:.0f}s ({k + 1}/5)")
            time.sleep(e.wait)
        except (requests.ConnectionError, requests.Timeout) as e:
            if k >= 3:
                raise
            time.sleep(3 * (k + 1))
    raise RuntimeError("unreachable")


_LAST_COMMONS = [0.0]


def api_commons(params: dict) -> dict:
    """Commons is strict about bursts: one call at a time, a pause between them, and a longer wait when it says 'too many requests'."""
    for k in range(5):
        wait = 0.8 - (time.time() - _LAST_COMMONS[0])
        if wait > 0:
            time.sleep(wait)
        _LAST_COMMONS[0] = time.time()
        r = requests.get(COMMONS_API, params={**params, "format": "json", "formatversion": 2}, headers={"User-Agent": _ua()}, timeout=30)
        if r.status_code in (429, 503):
            time.sleep(10 * (k + 1) if "retry-after" not in (getattr(r, "headers", None) or {}) else max(10 * (k + 1), _retry_after(r)))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Wikimedia Commons keeps answering 'too many requests' or 'busy'")


def page_links(page: str, cache: Path | None = None, max_age_days: float = 7) -> list[str]:
    """Article (namespace 0) links on a page, cached on disk."""
    if cache:
        c = read_json(cache)
        if c and c.get("page") == page and time.time() - c.get("at", 0) < max_age_days * 86400:
            return c["titles"]
    data = api({"action": "parse", "page": page, "prop": "links", "redirects": 1})
    titles = sorted({ln["title"] for ln in data.get("parse", {}).get("links", [])
                     if ln.get("ns") == 0 and ln.get("exists", True)})
    if cache and titles:
        atomic_write_json(cache, {"page": page, "at": time.time(), "titles": titles})
    return titles


def extract(title: str, max_chars: int = 12000) -> dict | None:
    data = api({"action": "query", "prop": "extracts|info", "explaintext": 1, "inprop": "url",
                "titles": title, "redirects": 1})
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return None
    p = pages[0]
    text = (p.get("extract") or "").strip()
    for cut in ("\n== See also ==", "\n== References ==", "\n== Notes ==", "\n== External links =="):
        text = text.split(cut)[0]
    return {"title": p.get("title", title), "url": p.get("fullurl", f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"),
            "text": text[:max_chars]}
