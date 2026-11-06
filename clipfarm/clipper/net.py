"""Is this Mac online? (The user, 2026-11-05: "make sure everything works properly when it's offline. It realizes it's offline.")

One cheap answer shared by the whole app, refreshed at most every CHECK_S seconds: a 1.5-second TCP connection to a couple of
well-known hosts (no DNS needed for the second). What offline changes:
    - editing decisions go to this Mac's model when it is downloaded (localai), instead of failing call after call;
    - describing footage stays on this Mac;
    - speech to text (Parakeet), footage search and music search load their models from disk only (no update check that hangs);
    - the page says so, and hides what needs the internet (stock, online music, links) behind a plain "you're offline".

    online() -> bool        cached
    changed() -> None       call after a network error: the next online() checks again at once
"""

from __future__ import annotations

import socket
import threading
import time

CHECK_S = 15.0
HOSTS = (("api.anthropic.com", 443), ("1.1.1.1", 443), ("8.8.8.8", 53))

_lock = threading.Lock()
_state = {"at": 0.0, "online": True}


def _reach(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:  # noqa: BLE001 - any failure is "not that way"
        return False


def online(max_age: float = CHECK_S) -> bool:
    with _lock:
        if time.time() - _state["at"] < max_age:
            return _state["online"]
    ok = any(_reach(h, p) for h, p in HOSTS)
    with _lock:
        _state.update(at=time.time(), online=ok)
    return ok


def peek() -> bool:
    """The last answer, without checking (for the page's frequent status request)."""
    return _state["online"]


def changed() -> None:
    with _lock:
        _state["at"] = 0.0


NET_ERROR = ("getaddrinfo", "enotfound", "econnrefused", "econnreset", "etimedout", "network", "unable to connect", "fetch failed",
             "connection error", "connection refused", "name or service not known", "nodename nor servname", "internet", "offline",
             "timed out", "no route to host", "could not reach", "couldn't reach", "can't reach", "cannot reach")


def looks_offline(err) -> bool:
    """An error that smells of a missing connection (then online() is asked again)."""
    s = str(err).lower()
    return any(k in s for k in NET_ERROR)
