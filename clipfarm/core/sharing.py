"""Sharing the M2's memory between night cartoon renders and the server's Clipper.

A Wan 2.2 14B scene uses nearly all 32 GB, so an Ollama model loading mid-scene would crash it.
The two sides take turns:

  M2 (render machine)  before each video-model scene: wait until its Ollama has no model loaded
                       (a Clipper request in flight), then tell the posting server "busy" over SSH;
                       "free" again once the scene is done.
  Server (Clipper)     uses the M2's Ollama only inside the allowed window, when it answers, and
                       while it isn't marked busy; otherwise its own Ollama.

The busy flag is a small JSON file on the server (data/state/m2_busy.json). A stale flag (the M2
crashed or slept mid-scene) expires on its own after its estimate plus a margin.
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime

import requests

from .log import get_logger

log = get_logger("sharing")
FLAG = "data/state/m2_busy.json"


# ------------------------------------------------------------------ render machine side
def ollama_loaded(base: str = "http://127.0.0.1:11434") -> list[str]:
    try:
        r = requests.get(f"{base}/api/ps", timeout=3)
        r.raise_for_status()
        return [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:  # noqa: BLE001 - Ollama not running: nothing to wait for
        return []


def wait_ollama_idle(max_wait_s: float = 1200, poll_s: float = 10, base: str = "http://127.0.0.1:11434") -> bool:
    """Block while another client (Clipper) has a model loaded. Returns True if idle."""
    t0 = time.monotonic()
    loaded = ollama_loaded(base)
    if loaded:
        log.info(f"waiting for Ollama to go idle before the next scene (loaded: {', '.join(loaded)})")
    while loaded and time.monotonic() - t0 < max_wait_s:
        time.sleep(poll_s)
        loaded = ollama_loaded(base)
    return not loaded


def _signal_host(cfg) -> str | None:
    host = cfg.get("gpu_share.signal_host")
    if host:
        return host
    target = cfg.get("handoff.target") or ""            # user@server:clipfarm/inbox/handoff/
    return target.split(":", 1)[0] if ":" in target else None


def signal_busy(cfg, busy: bool, what: str = "", est_s: float = 0) -> None:
    host = _signal_host(cfg)
    if not host or not cfg.get("gpu_share.enabled", True):
        return
    payload = json.dumps({"busy": busy, "what": what, "est_s": round(est_s), "at": time.time(),
                          "at_local": datetime.now().isoformat(timespec="seconds")})
    try:
        subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
                        f"mkdir -p clipfarm/data/state && cat > clipfarm/{FLAG}"],
                       input=payload, text=True, capture_output=True, timeout=30)
    except Exception as e:  # noqa: BLE001 - best effort; the server falls back to its own model anyway
        log.warning(f"could not signal busy={busy} to {host}: {e}")


class BusyLease:
    """with BusyLease(cfg, "scene 3", est_s): ... - wait for Ollama idle, flag busy, flag free after."""

    def __init__(self, cfg, what: str, est_s: float):
        self.cfg, self.what, self.est_s = cfg, what, est_s

    def __enter__(self):
        if self.cfg.get("gpu_share.enabled", True):
            wait_ollama_idle(float(self.cfg.get("gpu_share.max_wait_s", 1200)))
            signal_busy(self.cfg, True, self.what, self.est_s)
        return self

    def __exit__(self, *exc):
        if self.cfg.get("gpu_share.enabled", True):
            signal_busy(self.cfg, False, self.what)
        return False


# ------------------------------------------------------------------ server side
def m2_busy(root, margin_s: float = 1800) -> bool:
    p = root / FLAG
    try:
        d = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not d.get("busy"):
        return False
    return time.time() < float(d.get("at", 0)) + float(d.get("est_s", 0)) + margin_s


def in_window(now: datetime, start: str, end: str) -> bool:
    def mins(s):
        h, m = s.split(":")
        return int(h) * 60 + int(m)
    t, a, b = now.hour * 60 + now.minute, mins(start), mins(end)
    return a <= t < b if a <= b else (t >= a or t < b)
