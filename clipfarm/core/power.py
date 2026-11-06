"""Power-source check: heavy work only runs on AC power (config power.require_ac_for)."""

from __future__ import annotations

import subprocess

from .log import get_logger

log = get_logger("power")


def parse_pmset_ps(text: str) -> bool | None:
    """True = AC, False = battery, None = unknown. Input: output of `pmset -g ps`."""
    first = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    if "AC Power" in first:
        return True
    if "Battery Power" in first:
        return False
    return None


def on_ac_power() -> bool:
    """Desktops always report AC. If the state can't be read we allow the run (and log it)."""
    try:
        out = subprocess.run(["pmset", "-g", "ps"], capture_output=True, text=True, timeout=10).stdout
    except Exception as e:  # noqa: BLE001
        log.warning(f"could not read power source ({e}); assuming AC")
        return True
    state = parse_pmset_ps(out)
    if state is None:
        log.warning("unknown power source; assuming AC")
        return True
    return state


def required_for(cfg, task: str) -> bool:
    return task in (cfg.get("power.require_ac_for", ["night"]) or [])


def allowed(cfg, task: str) -> bool:
    """False when `task` requires AC power and the Mac is on battery."""
    return not required_for(cfg, task) or on_ac_power()


def parse_adapter_watts(text: str) -> int | None:
    """Watts of the connected charger from `pmset -g adapter` ("Wattage = 140W"); None when nothing is connected or it cannot be read."""
    import re
    m = re.search(r"Wattage\s*=\s*(\d+)\s*W", text or "")
    return int(m.group(1)) if m else None


def adapter_watts() -> int | None:
    try:
        out = subprocess.run(["pmset", "-g", "adapter"], capture_output=True, text=True, timeout=10).stdout
    except Exception as e:  # noqa: BLE001
        log.warning(f"could not read the charger ({e})")
        return None
    return parse_adapter_watts(out)


def in_window(cfg, now=None) -> bool:
    """True inside power.heavy_window (["00:30", "06:00"] = from half past midnight to six; a window that crosses midnight works too). No window set = always."""
    from datetime import datetime
    w = cfg.get("power.heavy_window")
    if not w:
        return True
    now = now or datetime.now()
    try:
        sh, sm = (int(x) for x in str(w[0]).split(":"))
        eh, em = (int(x) for x in str(w[1]).split(":"))
        if not (0 <= sh < 24 and 0 <= eh < 24 and 0 <= sm < 60 and 0 <= em < 60):
            raise ValueError
    except (ValueError, IndexError, TypeError):                    # a malformed window means "no window"
        return True
    t, a, b = now.hour * 60 + now.minute, sh * 60 + sm, eh * 60 + em
    return a <= t < b if a <= b else (t >= a or t < b)


def heavy_ok(cfg, now=None) -> tuple[bool, str]:
    """May a long, power-hungry job (cartoon renders, explainer builds) run on THIS Mac right now? On AC, with a charger strong enough to keep up with a full
    load (power.min_adapter_watts: a weak dock cannot, and the Mac would drain and shut down mid-render), inside the window. -> (ok, why not)."""
    if not in_window(cfg, now):
        return False, f"outside the heavy-work window {cfg.get('power.heavy_window')}"
    if not on_ac_power():
        return False, "on battery power"
    need = int(cfg.get("power.min_adapter_watts", 0) or 0)
    if need:
        w = adapter_watts()
        if w is None:
            return False, "no charger reading (not plugged in?)"
        if w < need:
            return False, f"the charger is {w} W, below the {need} W this machine needs for full load"
    return True, "ok"


# ----------------------------------------------------------------------------------------- governor
# At night the Mac builds as many videos at once as it can; when the battery runs low it backs off to what the charger can carry (so the battery climbs again),
# and below a hard floor it pauses everything until it has recharged.
def parse_battery(text: str) -> dict:
    """From `ioreg -rn AppleSmartBattery -w0`: {"pct", "ac", "adapter_w", "load_w"} (load_w = what the whole Mac draws right now); missing values are None."""
    import re

    def num(pat):
        m = re.search(pat, text or "")
        return float(m.group(1)) if m else None
    ext = re.search(r'"ExternalConnected"\s*=\s*(Yes|No)', text or "")
    load = num(r'"SystemLoad"=(\d+)')
    return {"pct": num(r'"CurrentCapacity"\s*=\s*(\d+)'), "ac": (ext.group(1) == "Yes") if ext else None,
            "adapter_w": num(r'"AdapterDetails"\s*=\s*\{[^}]*?"Watts"=(\d+)'), "load_w": round(load / 1000, 1) if load is not None else None}


def battery_state() -> dict:
    try:
        out = subprocess.run(["ioreg", "-rn", "AppleSmartBattery", "-w0"], capture_output=True, text=True, timeout=15).stdout
    except Exception as e:  # noqa: BLE001
        log.warning(f"could not read the battery ({e})")
        return {"pct": None, "ac": None, "adapter_w": None, "load_w": None}
    return parse_battery(out)


class Governor:
    """decide(state, running) -> {"workers": how many builds may run, "paused": freeze them all, "mode": full|recover|paused|battery}.
    full: up to power.max_builds. At or below power.low_battery (25 %) it drops to recover: only as many builds as the charger can carry (adapter watts minus
    power.headroom_w, judged from the measured load per build) until the battery is back to power.resume_battery (60 %). At or below power.pause_battery (10 %) everything is
    paused (and resumes once recovered). On battery power nothing runs."""

    def __init__(self, cfg):
        g = lambda k, d: float(cfg.get(f"power.{k}", d) or d)  # noqa: E731
        self.max_builds, self.low, self.resume, self.pause = int(g("max_builds", 4)), g("low_battery", 25), g("resume_battery", 60), g("pause_battery", 10)
        self.headroom = g("headroom_w", 6)
        self.mode = "full"

    def decide(self, st: dict, running: int) -> dict:
        pct, ac, adapter, load = st.get("pct"), st.get("ac"), st.get("adapter_w"), st.get("load_w")
        if ac is False:
            self.mode = "battery"
            return {"workers": 0, "paused": True, "mode": "battery"}
        if pct is None:
            return {"workers": min(self.max_builds, 1), "paused": False, "mode": self.mode}
        if self.mode in ("battery", "paused") and pct >= self.resume:
            self.mode = "full"
        if self.mode == "full" and pct <= self.low:
            self.mode = "recover"
        if self.mode == "recover" and pct >= self.resume:
            self.mode = "full"
        if pct <= self.pause:
            self.mode = "paused"
        if self.mode == "paused":
            return {"workers": 0, "paused": True, "mode": "paused"}
        if self.mode == "full":
            return {"workers": self.max_builds, "paused": False, "mode": "full"}
        base = 9.0                                                         # the Mac's own draw with nothing building (watts, rough)
        per = max(8.0, ((load or base) - base) / running) if running else 18.0
        room = (adapter or 60) - self.headroom - base
        return {"workers": max(1, min(self.max_builds, int(room // per))), "paused": False, "mode": "recover"}
