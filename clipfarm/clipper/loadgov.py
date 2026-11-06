"""How many files the catalogue reads at once: as many as this Mac can spare right now.

The first pass of the catalogue (frames and picture embeddings) is mostly waiting on a drive and on ffmpeg, so several files at once is
several times faster. But the Mac is also the person's computer: it should use everything when they are away, back off while they work
(or watch a shot play in FirstCut), and get out of the way entirely when something else heavy (a game) needs it.

  Governor(mode).target()  -> how many files may be in flight now (0 = paused)
  Governor.note()          -> the same in plain words, for the page ("Using the whole Mac while you're away")

Modes (Settings > Footage library): "adapt" (default), "gentle" (one file at a time), "full" (always the most).

Readings come every ~2 s from macOS itself, no packages: `top` (CPU, ours counted apart from everyone else's), IOHIDSystem (seconds since the
last key or mouse), IOAccelerator (GPU use), `pmset` (battery, low power mode, thermal throttling) and `lsappinfo` (frontmost app). The page
also tells us when a video is playing or being scrubbed (touch()), since watching a shot involves no keyboard or mouse for a while.
Going down is fast (a couple of samples); going up is slow (one more worker every few seconds, after a pause), so it never flaps.
"""
from __future__ import annotations

import os
import re
from collections import deque
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

from ..core.log import get_logger

log = get_logger("clipper.loadgov")

MODES = ("adapt", "gentle", "full")
MAX_WORKERS = 8                 # measured on an M2 Max (8 fast cores): files a minute grow in step with the workers up to the fast cores, then flatten
PROBE_S = 45.0                  # a worker count is judged on this long at it: did one more worker read more files a minute?
GAIN = 1.10                     # ... it has to read at least 10% more, or the drive (not the CPU) is the limit and it stays one lower
CEILING_S = 600.0               # how long a limit found that way is believed before it tries one more again
CORES_PER_WORKER = 2.0          # a 4K camera file's decode takes about two cores' worth while it runs
PLAYING_S = 10.0                # the page said a shot is playing: that long since its last message counts as still playing
OUR_APPS = ("firstcut", "video builder")

NOTES = {                       # what the person reads (Find footage's status line)
    "away": "Using the whole Mac while you're away",
    "working": "Using a little: you're working",
    "playing": "Using a little: a shot is playing",
    "busy": "Using a little: something else is busy",
    "battery": "Using a little: on battery",
    "hot": "Using a little: this Mac is warm",
    "game": "Paused: another app needs the Mac",
    "gentle": "Reading one file at a time",
    "full": "Using the whole Mac",
}


# ------------------------------------------------------------------------------------------------------------------ readings
@dataclass
class Reading:
    idle_s: float = 1e9             # seconds since the last key or mouse movement
    busy: float = 0.0               # fraction of the whole CPU in use (0-1)
    ours: float = 0.0               # of which FirstCut's own processes (0-1 of the whole CPU)
    gpu: float = 0.0                # GPU device utilisation, 0-100
    on_battery: bool = False
    low_power: bool = False
    hot: bool = False               # the Mac is slowing itself down
    front_ours: bool = False        # FirstCut is the app in front
    app_active: bool = False        # the page says a shot is playing / scrubbing

    @property
    def other(self) -> float:
        """The share of the whole CPU used by everything that is not us."""
        return max(0.0, self.busy - self.ours)


def perf_cores() -> int:
    for key in ("hw.perflevel0.logicalcpu", "hw.physicalcpu", "hw.ncpu"):
        try:
            n = int(subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=3).stdout.strip())
            if n > 0:
                return n
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
    return os.cpu_count() or 4


def worker_cap(cores: int | None = None) -> int:
    """The most files worth reading at once on this Mac: its fast cores, never above what the drive can feed (MAX_WORKERS)."""
    return max(2, min(MAX_WORKERS, int(cores or perf_cores())))


def _run(cmd: list[str], timeout: float = 6.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _our_tree() -> set[int]:
    me = os.getpid()
    kids: dict[int, list[int]] = {}
    for line in _run(["ps", "-A", "-o", "pid=,ppid="]).splitlines():
        p = line.split()
        if len(p) == 2 and p[0].isdigit() and p[1].isdigit():
            kids.setdefault(int(p[1]), []).append(int(p[0]))
    out, todo = {me}, [me]
    while todo:
        for c in kids.get(todo.pop(), []):
            if c not in out:
                out.add(c)
                todo.append(c)
    return out


def parse_top(text: str, ncpu: int, tree: set[int]) -> tuple[float, float]:
    """(busy, ours) from the LAST sample of `top -l 2 ... -stats pid,cpu`: the whole CPU in use (0-1), and our processes' part of it."""
    last = text.rsplit("Processes:", 1)[-1]
    m = re.search(r"CPU usage:\s*([\d.]+)% user,\s*([\d.]+)% sys,\s*([\d.]+)% idle", last)
    busy = max(0.0, min(1.0, 1.0 - float(m.group(3)) / 100.0)) if m else 0.0
    ours = 0.0
    for line in last.splitlines():
        p = line.split()
        if len(p) == 2 and p[0].isdigit() and int(p[0]) in tree:
            try:
                ours += float(p[1])
            except ValueError:
                pass
    return busy, min(busy, ours / 100.0 / max(1, ncpu))


class SystemSampler:
    """Reads the Mac. `sample()` blocks for about a second and a half (top needs two samples to show a rate)."""

    def __init__(self):
        self.ncpu = os.cpu_count() or 4
        self._slow = (0.0, False, False, False)          # (when, on_battery, low_power, hot): pmset changes rarely, so every 10 s
        self.touched = 0.0                               # the page's last 'a shot is playing' message (monotonic seconds)

    def touch(self):
        self.touched = time.monotonic()

    def sample(self) -> Reading:
        r = Reading()
        busy_ours = _run(["top", "-l", "2", "-s", "1", "-n", "40", "-o", "cpu", "-stats", "pid,cpu"], timeout=8)
        r.busy, r.ours = parse_top(busy_ours, self.ncpu, _our_tree())
        m = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', _run(["ioreg", "-c", "IOHIDSystem", "-d", "4"]))
        r.idle_s = int(m.group(1)) / 1e9 if m else 1e9
        gp = re.findall(r'"Device Utilization %"\s*=\s*(\d+)', _run(["ioreg", "-r", "-d", "1", "-w0", "-c", "IOAccelerator"]))
        r.gpu = float(max((int(x) for x in gp), default=0))
        now = time.monotonic()
        if now - self._slow[0] > 10:
            batt = _run(["pmset", "-g", "batt"])
            therm = _run(["pmset", "-g", "therm"])
            low = re.search(r"lowpowermode\s+1", _run(["pmset", "-g"]))
            lim = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", therm)
            self._slow = (now, "Battery Power" in batt, bool(low), bool(lim and int(lim.group(1)) < 100))
        r.on_battery, r.low_power, r.hot = self._slow[1:]
        front = _run(["lsappinfo", "front"]).strip()
        name = _run(["lsappinfo", "info", "-only", "name", front]).lower() if front else ""
        r.front_ours = any(a in name for a in OUR_APPS)
        r.app_active = now - self.touched < PLAYING_S
        return r


# ------------------------------------------------------------------------------------------------------------------ the decision
def decide(r: Reading, cap: int, ncpu: int) -> tuple[int, str]:
    """How many files may be read at once right now, and why (a key of NOTES). 0 = something else needs the whole Mac."""
    foreign_heavy = (r.other >= 0.5 or r.gpu >= 70) and not r.front_ours and r.idle_s < 300     # (a game, or a render in front of someone)
    if foreign_heavy:
        return 0, "game"
    if r.hot:
        return 1, "hot"
    if r.on_battery or r.low_power:
        return 1, "battery"
    if r.app_active:
        return 1, "playing"
    if r.idle_s < 5:
        n, why = (1 if r.front_ours else 2), "working"
    elif r.idle_s < 60:
        n, why = 2, "working"
    elif r.idle_s < 180:
        n, why = max(3, cap // 2), "away"
    else:
        n, why = cap, "away"
    room = max(1, int(ncpu * (1.0 - r.other) / CORES_PER_WORKER + 0.5))    # what the others leave free, two cores to a worker
    if room < n:
        n, why = room, "busy"
    return max(1, min(cap, n)), why


class Governor:
    """Turns readings into a worker count that goes down fast and up slowly. `target()` is read by the catalogue between files."""

    def __init__(self, mode: str = "adapt", sampler=None, cap: int | None = None, ncpu: int | None = None, clock=time.monotonic,
                 interval: float = 0.5, down_samples: int = 2, up_hold: float = 10.0, up_step: float = 5.0, resume_hold: float = 20.0):
        self.mode = mode if mode in MODES else "adapt"
        self.cap = int(cap) if cap else worker_cap()
        self.ncpu = int(ncpu) if ncpu else (os.cpu_count() or 4)
        self.clock, self.interval = clock, interval
        self.down_samples, self.up_hold, self.up_step, self.resume_hold = down_samples, up_hold, up_step, resume_hold
        self.fixed = None
        if self.mode == "gentle":
            self.fixed = 1
        elif self.mode == "full":
            self.fixed = self.cap
        elif sampler is None and os.environ.get("PYTEST_CURRENT_TEST"):
            self.fixed = min(3, self.cap)                       # (tests never sample the real machine)
        self.sampler = sampler if sampler is not None else (SystemSampler() if self.fixed is None else None)
        self._target = self.fixed if self.fixed is not None else 1
        self._why = {"gentle": "gentle", "full": "full"}.get(self.mode, "working")
        self._up_since: float | None = None                    # when the Mac first had room for one more (it must keep having it)
        self._hold = up_hold
        self._done: deque = deque(maxlen=2000)                  # when each file finished (the catalogue calls note_done())
        self._level_since = clock()                             # when the current worker count was reached
        self._rates: dict[int, float] = {}                      # files a second measured at each count (cleared when it goes down)
        self._ceiling: int | None = None                        # a count one more than which read no more files a minute (the drive is the limit)
        self._ceiling_until = 0.0
        self._game_n = 0
        self._resume_since: float | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- what the catalogue asks
    def target(self) -> int:
        return self._target

    def note(self) -> str:
        return NOTES.get(self._why, "")

    def note_done(self) -> None:
        """A file was finished: how fast it reads is how the right count is found."""
        self._done.append(self.clock())

    def _rate(self, window: float) -> tuple[float, int]:
        now = self.clock()
        n = sum(1 for t in self._done if now - t <= window)
        return n / window, n

    def touch(self) -> None:
        """The page says a shot is playing or being scrubbed: back off for a while."""
        if self.sampler is not None and hasattr(self.sampler, "touch"):
            self.sampler.touch()

    # -- the state machine (called by the thread, and directly by tests)
    def update(self, r: Reading) -> int:
        with self._lock:
            if self.fixed is not None:
                return self._target
            now = self.clock()
            want, why = decide(r, self.cap, self.ncpu)
            if self._ceiling is not None:
                if now < self._ceiling_until:
                    want = min(want, self._ceiling)
                else:
                    self._ceiling = None                        # (long enough: it tries one more again)
            if why == "game":
                self._game_n += 1
                if self._game_n < self.down_samples:            # one reading is not enough to stop: back off to one file first
                    want, why = 1, "busy"
            else:
                self._game_n = 0
            cur = self._target
            if cur == 0:                                        # paused: only resume once the others have been quiet for a while
                if want == 0:
                    self._resume_since = None
                else:
                    self._resume_since = self._resume_since if self._resume_since is not None else now
                    if now - self._resume_since >= self.resume_hold:
                        self._set(1, why, now, hold=self.up_hold)
                        self._resume_since = None
                    else:
                        self._why = "game"
            elif want < cur:
                self._up_since = None
                self._set(want, why, now, hold=self.up_hold)    # down: at once
            elif want > cur:
                if self._up_since is None:
                    self._up_since = now
                if now - self._up_since >= self._hold and self._worth_more(cur, now):    # up: one more every few seconds, while there is room
                    self._up_since = now
                    self._set(cur + 1, why, now, hold=self.up_step)
            else:
                self._up_since = None                           # (no room for more right now: the wait starts again)
                self._why = why
            return self._target

    def _worth_more(self, cur: int, now: float) -> bool:
        """From three workers up: judge the count we are at before adding one. If it read no more files a minute than one fewer did, the
        drive (not the CPU) is the limit: go back to that smaller count and keep to it for a while."""
        if cur < 3 or not self._done:                           # (below three nothing is judged; with no file finished yet there is nothing to judge by)
            return True
        dwell = now - self._level_since
        rate, n = self._rate(PROBE_S)
        if dwell < PROBE_S or (n < 6 and dwell < 3 * PROBE_S):
            return False                                        # (not enough evidence yet)
        before = self._rates.get(cur - 1)
        if before is not None and rate < before * GAIN:
            self._ceiling, self._ceiling_until = cur - 1, now + CEILING_S
            log.info(f"catalogue: {cur} files at once read no more than {cur - 1} ({rate * 60:.0f} vs {before * 60:.0f} a minute): staying at {cur - 1}")
            self._set(cur - 1, "busy", now, hold=self.up_hold, keep_rates=True)
            return False
        self._rates[cur] = rate
        return True

    def _set(self, n: int, why: str, now: float, hold: float, keep_rates: bool = False) -> None:
        if n != self._target:
            log.info(f"catalogue workers {self._target} -> {n} ({why})")
            self._level_since = now
            if n < self._target and not keep_rates:
                self._rates.clear()                             # (the Mac changed: what was measured no longer holds)
        self._target, self._why, self._hold = n, why, hold

    # -- the sampling thread
    def start(self) -> "Governor":
        if self.sampler is None or self._thread is not None:
            return self
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                try:
                    self.update(self.sampler.sample())
                except Exception as e:  # noqa: BLE001 - a failed reading keeps the last decision
                    log.info(f"load reading failed: {str(e)[:120]}")
                    self._stop.wait(2.0)
                self._stop.wait(self.interval)
        self._thread = threading.Thread(target=loop, name="catalog-load", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None:
            t.join(timeout=10)


def which_tools_missing() -> list[str]:
    return [t for t in ("top", "ioreg", "pmset", "lsappinfo") if not shutil.which(t)]
