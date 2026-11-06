"""How many files the catalogue reads at once (loadgov.py): the decision for idle / working / watching / a game / battery / hot, going down
fast and up slowly, the three modes, the first pass honouring a falling and rising target without losing a file, pausing and resuming, the
setting, and reading the real machine. All of it offline: the readings are faked.

Measured on an M2 Max (8 fast cores, files on the internal SSD) with synthetic 4K 10-bit 4:2:2 H.264 clips, keyframes-only frame grabs as the
catalogue does them (scripts in the session scratchpad: files a minute at 1/2/3/4/6/8 at once): see MAX_WORKERS in loadgov.py for the cap chosen.
"""
import threading
import time
from types import SimpleNamespace as NS

import pytest
from test_catalog import FakeEmbedder, _cfg, _photo

from clipfarm.clipper import catalog as C
from clipfarm.clipper import loadgov as L
from clipfarm.clipper import studio as SD

CAP, NCPU = 6, 12


def R(**kw) -> L.Reading:
    return L.Reading(**kw)


# ------------------------------------------------------------------------------------------------------------- the decision
def test_away_uses_everything_it_may():
    assert L.decide(R(idle_s=600, busy=0.05), CAP, NCPU) == (CAP, "away")


def test_typing_in_firstcut_reads_one_file_and_elsewhere_two():
    assert L.decide(R(idle_s=1, front_ours=True), CAP, NCPU) == (1, "working")
    assert L.decide(R(idle_s=1, front_ours=False), CAP, NCPU) == (2, "working")
    assert L.decide(R(idle_s=30), CAP, NCPU) == (2, "working")


def test_a_minute_or_two_away_is_in_between():
    n, why = L.decide(R(idle_s=100), CAP, NCPU)
    assert 3 <= n < CAP and why == "away"


def test_a_shot_playing_backs_off_even_with_no_keys_touched():
    assert L.decide(R(idle_s=900, app_active=True), CAP, NCPU) == (1, "playing")


def test_a_game_pauses_it():
    assert L.decide(R(idle_s=2, busy=0.8, ours=0.0, gpu=40), CAP, NCPU) == (0, "game")        # the CPU is taken by something else
    assert L.decide(R(idle_s=2, busy=0.2, gpu=90), CAP, NCPU) == (0, "game")                  # or the GPU
    assert L.decide(R(idle_s=2, busy=0.9, ours=0.8, gpu=10), CAP, NCPU)[0] > 0                # ... but not when it is our own load
    assert L.decide(R(idle_s=2, busy=0.8, gpu=90, front_ours=True), CAP, NCPU)[0] > 0         # FirstCut in front is not a game


def test_a_render_while_you_are_away_is_shared_not_stopped():
    n, why = L.decide(R(idle_s=900, busy=0.7), CAP, NCPU)                                     # (nobody is there: someone left a job running)
    assert n >= 1 and why in ("busy", "away")


def test_battery_low_power_and_heat_read_one_file():
    assert L.decide(R(idle_s=900, on_battery=True), CAP, NCPU) == (1, "battery")
    assert L.decide(R(idle_s=900, low_power=True), CAP, NCPU) == (1, "battery")
    assert L.decide(R(idle_s=900, hot=True), CAP, NCPU) == (1, "hot")


def test_what_others_leave_free_limits_the_workers():
    n, why = L.decide(R(idle_s=900, busy=0.4), CAP, NCPU)                                     # 60% of 12 cores free: about 4 workers of 2 cores
    assert n == 4 and why == "busy"


def test_cap_is_the_fast_cores_up_to_the_measured_limit():
    assert L.worker_cap(4) == 4
    assert L.worker_cap(6) == 6                               # (an M2 Pro: six fast cores)
    assert L.worker_cap(8) == 8 and L.worker_cap(12) == L.MAX_WORKERS == 8
    assert L.worker_cap(1) == 2


# ------------------------------------------------------------------------------------------------------------- the governor
class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def tick(self, s):
        self.t += s


def gov(mode="adapt", **kw):
    c = Clock()
    g = L.Governor(mode, sampler=NS(sample=lambda: None), cap=CAP, ncpu=NCPU, clock=c, **kw)
    return g, c


def test_it_goes_up_slowly_and_down_at_once():
    g, c = gov()
    away = R(idle_s=900)
    assert g.update(away) == 1                                # (starts at one: nothing is known yet)
    c.tick(2); assert g.update(away) == 1                     # not yet: it holds for a while first
    c.tick(9); assert g.update(away) == 2                     # then one more ...
    c.tick(2); assert g.update(away) == 2
    c.tick(4); assert g.update(away) == 3                     # ... every few seconds
    for _ in range(10):
        c.tick(5)
        g.update(away)
    assert g.target() == CAP
    assert g.update(R(idle_s=1, front_ours=True)) == 1        # you touch the keyboard: down in one reading
    assert g.note().startswith("Using a little")


def test_it_does_not_flap():
    g, c = gov()
    seq = [R(idle_s=900), R(idle_s=1), R(idle_s=900), R(idle_s=1)] * 6
    seen = []
    for r in seq:
        c.tick(2)
        seen.append(g.update(r))
    assert max(seen) <= 2                                     # (it never gets the chance to climb)


def test_a_game_takes_two_readings_to_pause_and_twenty_seconds_quiet_to_resume():
    g, c = gov()
    for _ in range(40):                                       # away: climb to the top
        c.tick(5); g.update(R(idle_s=900))
    assert g.target() == CAP
    game = R(idle_s=2, busy=0.9, gpu=95)
    assert g.update(game) == 1                                # the first reading only backs off
    assert g.update(game) == 0 and g.note() == "Paused: another app needs the Mac"
    quiet = R(idle_s=900)
    c.tick(2); assert g.update(quiet) == 0                    # the game closed, but not yet trusted
    c.tick(10); assert g.update(quiet) == 0
    c.tick(11); assert g.update(quiet) == 1                   # twenty seconds quiet: one file again, then it climbs
    assert g.update(game) == 1 or g.target() == 0             # (and a game again goes straight back)


def test_one_busy_reading_in_a_quiet_stretch_is_not_a_game():
    g, c = gov()
    for _ in range(40):
        c.tick(5); g.update(R(idle_s=900))
    g.update(R(idle_s=2, busy=0.9, gpu=95))
    c.tick(2)
    g.update(R(idle_s=900))                                   # (a spike: Spotlight, a backup)
    c.tick(2)
    assert g.update(R(idle_s=2, busy=0.9, gpu=95)) >= 1       # the count started again: one reading backs off, it does not stop


def finish(g, c, per_sec, seconds):
    """`per_sec` files finish every second for `seconds` seconds, with a reading each second."""
    for _ in range(int(seconds)):
        c.tick(1)
        for _ in range(per_sec):
            g.note_done()
        g.update(R(idle_s=900))


def test_it_stops_adding_workers_when_more_read_no_more_files():
    g, c = gov()                                              # (cap 6)
    for _ in range(30):                                       # up to three at once, quickly (nothing is judged below three)
        c.tick(5); g.update(R(idle_s=900))
        if g.target() >= 3:
            break
    assert g.target() == 3
    finish(g, c, 4, 50)                                       # three read 4 files a second ...
    assert g.target() == 4                                    # ... so it tries four
    finish(g, c, 4, 50)                                       # ... which read the same: the drive is the limit
    assert g.target() == 3 and g._ceiling == 3
    finish(g, c, 4, 120)
    assert g.target() == 3                                    # (and it stays there)
    c.tick(L.CEILING_S + 1)
    finish(g, c, 4, 60)
    assert g.target() >= 3                                    # (after a while it tries again)


def test_it_keeps_adding_workers_while_each_one_helps():
    g, c = gov()
    for _ in range(30):
        c.tick(5); g.update(R(idle_s=900))
        if g.target() >= 3:
            break
    for k in range(12):                                       # every added worker adds the same again
        finish(g, c, g.target() * 2, 50)
    assert g.target() == CAP and g._ceiling is None


def test_gentle_is_one_and_full_is_the_cap_and_neither_samples():
    g = L.Governor("gentle", cap=CAP)
    f = L.Governor("full", cap=CAP)
    assert g.target() == 1 and g.sampler is None and g.note() == "Reading one file at a time"
    assert f.target() == CAP and f.sampler is None and f.note() == "Using the whole Mac"
    assert f.update(R(idle_s=1, busy=0.9, gpu=99)) == CAP     # (full speed ignores everything)


def test_tests_never_sample_the_real_machine():
    g = L.Governor("adapt", cap=CAP)                          # (PYTEST_CURRENT_TEST is set)
    assert g.sampler is None and g.target() == 3


def test_a_page_heartbeat_reaches_the_sampler():
    s = L.SystemSampler()
    assert s.touched == 0.0
    g = L.Governor("adapt", sampler=s, cap=CAP)
    g.touch()
    assert time.monotonic() - s.touched < 2


def test_the_governor_thread_applies_readings_and_stops():
    readings = iter([R(idle_s=900)] * 100)
    g = L.Governor("adapt", sampler=NS(sample=lambda: next(readings)), cap=CAP, ncpu=NCPU, interval=0.01, up_hold=0.0, up_step=0.01)
    g.start()
    deadline = time.time() + 5
    while g.target() < CAP and time.time() < deadline:
        time.sleep(0.02)
    g.stop()
    assert g.target() == CAP and g._thread is None


def test_top_output_is_read_and_our_processes_counted_apart():
    top = ("Processes: 300 total\nCPU usage: 50.0% user, 5.0% sys, 45.0% idle\nPID    %CPU\n111    240.0\n222    55.0\n333    10.0\n"
           "Processes: 300 total\nCPU usage: 30.0% user, 10.0% sys, 60.0% idle\nPID    %CPU\n111    240.0\n222    55.0\n333    10.0\n")
    busy, ours = L.parse_top(top, 12, {111, 333})
    assert busy == pytest.approx(0.40)
    assert ours == pytest.approx(250.0 / 100 / 12)
    assert L.parse_top("garbage", 12, {1}) == (0.0, 0.0)


@pytest.mark.skipif(not L.shutil.which("top"), reason="macOS tools")
def test_the_real_machine_can_be_read():
    r = L.SystemSampler().sample()
    assert 0.0 <= r.busy <= 1.0 and 0.0 <= r.ours <= r.busy and r.idle_s >= 0
    n, why = L.decide(r, L.worker_cap(), 12)
    assert 0 <= n <= L.MAX_WORKERS and why in L.NOTES


# ------------------------------------------------------------------------------------------------------------- the first pass
class FakeGov:
    """A governor whose answer the test changes."""

    def __init__(self, n):
        self.n, self.cap, self.stopped, self.started = n, 6, False, False

    def target(self):
        return self.n

    def note(self):
        return "Paused: another app needs the Mac" if self.n == 0 else "Using a little"

    def start(self):
        self.started = True
        return self

    def stop(self):
        self.stopped = True

    def touch(self):
        pass


def library(tmp_path, n=14):
    root = tmp_path / "Drive" / "Photos"
    for k in range(n):
        _photo(root / f"p{k:02d}.jpg", (k * 15 % 255, 40, 200 - k * 10 % 200))
    cat = C.Catalog(_cfg(tmp_path, [root]), embedder=FakeEmbedder())
    cat.workers = 0
    return cat


class Watch:
    """Wraps index_one: records how many are in flight, and lets the test react to a file finishing."""

    def __init__(self, cat, after=None, pause=0.08):
        self.inflight = 0
        self.log: list[tuple[int, int]] = []             # (files finished before this one started, in flight when it started)
        self.finished = 0
        self.lock = threading.Lock()
        self.orig, self.after, self.pause = cat.index_one, after, pause
        cat.index_one = self

    def __call__(self, f):
        with self.lock:
            self.inflight += 1
            self.log.append((self.finished, self.inflight))
        try:
            time.sleep(self.pause)
            return self.orig(f)
        finally:
            with self.lock:
                self.inflight -= 1
                self.finished += 1
                n = self.finished
            if self.after:
                self.after(n)


def run(cat, gov, monkeypatch):
    monkeypatch.setattr(L, "Governor", lambda mode, **kw: gov)
    return cat.index()


def test_the_pass_follows_a_falling_then_rising_target_and_loses_nothing(tmp_path, monkeypatch):
    cat = library(tmp_path, 16)
    g = FakeGov(4)

    def after(n):
        g.n = {3: 1, 9: 3}.get(n, g.n)                    # 4 at first, one from the third file, three from the ninth
    w = Watch(cat, after)
    r = run(cat, g, monkeypatch)
    assert r["new"] == 16 and r["failed"] == 0 and r["left"] == 0
    assert cat.stats()["photos"] == 16
    early = max(b for a, b in w.log if a < 3)
    mid = max(b for a, b in w.log if 4 <= a < 9)
    late = max(b for a, b in w.log if a >= 10)
    assert early >= 3                                     # (several at once while it was allowed)
    assert mid <= 2                                       # (down to one: at most the straggler plus the next)
    assert 2 <= late <= 3                                 # (back up, never past the target)
    assert g.stopped and g.started


def test_a_target_of_zero_pauses_and_it_carries_on_when_the_mac_is_free(tmp_path, monkeypatch):
    cat = library(tmp_path, 8)
    g = FakeGov(2)
    paused = threading.Event()

    def after(n):
        if n == 2:
            g.n = 0
    w = Watch(cat, after, pause=0.05)
    out = {}
    monkeypatch.setattr(L, "Governor", lambda mode, **kw: g)
    t = threading.Thread(target=lambda: out.update(r=cat.index()))
    t.start()
    deadline = time.time() + 10
    while time.time() < deadline and not str(cat.status.get("stage", "")).startswith("Paused"):
        time.sleep(0.05)
    assert str(cat.status.get("stage", "")).startswith("Paused") and cat.status.get("workers") == 0
    started = len(w.log)
    time.sleep(1.5)
    assert len(w.log) == started and w.inflight == 0      # nothing new starts while paused
    g.n = 3
    t.join(30)
    assert out["r"]["new"] == 8 and out["r"]["left"] == 0
    assert cat.status.get("workers") == 0                 # (reset when the pass ends)


def test_stop_while_paused_ends_the_pass_at_once(tmp_path, monkeypatch):
    cat = library(tmp_path, 6)
    g = FakeGov(0)
    monkeypatch.setattr(L, "Governor", lambda mode, **kw: g)
    out = {}
    t = threading.Thread(target=lambda: out.update(r=cat.index()))
    t.start()
    time.sleep(0.6)
    cat.stop.set()
    t.join(8)
    assert not t.is_alive() and out["r"]["new"] == 0 and out["r"]["left"] == 6


def test_a_number_set_by_hand_is_exactly_that_many(tmp_path, monkeypatch):
    cat = library(tmp_path, 10)
    cat.workers = 2
    w = Watch(cat, pause=0.05)
    r = cat.index()
    assert r["new"] == 10 and max(b for _, b in w.log) == 2


def test_the_status_says_what_it_is_doing(tmp_path, monkeypatch):
    cat = library(tmp_path, 4)
    seen = []
    Watch(cat, lambda n: seen.append((cat.status.get("workers"), cat.status.get("load"))), pause=0.02)
    run(cat, FakeGov(2), monkeypatch)
    assert any(w == 2 and load == "Using a little" for w, load in seen)
    assert "workers" in cat.stats()["status"] and "load" in cat.stats()["status"]


# ------------------------------------------------------------------------------------------------------------- the setting
@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "VB"
    monkeypatch.setenv("VIDEOBUILDER_HOME", str(h))
    monkeypatch.setattr(SD, "_answers", lambda url: False)
    monkeypatch.setattr(SD, "ram_gb", lambda: 16.0)
    return h


def test_library_load_setting_is_kept_checked_and_passed_to_the_catalogue(home):
    s = SD.Settings()
    assert s.get()["library"]["load"] == "adapt"
    assert s.update({"library": {"load": "gentle"}})["library"]["load"] == "gentle"
    assert s.update({"library": {"load": "rm -rf /"}})["library"]["load"] == "gentle"      # only the three values
    assert s.update({"library": {"load": "full"}})["library"]["load"] == "full"
    assert SD.overlay(s.get(), home)["catalog"]["load"] == "full"
    cat = C.Catalog(SD.overlay(s.get(), home) and __import__("clipfarm.core.config", fromlist=["load_config"]).load_config())
    assert cat.load_mode == "adapt"                                                         # (the default when the config says nothing)


def test_the_catalogue_reads_load_from_its_config(tmp_path):
    cat = C.Catalog(_cfg(tmp_path, [], load="gentle"))
    assert cat.load_mode == "gentle" and cat.workers == 0
