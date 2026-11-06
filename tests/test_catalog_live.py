"""Find footage's live panel (2026-11-06: "show the clips coming in and show what clip it's currently working on"): the catalogue's
own events, in a small ring, with stills that are its own frames (nothing extra decoded or stored)."""
import threading
from types import SimpleNamespace

from test_catalog import FakeEmbedder, FakeVision, _cfg, drive  # noqa: F401 - the fixture

from clipfarm.clipper import catalog as C


def test_the_ring_keeps_the_last_events_and_answers_what_is_new(tmp_path):
    cat = C.Catalog(_cfg(tmp_path, []))
    for k in range(40):
        cat._live("done", id=f"f{k}", rel=f"B-roll/{k}.mov", k=0)
    got = cat.live_since(0)
    assert len(got["events"]) == C.LIVE_N and got["last"] == 40 and got["events"][0]["n"] == 40 - C.LIVE_N + 1
    assert [e["n"] for e in cat.live_since(38)["events"]] == [39, 40]
    assert cat.live_since(40)["events"] == [] and got["status"]["running"] is False
    cat.live = None                                                                        # a broken ring never breaks the catalogue
    cat._live("now", id="x")


def test_who_described_it_in_plain_words():
    assert C.Catalog.who("claude-haiku-5-5") == "Claude Haiku" and C.Catalog.who("claude-sonnet-5-5") == "Claude Sonnet"
    assert C.Catalog.who("qwen3-vl:8b") == "this Mac" and C.Catalog.who(None) == "this Mac"
    assert C.Catalog.who("gpt-5.4-mini") == "ChatGPT"


def test_a_real_run_reports_the_file_in_hand_and_each_finished_file_with_its_own_frame(tmp_path, drive):  # noqa: F811
    cat = C.Catalog(_cfg(tmp_path, [drive]), embedder=FakeEmbedder(), vision=FakeVision())
    r = cat.index()
    assert r["new"] >= 2
    ev = cat.live_since(0)["events"]
    framed = [e for e in ev if e["kind"] == "done" and e["step"] == "frames"]
    assert framed and all(cat.thumb(e["id"], e["k"]).exists() for e in framed)              # the still is a frame it already keeps
    assert any(e["kind"] == "now" and e["step"] == "frames" and e["k"] is None for e in ev)  # (no frame yet while reading the file)
    after = cat.live_since(0)["last"]
    d = cat.describe()
    assert d["described"] >= 1
    new = cat.live_since(after)["events"]
    told = [e for e in new if e["kind"] == "done" and e["step"] == "describe"]
    assert told and any("red wall" in " ".join(e["what"]) for e in told)
    assert all(e["by"] == "this Mac" for e in told) and all(cat.thumb(e["id"], e["k"]).exists() for e in told if e["media"] == "video")
    assert any(e["kind"] == "now" and e["step"] == "describe" for e in new)
    before = {p.name for p in (tmp_path).rglob("*.jpg")}
    cat._live("done", id="x", rel="x", k=0)
    assert {p.name for p in (tmp_path).rglob("*.jpg")} == before                             # events never write pictures


def test_the_page_route(tmp_path):
    from clipfarm.clipper import studio_web as W
    api = W.StudioAPI.__new__(W.StudioAPI)
    api.runner = SimpleNamespace(library=lambda: None)
    assert api.get(["library", "live", "0"]) == (200, {"events": [], "last": 0, "status": {"running": False}})
    cat = C.Catalog(_cfg(tmp_path, []))
    cat._live("now", id="a", rel="B-roll/a.mov", step="describe", k=3)
    cat._live("done", id="a", rel="B-roll/a.mov", step="describe", k=3, what=["a gym"], by="Claude Haiku")
    api.runner = SimpleNamespace(library=lambda: cat)
    code, body = api.get(["library", "live", "1"])
    assert code == 200 and [e["kind"] for e in body["events"]] == ["done"] and body["last"] == 2
    assert api.get(["library", "live"])[1]["last"] == 2 and len(api.get(["library", "live", "zz"])[1]["events"]) == 2


def test_events_from_two_threads_keep_their_order(tmp_path):
    cat = C.Catalog(_cfg(tmp_path, []))
    ts = [threading.Thread(target=lambda: [cat._live("done", id="x", k=0) for _ in range(200)]) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    ns = [e["n"] for e in cat.live_since(0)["events"]]
    assert cat.live_n == 800 and ns == sorted(ns) and len(set(ns)) == len(ns)


def test_new_shots_are_searchable_within_a_minute(tmp_path, monkeypatch):
    """2026-11-06: "people cataloguing tons of footage can see the footage coming in real time": the index is refreshed after the
    1st, 5th and 20th file and then at least every COMPILE_S seconds (it was every 300 files; descriptions only at the very end)."""
    from clipfarm.clipper import catalog as C
    cat = C.Catalog.__new__(C.Catalog)
    built = []
    cat.compile = lambda: built.append(1)
    t = [1000.0]
    monkeypatch.setattr(C.time, "time", lambda: t[0])
    for n in range(1, 31):
        cat.compile_soon(n)
        t[0] += 1.0
    assert len(built) == 3                                                               # 1, 5, 20 (30 s: not yet the minute)
    t[0] += C.COMPILE_S
    cat.compile_soon(31)
    assert len(built) == 4
