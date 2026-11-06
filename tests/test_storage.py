"""Storage (storage.py): the videos of old projects come off the disk, the plan stays, and Re-render makes them again from the plan alone.
Real small projects (a real make), real files, fake AI."""
import json
import os
import threading
import time
from pathlib import Path

import pytest

from clipfarm.clipper import storage as ST
from clipfarm.clipper import studio as SD
from clipfarm.clipper.engine import Engine
from clipfarm.clipper.store import Store
from clipfarm.core.config import load_config
from test_replace_shot import _latest, made  # noqa: F401 - the fixture: a real project, made once


def _st(made, days=0, fc=False, now=time.time):
    return ST.Storage(None, made["store"], made["projects"], made["runner"], settings=lambda: {"storage": {"auto_days": days, "also_finalcut": fc}}, now=now)


def _size(paths):
    return sum(p.stat().st_size for p in paths)


def _wait_idle(runner, secs=240):
    end = time.time() + secs
    while runner.running and time.time() < end:
        time.sleep(0.2)
    assert not runner.running


# ------------------------------------------------------------------------------------------------ a long video
def test_usage_counts_the_videos_the_final_cut_folder_and_never_the_plan(made):
    st, pid, folder, job = _st(made), made["proj"]["id"], made["folder"], made["proj"]["job"]
    row = next(r for r in st.usage()["rows"] if r["id"] == pid)
    out_mp4 = list((folder / "Output").glob("*.mp4"))
    job_mp4 = [p for p in made["store"].dir(job).iterdir() if p.suffix == ".mp4"]
    fc = [p for p in (folder / "Final Cut").rglob("*") if p.is_file()] + list(folder.glob("*.fcpxml"))
    assert out_mp4 and job_mp4 and fc
    assert row["sizes"]["videos"] == _size(out_mp4) + _size(job_mp4)
    assert row["sizes"]["finalcut"] == _size(fc)
    assert row["kind"] == "long" and row["can"] and row["gain"] == row["sizes"]["videos"] + row["sizes"]["working"]      # (Final Cut stays by default)
    assert st.usage()["bytes"] >= row["bytes"] > 0


def test_archive_removes_the_videos_and_keeps_the_plan_the_notes_and_the_footage(made):
    st, pid, folder, job = _st(made), made["proj"]["id"], made["folder"], made["proj"]["job"]
    plan0 = (folder / ".aivideo" / "plan.json").read_text()
    updated0 = made["projects"].get(pid)["updated"]
    st.pin("long", pid, True)
    st.pin("long", pid, False)
    assert made["projects"].get(pid)["updated"] == updated0
    res = st.archive("long", pid)
    assert res["freed"] > 0 and res["archived"]["what"] == ["videos", "working"]
    assert not list((folder / "Output").glob("*.mp4"))
    assert not [p for p in made["store"].dir(job).iterdir() if p.suffix == ".mp4"]
    assert [p for p in (folder / "Final Cut").rglob("*.fcpxml")]                                   # (imported into Final Cut already? it stays)
    assert (folder / ".aivideo" / "plan.json").read_text() == plan0                                 # the plan, byte for byte
    assert json.loads((folder / ".aivideo" / "plan_words.json").read_text())["words"]
    assert (folder / "Notes" / "plan.md").exists() and [p for p in made["store"].dir(job).iterdir() if p.suffix == ".jpg"]   # notes and the poster stay
    assert (folder / "talk.mov").exists() and len(list((folder / "B-roll").glob("*.mov"))) == 3     # the footage is never touched
    assert made["projects"].get(pid)["archived"]["freed"] == res["freed"]
    assert made["projects"].get(pid)["updated"] == updated0                                        # (an old project doesn't jump to the top of the lists)
    assert made["projects"].view(made["projects"].get(pid))["archived"]["files"] == res["files"]
    with pytest.raises(ValueError, match="already removed|nothing to remove"):
        st.archive("long", pid)


def test_the_final_cut_toggle_removes_those_folders_too(made):
    st, pid, folder = _st(made, fc=True), made["proj"]["id"], made["folder"]
    st.archive("long", pid)
    assert not list((folder / "Final Cut").rglob("*.fcpxml")) and not list(folder.glob("*.fcpxml"))
    assert (folder / ".aivideo" / "plan.json").exists() and (folder / "talk.mov").exists()


def test_it_refuses_what_could_not_be_made_again_or_is_in_use(made):
    st, pid, folder, job = _st(made), made["proj"]["id"], made["folder"], made["proj"]["job"]
    plan = folder / ".aivideo" / "plan.json"
    text = plan.read_text()
    plan.write_text("{ not json")                                                                  # a plan that can't be read back
    assert not st.can_archive("long", pid)["ok"] and "plan" in st.can_archive("long", pid)["why"]
    with pytest.raises(ValueError, match="plan"):
        st.archive("long", pid)
    plan.write_text(text)
    assert list((folder / "Output").glob("*.mp4"))                                                  # nothing was removed
    made["store"].update(job, status="rendering")                                                   # being made
    assert "being made" in st.can_archive("long", pid)["why"]
    made["store"].update(job, status="done")
    st.pin("long", pid, True)                                                                       # pinned
    assert "kept" in st.can_archive("long", pid)["why"]
    st.pin("long", pid, False)
    p = json.loads(text)
    p["pending_render"], p["pending_changes"] = True, [{"kind": "replaced", "at": 4.0}]             # changes waiting for Re-render
    plan.write_text(json.dumps(p))
    assert "waiting" in st.can_archive("long", pid)["why"]
    plan.write_text(text)
    assert st.can_archive("long", pid)["ok"]
    (folder / ".aivideo" / "plan_words.json").unlink()                                              # its words gone: no
    assert "words" in st.can_archive("long", pid)["why"]


def test_the_age_rule_and_the_default_of_never(made):
    pid, folder = made["proj"]["id"], made["folder"]
    t0 = max(p.stat().st_mtime for p in (folder / "Output").glob("*.mp4"))
    later = lambda days: (lambda: t0 + days * 86400 + 60)                                           # noqa: E731
    row = lambda st, d: next(r for r in st.usage(d)["rows"] if r["id"] == pid)                      # noqa: E731
    assert row(_st(made, now=later(8)), 7)["due"] and not row(_st(made, now=later(8)), 14)["due"]
    assert row(_st(made, now=later(31)), 30)["due"] and not row(_st(made, now=later(31)), 60)["due"]
    assert not row(_st(made, now=later(100)), 0)["due"]                                              # (never: nothing is ever due)
    off = _st(made, days=0, now=later(500))                                                          # the default: nothing is removed
    assert off.auto_pass()["skipped"] == "off" and list((folder / "Output").glob("*.mp4"))
    busy = _st(made, days=7, now=later(20))
    assert busy.auto_pass(busy=lambda: "a transcript is being written")["skipped"] == "busy" and list((folder / "Output").glob("*.mp4"))
    young = _st(made, days=30, now=later(3))                                                         # too new
    assert young.auto_pass()["removed"] == 0 and list((folder / "Output").glob("*.mp4"))
    on = _st(made, days=14, now=later(20))
    res = on.auto_pass()
    assert res["removed"] == 1 and res["freed"] > 0 and not list((folder / "Output").glob("*.mp4"))


def test_an_archived_long_video_renders_again_from_the_plan_without_planning(made, monkeypatch):
    r, pid, folder, job = made["runner"], made["proj"]["id"], made["folder"], made["proj"]["job"]
    st = _st(made)
    plan0 = json.loads((folder / ".aivideo" / "plan.json").read_text())
    cutaways0 = json.loads((folder / ".aivideo" / "render_cache.json").read_text())["final"]["cutaways"]
    v1 = _latest(folder)
    from clipfarm.core import ffmpeg
    dur1 = ffmpeg.duration(v1)
    st.archive("long", pid)
    assert not list((folder / "Output").glob("*.mp4"))
    assert st.missing_footage("long", pid) == ""
    called = []
    ai = r.cur_ai or None
    import clipfarm.clipper.localproj as LP
    orig = LP.LocalAI

    def spy(cfg):
        a = orig(cfg)
        real = a.complete_json

        def cj(task, *args, **kw):
            called.append(task)
            return real(task, *args, **kw)
        a.complete_json = cj
        return a
    monkeypatch.setattr(LP, "LocalAI", spy)
    r.start(pid, "rerender")                                                                         # the page's Re-render (the real path, with its hook)
    _wait_idle(r)
    assert made["store"].get(job)["status"] == "done", made["store"].get(job)
    v2 = _latest(folder)
    assert abs(ffmpeg.duration(v2) - dur1) < 0.15                                                    # the same cut
    plan1 = json.loads((folder / ".aivideo" / "plan.json").read_text())
    assert [i for i in plan1["items"]] == [i for i in plan0["items"]]
    assert json.loads((folder / ".aivideo" / "render_cache.json").read_text())["final"]["cutaways"] == cutaways0
    assert not ({"broll_plan", "flow_plan", "highlight_candidates", "highlight_assemble"} & set(called)), called      # nothing was planned again
    assert "archived" not in made["projects"].get(pid) or not made["projects"].get(pid)["archived"]  # no longer marked as removed
    assert st.can_archive("long", pid)["ok"]                                                         # and it can be removed again


def test_the_missing_footage_is_named_before_a_re_render_starts(made):
    st, pid, folder = _st(made), made["proj"]["id"], made["folder"]
    away = folder.with_name("Park day (unplugged)")
    folder.rename(away)
    try:
        why = st.missing_footage("long", pid)
        assert "Park day" in why and "isn't there" in why and "Re-render" in why
    finally:
        away.rename(folder)
    (folder / "B-roll" / "red wall.mov").unlink()                                                    # one original clip gone
    why = st.missing_footage("long", pid)
    assert "red wall.mov" in why and "B-roll" in why


# ------------------------------------------------------------------------------------------------ short clips
def _short_job(tmp_path, store, name="Talk", url=False, status="done", clips=2):
    src = tmp_path / "Originals" / f"{name}.mov"
    src.parent.mkdir(exist_ok=True)
    src.write_bytes(b"x" * 20000)
    source = {"app": "studio", "url": "https://example.com/watch?v=abc"} if url else {"app": "studio", "path": str(src), "filename": src.name}
    job = store.create(source, {"clips": clips, "caption_style": "bold", "music": ""}, status=status)
    d = store.dir(job["id"])
    if url:
        (d / "source.mp4").write_bytes(b"d" * 30000)                                                  # a downloaded copy
    else:
        os.symlink(src, d / "source.mov")                                                             # a link to the original
    home = tmp_path / "Videos" / name
    (home / "Clips").mkdir(parents=True)
    (home / "Clips (before 10.05)").mkdir()
    (home / "Raw footage").mkdir()
    os.symlink(src, home / "Raw footage" / src.name)
    (d / "transcript.json").write_text(json.dumps({"words": [{"text": "hello", "start": 0.0, "end": 0.4}]}))
    for k in range(1, clips + 1):
        (d / f"clip_{k:02d}.mp4").write_bytes(b"v" * 50000)
        (d / f"clip_{k:02d}.jpg").write_bytes(b"j" * 900)
        (d / f"clip_{k:02d}.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")
        (home / "Clips" / f"{k:02d} Clip.mp4").write_bytes(b"v" * 50000)
        (home / "Clips" / f"{k:02d} Clip.srt").write_text("1\n")
        (home / "Clips (before 10.05)" / f"{k:02d} Old.mp4").write_bytes(b"o" * 40000)
        store.set_clip(job["id"], {"index": k, "file": f"clip_{k:02d}.mp4", "thumb": f"clip_{k:02d}.jpg", "srt": f"clip_{k:02d}.srt", "title": f"Clip {k}",
                                   "duration": 20.0, "rendered_at": time.time() - 40 * 86400, "layout": "auto", "caption_style": "bold", "pace": "tight",
                                   "moment": {"start": 1.0 * k, "end": 21.0 * k, "first": k, "last": k + 3, "title": f"Clip {k}", "hook": "", "score": 7.0, "reason": ""}})
    store.update(job["id"], app_home=str(home), title=name)
    return store.get(job["id"]), src, home


@pytest.fixture
def shorts(tmp_path):
    store = Store(tmp_path / "jobs")

    class P:                                                                                          # (no long projects here)
        def ids(self):
            return []

        def get(self, pid):
            raise KeyError(pid)

    return {"store": store, "tmp": tmp_path, "st": ST.Storage(None, store, P(), None, settings=lambda: {"storage": {"auto_days": 30, "also_finalcut": False}})}


def test_short_clips_lose_their_mp4s_and_keep_cuts_words_posters_and_the_original(shorts):
    store, tmp, st = shorts["store"], shorts["tmp"], shorts["st"]
    job, src, home = _short_job(tmp, store)
    row = next(r for r in st.usage()["rows"] if r["id"] == job["id"])
    assert row["kind"] == "short" and row["sizes"]["videos"] == 280000 and row["sizes"]["working"] == 0 and row["due"]
    res = st.archive("short", job["id"])
    assert res["freed"] == 280000
    d = store.dir(job["id"])
    assert not list(d.glob("clip_*.mp4")) and not list((home / "Clips").glob("*.mp4")) and not list((home / "Clips (before 10.05)").glob("*.mp4"))
    assert list(d.glob("clip_*.jpg")) and list(d.glob("clip_*.srt")) and (d / "transcript.json").exists()     # the posters, subtitles and words stay
    assert (home / "Clips" / "01 Clip.srt").exists()
    assert src.exists() and (d / "source.mov").is_symlink() and (home / "Raw footage" / src.name).is_symlink()   # the original and its links: untouched
    j = store.get(job["id"])
    assert j["archived"]["freed"] == 280000 and [c["moment"]["title"] for c in j["clips"]] == ["Clip 1", "Clip 2"]


def test_a_downloaded_copy_is_working_and_a_local_link_is_not(shorts):
    store, tmp, st = shorts["store"], shorts["tmp"], shorts["st"]
    job, src, home = _short_job(tmp, store, name="From a link", url=True)
    row = next(r for r in st.usage()["rows"] if r["id"] == job["id"])
    assert row["sizes"]["working"] == 30000
    st.archive("short", job["id"])
    assert not (store.dir(job["id"]) / "source.mp4").exists()
    assert "link" not in st.missing_footage("short", job["id"])                                       # (downloaded again: nothing to plug in)


def test_short_jobs_in_use_without_cuts_or_pinned_are_left_alone(shorts):
    store, tmp, st = shorts["store"], shorts["tmp"], shorts["st"]
    job, _, _ = _short_job(tmp, store)
    store.update(job["id"], status="rendering")
    assert "being made" in st.can_archive("short", job["id"])["why"]
    store.update(job["id"], status="done")
    st.pin("short", job["id"], True)
    assert "kept" in st.can_archive("short", job["id"])["why"]
    st.pin("short", job["id"], False)
    (store.dir(job["id"]) / "transcript.json").unlink()
    assert "words" in st.can_archive("short", job["id"])["why"]
    other, _, _ = _short_job(tmp, store, name="Other")
    store.set_clip(other["id"], {**store.get(other["id"])["clips"][0], "moment": None})
    assert "cuts" in st.can_archive("short", other["id"])["why"]


def test_the_original_of_a_short_is_named_when_it_is_missing(shorts):
    store, tmp, st = shorts["store"], shorts["tmp"], shorts["st"]
    job, src, _ = _short_job(tmp, store, name="Gone")
    src.unlink()
    why = st.missing_footage("short", job["id"])
    assert "Gone.mov" in why and "isn't there" in why


def test_short_clips_render_again_from_what_was_saved(shorts, monkeypatch):
    store, tmp, st = shorts["store"], shorts["tmp"], shorts["st"]
    job, src, home = _short_job(tmp, store, name="From a link", url=True)
    st.archive("short", job["id"])
    cfg = load_config()
    cfg.root = tmp
    eng = Engine(cfg, store)
    seen = {"ingest": 0, "clips": []}

    def ingest(self, jid, d):
        seen["ingest"] += 1                                                                          # (the download comes down again)
        (d / "source.mp4").write_bytes(b"d" * 30000)
        return d / "source.mp4"

    def render_clip(self, jid, index, m, words, srcp, layout="auto", style="bold", pace="tight", hook_title=True, **kw):
        seen["clips"].append((index, round(m.start, 1), layout, style, pace, kw.get("segments")))
        d = self.store.dir(jid)
        (d / f"clip_{index:02d}.mp4").write_bytes(b"n" * 1000)
        self.store.set_clip(jid, {**next(c for c in self.store.get(jid)["clips"] if c["index"] == index), "file": f"clip_{index:02d}.mp4", "rendered_at": time.time()})
        return {}
    monkeypatch.setattr(Engine, "_ingest", ingest)
    monkeypatch.setattr(Engine, "render_clip", render_clip)
    eng.rerender_saved(job["id"])
    end = time.time() + 30
    while store.get(job["id"])["status"] != "done" and time.time() < end:
        time.sleep(0.1)
    j = store.get(job["id"])
    assert j["status"] == "done" and not j.get("archived"), j
    assert seen["ingest"] == 1 and [c[:5] for c in seen["clips"]] == [(1, 1.0, "auto", "bold", "tight"), (2, 2.0, "auto", "bold", "tight")]
    assert len(list(store.dir(job["id"]).glob("clip_*.mp4"))) == 2
    with pytest.raises(ValueError, match="still being made"):
        store.update(job["id"], status="rendering")
        eng.rerender_saved(job["id"])


# ------------------------------------------------------------------------------------------------ the setting
def test_the_setting_is_never_by_default_and_only_takes_the_offered_days(tmp_path):
    s = SD.Settings(tmp_path)
    assert s.get()["storage"] == {"auto_days": 0, "also_finalcut": False}
    for d in (7, 14, 30, 60, 0):
        assert s.update({"storage": {"auto_days": d}})["storage"]["auto_days"] == d
    assert s.update({"storage": {"also_finalcut": True}})["storage"] == {"auto_days": 0, "also_finalcut": True}
    for bad in (3, "soon", -1, 90):
        with pytest.raises(ValueError):
            s.update({"storage": {"auto_days": bad}})
    assert s.get()["storage"]["auto_days"] == 0                                                        # (a bad value changes nothing)
    assert s.get()["library"]["when"] and s.get()["updates"]                                          # (the other settings are intact)
