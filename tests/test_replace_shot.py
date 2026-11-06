"""Replace one B-roll shot of a finished long video (2026-11-05: "when you're watching back a video in long form and you don't like a B-roll
shot or clip or photo that it added, you should have the option to replace it with something else"). A real make, then real re-renders:
only that shot changes, the others and the cut stay, Final Cut follows."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from clipfarm.clipper import aivideo as AV
from clipfarm.clipper.store import Store
from clipfarm.core import ffmpeg
from clipfarm.core.config import load_config
from test_aivideo import _FakeAI, _Res

FF = ffmpeg.bin_path("ffmpeg")
COLORS = {"red wall": (255, 0, 0), "green trees": (0, 160, 0), "blue sea": (0, 0, 255)}


def _clip(path: Path, secs: int, color: str | None = None) -> None:
    src = f"color=c={color}:size=480x270:rate=30:duration={secs}" if color else f"testsrc2=size=320x240:rate=30:duration={secs}"
    cmd = [FF, "-y", "-v", "error", "-f", "lavfi", "-i", src]
    if not color:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=300:duration={secs}", "-c:a", "aac", "-shortest"]
    subprocess.run(cmd + ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)], check=True, timeout=120)


def _rgb_at(video: Path, t: float, tmp: Path) -> tuple:
    out = tmp / f"f{t:.1f}.png"
    subprocess.run([FF, "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1", str(out)], check=True, timeout=60)
    a = np.asarray(Image.open(out).convert("RGB"), dtype=float)
    h, w = a.shape[:2]
    return tuple(a[h // 3: 2 * h // 3, w // 3: 2 * w // 3].reshape(-1, 3).mean(axis=0))


class AI(_FakeAI):
    """Describes each B-roll clip by its colour; places the red wall at 4 s."""

    def complete_json(self, task, system, prompt, schema, validate=None, images=None, **kw):
        if task == "broll_describe":
            name = next((n for n in COLORS if n in prompt), "red wall")
            what = {"red wall": "a plain red brick wall", "green trees": "green trees in a sunny park", "blue sea": "the blue sea at the beach"}[name]
            return _Res({"scenes": [{"start_s": 0, "end_s": 6, "what": what, "quality": 8}]})
        if task == "broll_plan":
            import re
            red = next(int(i) for i, w in re.findall(r"^\s+S(\d+): (.*)$", prompt, re.M) if "red" in w)
            return _Res({"cutaways": [{"at": 4.0, "scene": red, "length": 2.5, "why": "the wall"}]})
        if task in ("broll_pick", "broll_suitable"):
            return _Res({"scene": -1})
        return super().complete_json(task, system, prompt, schema, validate=validate, images=images, **kw)


@pytest.fixture
def made(tmp_path, monkeypatch):
    from clipfarm.clipper import localproj as LP
    from clipfarm.clipper.engine import Engine
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["paths"].update({"state": str(tmp_path / "state")})
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "x"), "local": True, "data_dir": "ai", "local_projects": True, "verify_freeze": False}
    folder = tmp_path / "Park day"
    (folder / "B-roll").mkdir(parents=True)
    _clip(folder / "talk.mov", 16)
    for name in COLORS:
        _clip(folder / "B-roll" / f"{name}.mov", 6, color=name.split()[0])
    said = ("We met at the park this morning. The trees were green and tall and everyone sat under them. "
            "Then we walked down to the beach. The sea was calm and blue. It was a lovely day out with friends.").split()
    words = [{"text": w, "start": 0.4 + i * 0.42, "end": 0.4 + i * 0.42 + 0.3} for i, w in enumerate(said)]
    store = Store(tmp_path / "jobs")
    projects = AV.Projects(cfg, store, Engine(cfg, store))
    fake = AI()
    monkeypatch.setattr(LP, "LocalAI", lambda c: fake)
    monkeypatch.setattr(LP.LocalRunner, "_openai_words", lambda self, f, ai: words if f["name"] == "talk.mov" else [])
    runner = LP.LocalRunner(cfg, store, projects)
    proj = projects.create_local(str(folder), "Park day", "", {"transcriber": "openai", "music": ""}, {"clips": 0})
    projects.prepare_local(proj["id"], {"graphics": "", "music": ""})
    runner.make(proj["id"], {})
    assert store.get(proj["job"])["status"] == "done", store.get(proj["job"])
    return {"LP": LP, "runner": runner, "store": store, "projects": projects, "proj": proj, "folder": folder, "tmp": tmp_path}


def _latest(folder: Path) -> Path:
    return max((folder / "Output").glob("*.mp4"), key=lambda p: p.stat().st_mtime)


def _fcp_broll(folder: Path) -> list[str]:
    """The names of the clips on the B-roll lane of the newest Final Cut project (the closing shot on the main track isn't B-roll)."""
    import xml.etree.ElementTree as ET
    root = ET.parse(max((folder / "Final Cut").rglob("*.fcpxml"), key=lambda p: p.stat().st_mtime)).getroot()
    names = {a.get("id"): a.get("name") for a in root.iter("asset")}
    return [names[c.get("ref")] for c in root.iter("asset-clip") if c.get("lane")]


def test_a_shot_is_swapped_for_one_that_fits_the_words_and_nothing_else_moves(made):
    r, pid, folder, tmp = made["runner"], made["proj"]["id"], made["folder"], made["tmp"]
    shots = r.shots(pid)
    assert [round(s["at"], 1) for s in shots["shots"]] == [4.0] and "red" in shots["shots"][0]["what"]
    assert shots["shots"][0]["thumb"].startswith("shot_") and (made["store"].dir(made["proj"]["job"]) / shots["shots"][0]["thumb"]).exists()
    v1 = _latest(folder)
    red = _rgb_at(v1, 5.0, tmp)
    assert red[0] > 180 and red[1] < 60                                                # the red wall is on screen at 5 s
    ch = r.shot_choices(pid, 0)
    assert "trees" in ch["spoken"] and [("green" in c["what"]) for c in ch["choices"]][0]          # what's said ranks the park first
    assert all("red" not in c["what"] for c in ch["choices"]) and all(c["thumb"] for c in ch["choices"])
    rc_before = json.loads((folder / ".aivideo" / "render_cache.json").read_text())
    green = ch["choices"][0]["choice"]
    r.replace_shot(pid, 0, green)
    assert made["store"].get(made["proj"]["job"])["status"] == "done"
    v2 = _latest(folder)
    assert v2 != v1 and abs(ffmpeg.duration(v2) - ffmpeg.duration(v1)) < 0.1             # the same cut
    g = _rgb_at(v2, 5.0, tmp)
    assert g[1] > 120 and g[0] < 60                                                    # green trees now
    talk_before, talk_after = _rgb_at(v1, 9.0, tmp), _rgb_at(v2, 9.0, tmp)
    assert np.allclose(talk_before, talk_after, atol=12)                               # the talking around it is untouched
    rc = json.loads((folder / ".aivideo" / "render_cache.json").read_text())
    assert rc["sig"] == rc_before["sig"] and rc["final"]["cutaways"][0]["scene"] == green and rc["final"]["cutaways"][0]["why"] == "chosen by you"
    assert _fcp_broll(folder) == ["green trees"]                                        # Final Cut gets the new shot
    # "Show the speaker instead": the shot is taken out
    r.replace_shot(pid, 0, "speaker")
    v3 = _latest(folder)
    s = _rgb_at(v3, 5.0, tmp)
    assert not (s[1] > 120 and s[0] < 60) and r.shots(pid)["shots"] == []
    assert _fcp_broll(folder) == []


def test_wrong_choices_and_ads_are_refused_in_plain_words(made):
    r, pid, folder = made["runner"], made["proj"]["id"], made["folder"]
    with pytest.raises(ValueError, match="isn't in the video"):
        r.check_replace(pid, 7, 0)
    used = r.shots(pid)["shots"][0]
    with pytest.raises(ValueError, match="pick one"):
        r.check_replace(pid, used["k"], "something")
    ch = r.shot_choices(pid, 0)["choices"]
    r.check_replace(pid, 0, ch[0]["choice"])                                           # fine: says nothing
    (folder / "B-roll" / "blue sea.mov").unlink()                                      # a file that went away is not offered
    assert all("blue" not in c["what"] for c in r.shot_choices(pid, 0)["choices"])
    st = made["LP"].State(made["LP"].home_of(made["projects"].get(pid)))
    st.put("plan.json", {**st.get("plan.json"), "style": "ad"})              # an ad's shots can be swapped too (the swap pins its B-roll)
    assert r.shots(pid)["shots"]


def test_the_page_routes(made, monkeypatch):
    from clipfarm.clipper import studio_web as W
    api = W.StudioAPI.__new__(W.StudioAPI)
    api.runner = made["runner"]
    pid = made["proj"]["id"]
    code, body = api.get(["long", pid, "shots"])
    assert code == 200 and len(body["shots"]) == 1
    code, body = api.get(["long", pid, "shots", "0", "choices"])
    assert code == 200 and body["choices"]
    code, body = api.get(["long", pid, "shots", "9", "choices"])
    assert code == 409 and "isn't in the video" in body["error"]
    started = []
    monkeypatch.setattr(made["runner"], "start", lambda p, kind, **kw: started.append((p, kind, kw)))
    code, body = api.post(["long", pid, "shots", "0"], {"choice": "speaker"})
    # (2026-11-06) Replace only changes the plan: nothing renders until Re-render
    assert code == 200 and body["staged"] and body["pending"] == 1 and started == []
    with pytest.raises(ValueError):
        api.post(["long", pid, "shots", "0"], {"choice": 999})
