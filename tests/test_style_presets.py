"""Editing-style presets: named, global, kept in the app's folder; rename, delete, add more videos (the numbers are combined), the person's own lines and
rules, "Use this" filling a mode's points box and settings, the HTTP routes, the settings for the fallback address, and the three MCP tools."""

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from clipfarm.clipper import mcp_server as MCP
from clipfarm.clipper import studio as SD
from clipfarm.clipper import stylefetch, stylelearn as SL, styles_api, stylepresets as SP
from clipfarm.clipper import styles_profile as SPF


def _m(**kw) -> dict:
    base = {k: None for k in SL.NUMBERS}
    base.update({k: None for k in SL.WORDS})
    base.update({"duration_s": 30.0, "aspect": "9:16", "faces_measured": False})
    base.update(kw)
    return base


def _src(title, **kw):
    return {"url": f"https://www.youtube.com/watch?v={title}", "title": title, "kind": "youtube_video", "added": 1, "measurements": _m(**kw)}


class FakeAI:
    def __init__(self, points):
        self.points, self.calls = points, 0

    def complete_json(self, task, system, prompt, schema, validate=None, images=None, **kw):
        self.calls += 1
        return SimpleNamespace(data={"points": self.points})


@pytest.fixture
def pr(tmp_path):
    return SP.Presets(tmp_path)


# ----------------------------------------------------------------------------------------------------------------- the store
def test_presets_are_created_renamed_deleted_and_survive_a_restart(tmp_path):
    a = SP.Presets(tmp_path)
    v = a.create("Fast reels", points=["Cut every 2 s."], rules=["Never use gameplay."])
    assert v["name"] == "Fast reels" and v["points"] == ["Cut every 2 s."] and v["rules"] == ["Never use gameplay."]
    b = SP.Presets(tmp_path)                                                              # a new process reads the same file
    assert [p["name"] for p in b.summary()] == ["Fast reels"]
    assert b.create("fast reels")["name"] == "fast reels 2"                               # names stay different
    b.rename(v["id"], "  Slow   talks ")
    assert SP.Presets(tmp_path).get(v["id"])["name"] == "Slow talks"
    with pytest.raises(ValueError):
        b.rename(v["id"], "   ")
    b.delete(v["id"])
    assert [p["name"] for p in SP.Presets(tmp_path).summary()] == ["fast reels 2"]
    with pytest.raises(KeyError):
        b.get(v["id"])
    (tmp_path / "styles" / "presets.json").write_text("not json")                         # a damaged file is an empty list, not a crash
    assert SP.Presets(tmp_path).summary() == []


def test_a_second_video_is_added_to_a_style_and_the_numbers_are_combined(pr):
    ai = FakeAI(["Cut every 2 s."])
    v1 = pr.add_learned(None, "", [_src("one", cuts_per_30s=10.0, shot_median_s=3.0, caption_share=0.9, music="under")], [], ai)
    assert v1["name"] == "one" and len(v1["sources"]) == 1 and v1["style"]["pace"]["name"] == "tight" and v1["points"] == ["Cut every 2 s."]
    v2 = pr.add_learned(v1["id"], "", [_src("two", cuts_per_30s=20.0, shot_median_s=1.5, caption_share=0.9, music="under")], [], ai)
    assert len(v2["sources"]) == 2 and v2["id"] == v1["id"]
    assert v2["style"]["pace"]["name"] == "fast" and any("2 videos" in e for e in v2["evidence"])        # the median of 10 and 20 is 15: fast
    assert len(pr.all()) == 1 and ai.calls == 2


def test_the_persons_own_lines_are_not_replaced_by_later_learning(pr):
    ai = FakeAI(["Learned line."])
    v = pr.add_learned(None, "x", [_src("one", cuts_per_30s=10.0)], [], ai)
    pr.set_points(v["id"], "My own line\nAnother of mine")
    assert pr.get(v["id"])["edited"] is True
    v2 = pr.add_learned(v["id"], "", [_src("two", cuts_per_30s=12.0)], [], FakeAI(["Newer learned line."]))
    assert v2["kept_edits"] is True and v2["points"] == ["My own line", "Another of mine"] and v2["learned_points"] == ["Newer learned line."]
    assert pr.reset_points(v["id"])["points"] == ["Newer learned line."] and pr.get(v["id"])["edited"] is False


def test_rules_are_added_removed_and_come_first_when_applied(pr):
    v = pr.create("S", points=["A line."])
    pr.add_rule(v["id"], "  Never put music under a prayer. ")
    pr.add_rule(v["id"], "Never put music under a prayer.")                               # the same rule once
    p = pr.get(v["id"])
    assert p["rules"] == ["Never put music under a prayer."] and pr.lines_for(p) == ["Never put music under a prayer.", "A line."]
    with pytest.raises(ValueError):
        pr.add_rule(v["id"], "")
    with pytest.raises(ValueError):
        pr.remove_rule(v["id"], 5)
    assert pr.remove_rule(v["id"], 0)["rules"] == []


def test_the_same_video_added_twice_is_one_source_and_frames_are_kept_small_and_deleted_with_the_style(pr, tmp_path):
    frames = []
    for k in range(12):
        f = tmp_path / f"f{k}.jpg"
        f.write_bytes(b"\xff\xd8x")
        frames.append(f)
    s = _src("one", cuts_per_30s=10.0)
    v = pr.add_learned(None, "x", [s], frames, None)
    assert len(v["frames"]) == SL.KEEP_FRAMES and len(list((pr.frames / v["id"]).glob("*.jpg"))) == SL.KEEP_FRAMES
    v = pr.add_learned(v["id"], "", [s], [], None)
    assert len(v["sources"]) == 1
    pr.delete(v["id"])
    assert not (pr.frames / v["id"]).exists()


# ----------------------------------------------------------------------------------------------------------------- applying
class FakeAPI:
    def __init__(self, root):
        self.root = Path(root)
        self.settings, self.styles = SD.Settings(root), SD.Styles(root)
        self.cfg = SimpleNamespace(get=lambda k, d=None: d)
        self.synced = []

    def sync_long_style(self, extra=None):
        self.synced.append(("long", self.styles.get("long")["points"], extra))

    def sync_points_style(self, mode, pts, extra=None):
        self.synced.append((mode, list(pts), extra))

    def style_view(self, kind):
        d = self.styles.get(kind)
        return {"points": d["points"], "updated": d["updated"], "added": d["added"], "can_undo": bool(d["history"])}


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEOBUILDER_HOME", str(tmp_path / "h"))
    return FakeAPI(tmp_path / "h")


def test_use_this_fills_the_points_box_and_the_settings_for_that_mode_only(api):
    pr = styles_api.presets(api)
    v = pr.add_learned(None, "Tight reels", [_src("one", cuts_per_30s=12.0, shot_median_s=2.0, caption_share=0.9, caption_words_median=2, music="under")], [],
                       FakeAI(["Cut every 2 s.", "Two-word captions."]))
    pr.add_rule(v["id"], "Keep the speaker's name on screen.")
    api.styles._save("short", ["An old line."], [], [])
    out = styles_api.apply_preset(api, v["id"], "short")
    assert out["points"] == ["Keep the speaker's name on screen.", "Cut every 2 s.", "Two-word captions."]
    assert api.styles.get("short")["points"] == out["points"] and api.styles.get("long")["points"] == []        # the other mode is untouched
    ov = api.settings.get()["style_overrides"]
    assert ov["short"]["pace"] == "tight" and ov["short"]["captions"] == "karaoke" and ov["short"]["music"] == "throughout" and ov["long"] == {}
    assert api.synced[-1][0] == "short" and api.synced[-1][2]["pace"] == "tight"           # the points are read again with the preset's numbers under them
    assert api.styles.undo("short")["points"] == ["An old line."]                          # one undo brings the old box back
    styles_api.apply_preset(api, v["id"], "long")
    assert api.synced[-1][0] == "long" and api.styles.get("long")["points"] == out["points"]
    with pytest.raises(ValueError):
        styles_api.apply_preset(api, v["id"], "medium")
    empty = pr.create("Empty")
    with pytest.raises(ValueError, match="no lines"):
        styles_api.apply_preset(api, empty["id"], "short")


def test_the_length_of_the_learned_videos_is_not_applied_as_a_default(api):
    pr = styles_api.presets(api)
    v = pr.add_learned(None, "x", [_src("one", cuts_per_30s=9.0, duration_s=28.0)], [], None)
    p = pr.get(v["id"])
    assert p["style"]["length"]["target_s"] == 28 and "target_seconds" not in pr.settings_for(p)
    styles_api.apply_preset(api, v["id"], "long")
    assert api.settings.get()["style_overrides"]["long"]["target_seconds"] is None


def test_a_preset_cannot_carry_anything_but_editing_settings(api):
    pr = styles_api.presets(api)
    v = pr.add_learned(None, "x", [_src("one", cuts_per_30s=9.0, caption_share=0.9, caption_words_median=3)], [], None)
    p = pr.get(v["id"])
    assert set(p["style"]) <= {"pace", "broll", "captions", "music", "length"} and set(pr.settings_for(p)) <= set(SPF.SETTING_KEYS) - {"profile"}


# ----------------------------------------------------------------------------------------------------------------- the routes
def _wait(api, job, seconds=20):
    t0 = time.time()
    while time.time() - t0 < seconds:
        v = styles_api.job_view(api, job)
        if v["state"] != "running":
            return v
        time.sleep(0.05)
    raise AssertionError("the job never finished")


def test_learning_runs_in_the_background_and_ends_in_a_preset(api, monkeypatch):
    seen = {"n": 0}

    def fake_learn(source, **kw):
        seen["source"], seen["endpoint"] = source, kw.get("endpoint")
        seen["n"] += 1
        kw["progress"]("Measuring the video")
        return {"sources": [_src(f"Cooking short {seen['n']}", cuts_per_30s=15.0, shot_median_s=1.8)], "frames": []}
    monkeypatch.setattr(styles_api.stylelearn, "learn", fake_learn)
    api.settings.update({"styles": {"fetch_url": "https://my.example/api"}})
    code, out = styles_api.post(api, ["learn"], {"url": "https://www.youtube.com/watch?v=abc", "name": "Kitchen"})
    assert code == 200 and out["job"]
    v = _wait(api, out["job"])
    assert v["state"] == "done" and v["preset"]["name"] == "Kitchen" and v["preset"]["style"]["pace"]["name"] == "fast"
    assert seen["source"].startswith("https://www.youtube.com") and seen["endpoint"] == "https://my.example/api"
    code, listing = styles_api.get(api, ["presets"])
    assert code == 200 and [p["name"] for p in listing["presets"]] == ["Kitchen"] and listing["fetch_url"] == "https://my.example/api"
    # adding another video to the same style
    code, out = styles_api.post(api, ["learn"], {"url": "https://www.youtube.com/watch?v=def", "preset_id": v["preset"]["id"]})
    v2 = _wait(api, out["job"])
    assert v2["state"] == "done" and len(v2["preset"]["sources"]) == 2 and len(styles_api.presets(api).all()) == 1


def test_a_failed_link_gives_the_plain_message_and_makes_no_preset(api, monkeypatch):
    def boom(source, **kw):
        raise stylefetch.FetchError(stylefetch.INSTAGRAM_FAILED)
    monkeypatch.setattr(styles_api.stylelearn, "learn", boom)
    code, out = styles_api.post(api, ["learn"], {"url": "https://www.instagram.com/reel/Cx/"})
    v = _wait(api, out["job"])
    assert v["state"] == "error" and v["error"] == "Instagram would not give this reel without signing in: drop the video file instead." and styles_api.presets(api).all() == []
    monkeypatch.setattr(styles_api.stylelearn, "learn", lambda s, **kw: (_ for _ in ()).throw(RuntimeError("secret path /Users/x")))
    code, out = styles_api.post(api, ["learn"], {"url": "https://www.youtube.com/watch?v=abc"})
    v = _wait(api, out["job"])
    assert v["state"] == "error" and "/Users" not in v["error"] and "Try another" in v["error"]


def test_bad_requests_are_plain_400s_and_unknown_styles_404(api):
    assert styles_api.post(api, ["learn"], {})[0] == 400
    assert styles_api.post(api, ["learn"], {"url": "https://example.com/page"})[0] == 400
    assert styles_api.post(api, ["learn"], {"file": "/nope/missing.mp4"})[0] == 400
    assert styles_api.post(api, ["learn"], {"url": "https://www.youtube.com/watch?v=a", "preset_id": "stdoesnotexist"})[0] == 404
    assert styles_api.post(api, ["presets", "stdoesnotexist", "rename"], {"name": "x"})[0] == 404
    code, out = styles_api.post(api, ["presets"], {"name": "A"})
    pid = out["preset"]["id"]
    assert styles_api.post(api, ["presets", pid, "rename"], {"name": ""})[0] == 400
    assert styles_api.post(api, ["presets", pid, "apply"], {"mode": "short"})[0] == 400          # nothing to apply yet
    assert styles_api.post(api, ["presets", pid, "nonsense"], {}) is None
    assert styles_api.get(api, ["learn", "slnotreal0"]) is None and styles_api.get(api, ["learn", "sl00000000"])[0] == 404
    for what, body in (("points", {"points": "one\ntwo"}), ("rule", {"text": "a rule"}), ("remove-rule", {"index": 0}), ("reset-points", {}), ("rename", {"name": "B"})):
        assert styles_api.post(api, ["presets", pid, what], body)[0] == 200
    assert styles_api.post(api, ["presets", pid, "delete"], {})[0] == 200 and styles_api.presets(api).all() == []


def test_only_two_learning_jobs_run_at_once(api, monkeypatch):
    gate = threading.Event()

    def slow(source, **kw):
        gate.wait(5)
        return {"sources": [_src("v", cuts_per_30s=9.0)], "frames": []}
    monkeypatch.setattr(styles_api.stylelearn, "learn", slow)
    first = [styles_api.post(api, ["learn"], {"url": f"https://www.youtube.com/watch?v={k}"}) for k in ("a", "b")]
    assert [c for c, _ in first] == [200, 200]
    code, out = styles_api.post(api, ["learn"], {"url": "https://www.youtube.com/watch?v=c"})
    assert code == 400 and "wait" in out["error"] and len(styles_api.summary(api)["running"]) == 2
    gate.set()
    for _, o in first:
        _wait(api, o["job"])


# ----------------------------------------------------------------------------------------------------------------- the app's own settings and page state
def test_the_fallback_address_is_a_plain_setting_and_empty_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEOBUILDER_HOME", str(tmp_path / "h2"))
    s = SD.Settings(tmp_path / "h2")
    assert s.get()["styles"] == {"fetch_url": ""}
    assert s.update({"styles": {"fetch_url": "https://my.example/api"}})["styles"]["fetch_url"] == "https://my.example/api"
    assert s.update({"styles": {"fetch_url": "javascript:alert(1)"}})["styles"]["fetch_url"] == ""        # only a web address is kept
    assert s.update({"styles": {"fetch_url": "https://a b"}})["styles"]["fetch_url"] == ""


def test_the_style_replace_keeps_the_old_box_one_undo_away(tmp_path):
    st = SD.Styles(tmp_path)
    st._save("long", ["old"], [], [])
    assert st.replace("long", ["new one", "new two"])["points"] == ["new one", "new two"]
    assert st.undo("long")["points"] == ["old"]


def test_the_app_state_and_routes_carry_the_presets(tmp_path, monkeypatch):
    from clipfarm.clipper import studio_web as W
    h = tmp_path / "VB"
    monkeypatch.setenv("VIDEOBUILDER_HOME", str(h))
    a = W.StudioAPI.__new__(W.StudioAPI)
    a.cfg = SimpleNamespace(get=lambda k, d=None: d)
    a.root, a.settings, a.styles = h, SD.Settings(h), SD.Styles(h)
    a.projects = SimpleNamespace(rules_path=lambda: tmp_path / "rules.md", points_sig=lambda: "", points_style=lambda: {}, set_points_style=lambda *x: None)
    code, out = a.post(["styles", "presets"], {"name": "Mine", "points": ["Cut fast."]})
    assert code == 200 and out["preset"]["name"] == "Mine"
    assert a.style_presets_state()["presets"][0]["points"] == ["Cut fast."]
    assert a.get(["styles", "presets"])[1]["presets"][0]["name"] == "Mine"
    code, out = a.post(["styles", "presets", out["preset"]["id"], "apply"], {"mode": "short"})
    assert code == 200 and a.styles.get("short")["points"] == ["Cut fast."] and out["styles"]["short"]["points"] == ["Cut fast."]
    assert a.post(["styles", "short", "undo"], {})[0] == 200                              # the older routes still work next to the new ones


# ----------------------------------------------------------------------------------------------------------------- the MCP tools
class FakeEngine:
    def __init__(self):
        self.log, self.preset = [], {"id": "st1", "name": "Tight reels", "points": ["Cut every 2 s."], "rules": ["No gameplay."], "evidence": ["12 cuts in 30 s."],
                                     "sources": [{"title": "A video", "url": "u"}]}

    def get(self, path, **kw):
        self.log.append(("GET", path))
        if path == "/api/app/styles/presets":
            return {"presets": [self.preset]}
        return {"state": "done", "preset": self.preset}

    def post(self, path, body=None, **kw):
        self.log.append(("POST", path, body))
        if path == "/api/app/styles/learn":
            return {"job": "sl12345678"}
        return {"ok": True, "points": ["No gameplay.", "Cut every 2 s."]}


def test_the_three_style_tools_have_valid_schemas_and_work(monkeypatch):
    names = {"learn_style", "list_style_presets", "apply_style_preset"}
    tools = {t["name"]: t for t in MCP.TOOLS}
    assert names <= set(tools) and names <= set(MCP.HANDLERS)
    for n in names:
        sch = tools[n]["inputSchema"]
        assert sch["type"] == "object" and sch["additionalProperties"] is False and tools[n]["description"]
    assert tools["learn_style"]["inputSchema"]["required"] == ["url"] and tools["apply_style_preset"]["inputSchema"]["properties"]["mode"]["enum"] == ["short", "long"]
    monkeypatch.setattr(MCP.time, "sleep", lambda s: None)
    eng = FakeEngine()
    text = MCP.tool_learn_style(eng, {"url": "https://www.youtube.com/watch?v=abc", "preset": "tight", "name": "x"})[0]["text"]
    assert "Tight reels" in text and "Cut every 2 s." in text and "12 cuts in 30 s." in text and "Nothing is applied yet" in text
    post = next(x for x in eng.log if x[0] == "POST")
    assert post[2] == {"url": "https://www.youtube.com/watch?v=abc", "preset_id": "st1", "name": "x"}
    eng2 = FakeEngine()
    MCP.tool_learn_style(eng2, {"url": "/Users/me/clip.mp4"})
    assert next(x for x in eng2.log if x[0] == "POST")[2] == {"file": "/Users/me/clip.mp4"}
    assert "Tight reels" in MCP.tool_list_style_presets(eng, {})[0]["text"]
    out = MCP.tool_apply_style_preset(eng, {"preset": "Tight Reels", "mode": "long"})[0]["text"]
    assert "long videos" in out and "- Cut every 2 s." in out and eng.log[-1][1] == "/api/app/styles/presets/st1/apply" and eng.log[-1][2] == {"mode": "long"}
    with pytest.raises(MCP.ToolError):
        MCP.tool_apply_style_preset(eng, {"preset": "nothing like it", "mode": "long"})
    with pytest.raises(MCP.ToolError):
        MCP.tool_apply_style_preset(eng, {"preset": "st1", "mode": "medium"})
    with pytest.raises(MCP.ToolError):
        MCP.tool_learn_style(eng, {"url": " "})
