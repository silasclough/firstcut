"""Phase 2 (WP3/WP5/WP6 and learn-a-style) attacked with awkward input. Tests marked xfail(strict) document a real defect: when it is fixed the test
flips to a pass and the marker must go."""
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from clipfarm.clipper import constraints as CN
from clipfarm.clipper import moments as M
from clipfarm.clipper import poolrules as PR
from clipfarm.clipper import speechtrim as ST
from clipfarm.clipper import stylefetch, styles_api, stylepresets as SP
from clipfarm.clipper import transcript_search as TS

TEXTS = ["Welcome everyone to our summer camp week", "We were wrong about the schedule and we say sorry", "The river flooded our camp on Tuesday night",
         "Nobody was hurt and everyone helped each other", "We were wrong about the schedule and we say sorry"]


def _words(texts=TEXTS, gap=1.4):
    out, t = [], 0.0
    for tx in texts:
        for w in tx.split():
            out.append({"text": w, "start": round(t, 3), "end": round(t + 0.4, 3)})
            t += 0.45
        t += gap
    return out


class _R:
    def __init__(self, d):
        self.data = d


class AI:
    def __init__(self, **ans):
        self.ans, self.calls = ans, []

    def complete_json(self, task, system, prompt, schema, validate=None, images=None, **kw):
        self.calls.append(task)
        a = self.ans[task]
        return _R(a(prompt) if callable(a) else a)


def _con(**kw):
    c = {"id": "c1", "type": "include", "hard": True, "source": 'keep "we were wrong about the schedule"', "status": "ungrounded",
         "anchor": {"words": None, "spans": [], "file_ids": [], "scene_ids": [], "seconds": None, "text": ""}, "rule": {"text": "x"}, "candidates": [], "note": ""}
    c.update(kw)
    return c


# ------------------------------------------------------------------------------------------------------------- constraints.extract
def test_extract_keeps_only_quotes_that_are_in_the_notes_and_survives_junk_rows():
    notes = "Please do not include the part about the flood. Make it 2 minutes."
    ai = AI(constraints_extract={"constraints": [
        {"type": "exclude_words", "source": "do not include the part about the flood", "want": "x"},
        {"type": "include", "source": "keep the grace part", "want": "x"},              # not in the notes
        {"type": "bogus", "source": "Make it 2 minutes"}, "junk", None, {"type": "include", "source": ""},
        {"type": "duration", "source": "2 minutes", "hard": "yes"}]})
    got = CN.extract(ai, notes)
    assert [(c["type"], c["hard"]) for c in got] == [("exclude_words", True), ("duration", True)]


def test_extract_survives_a_constraints_value_that_is_not_a_list():
    assert CN.extract(AI(constraints_extract={"constraints": 5}), "make it short please") == []


def test_extract_with_no_notes_or_only_an_outline_asks_nothing():
    ai = AI()
    assert CN.extract(ai, "") == [] and CN.extract(ai, "1. Intro\n2. Middle\n- bullet") == [] and ai.calls == []


# ------------------------------------------------------------------------------------------------------------- ground
def test_a_quote_said_twice_far_apart_is_ambiguous_but_a_quote_said_twice_close_by_is_not():
    far = _words(gap=40.0)
    got = CN.ground([_con()], far, M.sentences_from_words(far), [], [])[0]
    assert got["status"] == "ambiguous" and len(got["candidates"]) == 2
    close = _words(TEXTS[:2] + TEXTS[1:2], gap=1.0)
    got = CN.ground([_con()], close, M.sentences_from_words(close), [], [])[0]
    assert got["status"] == "grounded"


def test_a_quote_that_is_not_in_the_transcript_is_ungrounded_not_guessed():
    w = _words()
    got = CN.ground([_con(source='keep "the dragon ate the moon"')], w, M.sentences_from_words(w), [], [])[0]
    assert got["status"] == "ungrounded" and not got["anchor"]["spans"]
    assert CN.ground([_con()], [], [], [], [])[0]["status"] == "ungrounded"           # no transcript at all


def test_a_zero_second_length_is_not_a_band():
    d = _con(type="duration", source="0 seconds", rule={"text": "0 seconds"})
    assert CN.ground([d], _words(), [], [], [])[0]["status"] == "ungrounded"


def test_an_exclude_that_is_ambiguous_makes_no_filter_and_stops_for_the_person():
    far = _words(gap=40.0)
    sents = M.sentences_from_words(far)
    c = CN.ground([_con(type="exclude_words", source='cut "we were wrong about the schedule"')], far, sents, [], [])
    assert c[0]["status"] == "ambiguous" and CN.filters(c, sents, far)["exclude_sentences"] == set()
    plan = CN.finish(None, {"items": [{"kind": "talk", "a": 0, "b": 4}]}, {"constraints": c, "filters": CN.filters(c, sents, far)}, far, sents,
                     [{"name": "a.mp4", "start": 0, "end": far[-1]["end"] + 1}], [])
    assert plan["awaiting_approval"] is True and plan["approval"]["questions"]


# ------------------------------------------------------------------------------------------------------------- the repair loop
def _files(w):
    return [{"name": "a.mp4", "start": 0, "end": w[-1]["end"] + 2}]


def test_a_repair_that_invents_ids_is_thrown_away_and_the_plan_is_kept():
    w = _words()
    sents = M.sentences_from_words(w)
    c = CN.ground([_con(type="duration", source="10 seconds", rule={"text": "10 seconds"})], w, sents, [], [])
    plan = {"items": [{"kind": "talk", "a": 0, "b": 4}]}
    ai = AI(plan_repair={"items": [{"kind": "talk", "a": 999, "b": 1000}], "unmet": [{"constraint": "c1", "why": "no way"}]})
    new, res, calls = CN.run_repair(ai, plan, c, lambda p: CN.check_plan(p, c, _files(w), w, [], sents=sents))
    assert new["items"] == plan["items"] and calls == CN.REPAIR_TRIES and not res["ok"]
    assert new["unmet"] == [{"constraint": "c1", "why": "no way"}]


def test_a_repair_with_wrong_types_or_a_failing_call_never_raises():
    w = _words()
    sents = M.sentences_from_words(w)
    c = CN.ground([_con(type="duration", source="10 seconds", rule={"text": "10 seconds"})], w, sents, [], [])
    plan = {"items": [{"kind": "talk", "a": 0, "b": 4}]}
    chk = lambda p: CN.check_plan(p, c, _files(w), w, [], sents=sents)       # noqa: E731
    for bad in ({"items": "nope"}, {"items": [{"kind": "talk", "a": "1", "b": 2}]}, {"items": [{"kind": "visual", "start": "x", "end": 2}]}, {"items": []}, {}, []):
        new, res, calls = CN.run_repair(AI(plan_repair=bad), plan, c, chk, tries=1)
        assert new["items"] == plan["items"]

    class Boom:
        def complete_json(self, *a, **k):
            raise RuntimeError("down")
    assert CN.run_repair(Boom(), plan, c, chk)[2] == 1


def test_an_unmet_claim_by_the_repair_model_for_a_satisfied_rule_is_not_believed():
    w = _words()
    sents = M.sentences_from_words(w)
    inc = CN.ground([_con(source='keep "welcome everyone to our summer camp"')], w, sents, [], [])
    dur = CN.ground([_con(id="c2", type="duration", source="10 seconds", rule={"text": "10 seconds"})], w, sents, [], [])
    cons = inc + dur
    ai = AI(plan_repair={"items": [{"kind": "talk", "a": 0, "b": 4}], "unmet": [{"constraint": "c1", "why": "x"}]})
    plan = CN.finish(ai, {"items": [{"kind": "talk", "a": 0, "b": 4}]}, {"constraints": cons, "filters": {}}, w, sents, _files(w), [])
    assert "c1" not in {u["constraint"] for u in plan["unmet"]}


# ------------------------------------------------------------------------------------------------------------- check_plan on odd stored plans
def test_check_plan_on_a_plan_without_any_of_the_new_keys():
    w = _words()
    sents = M.sentences_from_words(w)
    r = CN.validate_edited_plan({"items": [{"kind": "talk", "a": 0, "b": 3}]}, _files(w), w, [], sents)
    assert r["ok"] and r["totals"]["length_s"] > 0
    assert CN.check_plan({}, None, [], [], None)["ok"] is True


@pytest.mark.parametrize("bad", [{"items": [{"kind": "visual", "start": 1}]}, {"items": [{"kind": "talk", "a": 0, "b": 1, "t0": "abc"}]}, {"items": ["x"]},
                                 {"items": [{"kind": "talk", "a": [0], "b": 1}]}])
def test_check_plan_does_not_crash_on_a_damaged_saved_plan(bad):
    w = _words()
    sents = M.sentences_from_words(w)
    dur = CN.ground([_con(type="duration", source="1 minute", rule={"text": "1 minute"})], w, sents, [], [])
    CN.check_plan(bad, dur, _files(w), w, [], sents=sents)


def test_planapi_cleans_items_before_the_validator_sees_them():
    from clipfarm.clipper import planapi
    for bad in (None, [], "x", [1], [{"kind": "talk", "a": "1", "b": 2}], [{"kind": "visual", "start": 1}], [{"kind": "talk", "a": 1}]):
        with pytest.raises(ValueError):
            planapi._clean_items(bad)
    ok = planapi._clean_items([{"id": "keep-me", "kind": "talk", "a": 1, "b": 2, "satisfies": ["c1"]}])
    assert ok[0]["id"] == "keep-me" and ok[0]["satisfies"] == ["c1"]


# ------------------------------------------------------------------------------------------------------------- transcript search
def test_quote_search_edge_inputs():
    w = _words()
    assert TS.quote_search(w, "") == [] and TS.quote_search(w, "welcome") == [] and TS.quote_search([], "we were wrong") == []
    assert TS.quote_search(w, "!!! ??? ...") == []
    hits = TS.quote_search(w, "WE WERE wrong, about the schedule!!")
    assert len(hits) == 2 and hits[0]["first_word"] < hits[1]["first_word"]
    assert TS.far_apart([]) is False and TS.far_apart(hits[:1]) is False
    assert TS.search(w, [], "x y z", "topic", None) == []                           # topic without an AI


# ------------------------------------------------------------------------------------------------------------- speech trim
def test_snap_cuts_never_cuts_inside_a_word_on_random_speech():
    import random
    rnd = random.Random(7)
    for _ in range(600):
        t, words = 0.0, []
        for _i in range(rnd.randint(2, 40)):
            t += rnd.choice([0.01, 0.05, 0.1, 0.2, 0.5, 1.5])
            d = rnd.choice([0.05, 0.2, 0.4, 0.8])
            words.append({"text": "w", "start": round(t, 3), "end": round(t + d, 3)})
            t += d
        ivs = [(a, a + rnd.uniform(0.1, 5)) for a in (rnd.uniform(0, t) for _k in range(3))]
        for a2, b2 in ST.snap_cuts(ivs, words, 0.08):
            for x in (a2, b2):
                assert not any(w["start"] + 1e-3 < x < w["end"] - 1e-3 for w in words), (x, words)


def test_a_cut_end_just_inside_the_next_words_start_does_not_pull_that_word_in():
    w = [{"text": "done.", "start": 0.0, "end": 0.9}, {"text": "Next", "start": 1.0, "end": 2.0}]
    (a, b), = ST.snap_cuts([(0.0, 1.04)], w, 0.08)
    assert b < 1.0 + 1e-6


def test_snap_cuts_with_no_words_or_no_intervals_is_a_no_op():
    assert ST.snap_cuts([(1.0, 2.0)], []) == [(1.0, 2.0)] and ST.snap_cuts([], [{"start": 0, "end": 1}]) == []


def test_repace_style_on_empty_and_single_word_input():
    style = {"pace": {"gap_shrink_over_s": 0.5}}
    assert ST.repace_style([], 0.0, 5.0, style) == [(0.0, 5.0)]
    one = ST.repace_style([{"text": "hi", "start": 1.0, "end": 1.3}], 0.0, 5.0, style)
    assert len(one) == 1 and one[0][0] <= 1.0 < 1.3 <= one[0][1]


# ------------------------------------------------------------------------------------------------------------- pool rules
def test_rules_from_survives_garbage_options():
    r = PR.rules_from({"avoid": "str", "exclude_files": [None, " ", "C0003"], "exclude_windows": [[1], "ab", [5, 2], [-1, 3], [1, "x"], [2, 4]], "min_shot_s": "abc"})
    assert r == {"avoid": [], "exclude_files": ["None", "C0003"], "exclude_windows": [[2.0, 4.0]], "min_shot_s": 0.0} or r["exclude_windows"] == [[2.0, 4.0]]
    assert PR.rules_from(None)["exclude_windows"] == []


def test_file_excluded_matches_name_stem_and_scene_id_case_blind_and_never_the_empty_name():
    sc = {"sid": "s0041", "file": "Cam A - C0003.MP4"}
    assert PR.file_excluded(sc, ["S0041"]) and PR.file_excluded(sc, ["cam a - c0003"]) and not PR.file_excluded(sc, ["C0003"])
    assert not PR.file_excluded({"file": ""}, [""]) and not PR.file_excluded(sc, [])


def test_cache_lookup_keeps_two_same_named_files_apart(tmp_path):
    a, b = tmp_path / "A" / "C0001.MP4", tmp_path / "B" / "C0001.MP4"
    for p, body in ((a, b"x" * 100), (b, b"y" * 100)):
        p.parent.mkdir()
        p.write_bytes(body)
    fa, fb = {"name": "C0001.MP4", "path": str(a), "size": 100}, {"name": "C0001.MP4", "path": str(b), "size": 100}
    cache = {"C0001.MP4": {"size": 100, "words": ["a"], "key": PR.cache_key(fa)}}
    assert PR.cache_lookup(cache, fa)["words"] == ["a"]
    assert PR.cache_lookup(cache, fb) is None                                      # another camera's file of the same name and size
    assert PR.cache_lookup({}, fa) is None and PR.cache_lookup({"x": "junk"}, {"name": "x"}) is None


def test_music_like_names():
    assert PR.music_like(["Music", "a.wav"]) and PR.music_like(["x", "Soundtrack 2.mp3"]) and not PR.music_like(["Track 1.wav"]) and not PR.music_like([])


# ------------------------------------------------------------------------------------------------------------- stylefetch
@pytest.mark.parametrize("src,kind", [
    ("https://www.instagram.com/reel/Cx12/?igsh=abc", "instagram"), ("instagram.com/reel/Cx12", "instagram"), ("https://instagr.am/p/x", "instagram"),
    ("https://www.instagram.com/someone/", "instagram"), ("https://youtu.be/abc", "youtube_video"), ("https://youtu.be/", "other"),
    ("https://www.youtube.com/shorts/abc", "youtube_video"), ("https://www.youtube.com/watch?v=a&list=b", "youtube_video"),
    ("https://www.youtube.com/playlist?list=b", "youtube_playlist"), ("https://www.youtube.com/@name/shorts", "youtube_channel"),
    ("https://www.youtube.com/", "other"), ("https://youtube.com.evil.com/watch?v=a", "other"), ("https://notinstagram.com/x", "other"),
    ("javascript:alert(1)", "other"), ("", "other"), ("   ", "other"), ("/tmp/x.mp4", "file"), ("~/clip.MOV", "file"), ("file:///tmp/x.mp4", "file"),
    ("https://example.com/x.mp4", "other"), ("http://[::1", "other")])
def test_classify_link_variants(src, kind):
    assert stylefetch.classify(src) == kind


def _fake_ydl_factory(make):
    class Y:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            return make(self.opts, url)
    return Y


def test_a_downloader_that_returns_nothing_or_no_file_is_a_plain_error(tmp_path):
    f = _fake_ydl_factory(lambda o, u: None)
    with pytest.raises(stylefetch.FetchError):
        stylefetch.fetch("https://www.youtube.com/watch?v=abc", tmp_path / "w", ydl_factory=f)
    f = _fake_ydl_factory(lambda o, u: {"entries": []})
    with pytest.raises(stylefetch.FetchError):
        stylefetch.fetch("https://www.youtube.com/watch?v=abc", tmp_path / "w2", ydl_factory=f)
    f = _fake_ydl_factory(lambda o, u: (_ for _ in ()).throw(RuntimeError("HTTP 403 " + "x" * 900)))
    with pytest.raises(stylefetch.FetchError) as e:
        stylefetch.fetch("https://www.youtube.com/watch?v=abc", tmp_path / "w3", ydl_factory=f)
    assert len(str(e.value)) <= 240


def test_a_channel_with_no_videos_or_only_long_ones_is_a_plain_error(tmp_path):
    f = _fake_ydl_factory(lambda o, u: {"entries": [{"id": "a", "duration": 99999}]})
    with pytest.raises(stylefetch.FetchError):
        stylefetch.fetch("https://www.youtube.com/@x", tmp_path / "w", ydl_factory=f)
    f = _fake_ydl_factory(lambda o, u: {"entries": []})
    with pytest.raises(stylefetch.FetchError):
        stylefetch.fetch("https://www.youtube.com/playlist?list=a", tmp_path / "w2", ydl_factory=f)


def test_the_fallback_download_stops_at_the_size_limit_and_never_signs_in(tmp_path):
    class Resp:
        def __init__(self):
            self.n = 0

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, k):
            self.n += 1
            return b"0" * k

    import urllib.request
    orig = urllib.request.urlopen
    urllib.request.urlopen = lambda *a, **k: Resp()
    try:
        with pytest.raises(stylefetch.FetchError):
            stylefetch.http_download("http://x/y.mp4", tmp_path / "o.mp4", max_bytes=3 * 1024 * 1024)
    finally:
        urllib.request.urlopen = orig
    seen = {}
    fac = _fake_ydl_factory(lambda o, u: seen.update(o) or None)
    with pytest.raises(stylefetch.FetchError):
        stylefetch._download_one("https://www.instagram.com/reel/x/", tmp_path / "w", "v0", fac)
    assert "cookiefile" not in stylefetch._ydl({"cookiefile": "/x", "cookiesfrombrowser": ("chrome",), "username": "u"}).params


def test_the_endpoint_answer_with_a_non_http_media_address_is_refused(tmp_path):
    got = []
    with pytest.raises(stylefetch.FetchError):
        stylefetch._via_endpoint("https://www.instagram.com/reel/x/", "https://dl.example/api", tmp_path, "v0",
                                 lambda u, b: {"status": "tunnel", "url": "file:///etc/hosts"}, lambda u, d: got.append(u))
    assert got == []


# ------------------------------------------------------------------------------------------------------------- presets and the routes
class FakeAPI:
    def __init__(self, root):
        from clipfarm.clipper import studio as SD
        self.root = Path(root)
        self.settings, self.styles = SD.Settings(root), SD.Styles(root)
        self.cfg = SimpleNamespace(get=lambda k, d=None: d)

    def sync_long_style(self, extra=None):
        pass

    def sync_points_style(self, mode, pts, extra=None):
        pass

    def style_view(self, kind):
        return {}


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEOBUILDER_HOME", str(tmp_path / "h"))
    return FakeAPI(tmp_path / "h")


def test_preset_names_with_unicode_emoji_and_huge_length(api):
    pr = styles_api.presets(api)
    long_ = "é" * 500
    a = pr.create(long_)
    assert len(a["name"]) <= 60
    b = pr.create(long_)
    assert b["name"] != a["name"]
    for n in ("日本語のスタイル", "Fast 🎬 reels", "<script>alert(1)</script>", "a\nb\tc", "‮RTL"):
        v = pr.create(n)
        assert v["name"] and "\n" not in v["name"] and "\t" not in v["name"]
    assert len(pr.summary()) == 7
    for k in range(SP.MAX_PRESETS):
        try:
            pr.create(f"n{k}")
        except ValueError:
            break
    with pytest.raises(ValueError):
        pr.create("one too many")


def test_the_style_routes_reject_bad_bodies_without_a_500(api):
    pid = styles_api.post(api, ["presets"], {"name": "A"})[1]["preset"]["id"]
    cases = [(["presets", pid, "remove-rule"], {"index": "abc"}), (["presets", pid, "remove-rule"], {"index": None}), (["presets", pid, "remove-rule"], {"index": 99}),
             (["presets", pid, "rule"], {}), (["presets", pid, "rename"], {"name": None}), (["presets", pid, "apply"], {"mode": "weird"}),
             (["presets", pid, "apply"], {}), (["presets", pid, "points"], {"points": {"a": 1}}), (["learn"], {"url": None}), (["learn"], {"url": 5}),
             (["learn"], {"file": "relative/x.mp4"}), (["learn"], {"file": "/tmp"}), (["learn"], {"file": "/etc/passwd"}),
             (["presets", "../../etc", "delete"], {})]
    for rest, body in cases:
        code, out = styles_api.post(api, rest, body)
        assert code in (200, 400, 404) and (code == 200 or "error" in out), (rest, body, code)
    assert styles_api.post(api, ["presets", pid, "nonsense"], {}) is None and styles_api.post(api, ["nothing"], {}) is None
    assert styles_api.get(api, ["learn", "../x"]) is None and styles_api.get(api, ["learn", "sl00000000"])[0] == 404
    code, out = styles_api.post(api, ["presets", pid, "points"], {"points": ["x" * 5000, 7, True, None, ["n"]]})
    assert code == 200


def test_a_dropped_non_video_zero_byte_or_audio_only_file_is_a_plain_error(api, tmp_path):
    z = tmp_path / "zero.mp4"
    z.write_bytes(b"")
    t = tmp_path / "a.txt"
    t.write_text("hi")
    m = tmp_path / "m.mp3"
    m.write_bytes(b"x")
    assert styles_api.post(api, ["learn"], {"file": str(t)})[0] == 400 and styles_api.post(api, ["learn"], {"file": str(m)})[0] == 400
    code, out = styles_api.post(api, ["learn"], {"file": str(z)})                    # a .mp4 name with nothing in it: the job fails in words, not a trace
    assert code == 200
    for _ in range(100):
        v = styles_api.job_view(api, out["job"])
        if v["state"] != "running":
            break
        time.sleep(0.1)
    assert v["state"] == "error" and v["error"] and "Traceback" not in v["error"] and str(tmp_path) not in v["error"]
    assert styles_api.presets(api).all() == []


def test_three_learning_jobs_at_once_are_refused_beyond_the_limit(api, monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr(styles_api.stylelearn, "learn", lambda s, **kw: gate.wait(5) and (_ for _ in ()).throw(stylefetch.FetchError("stopped")))
    ids = []
    for k in range(styles_api.MAX_RUNNING):
        code, out = styles_api.post(api, ["learn"], {"url": f"https://www.youtube.com/watch?v=a{k}"})
        assert code == 200
        ids.append(out["job"])
    code, out = styles_api.post(api, ["learn"], {"url": "https://www.youtube.com/watch?v=zz"})
    assert code == 400 and "already" in out["error"]
    assert len(styles_api.summary(api)["running"]) == styles_api.MAX_RUNNING
    gate.set()
    for _ in range(100):
        if not styles_api.summary(api)["running"]:
            break
        time.sleep(0.05)
    assert all(styles_api.job_view(api, i)["state"] == "error" for i in ids)


def test_one_job_finishing_does_not_delete_the_frames_of_another_running_job(api, monkeypatch):
    pr = styles_api.presets(api)
    mid, go = threading.Event(), threading.Event()

    kept = {}

    def learn(src, frames_dir=None, **kw):
        frames_dir.mkdir(parents=True, exist_ok=True)
        kept[src[-1]] = frames_dir
        f = frames_dir / f"{src[-1]}.jpg"
        f.write_bytes(b"\xff\xd8x")
        meas = {k: None for k in ("duration_s",)}
        if src.endswith("B"):
            mid.set()
            go.wait(5)
        return {"sources": [{"url": src, "title": src[-1], "kind": "youtube_video", "added": 1, "measurements": meas}], "frames": [f]}
    monkeypatch.setattr(styles_api.stylelearn, "learn", learn)
    b = styles_api.post(api, ["learn"], {"url": "https://www.youtube.com/watch?v=B"})[1]["job"]
    mid.wait(5)
    a = styles_api.post(api, ["learn"], {"url": "https://www.youtube.com/watch?v=A", "name": "first"})[1]["job"]
    for _ in range(100):
        if styles_api.job_view(api, a)["state"] != "running":
            break
        time.sleep(0.05)
    assert (kept["B"] / "B.jpg").exists()                                  # job A is done; job B has not finished yet
    go.set()
    for _ in range(100):
        if styles_api.job_view(api, b)["state"] != "running":
            break
        time.sleep(0.05)
    v = styles_api.job_view(api, b)["preset"]
    assert v and v["frames"]


# ------------------------------------------------------------------------------------------------------------- the MCP tools with bad arguments
def test_phase2_mcp_tools_exist_with_schemas_and_handlers():
    from clipfarm.clipper import mcp_server as MCP
    names = {t["name"] for t in MCP.TOOLS}
    for n in ("plan_long_video", "check_plan", "transcript_search", "shot_details", "render_preview", "rerender", "learn_style", "list_style_presets", "apply_style_preset"):
        assert n in names and n in MCP.HANDLERS
    for t in MCP.TOOLS:
        assert t["inputSchema"]["type"] == "object", t["name"]


class _Eng:
    """An engine whose server answers from a script, to see what a tool sends and how it copes with odd answers."""
    def __init__(self, get=None, post=None):
        self._get, self._post, self.sent = get or {}, post or {}, []

    def get(self, path, *a, **k):
        self.sent.append(("GET", path))
        v = self._get.get(path)
        if isinstance(v, Exception):
            raise v
        return v if v is not None else {}

    def post(self, path, body=None, *a, **k):
        self.sent.append(("POST", path, body))
        v = self._post.get(path)
        if isinstance(v, Exception):
            raise v
        return v if v is not None else {}


def test_learn_style_tool_bad_arguments():
    from clipfarm.clipper import mcp_server as MCP
    for a in ({}, {"url": ""}, {"url": "   "}, {"url": None}):
        with pytest.raises(MCP.ToolError):
            MCP.tool_learn_style(_Eng(), a)
    with pytest.raises(MCP.ToolError):
        MCP.tool_learn_style(_Eng(get={"/api/app/styles/presets": {"presets": []}}), {"url": "https://youtu.be/a", "preset": "nothing"})
    with pytest.raises(MCP.ToolError):
        MCP.tool_apply_style_preset(_Eng(get={"/api/app/styles/presets": {"presets": [{"id": "st1", "name": "Fast"}]}}), {"preset": "fast", "mode": "medium"})
    with pytest.raises(MCP.ToolError):
        MCP.tool_apply_style_preset(_Eng(get={"/api/app/styles/presets": {"presets": [{"id": "st1", "name": "Fast"}]}}), {"preset": "", "mode": "short"})


def test_learn_style_tool_sends_a_path_as_a_file_and_a_link_as_a_url(monkeypatch):
    from clipfarm.clipper import mcp_server as MCP
    monkeypatch.setattr(MCP.time, "sleep", lambda s: None)
    e = _Eng(post={"/api/app/styles/learn": {"job": "sl12345678"}}, get={"/api/app/styles/learn/sl12345678": {"state": "error", "error": "no"}})
    for url, key in (("/Users/x/v.mp4", "file"), ("~/v.mp4", "file"), ("https://youtu.be/a", "url")):
        with pytest.raises(MCP.ToolError):
            MCP.tool_learn_style(e, {"url": url})
        assert key in e.sent[-2][2]


def test_the_plan_tools_pass_a_missing_plan_to_the_server_which_refuses_it():
    from clipfarm.clipper import mcp_server as MCP
    with pytest.raises(Exception):
        MCP.tool_check_plan(_Eng(), {"plan": {"items": []}})                        # no project: refused before any request


def test_a_clean_item_keeps_protect_and_planned_seconds():
    from clipfarm.clipper import planapi
    got = planapi._clean_items([{"id": "x", "kind": "talk", "a": 1, "b": 2, "protect": True, "t0": 4.5, "t1": "9.25"}])
    assert got[0]["protect"] is True and got[0]["t0"] == 4.5 and got[0]["t1"] == 9.25


def test_preview_with_a_start_that_is_not_a_number_says_so_in_plain_words(monkeypatch):
    from clipfarm.clipper import planapi
    monkeypatch.setattr(planapi, "_final", lambda runner, pid: Path("/nonexistent.mp4"))
    monkeypatch.setattr(planapi.ffmpeg, "duration", lambda p: 30.0)
    with pytest.raises(ValueError) as e:
        planapi.preview(None, "p", "abc")
    assert "numbers of seconds" in str(e.value) and "could not convert" not in str(e.value)


def test_scene_ids_chosen_from_the_first_list_follow_their_scenes_into_the_final_pool():
    from clipfarm.clipper import localproj as LP
    before = [{"sid": "s0000", "file": "A.mp4", "fstart": 0.0}, {"sid": "s0001", "file": "B.mp4", "fstart": 5.0}]
    after = [{"sid": "s0000", "file": "B.mp4", "fstart": 5.0}]
    cons = [{"anchor": {"scene_ids": ["s0001", "s0000", "zzz"]}}, {"anchor": {}}, "x"]
    LP.remap_scene_ids(cons, before, after)
    assert cons[0]["anchor"]["scene_ids"] == ["s0000"]
