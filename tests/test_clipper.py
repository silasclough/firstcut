import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError

import pytest

from clipfarm.clipper import moments as M
from clipfarm.clipper import reframe as R
from clipfarm.clipper.engine import remap_words, remapper, repace
from clipfarm.clipper.store import Store


def words_at(times):
    return [{"text": f"w{i}", "start": a, "end": b} for i, (a, b) in enumerate(times)]


def test_repace_cuts_long_pauses_only():
    w = words_at([(10.0, 10.4), (10.5, 10.9), (13.0, 13.4), (13.5, 13.9)])   # 2.1 s pause in the middle
    iv = repace(w, 9.0, 15.0, gap_max=0.5)
    assert len(iv) == 2
    assert iv[0][0] == pytest.approx(9.88) and iv[0][1] == pytest.approx(11.12)
    assert iv[1][0] < 13.0 < iv[1][1]
    assert repace(w, 9.0, 15.0, gap_max=5.0) == [(pytest.approx(9.88), pytest.approx(14.57))]


def test_remap_and_word_retiming_through_cuts():
    iv = [(10.0, 11.0), (13.0, 14.0)]
    f = remapper(iv)
    assert f(10.5) == pytest.approx(0.5) and f(13.5) == pytest.approx(1.5) and f(12.0) == pytest.approx(1.0)
    assert f(20.0) is None
    assert remapper(iv, speed=2.0)(13.5) == pytest.approx(0.75)
    out = remap_words(words_at([(10.2, 10.6), (13.1, 13.4), (16, 17)]), iv, 1.0)
    assert [(w["start"], w["end"]) for w in out] == [(0.2, 0.6), (1.1, 1.4)]


def test_sentences_split_on_punctuation_and_pauses():
    w = [{"text": "Hello", "start": 0, "end": .3}, {"text": "there.", "start": .35, "end": .7},
         {"text": "So", "start": .8, "end": 1}, {"text": "anyway", "start": 1.05, "end": 1.4},
         {"text": "later", "start": 3.0, "end": 3.3}]
    s = M.sentences_from_words(w)
    assert [x.text for x in s] == ["Hello there.", "So anyway", "later"]


def _sents(n, each=5.0):
    return [M.Sentence(i, i * each, i * each + each - 0.2, f"s{i}") for i in range(n)]


def test_fit_length_grows_and_shrinks():
    s = _sents(40)
    m = M.fit_length(M.Moment(0, 0, 10, 10), s, 20, 40)
    assert 20 <= m.duration <= 40 and m.first >= 8
    m = M.fit_length(M.Moment(0, 0, 0, 9), s, 20, 40)              # 50 s: a bit long, kept whole (payoff)
    assert m.last == 9 and 40 < m.duration <= M.hard_cap(40)
    m = M.fit_length(M.Moment(0, 0, 0, 30), s, 20, 40)             # way too long: trimmed to the hard cap
    assert m.duration <= M.hard_cap(40) and m.first == 0


def test_dedupe_and_top_up_prefer_llm_picks():
    a = M.Moment(0, 30, 0, 5, score=8)
    b = M.Moment(5, 35, 1, 6, score=6)           # overlaps a heavily
    c = M.Moment(100, 130, 20, 25, score=5)
    assert [m.score for m in M.dedupe([b, a, c])] == [8, 5]
    h = [M.Moment(10, 40, 2, 7, score=9.5, source="heuristic"), M.Moment(200, 230, 40, 45, score=4, source="heuristic")]
    out = M.top_up([a], h, 3)
    assert [m.start for m in out] == [0, 200]       # the overlapping heuristic one is skipped despite its score


def test_chunks_cover_everything_with_overlap():
    s = _sents(300, each=5.0)                        # 25 minutes
    blocks = M.chunks(s, chunk_s=480, overlap_s=45)
    assert blocks[0][0].idx == 0 and blocks[-1][-1].idx == 299
    assert all(b[0].start < a[-1].end for a, b in zip(blocks, blocks[1:]))   # consecutive chunks overlap


def test_crop_path_hysteresis_and_expression():
    F = R.Face
    samples = [[F(0, .30, .5, .1, .9)]] * 6 + [[F(0, .32, .5, .1, .9)]] * 2 + [[F(0, .75, .5, .1, .9)]] * 6
    path = R.crop_path(samples, rate=2.0, crop_frac=0.3164, dead_zone=0.10, min_shot_s=1.6)
    assert len(path) == 2 and path[0][1] == pytest.approx(.30) and path[1] == (4.0, pytest.approx(.75))
    expr = R.x_expression(path, 1920, 608, lambda t: t)
    assert expr.startswith("if(lt(t,4.000),272,") and expr.endswith("1136)")
    assert R.x_expression([], 1920, 608, lambda t: t) == "656"


def test_title_card_renders(tmp_path):
    from PIL import Image

    from clipfarm.clipper.engine import title_card
    title_card("This is a long hook title that should wrap onto two lines", tmp_path / "t.png", 1080,
               ["/System/Library/Fonts/Supplemental/Arial Black.ttf"])
    im = Image.open(tmp_path / "t.png")
    assert im.mode == "RGBA" and im.width < 1080 and im.height > 100


# ----------------------------------------------------------------------------- HTTP API
class FakeFeed:
    on = True

    def queue_info(self, qid):
        return {"state": "review"}


class FakeEngine:
    ccfg = {"default_clips": 10, "min_len": 20, "max_len": 55}

    @property
    def cfg(self):                                    # (the server reads the config for ChatGPT sign-in state)
        from clipfarm.core.config import load_config
        return load_config()

    def status(self):
        return {"llm": [], "openai_key": False, "transcriber": "local", "whisper_model": "base.en",
                "caption_styles": {"bold": "Bold"}, "layouts": {"auto": "Auto"}, "paces": ["tight"], }

    def process(self, job_id):
        pass


@pytest.fixture
def api(tmp_path):
    from clipfarm.clipper.server import Worker, make_handler
    store = Store(tmp_path / "clipper")
    worker = Worker(FakeEngine(), store, FakeFeed())                     # not started: jobs stay queued
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(FakeEngine(), store, worker, FakeFeed(), 1))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def call(path, data=None, method=None, raw=None, headers=None):
        body = raw if raw is not None else (json.dumps(data).encode() if data is not None else None)
        req = urllib.request.Request(base + path, data=body, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, r.read(), dict(r.headers)
        except HTTPError as e:
            return e.code, e.read(), dict(e.headers)
    yield call, store
    httpd.shutdown()


def test_api_create_requires_a_url(api):
    call, store = api
    assert call("/api/jobs", {"url": "notaurl"})[0] == 400
    code, body, _ = call("/api/jobs", {"url": "https://youtu.be/x", "clips": 99})
    job = json.loads(body)
    assert code == 201 and job["status"] == "queued" and job["options"]["clips"] == 30
    assert json.loads(call("/api/jobs")[1])[0]["id"] == job["id"]
    assert call("/", None)[0] == 200


def test_api_upload_media_range_and_delete(api):
    call, store = api
    code, body, _ = call("/api/upload?name=talk.mp4", raw=b"\x00" * 1000, method="POST",
                         headers={"Content-Type": "application/octet-stream"})
    job = json.loads(body)
    assert code == 201 and (store.dir(job["id"]) / "source.mp4").stat().st_size == 1000
    assert job["status"] == "draft" and store.next_queued() is None                 # an upload doesn't start processing by itself
    (store.dir(job["id"]) / "clip_01.mp4").write_bytes(bytes(range(256)) * 4)
    code, data, hdr = call(f"/media/{job['id']}/clip_01.mp4", headers={"Range": "bytes=10-19"})
    assert code == 206 and data == bytes(range(10, 20)) and hdr["Content-Range"] == "bytes 10-19/1024"
    assert call(f"/media/{job['id']}/source.mp4")[0] == 404                         # only clip files are served
    assert call(f"/media/{job['id']}/..%2Fjob.json")[0] == 404
    assert call("/api/upload?name=x.exe", raw=b"x", method="POST")[0] == 400
    assert call(f"/api/jobs/{job['id']}", method="DELETE")[0] == 200
    assert not store.dir(job["id"]).exists()


def test_store_requeues_interrupted_jobs(tmp_path):
    s = Store(tmp_path)
    j = s.create({"url": "https://x"}, {})
    s.update(j["id"], status="rendering")
    assert s.requeue_interrupted() == [j["id"]] and s.get(j["id"])["status"] == "queued"


def test_faceless_spans():
    F = R.Face
    face = [F(0, .5, .5, .1, .9)]
    samples = [face] * 3 + [[]] * 4 + [face] * 2 + [[]] * 1 + [[]] * 3
    assert R.faceless_spans(samples, rate=2.0, min_s=1.0) == [(1.5, 3.5), (4.5, 6.5)]


def test_window_and_busy_flag(tmp_path):
    import time
    from datetime import datetime

    from clipfarm.core import sharing
    assert sharing.in_window(datetime(2026, 1, 1, 23, 0), "17:00", "08:00")
    assert sharing.in_window(datetime(2026, 1, 1, 7, 59), "17:00", "08:00")
    assert not sharing.in_window(datetime(2026, 1, 1, 12, 0), "17:00", "08:00")
    (tmp_path / "data/state").mkdir(parents=True)
    flag = tmp_path / sharing.FLAG
    assert not sharing.m2_busy(tmp_path)
    flag.write_text(json.dumps({"busy": True, "at": time.time(), "est_s": 3000}))
    assert sharing.m2_busy(tmp_path)
    flag.write_text(json.dumps({"busy": True, "at": time.time() - 9000, "est_s": 3000}))
    assert not sharing.m2_busy(tmp_path)                       # stale: the M2 died mid-scene
    flag.write_text(json.dumps({"busy": False, "at": time.time()}))
    assert not sharing.m2_busy(tmp_path)


def test_router_prefers_m2_only_when_allowed(tmp_path, monkeypatch):
    from datetime import datetime

    from clipfarm.clipper.engine import Engine
    from clipfarm.core import llm, sharing
    from clipfarm.core.config import load_config
    cfg = load_config()
    cfg.data["clipper"]["remote_ollama"] = {"base_url": "http://m2:11435", "window": ["17:00", "08:00"]}
    eng = Engine(cfg, Store(tmp_path / "c"))
    monkeypatch.setattr(llm.OllamaLLM, "available", lambda self: (True, "ok"))
    monkeypatch.setattr(sharing, "m2_busy", lambda root: False)
    monkeypatch.setattr(type(cfg), "now", lambda self: datetime(2026, 1, 1, 22, 0))
    router, where = eng.router_for_now()
    assert where == "M2" and router.chain[0] == "ollama_remote" and router.chain.index("openai") < router.chain.index("ollama")
    assert router.cfg["ollama_remote"]["base_url"] == "http://m2:11435"
    monkeypatch.setattr(sharing, "m2_busy", lambda root: True)
    assert eng.router_for_now()[1].endswith("(M2 busy rendering)")
    monkeypatch.setattr(sharing, "m2_busy", lambda root: False)
    monkeypatch.setattr(type(cfg), "now", lambda self: datetime(2026, 1, 1, 12, 0))
    assert "(M2 allowed 17:00-08:00)" in eng.router_for_now()[1]


def test_interrupted_upload_marked_failed(tmp_path):
    s = Store(tmp_path)
    j = s.create({"filename": "big.mp4"}, {})
    (s.dir(j["id"]) / "source.mp4").write_bytes(b"partial")
    s.update(j["id"], status="uploading")
    s.requeue_interrupted()
    j = s.get(j["id"])
    assert j["status"] == "failed" and "upload" in j["error"].lower()
    assert not (s.dir(j["id"]) / "source.mp4").exists()


def test_vision_router_uses_m2_vision_at_night(tmp_path, monkeypatch):
    from datetime import datetime

    from clipfarm.clipper.engine import Engine
    from clipfarm.core import llm, sharing
    from clipfarm.core.config import load_config
    cfg = load_config()
    cfg.data["clipper"]["remote_ollama"] = {"base_url": "http://m2:11435", "vision_model": "qwen3-vl:8b"}
    eng = Engine(cfg, Store(tmp_path / "c"))
    monkeypatch.setattr(llm.OllamaLLM, "available", lambda self: (True, "ok"))
    monkeypatch.setattr(llm.OpenAILLM, "available", lambda self: (True, "ok"))
    monkeypatch.setattr(sharing, "m2_busy", lambda root: False)
    monkeypatch.setattr(type(cfg), "now", lambda self: datetime(2026, 1, 1, 23, 0))
    r, where = eng.vision_router()
    assert where == "M2" and r.chain[:2] == ["ollama_remote", "openai"]
    assert r.cfg["ollama_remote"]["vision_model"] == "qwen3-vl:8b"
    monkeypatch.setattr(type(cfg), "now", lambda self: datetime(2026, 1, 1, 12, 0))
    r, where = eng.vision_router()
    assert where == "OpenAI" and r.chain == ["openai"]            # server Ollama has no vision model


def test_api_batch_upload_becomes_one_job(api):
    call, store = api
    assert call("/api/jobs/batch", {"names": ["a.mp4"]})[0] == 400
    code, body, _ = call("/api/jobs/batch", {"names": ["a.mp4", "b.mov"], "source_type": "glasses"})
    job = json.loads(body)
    assert code == 201 and job["status"] == "uploading" and job["options"]["source_type"] == "glasses"
    up = lambda k, name: call(f"/api/upload?job={job['id']}&index={k}&name={name}", raw=b"\x00" * 50, method="POST")  # noqa: E731
    assert up(5, "x.mp4")[0] == 400
    assert json.loads(up(0, "a.mp4")[1])["stage"] == "Uploading 1 of 2"
    j = json.loads(up(1, "b.mov")[1])
    assert j["status"] == "draft"                                                   # waits for you to set options and press Clip it
    assert call(f"/api/jobs/{job['id']}/start", {"prompt": "only the funny bits", "mode": "long", "clips": "0"})[0] == 200
    started = store.get(job["id"])
    assert started["status"] == "queued" and started["options"]["prompt"] == "only the funny bits" and started["options"]["mode"] == "long"
    assert call(f"/api/jobs/{job['id']}/start", {})[0] == 409                      # can't start it twice
    assert sorted(p.name for p in (store.dir(job["id"]) / "parts").iterdir()) == ["part_000__a.mp4", "part_001__b.mov"]


def test_one_bad_proposal_does_not_sink_the_answer():
    class R:
        def complete_json(self, task, system, prompt, schema, validate=None, images=None):
            d = {"clips": [{"first": 0, "last": 3, "title": "ok", "hook": "", "caption": "", "hashtags": [],
                            "score": 7, "reason": "", "mood": "chill"},
                           {"first": 900, "last": 950, "title": "bad", "hook": "", "caption": "", "hashtags": [],
                            "score": 9, "reason": "", "mood": "epic"}]}
            validate(d)                                   # must not raise for a partly-bad answer

            class Res:
                provider = "fake"
                data = d
            return Res()
    sents = _sents(20)
    out = M.llm_moments(R(), sents, 3, 10, 40)
    assert [m.title for m in out] == ["ok"] and out[0].mood == "chill"



def _talk(texts, gap=0.1, each=3.0):
    out, t = [], 0.0
    for i, x in enumerate(texts):
        out.append(M.Sentence(i, t, t + each, x))
        t += each + gap
    return out


def test_complete_endings_follows_the_thought():
    sents = _talk(["Setup line.", "I dealt with racism from a very young age.", "And not like a little bit of racism.",
                   "It was constant.", "Totally different topic now."])
    m = M.Moment(sents[0].start, sents[1].end, 0, 1)
    M.complete_endings([m], sents, 40)
    assert m.last == 2                             # "And ..." carries on; stops before a sentence that doesn't
    cut = _talk(["So my son will do something", "Hey, stop that.", "Unrelated."])
    m2 = M.Moment(0, cut[0].end, 0, 0)
    M.complete_endings([m2], cut, 40)
    assert m2.last == 1                            # unfinished last sentence -> extend to where it ends


def test_llm_fix_endings_moves_end_within_cap():
    sents = _talk(["Hook?", "Setup.", "More setup.", "Punchline!", "After.", "x."] + ["y."] * 30, each=5.0)

    class R:
        def complete_json(self, task, system, prompt, schema, **k):
            assert "what is said next" in prompt

            class Res:
                data = {"clips": [{"clip": 0, "last": 3, "why": "payoff"}, {"clip": 1, "last": 35, "why": "too far"}]}
            return Res()
    a = M.Moment(sents[0].start, sents[2].end, 0, 2, title="a")
    b = M.Moment(sents[6].start, sents[8].end, 6, 8, title="b")
    M.llm_fix_endings(R(), [a, b], sents, 30)
    assert a.last == 3 and a.end == sents[3].end
    assert b.last == 8                              # beyond the hard cap -> ignored


def test_repace_keeps_the_ends_of_words():
    from clipfarm.clipper.engine import repace
    words = [{"text": "hello", "start": 1.0, "end": 1.5}, {"text": "there", "start": 3.0, "end": 3.4}]
    (a1, b1), (a2, b2) = repace(words, 0.8, 3.6, 0.4)
    assert b1 >= 1.5 + 0.12 + 0.10 - 1e-6               # cut after "hello" leaves its tail (+100 ms)
    assert b2 >= 3.4 + 0.12 + 0.10 + 0.45 - 1e-6        # and the clip end rings out even longer
    assert a2 > b1                                       # stretches still never overlap


# ----------------------------------------------------------------------------- framing: show more of the scene
def _faces(*cxw):
    from clipfarm.clipper.reframe import Face
    return [Face(0, cx, 0.4, w, 0.9) for cx, w in cxw]


def test_wide_window_centres_on_the_whole_group_and_flags_far_apart_people():
    from clipfarm.clipper import reframe as R
    two = [_faces((0.35, 0.18), (0.55, 0.18))] * 8                     # speaker + listener, close together
    path = R.group_path(two, 2.0, 0.56)
    assert abs(path[0][1] - 0.45) < 0.01                               # centred BETWEEN them, not on one face
    far = [_faces((0.1, 0.12), (0.9, 0.12))] * 6                       # sitting at opposite ends: no window holds both
    assert R.too_wide_spans(far, 2.0, 0.56) == [(0.0, 3.0)]
    assert R.too_wide_spans(two, 2.0, 0.56) == []
    assert R.typical_face_width([_faces((0.9, 0.06))] * 5) < 0.13      # small corner facecam over gameplay
    assert R.typical_face_width(two) > 0.13


def _render(tmp_path, monkeypatch, faces, layout="auto"):
    import subprocess
    from clipfarm.clipper import moments as M, reframe as R
    from clipfarm.clipper.engine import Engine
    from clipfarm.core import ffmpeg
    from clipfarm.core.config import load_config
    cfg = load_config()
    cfg.root = tmp_path
    for k in ("work", "logs", "state", "queue"):
        cfg.data["paths"][k] = str(tmp_path / k)
    store = Store(tmp_path / "clipper")
    job = store.create({"filename": "x.mp4"}, {})
    src = store.dir(job["id"]) / "source.mp4"
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=25:d=4",
                    "-f", "lavfi", "-i", "sine=f=300:d=4", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(src)], check=True)
    monkeypatch.setattr(R, "detect_faces", lambda *a, **k: ([faces] * 8, 8))
    words = [{"text": f"w{i}", "start": 0.3 + i * 0.4, "end": 0.6 + i * 0.4} for i in range(8)]
    m = M.Moment(start=0.2, end=3.6, first=0, last=7, title="T", caption="c", score=8)
    clip = Engine(cfg, store).render_clip(job["id"], 1, m, words, src, layout=layout, style="none", pace="natural", hook_title=False)
    info = ffmpeg.probe(store.dir(job["id"]) / clip["file"])
    v = ffmpeg.video_stream(info)
    return clip, (int(v["width"]), int(v["height"]))


def test_talking_heads_get_the_wide_window(tmp_path, monkeypatch):
    clip, dims = _render(tmp_path, monkeypatch, _faces((0.4, 0.2), (0.6, 0.2)))
    assert clip["layout"] == "wide" and dims == (1080, 1920) and clip["framing_v"] == 3


def test_gameplay_with_a_corner_facecam_puts_the_face_on_top_and_the_game_below(tmp_path, monkeypatch):
    clip, dims = _render(tmp_path, monkeypatch, _faces((0.9, 0.07)))
    assert clip["layout"] == "facecam" and dims == (1080, 1920)


def test_gameplay_with_a_corner_facecam_shows_the_whole_frame_when_the_facecam_layout_is_off(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "facecam_box", lambda *a, **k: None)
    clip, dims = _render(tmp_path, monkeypatch, _faces((0.9, 0.07)))
    assert clip["layout"] == "fit" and dims == (1080, 1920)


def test_people_far_apart_fall_back_to_the_whole_frame_for_that_stretch(tmp_path, monkeypatch):
    clip, dims = _render(tmp_path, monkeypatch, _faces((0.08, 0.15), (0.92, 0.15)))
    assert clip["layout"] == "wide" and clip["fit_spans"] >= 1 and dims == (1080, 1920)


def test_has_video_catches_empty_outputs(tmp_path):
    from clipfarm.core import ffmpeg
    empty = tmp_path / "e.mp4"
    empty.write_bytes(b"x" * 318)                                              # what a failed hardware encode left behind
    assert ffmpeg.has_video(empty) is False and ffmpeg.has_video(tmp_path / "missing.mp4") is False


def test_a_cut_off_video_file_is_detected(tmp_path):
    import subprocess
    from clipfarm.core import ffmpeg
    full = tmp_path / "full.mov"
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x240:r=25:d=20",
                    "-c:v", "libx264", "-movflags", "+faststart", str(full)], check=True)
    cut = tmp_path / "cut.mov"
    cut.write_bytes(full.read_bytes()[: int(full.stat().st_size * 0.45)])          # header intact, data stops part-way
    assert ffmpeg.readable_seconds(full) > 19
    r = ffmpeg.readable_seconds(cut)
    assert 4 < r < 14                                                              # nowhere near the 20 s the header promises


def test_framing_follows_whoever_is_talking():
    import numpy as np
    from clipfarm.clipper import reframe as R
    from clipfarm.clipper.reframe import Face
    rate = 6.0
    samples = []
    for k in range(60):                                             # 10 s: left person talks for 5 s, then the right one
        left_talking = k < 30
        samples.append([Face(k / rate, 0.25, 0.4, 0.14, 0.9, act=2.0 if left_talking else 0.1, track=0),
                        Face(k / rate, 0.75, 0.4, 0.14, 0.9, act=0.1 if left_talking else 2.0, track=1)])
    path = R.speaker_path(samples, rate, 0.4)
    assert path and path[0][1] < 0.45                              # starts on the left speaker (pulled slightly to the group)
    late = [cx for t, cx in path if t > 5.0]
    assert late and late[-1] > 0.55                                # then moves to the right speaker
    assert len(path) <= 4                                          # a couple of deliberate cuts, not a jittery pan
    assert R.speaker_path([[Face(0, 0.5, 0.4, 0.2, 0.9)]] * 20, rate, 0.5) is None   # no lip signal at all -> caller falls back


def test_mouth_motion_ignores_head_movement_but_sees_talking():
    import numpy as np
    from clipfarm.clipper import reframe as R
    rng = np.random.default_rng(0)
    base = rng.random((200, 200)).astype(np.float32) * 50
    box = (50, 50, 100, 100)
    talking = base.copy()
    talking[50 + 65:50 + 95, 50 + 25:50 + 75] += 80                  # only the mouth area changes
    nodding = base + 40                                              # the whole face changes together
    assert R.mouth_motion(base, talking, box) > 5
    assert R.mouth_motion(base, nodding, box) < 1
    assert R.mouth_motion(None, talking, box) == 0.0


# ----------------------------------------------------------------------------- tell the AI what to change in a clip
def test_edit_plan_moves_a_sentence_to_the_front_and_cuts_another():
    from clipfarm.clipper import edit as E
    from clipfarm.clipper import moments as M
    sents = [M.Sentence(i, i * 5.0, i * 5.0 + 4.0, f"sentence {i}.") for i in range(10)]

    class LLM:
        def complete_json(self, task, system, prompt, schema):
            class R:
                data = {"segments": [{"first": 6, "last": 6}, {"first": 3, "last": 5}, {"first": 7, "last": 8}, {"first": 4, "last": 4}],
                        "title": "He fell off the roof", "summary": "Moved sentence 6 to the front, cut 9", "problem": ""}
            return R
    p = E.plan(LLM(), E.window(sents, 15.0, 45.0, pad=10), 15.0, 45.0, [(3, 8)], "put the roof line first, cut the last one", "Old")
    assert p.ranges == [(6, 6), (3, 5), (7, 8)]                                  # moved to the front; nothing plays twice
    assert p.segments[0] == (30.0, 34.0) and p.title == "He fell off the roof" and "front" in p.summary

    class Nope:
        def complete_json(self, *a, **k):
            class R:
                data = {"segments": [], "title": "", "summary": "", "problem": "no sentence like that"}
            return R
    q = E.plan(Nope(), sents, 15.0, 45.0, [(3, 8)], "x", "Old")
    assert q.problem and q.ranges == [(3, 8)] and not q.summary                   # can't do it -> nothing changes, says why


def test_reordered_edit_renders_with_the_hook_first(tmp_path, monkeypatch):
    import subprocess
    from clipfarm.clipper import moments as M, reframe as R
    from clipfarm.clipper.engine import Engine, remapper
    from clipfarm.core import ffmpeg
    from clipfarm.core.config import load_config
    iv = [(30.0, 34.0), (15.0, 30.0)]                                            # a later sentence first, then the start
    f = remapper(iv)
    assert f(31.0) == pytest.approx(1.0) and f(16.0) == pytest.approx(5.0)      # time maps through the interval that holds it
    cfg = load_config()
    cfg.root = tmp_path
    for k in ("work", "logs", "state", "queue"):
        cfg.data["paths"][k] = str(tmp_path / k)
    store = Store(tmp_path / "clipper")
    job = store.create({"filename": "x.mp4"}, {})
    src = store.dir(job["id"]) / "source.mp4"
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=40",
                    "-f", "lavfi", "-i", "sine=f=300:d=40", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(src)], check=True)
    monkeypatch.setattr(R, "detect_faces", lambda *a, **k: ([[]] * 4, 4))
    words = [{"text": f"w{i}", "start": 15.5 + i * 0.5, "end": 15.8 + i * 0.5} for i in range(40)]
    m = M.Moment(start=15.0, end=34.0, first=0, last=1, title="T", caption="c", score=8)
    clip = Engine(cfg, store).render_clip(job["id"], 1, m, words, src, layout="auto", style="none", pace="natural",
                                          hook_title=False, segments=[(30.0, 34.0), (15.0, 30.0)])
    dur = ffmpeg.duration(store.dir(job["id"]) / clip["file"])
    assert 18 < dur < 21                                                          # all 19 s kept, just reordered
