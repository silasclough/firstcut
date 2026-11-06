"""Long videos: stock B-roll (Pixabay / Pexels, opt-in, only where your own footage doesn't show what is said) and the graphics in the footage
folder (a logo -> the end card, a lower third while that person talks, a title / a see-through video over the opening). Offline: the AI and
every HTTP call are faked."""

import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from clipfarm import stock as SK
from clipfarm.clipper import aivideo as AV
from clipfarm.clipper import foldergfx as FG
from clipfarm.clipper import photos as PH
from clipfarm.clipper.store import Store
from clipfarm.core import ffmpeg
from clipfarm.core.config import load_config
from test_aivideo import _FakeAI, _Res

FF = ffmpeg.bin_path("ffmpeg")


# ------------------------------------------------------------------------------------------------ helpers
def _talk(path: Path, secs: int, size: str = "640x360", color: str | None = None) -> None:
    src = f"color=c={color}:size={size}:rate=30:duration={secs}" if color else f"testsrc2=size={size}:rate=30:duration={secs}"
    subprocess.run([FF, "-y", "-v", "error", "-f", "lavfi", "-i", src, "-f", "lavfi", "-i", f"sine=frequency=300:duration={secs}",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True, timeout=120)


def _alpha_mov(path: Path, secs: float = 3.0) -> None:
    """A ProRes 4444 clip with alpha: an orange disc on nothing."""
    png = path.with_suffix(".src.png")
    im = Image.new("RGBA", (480, 270), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse((140, 35, 340, 235), fill=(255, 140, 0, 255))
    im.save(png)
    subprocess.run([FF, "-y", "-v", "error", "-loop", "1", "-framerate", "30", "-t", str(secs), "-i", str(png), "-vf", "format=rgba",
                    "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", str(path)], check=True, timeout=120)
    png.unlink()


def _noise_photo(path: Path, size=(800, 600)) -> None:
    import numpy as np
    rng = np.random.default_rng(3)
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path, quality=90)


def _setup(tmp_path, monkeypatch, ai):
    from clipfarm.clipper import localproj as LP
    from clipfarm.clipper.engine import Engine
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["paths"].update({"state": str(tmp_path / "state")})
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "x"), "local": True, "data_dir": "ai", "local_projects": True, "verify_freeze": False}
    monkeypatch.setenv("VIDEOBUILDER_HOME", str(tmp_path / "apphome"))       # <home>: where the app keeps its Stock folder
    store = Store(tmp_path / "jobs")
    projects = AV.Projects(cfg, store, Engine(cfg, store))
    monkeypatch.setattr(LP, "LocalAI", lambda c: ai)
    return LP, cfg, store, projects


class _Resp:
    def __init__(self, data=None, body: bytes = b"", status: int = 200):
        self.data, self.body, self.status_code, self.headers = data, body, status, {}

    def json(self):
        return self.data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, n):
        for i in range(0, len(self.body), n):
            yield self.body[i:i + n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeStockWeb:
    """Pixabay and Pexels as far as the code can tell: searches answer from small tables, downloads serve real media files."""

    def __init__(self, video: Path, photo: Path):
        self.calls: list[str] = []
        self.video, self.photo = video.read_bytes(), photo.read_bytes()

    def get(self, url, params=None, **kw):
        params = params or {}
        self.calls.append(url)
        q = str(params.get("q") or params.get("query") or "")
        if url == "https://pixabay.com/api/videos/":
            hits = [{"id": 111, "tags": "library, students, study", "duration": 12, "pageURL": "https://pixabay.com/videos/library-111/", "user": "Anna",
                     "videos": {"large": {"url": "https://cdn.pixabay.com/video/111_large.mp4", "width": 3840, "height": 2160},
                                "medium": {"url": "https://cdn.pixabay.com/video/111_medium.mp4", "width": 1920, "height": 1080},
                                "small": {"url": "https://cdn.pixabay.com/video/111_small.mp4", "width": 1280, "height": 720}}},
                    {"id": 112, "tags": "library, students", "duration": 9, "pageURL": "https://pixabay.com/videos/upright-112/", "user": "Tall",
                     "videos": {"medium": {"url": "https://cdn.pixabay.com/video/112_medium.mp4", "width": 1080, "height": 1920}}}] if "library" in q else []
            return _Resp({"hits": hits})
        if url == "https://api.pexels.com/videos/search":
            return _Resp({"videos": []})
        if url == "https://pixabay.com/api/":
            return _Resp({"hits": []})
        if url == "https://api.pexels.com/v1/search":
            ph = [{"id": 222, "alt": "Sunset over the city skyline", "url": "https://www.pexels.com/photo/sunset-over-the-city-222/", "photographer": "Ben",
                   "width": 1880, "height": 1253, "src": {"large2x": "https://images.pexels.com/photos/222/pexels-photo-222.jpeg?auto=compress&w=1880"}}]
            return _Resp({"photos": ph if "sunset" in q else []})
        if url.startswith("https://cdn.pixabay.com/video/"):
            return _Resp(body=self.video)
        if url.startswith("https://images.pexels.com/"):
            return _Resp(body=self.photo)
        return _Resp(status=404)

    def downloads(self) -> list[str]:
        return [u for u in self.calls if u.startswith(("https://cdn.", "https://images."))]


# ------------------------------------------------------------------------------------------------ stock search (no render)
def test_stock_broll_needs_a_key_picks_the_frame_shape_at_most_1080p_and_never_logs_the_key(tmp_path, monkeypatch, caplog):
    vid, pic = tmp_path / "v.mp4", tmp_path / "p.jpg"
    _talk(vid, 2)
    _noise_photo(pic)
    web = FakeStockWeb(vid, pic)
    monkeypatch.setattr(SK.requests, "get", web.get)
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    assert SK.keys() == [] and SK.broll("students in a library") == [] and web.calls == []      # no key: nothing is even asked
    monkeypatch.setenv("PIXABAY_API_KEY", "pb-secret-123")
    monkeypatch.setenv("PEXELS_API_KEY", "px-secret-456")
    got = SK.broll("students studying in a library", landscape=True)
    assert [x["id"] for x in got] == ["111"]                                       # the upright clip is the wrong shape for a 16:9 video
    assert got[0]["url"].endswith("111_medium.mp4") and got[0]["kind"] == "video"   # 1920x1080, not the 4K file
    assert SK.broll("sunset over the city")[0]["kind"] == "photo"
    tall = SK.broll("students studying in a library", landscape=False)
    assert [x["id"] for x in tall] == ["112"]

    def broken(url, params=None, **kw):
        raise RuntimeError(f"500 for {url}?key={params.get('key') if params else ''}")
    monkeypatch.setattr(SK.requests, "get", broken)
    with caplog.at_level("WARNING"):
        assert SK.broll("anything") == []
    assert "pb-secret-123" not in caplog.text and "px-secret-456" not in caplog.text
    # a download into a Stock folder: named by site + id, with its licence beside it
    monkeypatch.setattr(SK.requests, "get", web.get)
    item = {**got[0], "query": "students studying in a library"}
    f = SK.fetch(load_config(), item, into=tmp_path / "Stock")
    assert f == tmp_path / "Stock" / "pixabay_111.mp4" and f.read_bytes() == vid.read_bytes()
    meta = json.loads((tmp_path / "Stock" / "pixabay_111.stock.json").read_text())
    assert meta["provider"] == "pixabay" and meta["id"] == "111" and meta["page"].startswith("https://pixabay.com/") and "Pixabay" in meta["license"]
    assert meta["credit"] == "Video: Anna / Pixabay" and meta["query"] == "students studying in a library"
    n = len(web.downloads())
    assert SK.fetch(load_config(), item, into=tmp_path / "Stock") == f and len(web.downloads()) == n      # kept: never downloaded twice


# ------------------------------------------------------------------------------------------------ what is a graphic (no render)
def test_graphics_are_told_from_photos_and_never_offered_as_broll(tmp_path):
    from clipfarm.clipper import localproj as LP
    f = tmp_path / "Shoot"
    (f / "B-roll" / "photos").mkdir(parents=True)
    (f / "Graphics").mkdir()
    clear = Image.new("RGBA", (600, 300), (0, 0, 0, 0))
    ImageDraw.Draw(clear).rectangle((50, 50, 550, 250), fill=(20, 90, 200, 255))
    clear.save(f / "B-roll" / "photos" / "cutout.png")                             # see-through: a graphic even in a photos folder
    _noise_photo(f / "B-roll" / "photos" / "street.jpg")                          # a real photo
    flat = Image.new("RGB", (500, 200), (255, 255, 255))
    ImageDraw.Draw(flat).text((40, 80), "ACME", fill=(200, 0, 0))
    flat.save(f / "B-roll" / "photos" / "acme.jpg")                               # a flat wordmark on white: a graphic
    Image.new("RGB", (1920, 1080), (30, 30, 30)).save(f / "B-roll" / "photos" / "slide.png")   # a flat FULL frame: shown whole, like a photo
    ImageDraw.Draw(clear).text((60, 120), "Dr Smith", fill="white")
    clear.save(f / "Graphics" / "Lower third - Dr Smith.png")
    clear.save(f / "logo.png")
    _alpha_mov(f / "Graphics" / "Intro.mov")
    _talk(f / "talk.mov", 3)
    photos = {p["name"] for p in PH.scan(f, LP.skipped)}
    assert photos == {"street.jpg", "slide.png"}
    files = LP.scan(f)
    assert [x["name"] for x in files] == ["talk.mov"]                              # the see-through video is not footage
    gfx = {g["name"]: g for g in FG.scan(f, LP.skipped)}
    assert {"cutout.png", "acme.jpg", "Lower third - Dr Smith.png", "logo.png", "Intro.mov"} <= set(gfx) and "street.jpg" not in gfx
    assert gfx["Intro.mov"]["kind"] == "video" and gfx["Intro.mov"]["role"] == "title" and gfx["Intro.mov"]["dur"] == pytest.approx(3.0, abs=0.1)
    assert gfx["Lower third - Dr Smith.png"]["role"] == "lower" and gfx["Lower third - Dr Smith.png"]["label"] == "Dr Smith"
    assert FG.logo_of(list(gfx.values())) == str(f / "logo.png")
    # where they go: the lower third where the name is said, the title over the opening, the corner logo only when asked for
    from clipfarm.clipper.moments import Sentence
    sents = [Sentence(0, 0.5, 4.0, "Welcome to the open day."), Sentence(1, 4.5, 9.0, "My name is Doctor Smith and I teach here."),
             Sentence(2, 9.5, 15.0, "We study the stars.")]
    placed = FG.place([gfx["Lower third - Dr Smith.png"], gfx["Intro.mov"], gfx["logo.png"]], sents, 16.0, "", None)
    by = {o["name"]: o for o in placed["overlays"]}
    assert by["Intro.mov"]["at"] == 0.0 and by["Intro.mov"]["len"] == pytest.approx(3.0, abs=0.1) and placed["title"]
    assert by["Lower third - Dr Smith.png"]["at"] == pytest.approx(4.8, abs=0.01)
    assert placed["logo"] == str(f / "logo.png") and not placed["bug"] and "logo.png" not in by
    assert FG.place([gfx["logo.png"]], sents, 16.0, "Keep our logo in the corner please", None)["bug"]


# ------------------------------------------------------------------------------------------------ stock in a real render
def test_stock_broll_fills_what_the_footage_doesnt_show_downloads_only_what_is_placed_and_is_reused(tmp_path, monkeypatch):
    import xml.etree.ElementTree as ET

    from clipfarm.clipper import fcpxml as FX
    vid, pic = tmp_path / "stock_src.mp4", tmp_path / "stock_src.jpg"
    _talk(vid, 12, size="1280x720", color="green")
    _noise_photo(pic, (1200, 800))
    web = FakeStockWeb(vid, pic)
    monkeypatch.setattr(SK.requests, "get", web.get)
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)

    class AI(_FakeAI):
        def complete_json(self, task, system, prompt, schema, validate=None, images=None, **kw):
            import re
            if task == "broll_queries":
                return _Res({"passages": [{"n": 1, "covered": False, "looks": ["students studying in a library"]},
                                          {"n": 2, "covered": False, "looks": ["sunset over the city"]}]})
            if task == "broll_plan":                                     # place every stock shot offered: the video at 4 s, the photo at 13 s
                out = []
                for i, what in re.findall(r"^\s+S(\d+): (.*)$", prompt, re.M):
                    if "(stock footage)" in what:
                        out.append({"at": 4.0, "scene": int(i), "length": 2.5, "why": "library"})
                    elif "stock" in what:
                        out.append({"at": 13.0, "scene": int(i), "length": 2.5, "why": "sunset"})
                return _Res({"cutaways": out})
            if task == "broll_pick":
                return _Res({"scene": -1})
            return super().complete_json(task, system, prompt, schema, validate=validate, images=images, **kw)
    fake = AI()
    LP, cfg, store, projects = _setup(tmp_path, monkeypatch, fake)
    folder = tmp_path / "Open day"
    folder.mkdir()
    _talk(folder / "talk.mov", 25)
    said = ("Our students study together every evening. The library stays open late. Everyone finds a quiet corner to read. "
            "After class we walk across town. The sunset over the city is beautiful. It is the best part of the day.").split()
    words = [{"text": w, "start": 0.4 + i * 0.6, "end": 0.4 + i * 0.6 + 0.4} for i, w in enumerate(said)]
    monkeypatch.setattr(LP.LocalRunner, "_openai_words", lambda self, f, ai: words)
    runner = LP.LocalRunner(cfg, store, projects)
    proj = projects.create_local(str(folder), "Open day", "", {"transcriber": "openai"}, {"clips": 0})
    projects.prepare_local(proj["id"], {"stock": True, "graphics": "", "music": ""})
    assert store.get(proj["job"])["options"]["stock"] is True
    # 1) no key on this Mac: stock is silently left out, nothing is asked of any site
    runner.make(proj["id"], {})
    assert store.get(proj["job"])["status"] == "done"
    assert json.loads((folder / ".aivideo" / "stock.json").read_text())["files"] == [] and web.calls == []
    assert "broll_queries" not in fake.calls
    # 2) with keys: the two passages your footage doesn't show get stock (a video, and a photo where no video was found)
    monkeypatch.setenv("PIXABAY_API_KEY", "pb-key")
    monkeypatch.setenv("PEXELS_API_KEY", "px-key")
    runner.make(proj["id"], {})
    job = store.get(proj["job"])
    assert job["status"] == "done", job
    sj = json.loads((folder / ".aivideo" / "stock.json").read_text())
    assert [f["name"] for f in sj["files"]] == ["Stock - Pixabay 111"] and [p["file"] for p in sj["photos"]] == ["Stock - Pexels photo 222"]
    sdir = tmp_path / "apphome" / "Stock"
    assert (sdir / "pixabay_111.mp4").read_bytes() == vid.read_bytes() and (sdir / "pexels_222.jpeg").exists()
    side = json.loads((sdir / "pixabay_111.stock.json").read_text())
    assert {k: side[k] for k in ("provider", "id", "query")} == {"provider": "pixabay", "id": "111", "query": "students studying in a library"}
    assert side["page"] and side["license"] and side["credit"] == "Video: Anna / Pixabay"
    assert json.loads((sdir / "pexels_222.stock.json").read_text())["credit"] == "Photo: Ben / Pexels"
    assert sorted(web.downloads()) == ["https://cdn.pixabay.com/video/111_medium.mp4",
                                       "https://images.pexels.com/photos/222/pexels-photo-222.jpeg?auto=compress&w=1880"]
    meta = json.loads((tmp_path / "jobs" / proj["job"] / "plan.json").read_text())
    stock_file = sj["files"][0]
    assert any(stock_file["start"] - 0.01 <= c["src"] < stock_file["start"] + 30 for c in meta["cutaways"])      # the stock video is placed
    assert any("pexels_222" in str(c.get("photo")) for c in meta["cutaways"])                                     # ...and the stock photo
    credits = (folder / "Notes" / "credits.txt").read_text()
    assert "Video: Anna / Pixabay - https://pixabay.com/videos/library-111/" in credits and "Photo: Ben / Pexels" in credits
    out = sorted((folder / "Output").glob("*.mp4"))[-1]                            # (v1 was the make without a key)
    assert out.name.endswith("v2.mp4") and ffmpeg.duration(out) == pytest.approx(words[-1]["end"], abs=1.5)
    # the stock shot is really in the picture: green (the stock clip) at 5 s, not the test pattern
    png = tmp_path / "at5.png"
    subprocess.run([FF, "-y", "-v", "error", "-ss", "5.2", "-i", str(out), "-frames:v", "1", str(png)], check=True)
    r, g, b = Image.open(png).convert("RGB").resize((1, 1), Image.BOX).getpixel((0, 0))
    assert g > 90 and r < 60 and b < 60, (r, g, b)
    # Final Cut: the stock clip is referenced where it was downloaded, and the project is valid
    xml = sorted(folder.glob("*.fcpxml"))[-1]
    srcs = [mr.get("src") for mr in ET.parse(xml).getroot().iter("media-rep")]
    assert any(s.endswith("/Stock/pixabay_111.mp4") for s in srcs), srcs
    assert not FX.validate(xml)
    # 3) a re-render downloads nothing again and searches nothing again
    n_calls = len(web.calls)
    runner.rerender(proj["id"])
    assert store.get(proj["job"])["status"] == "done"
    assert len(web.calls) == n_calls


# ------------------------------------------------------------------------------------------------ folder graphics in a real render
def _frame(video: Path, t: float, out: Path) -> Image.Image:
    subprocess.run([FF, "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1", str(out)], check=True)
    return Image.open(out).convert("RGB")


def test_folder_graphics_logo_end_card_lower_third_while_they_talk_alpha_title_and_final_cut_lanes(tmp_path, monkeypatch):
    import xml.etree.ElementTree as ET

    from clipfarm.clipper import fcpxml as FX

    class AI(_FakeAI):
        def complete_json(self, task, system, prompt, schema, validate=None, images=None, **kw):
            if task == "broll_plan":
                return _Res({"cutaways": []})
            if task == "broll_pick":
                return _Res({"scene": -1})
            return super().complete_json(task, system, prompt, schema, validate=validate, images=images, **kw)
    fake = AI()
    LP, cfg, store, projects = _setup(tmp_path, monkeypatch, fake)
    folder = tmp_path / "Lecture"
    (folder / "Graphics").mkdir(parents=True)
    (folder / "B-roll" / "photos").mkdir(parents=True)
    _talk(folder / "talk.mov", 20, color="gray")
    _alpha_mov(folder / "Graphics" / "Intro.mov", 3.0)
    lower = Image.new("RGBA", (900, 150), (0, 0, 0, 0))
    ImageDraw.Draw(lower).rectangle((0, 0, 899, 149), fill=(0, 60, 255, 255))
    lower.save(folder / "Graphics" / "Lower third - Dr Smith.png")
    logo = Image.new("RGBA", (500, 500), (0, 0, 0, 0))
    ImageDraw.Draw(logo).ellipse((50, 50, 450, 450), fill=(220, 0, 0, 255))
    logo.save(folder / "logo.png")
    clear = Image.new("RGBA", (400, 300), (0, 0, 0, 0))
    ImageDraw.Draw(clear).rectangle((40, 40, 360, 260), fill=(255, 0, 255, 255))
    clear.save(folder / "B-roll" / "photos" / "sticker.png")                       # see-through in a photos folder: never full-screen B-roll
    said = ("Welcome to this short lecture about the night sky. I am Doctor Smith and I have studied stars for years. "
            "Tonight we look at three bright planets together.").split()
    words = [{"text": w, "start": 0.4 + i * 0.6, "end": 0.4 + i * 0.6 + 0.4} for i, w in enumerate(said)]
    monkeypatch.setattr(LP.LocalRunner, "_openai_words", lambda self, f, ai: words)
    runner = LP.LocalRunner(cfg, store, projects)
    proj = projects.create_local(str(folder), "Night sky", "", {"transcriber": "openai"}, {"clips": 0})
    projects.prepare_local(proj["id"], {"graphics": "title", "music": "", "kind": "story"})
    runner.make(proj["id"], {})
    job = store.get(proj["job"])
    assert job["status"] == "done", job
    plan = json.loads((folder / ".aivideo" / "plan.json").read_text())
    assert plan["style"] == "story"
    assert all(not str(s.get("photo") or "").endswith("sticker.png") for s in plan["pool"])     # never offered as a photo
    by = {g["name"]: g for g in plan["folder_graphics"]}
    assert by["Intro.mov"]["role"] == "title" and by["Intro.mov"]["at"] == 0.0
    t_said = next(w["start"] for w in words if w["text"] == "Smith")
    sent_start = next(w["start"] for w in words if w["text"] == "I")
    lt = by["Lower third - Dr Smith.png"]
    assert lt["role"] == "lower" and abs(lt["at"] - (sent_start + 0.3)) < 0.6 and lt["at"] <= t_said + 0.5    # while Dr Smith says who he is
    assert by["logo.png"]["role"] == "end card"
    out = next((folder / "Output").glob("*.mp4"))
    total = ffmpeg.duration(out)
    assert total == pytest.approx(words[-1]["end"] + LP.OUTRO_S, abs=2.0)            # the talking, then the end card
    # what is on screen: the see-through intro over the opening, the lower third while he talks, the logo card at the end
    f1 = _frame(out, 1.5, tmp_path / "f1.png")
    W, H = f1.size
    cx = f1.getpixel((W // 2, int(H * 0.4)))
    assert cx[0] > 180 and 80 < cx[1] < 190 and cx[2] < 90, cx                       # orange disc in the middle
    assert f1.getpixel((5, 5))[0] < 200                                               # ...and the picture still shows round it (see-through)
    f2 = _frame(out, lt["at"] + 1.5, tmp_path / "f2.png")
    blue = [f2.getpixel((x, y)) for x in range(int(W * 0.07), int(W * 0.4), 8) for y in range(int(H * 0.55), int(H * 0.84), 8)]
    assert sum(1 for p in blue if p[2] > 180 and p[0] < 80) > 20                      # the blue bar bottom-left, above the captions
    assert all(not (p[2] > 180 and p[0] < 80) for p in [f2.getpixel((x, int(H * 0.2))) for x in range(0, W, 16)])   # not up top
    f3 = _frame(out, total - 0.6, tmp_path / "f3.png")
    assert f3.getpixel((W // 2, H // 2))[0] > 180 and f3.getpixel((W // 2, H // 2))[1] < 60     # the red logo, centred
    corner = f3.getpixel((8, 8))
    assert abs(corner[0] - corner[1]) < 12 and abs(corner[1] - corner[2]) < 12          # on a plain (grey-scale) background
    # Final Cut: each graphic is its own clip ABOVE the B-roll lane, transparent (QuickTime Animation), and the project is valid
    xml = next(folder.glob("*.fcpxml"))
    root = ET.parse(xml).getroot()
    gfx_clips = [c for c in root.iter("asset-clip") if c.get("name") == "Graphic"]
    assert len(gfx_clips) >= 2 and all(int(c.get("lane")) >= 2 for c in gfx_clips)
    gdir = next((folder / "Final Cut").rglob("Graphics"))
    movs = sorted(gdir.glob("*.mov"))
    assert movs and all(ffmpeg.video_stream(ffmpeg.probe(m))["pix_fmt"] == "argb" for m in movs)
    assert not FX.validate(xml)
    titles = [t.get("name") for t in root.iter("title")]
    assert "Hook title" not in titles                                                 # your title replaced the made-up one
