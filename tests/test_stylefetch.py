"""Getting a video to learn a style from (stylefetch.py): the kind of link, YouTube video / channel with a fake yt-dlp, Instagram with and without the
person's own fallback address, a dropped file, and that no login of any kind is ever passed on. All offline: yt-dlp and the network are faked."""

import subprocess
from pathlib import Path

import pytest

from clipfarm.clipper import stylefetch as F
from clipfarm.core import ffmpeg


def _mp4(path: Path, secs: float = 1.0) -> Path:
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c=red:s=64x112:r=10:d={secs}", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


class FakeYDL:
    """A yt-dlp stand-in: extract_info writes a small mp4 where outtmpl says (download=True) or lists entries (a channel)."""
    calls: list = []

    def __init__(self, opts, entries=None, fail=None, title="A video"):
        self.opts, self.entries, self.fail, self.title = opts, entries, fail, title
        FakeYDL.calls.append(opts)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download=True):
        if self.fail:
            raise RuntimeError(self.fail)
        if not download:
            return {"entries": self.entries or []}
        out = Path(str(self.opts["outtmpl"]).replace("%(ext)s", "mp4"))
        _mp4(out)
        return {"title": f"{self.title} {out.stem}", "duration": 40, "view_count": 123}


def factory(entries=None, fail=None):
    return lambda opts: FakeYDL(opts, entries, fail)


@pytest.mark.parametrize("url,kind", [
    ("https://www.youtube.com/watch?v=abc123", "youtube_video"), ("https://youtu.be/abc123", "youtube_video"),
    ("https://www.youtube.com/shorts/xyz", "youtube_video"), ("https://www.youtube.com/@somebody", "youtube_channel"),
    ("https://www.youtube.com/@somebody/videos", "youtube_channel"), ("https://www.youtube.com/channel/UCabc", "youtube_channel"),
    ("https://www.youtube.com/playlist?list=PL1", "youtube_playlist"), ("https://www.instagram.com/reel/Cxyz/", "instagram"),
    ("https://example.com/page", "other"), ("not a link", "other"), ("", "other"), ("/Users/x/clip.mp4", "file"), ("youtube.com/watch?v=q", "youtube_video")])
def test_what_kind_of_link_it_is(url, kind):
    assert F.classify(url) == kind


def test_a_youtube_video_comes_down_small_and_with_no_login(tmp_path):
    FakeYDL.calls.clear()
    out = F.fetch("https://www.youtube.com/watch?v=abc", tmp_path / "dl", ydl_factory=factory())
    assert len(out) == 1 and out[0]["path"].exists() and out[0]["kind"] == "youtube_video" and out[0]["owned"] is True
    opts = FakeYDL.calls[0]
    assert "480" in opts["format"]                                                       # 480p or lower
    for key in ("cookiefile", "cookiesfrombrowser", "username", "password", "netrc", "videopassword"):
        assert key not in opts                                                           # never a login, a cookie or a browser's saved sign-in


def test_the_real_factory_strips_any_login_a_caller_passes(monkeypatch):
    seen = {}

    class Y:
        def __init__(self, o):
            seen.update(o)
    import yt_dlp
    monkeypatch.setattr(yt_dlp, "YoutubeDL", Y)
    F._ydl({"cookiefile": "/x", "cookiesfrombrowser": ("chrome",), "username": "u", "password": "p", "format": "b"})
    assert "cookiefile" not in seen and "cookiesfrombrowser" not in seen and "username" not in seen and "password" not in seen and seen["format"] == "b"


def test_a_channel_gives_the_newest_and_the_most_viewed(tmp_path):
    entries = [{"id": f"v{i}", "title": f"t{i}", "duration": 40, "view_count": (1000 - i * 10) if i % 5 == 0 else 5} for i in range(30)]
    got = F.pick(entries, 8)
    ids = [e["id"] for e in got]
    assert len(ids) == 8 and ids[:4] == ["v0", "v1", "v2", "v3"]                         # the newest four
    assert "v5" in ids and "v10" in ids and len(set(ids)) == 8                           # then the most viewed of the rest, no repeats
    longones = [{"id": "long", "duration": 99999}] + entries[:3]
    assert "long" not in [e["id"] for e in F.pick(longones, 8)]
    out = F.fetch("https://www.youtube.com/@somebody", tmp_path / "dl", ydl_factory=factory(entries=entries))
    assert len(out) == 8 and all(o["path"].exists() for o in out) and all(o["kind"] == "youtube_channel" for o in out)


def test_a_channel_with_no_videos_is_said_in_words(tmp_path):
    with pytest.raises(F.FetchError, match="no videos"):
        F.fetch("https://www.youtube.com/@somebody", tmp_path / "dl", ydl_factory=factory(entries=[]))


def test_instagram_that_wants_a_sign_in_says_so_and_asks_for_the_file(tmp_path):
    with pytest.raises(F.FetchError) as e:
        F.fetch("https://www.instagram.com/reel/Cx/", tmp_path / "dl", ydl_factory=factory(fail="login required"))
    assert str(e.value) == F.INSTAGRAM_FAILED and "drop the video file" in str(e.value)


def test_instagram_can_use_the_persons_own_fallback_address(tmp_path):
    sent = {}

    def post(url, body):
        sent.update({"url": url, "body": body})
        return {"status": "tunnel", "url": "https://media.example/x.mp4"}

    def download(url, dest):
        sent["download"] = url
        _mp4(dest)
    out = F.fetch("https://www.instagram.com/reel/Cx/", tmp_path / "dl", endpoint="https://my-cobalt.example/", ydl_factory=factory(fail="login required"),
                  post=post, download=download)
    assert out[0]["path"].exists() and out[0]["kind"] == "instagram" and sent["url"] == "https://my-cobalt.example/"
    assert set(sent["body"]) == {"url", "videoQuality"}                                  # nothing but the link is sent
    with pytest.raises(F.FetchError, match="drop the video file"):                        # a bad answer from it: the same plain message
        F.fetch("https://www.instagram.com/reel/Cx/", tmp_path / "dl2", endpoint="https://my-cobalt.example/", ydl_factory=factory(fail="login required"),
                post=lambda u, b: {"status": "error"}, download=download)
    with pytest.raises(F.FetchError, match="drop the video file"):
        F.fetch("https://www.instagram.com/reel/Cx/", tmp_path / "dl3", endpoint="not an address", ydl_factory=factory(fail="login required"))


def test_nothing_third_party_is_built_in():
    import inspect
    src = inspect.getsource(F)
    assert "cobalt.tools" not in src and "ssyoutube" not in src and F.fetch.__kwdefaults__["endpoint"] == ""


def test_a_dropped_file_is_read_in_place_and_never_deleted(tmp_path):
    mine = _mp4(tmp_path / "mine.mp4")
    out = F.fetch(str(mine), tmp_path / "dl")
    assert out[0]["path"] == mine and out[0]["owned"] is False
    F.cleanup(out, tmp_path / "dl")
    assert mine.exists() and not (tmp_path / "dl").exists()
    with pytest.raises(F.FetchError):
        F.fetch(str(tmp_path / "notes.txt"), tmp_path / "dl")
    with pytest.raises(F.FetchError, match="no video"):
        F.fetch("https://example.com/page", tmp_path / "dl")


def test_cleanup_removes_every_download(tmp_path):
    out = F.fetch("https://www.youtube.com/watch?v=abc", tmp_path / "dl", ydl_factory=factory())
    assert out[0]["path"].exists()
    F.cleanup(out, tmp_path / "dl")
    assert not (tmp_path / "dl").exists()
