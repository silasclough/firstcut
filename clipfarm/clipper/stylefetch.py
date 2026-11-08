"""Getting a video to learn an editing style from: a YouTube video, short, channel or playlist, an Instagram reel, or a file the person drops.

No sign-in, ever. Nothing here asks for, stores or uses a password, a cookie or a browser's saved login (yt-dlp is never given any). A video
that will not come down without signing in is said so in plain words, and the person is told to drop the file instead.

    youtube   yt-dlp, 480p or lower, only the first few minutes of each video (a style shows in minutes). A channel or playlist: up to 8 videos,
              the newest and the most viewed mixed.
    instagram yt-dlp without a login first; then, only when the person has set one, their own fallback endpoint (a self-hosted cobalt-style
              API url in Settings; nothing third-party is built in); else a plain message.
    file      a video on this Mac: read where it is (never copied, never deleted).

Everything is downloaded into one temporary folder that the caller deletes (stylelearn.learn does, in a `finally`). Every network call is behind a
function a test can replace (`ydl_factory`, `http_post`, `http_download`).
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from ..core.log import get_logger

log = get_logger("clipper.stylefetch")

MAX_VIDEOS = 8                    # a channel or playlist: how many videos are looked at
LOOK_S = 180                      # seconds from the start of each video that are downloaded and measured
MAX_MINUTES = 30                  # a longer video is skipped (its first minutes would not show how the channel cuts)
POOL = 40                         # a channel's newest videos that are listed before choosing
INSTAGRAM_FAILED = "Instagram would not give this reel without signing in: drop the video file instead."
NEED_VIDEO = "That link has no video I can read. Paste a YouTube video, short, channel or playlist link, or drop a video file."
FORMAT = "bv*[height<=480]+ba/b[height<=480]/worst[height>=240]/worst"

YT_HOST = re.compile(r"^(?:www\.|m\.|music\.)?(?:youtube\.com|youtu\.be|youtube-nocookie\.com)$", re.I)
IG_HOST = re.compile(r"^(?:www\.)?(?:instagram\.com|instagr\.am)$", re.I)
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".mts", ".m2ts", ".avi", ".3gp"}


class FetchError(Exception):
    """A plain-words reason a video could not be fetched (shown to the person as it is)."""


def classify(source: str) -> str:
    """'file' | 'youtube_video' | 'youtube_channel' | 'youtube_playlist' | 'instagram' | 'other' for a link or a path."""
    s = str(source or "").strip()
    if not s:
        return "other"
    if s.startswith(("/", "~", "file://")) or (not re.match(r"^[a-z][a-z0-9+.-]*://", s, re.I) and Path(s).expanduser().suffix.lower() in VIDEO_EXT):
        return "file"
    if not re.match(r"^[a-z][a-z0-9+.-]*://", s, re.I):
        s = "https://" + s
    try:
        u = urllib.parse.urlparse(s)
    except ValueError:
        return "other"
    host = (u.hostname or "").lower()
    if IG_HOST.match(host):
        return "instagram"
    if YT_HOST.match(host):
        path = u.path or ""
        q = urllib.parse.parse_qs(u.query)
        if host == "youtu.be" and path.strip("/"):
            return "youtube_video"
        if path.startswith(("/shorts/", "/live/", "/embed/")):
            return "youtube_video"
        if path == "/watch" and q.get("v"):
            return "youtube_video"
        if path == "/playlist" or (q.get("list") and not q.get("v")):
            return "youtube_playlist"
        if re.match(r"^/(?:@[^/]+|channel/[^/]+|c/[^/]+|user/[^/]+)(?:/.*)?$", path):
            return "youtube_channel"
        return "other"
    return "other"


def _plain(e: Exception) -> str:
    return " ".join(str(e).split())[:240]


def _ydl(opts: dict):
    import yt_dlp
    # never any login: these keys are not passed, whatever the caller's `opts` hold
    clean = {k: v for k, v in opts.items() if k not in ("cookiefile", "cookiesfrombrowser", "username", "password", "netrc", "videopassword")}
    return yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "socket_timeout": 20, "noplaylist": True, **clean})


def _sections():
    from yt_dlp.utils import download_range_func
    return download_range_func(None, [(0, LOOK_S)])


def _download_one(url: str, work: Path, name: str, ydl_factory: Callable = _ydl, limit_s: int = LOOK_S) -> dict:
    """One video (its first LOOK_S seconds) into `work`. -> {"path", "title", "url", "duration", "views"}; FetchError when it cannot."""
    opts = {"outtmpl": str(work / f"{name}.%(ext)s"), "format": FORMAT, "merge_output_format": "mp4", "retries": 2, "fragment_retries": 2,
            "match_filter": lambda info, *, incomplete=False: (f"longer than {MAX_MINUTES} minutes" if (info.get("duration") or 0) > MAX_MINUTES * 60 else None)}
    try:
        opts["download_ranges"] = _sections()
        opts["force_keyframes_at_cuts"] = False
    except Exception:  # noqa: BLE001 - an old yt-dlp downloads the whole video; the proxy step still reads only the first minutes
        pass
    try:
        with ydl_factory(opts) as y:
            info = y.extract_info(url, download=True)
    except Exception as e:  # noqa: BLE001
        raise FetchError(_plain(e)) from None
    if info and info.get("entries"):
        info = next((x for x in info["entries"] if x), None)
    path = next((p for p in sorted(work.glob(f"{name}.*")) if p.suffix.lower() in VIDEO_EXT), None)
    if not path:
        raise FetchError("the link gave no video file (it may be longer than %d minutes, private, or not a video)" % MAX_MINUTES)
    return {"path": path, "title": " ".join(str((info or {}).get("title") or path.stem).split())[:140], "url": url,
            "duration": float((info or {}).get("duration") or 0), "views": int((info or {}).get("view_count") or 0)}


def pick(entries: list[dict], n: int = MAX_VIDEOS) -> list[dict]:
    """From a channel or playlist listing (newest first): up to `n` videos, half the newest and half the most viewed, no repeats, none over the
    length limit. Entries without a view count are only picked as 'newest'."""
    ok = [e for e in entries if e and (e.get("id") or e.get("url")) and (e.get("duration") or 0) <= MAX_MINUTES * 60]
    newest = ok[: (n + 1) // 2]
    seen = {e.get("id") or e.get("url") for e in newest}
    viewed = sorted((e for e in ok if (e.get("id") or e.get("url")) not in seen and (e.get("view_count") or 0) > 0), key=lambda e: -(e.get("view_count") or 0))
    out = list(newest)
    for e in viewed:
        if len(out) >= n:
            break
        out.append(e)
    for e in ok:                                                              # (too few viewed ones: more of the newest)
        if len(out) >= n:
            break
        if e not in out:
            out.append(e)
    return out[:n]


def _entry_url(e: dict) -> str:
    u = str(e.get("webpage_url") or e.get("url") or "")
    if u.startswith("http"):
        return u
    return f"https://www.youtube.com/watch?v={e.get('id')}"


def _listing(url: str, ydl_factory: Callable) -> list[dict]:
    try:
        with ydl_factory({"extract_flat": "in_playlist", "playlistend": POOL, "noplaylist": False, "skip_download": True}) as y:
            info = y.extract_info(url, download=False) or {}
    except Exception as e:  # noqa: BLE001
        raise FetchError(_plain(e)) from None
    entries = [e for e in (info.get("entries") or []) if e]
    flat: list[dict] = []
    for e in entries:                                                         # (a channel's tabs are playlists of videos: one level down)
        if e.get("entries"):
            flat += [x for x in e["entries"] if x]
        else:
            flat.append(e)
    return flat


def http_post(url: str, body: dict, timeout: float = 25.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "FirstCut"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def http_download(url: str, dest: Path, timeout: float = 60.0, max_bytes: int = 120 * 1024 * 1024) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "FirstCut"})
    got = 0
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            got += len(chunk)
            if got > max_bytes:
                raise FetchError("that video is too big to read")
            f.write(chunk)


def _endpoint_ok(url: str) -> bool:
    return bool(re.match(r"^https?://[^\s]+$", str(url or "").strip()))


def _via_endpoint(url: str, endpoint: str, work: Path, name: str, post: Callable, download: Callable) -> dict:
    """The person's own cobalt-style endpoint: POST {url} -> {status: tunnel|redirect, url}. Nothing is sent but the link."""
    try:
        r = post(endpoint.strip(), {"url": url, "videoQuality": "480"})
    except Exception as e:  # noqa: BLE001
        log.info(f"the fallback address failed: {_plain(e)}")
        raise FetchError(INSTAGRAM_FAILED) from None
    media = (r or {}).get("url")
    if not media and isinstance((r or {}).get("picker"), list) and r["picker"]:
        media = (r["picker"][0] or {}).get("url")
    if (r or {}).get("status") not in ("tunnel", "redirect", "stream", "success", "picker") or not media:
        raise FetchError(INSTAGRAM_FAILED)
    if not re.match(r"(?i)^https?://[^\s/]", str(media).strip()):          # (only a web address: never file:// or anything else the endpoint answers)
        log.info("the fallback address answered with a link that is not a web address")
        raise FetchError(INSTAGRAM_FAILED)
    media = str(media).strip()
    dest = work / f"{name}.mp4"
    try:
        download(str(media), dest)
    except FetchError:
        raise
    except Exception:  # noqa: BLE001
        raise FetchError(INSTAGRAM_FAILED) from None
    return {"path": dest, "title": "Instagram reel", "url": url, "duration": 0.0, "views": 0}


def fetch(source: str, work: Path, *, max_videos: int = MAX_VIDEOS, endpoint: str = "", progress: Callable[[str], None] | None = None,
          ydl_factory: Callable = _ydl, post: Callable = http_post, download: Callable = http_download) -> list[dict]:
    """The videos to learn from, downloaded into `work` (the caller deletes it): [{"path", "title", "url", "kind", "duration", "views", "owned"}]
    (`owned` False = the person's own file, read in place and never deleted). Raises FetchError with a plain reason."""
    say = progress or (lambda m: None)
    work.mkdir(parents=True, exist_ok=True)
    kind = classify(source)
    s = str(source or "").strip()
    if kind == "file":
        p = Path(s[7:] if s.startswith("file://") else s).expanduser()
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXT:
            raise FetchError("that file is not a video I can open")
        return [{"path": p, "title": p.stem[:140], "url": "", "kind": "file", "duration": 0.0, "views": 0, "owned": False}]
    if kind == "other":
        raise FetchError(NEED_VIDEO)
    if not re.match(r"^[a-z][a-z0-9+.-]*://", s, re.I):
        s = "https://" + s
    if kind == "instagram":
        say("Fetching the reel")
        try:
            item = _download_one(s, work, "v0", ydl_factory)
        except FetchError as first:
            log.info(f"instagram without a login failed: {first}")
            if not _endpoint_ok(endpoint):
                raise FetchError(INSTAGRAM_FAILED) from None
            say("Trying your fallback address")
            item = _via_endpoint(s, endpoint, work, "v0", post, download)
        return [{**item, "kind": "instagram", "owned": True}]
    if kind == "youtube_video":
        say("Downloading the video")
        item = _download_one(s, work, "v0", ydl_factory)
        return [{**item, "kind": "youtube_video", "owned": True}]
    say("Looking at the channel" if kind == "youtube_channel" else "Looking at the playlist")
    chosen = pick(_listing(s, ydl_factory), max(1, min(int(max_videos or MAX_VIDEOS), MAX_VIDEOS)))
    if not chosen:
        raise FetchError("I found no videos there")
    out: list[dict] = []
    errors: list[str] = []
    for i, e in enumerate(chosen):
        say(f"Downloading video {i + 1} of {len(chosen)}")
        try:
            item = _download_one(_entry_url(e), work, f"v{i}", ydl_factory)
        except FetchError as ex:
            errors.append(str(ex))
            continue
        out.append({**item, "kind": kind, "owned": True, "views": item["views"] or int(e.get("view_count") or 0)})
    if not out:
        raise FetchError(errors[0] if errors else "none of those videos could be downloaded")
    return out


def cleanup(items: list[dict], work: Path) -> None:
    """Delete everything that was downloaded (the person's own files are never touched)."""
    import shutil
    for it in items or []:
        if it.get("owned") and it.get("path"):
            try:
                Path(it["path"]).unlink(missing_ok=True)
            except OSError:
                pass
    shutil.rmtree(work, ignore_errors=True)
