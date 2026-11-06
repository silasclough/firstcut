"""Music from online sources, inside FirstCut: search, preview, and fetch a song into the person's music library (where the catalogue,
the picker and the editor find it like any other song).

Openverse  (api.openverse.org, anonymous, free)  openly licensed music from Jamendo, ccMixter and Wikimedia, searched with category=music.
           Only licences fine for a monetised video are offered: CC0, the Public Domain Mark and CC BY (attribution). NC / ND / SA never.
           CC BY needs a credit line: it travels with the song (a sidecar JSON next to it, and the file's own tags, so a copy keeps it) and
           ends up in the video's description text (long form: clip_01.txt -> Notes/credits.txt; short clips: each clip's caption).
Epidemic   (partner-content-api.epidemicsound.com, an API key from the Epidemic Sound Developer Portal): search and preview. A key on the
           free developer plan comes with Epidemic's API Developer License: evaluation only, at most 50 downloads, no commercial licence
           (https://developers.epidemicsound.com/docs/ "Getting access"). So fetch() refuses to download with such a key and says why;
           a paid partner plan (or ES Connect for a subscriber, which needs a partner agreement) is what licenses songs for published
           videos. Pass licensed=True only for a key on a paid plan.

    openverse_search(q, n=30, page=1) -> [track]
    epidemic_status(key) -> {ok, error, name}
    epidemic_search(key, q, n=30) -> [track]          (preview "" : ask epidemic_preview(key, id) when the person presses play)
    epidemic_preview(key, track_id) -> url
    fetch(track, dest_dir, key=None, licensed=False) -> Path      the song in dest_dir/<source>/<Artist - Title>.mp3 + a sidecar .json
    credit_for(path) -> str | None                     the credit line of a fetched song ("" when none is needed), None for your own

A track: {source, id, title, artist, dur, preview, license, license_url, credit, page, tags, provider}.
The API key is passed in per call: it is never logged or written anywhere by this module.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from pathlib import Path

import requests

from ..core import ffmpeg
from ..core.log import get_logger

log = get_logger("clipper.musicsources")
UA = "FirstCut/1.0 (video editing app; https://openverse.org/ API client)"
OPENVERSE = "https://api.openverse.org/v1/audio/"
EPIDEMIC = "https://partner-content-api.epidemicsound.com"
CACHE_S = 600                                     # search results are kept 10 minutes (Openverse's anonymous limits are low)
MIN_S, MAX_S = 30, 900                            # a song: at least 30 s, at most 15 min
OK_LICENSES = {"cc0", "pdm", "by"}
NOT_MUSIC = re.compile(r"\b(sfx|sound ?effects?|foley|ringtone|sample pack|loop ?\d+|acapella|a cappella|karaoke)\b", re.I)
_BAD_NAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')
_cache: dict = {}
_lock = threading.Lock()


class MusicSourceError(RuntimeError):
    """Something the person can act on, in plain words (shown on the page)."""


# ------------------------------------------------------------------------------------------------ helpers
def known_preview(source: str, track_id: str) -> str | None:
    """The preview link of a song this app found in a recent search (the page only ever sends an id: never a link to fetch)."""
    with _lock:
        lists = [v for _, v in _cache.values() if isinstance(v, list)]
    for tracks in lists:
        for t in tracks:
            if isinstance(t, dict) and t.get("source") == source and str(t.get("id")) == str(track_id) and t.get("preview"):
                return str(t["preview"])
    return None


def _cached(key, fn):
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_S:
            return hit[1]
    val = fn()
    with _lock:
        _cache[key] = (time.time(), val)
        if len(_cache) > 300:                                         # (a small cache: drop the oldest)
            for k in sorted(_cache, key=lambda k: _cache[k][0])[:100]:
                _cache.pop(k, None)
    return val


def _license_name(lic: str, ver: str) -> str:
    lic = (lic or "").lower()
    if lic == "cc0":
        return "CC0 1.0"
    if lic == "pdm":
        return "Public Domain Mark 1.0"
    return f"CC BY {ver or '4.0'}"


def _credit(title: str, artist: str, provider: str, lic_name: str, lic_url: str, page: str) -> str:
    """The credit line CC BY asks for (title, creator, source, licence and its link); nothing for CC0 / public domain."""
    if not lic_name.startswith("CC BY"):
        return ""
    where = f" ({provider.replace('_', ' ').title()} via Openverse)" if provider else " (via Openverse)"
    return f'Music: "{title}" by {artist}{where} – {lic_name} {lic_url}' + (f" – {page}" if page else "")


def _openverse_track(r: dict) -> dict | None:
    lic = (r.get("license") or "").lower()
    if lic not in OK_LICENSES:
        return None
    dur = float(r.get("duration") or 0) / 1000
    title = " ".join(str(r.get("title") or "Untitled").split())[:120]
    tags = [str(t.get("name") or "") for t in (r.get("tags") or []) if isinstance(t, dict)]
    if not (MIN_S <= dur <= MAX_S) or not r.get("url") or NOT_MUSIC.search(title + " " + " ".join(tags)):
        return None
    artist = " ".join(str(r.get("creator") or "Unknown artist").split())[:80]
    name = _license_name(lic, str(r.get("license_version") or ""))
    lic_url = r.get("license_url") or ("https://creativecommons.org/licenses/by/4.0/" if lic == "by" else "")
    page = r.get("foreign_landing_url") or r.get("detail_url") or ""
    provider = str(r.get("source") or r.get("provider") or "")
    return {"source": "openverse", "id": str(r.get("id")), "title": title, "artist": artist, "dur": round(dur, 1),
            "preview": str(r["url"]), "license": name, "license_url": lic_url, "credit": _credit(title, artist, provider, name, lic_url, page),
            "page": page, "tags": tags[:12], "provider": provider, "filetype": str(r.get("filetype") or "")}


def _openverse_get(params: dict) -> list[dict]:
    for attempt in range(2):
        try:
            r = requests.get(OPENVERSE, params=params, headers={"User-Agent": UA}, timeout=25)
        except requests.RequestException as e:
            raise MusicSourceError("couldn't reach Openverse: check the internet connection") from e
        if r.status_code == 429:                                     # anonymous use is rate limited: one short wait, then say so
            wait = min(8.0, float(r.headers.get("Retry-After") or 4))
            if attempt == 0:
                time.sleep(wait)
                continue
            raise MusicSourceError("Openverse is busy right now (too many searches): try again in a minute")
        if not r.ok:
            raise MusicSourceError(f"Openverse didn't answer the search (HTTP {r.status_code})")
        try:
            return list(r.json().get("results") or [])
        except ValueError as e:
            raise MusicSourceError("Openverse sent something that isn't a search result") from e
    return []


# ------------------------------------------------------------------------------------------------ better matches (2026-11-06)
# "calm piano" found "Harsh Calm [Hardcore Techno]"; "upbeat corporate" a song with singing. So: the words become 2-3 music searches
# (stockfind.searches_for, one cheap AI call), songs with singing are left out unless the words ask for it, and the best candidates
# are LISTENED to: CLAP (the music library's model) scores 15 s from the middle of each preview against the words.
_CLAP = None                          # () -> musiclib.ClapEmbedder or None; the app sets it (configure)
_AVEC: dict = {}                      # track id -> its audio vector
LISTEN = 20                           # how many of the best-by-tags candidates are listened to
VOCAL_TAGS = {"vocal", "vocals", "male", "female", "singing", "singer", "voice", "choir", "rap", "lyrics", "songwriter"}
ASKS_VOCAL = re.compile(r"\b(vocals?|sing(ing|er)?|sung|lyrics|voice|choir|rap|words)\b", re.I)


class excerpt:
    """A ~600 KB slice from 40% into a song on the web, as a temporary local file (with: ... as path). The app's own ffmpeg has no
    certificate store, so it can't open https itself; a slice is enough to hear what the song is like."""

    def __init__(self, url: str, size: int = 600_000):
        self.url, self.size, self.path = url, size, None

    def __enter__(self):
        import tempfile
        if not str(self.url).startswith("https://"):
            raise MusicSourceError("not a song on the web")
        r = requests.get(self.url, headers={"User-Agent": UA, "Range": "bytes=0-1"}, timeout=15)
        total = int(str(r.headers.get("Content-Range", "/0")).rsplit("/", 1)[-1] or 0) if r.status_code == 206 else 0
        a = int(total * 0.4) if total > self.size * 2 else 0
        r = requests.get(self.url, headers={"User-Agent": UA, "Range": f"bytes={a}-{a + self.size - 1}"}, timeout=30)
        r.raise_for_status()
        f = tempfile.NamedTemporaryFile(prefix="clap-", suffix=".mp3", delete=False)
        f.write(r.content[: self.size])
        f.close()
        self.path = Path(f.name)
        return self.path

    def __exit__(self, *exc):
        if self.path:
            self.path.unlink(missing_ok=True)


def configure(clap=None) -> None:
    global _CLAP
    if clap is not None:
        _CLAP = clap


def _has_singing(t: dict) -> bool:
    return bool(VOCAL_TAGS & {str(x).lower() for x in t.get("tags") or []})


def listen_rank(q: str, tracks: list[dict], top: int = LISTEN) -> list[dict]:
    """The first `top` tracks re-ordered by how well they SOUND like the words (CLAP), the rest after them; unchanged without CLAP."""
    import concurrent.futures as cf
    emb = None
    if _CLAP:
        try:
            emb = _CLAP()
        except Exception as e:  # noqa: BLE001
            log.info(f"music ranking without listening ({str(e)[:120]})")
    if emb is None or not tracks:
        return tracks
    head, tail = tracks[:top], tracks[top:]
    try:
        import numpy as np
        todo = [t for t in head if t["id"] not in _AVEC and t.get("preview")]

        def one(t):
            try:
                with excerpt(t["preview"]) as f:
                    return t["id"], emb.audio(f, 0.0, 12.0)
            except Exception:  # noqa: BLE001 - one song that won't play: unranked
                return t["id"], None
        with cf.ThreadPoolExecutor(6) as ex:
            for tid, v in ex.map(one, todo):
                if v is not None:
                    _AVEC[tid] = v
        while len(_AVEC) > 3000:
            _AVEC.pop(next(iter(_AVEC)))
        txt = emb.text([q, "instrumental music without singing", "a person singing with lyrics"])
        for k, t in enumerate(head):
            v = _AVEC.get(t["id"])
            if v is None:
                t["match"] = None
                continue
            t["match"] = round(float(np.dot(v, txt[0])), 4)
            t["_sings"] = float(np.dot(v, txt[2])) - float(np.dot(v, txt[1])) > 0.08
        want_voice = bool(ASKS_VOCAL.search(q))
        words = [w for w in re.split(r"[^\w]+", q.lower()) if len(w) > 2]

        def tagged(t):                                                # the words in its tags: what it was labelled as (a harp
            tags = {str(x).lower() for x in t.get("tags") or []}      # and a piano sound alike to the listener)
            return sum(1 for w in words if w in tags)
        scored = sorted(head, key=lambda t: -((t["match"] if t.get("match") is not None else -1.0)
                                              - (0.15 if t.pop("_sings", False) and not want_voice else 0.0)
                                              + 0.04 * tagged(t)
                                              + 0.02 * (top - head.index(t)) / top))
        for k, t in enumerate(scored):
            t["best"] = k < 3 and t.get("match") is not None
        return scored + tail
    except Exception as e:  # noqa: BLE001 - listening is a bonus
        log.warning(f"music ranking failed: {str(e)[:200]}")
        return tracks


# ------------------------------------------------------------------------------------------------ Openverse
def openverse_search(q: str, n: int = 30, page: int = 1) -> list[dict]:
    """Openly licensed music for these words (only licences fine for a monetised video). Several words find few songs on Openverse, so
    when the whole phrase finds too few, each word is searched too and the songs that match more of them come first."""
    q = " ".join(str(q or "").split())[:120]
    n = max(1, min(60, int(n)))
    page = max(1, min(12, int(page)))

    def run():
        base = {"license": "by,cc0,pdm", "category": "music", "mature": "false", "page_size": 20}
        seen, out = set(), []

        def add(results, weight):
            for r in results:
                t = _openverse_track(r)
                if t and t["id"] not in seen:
                    seen.add(t["id"])
                    t["_w"] = weight
                    out.append(t)

        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(3) as ex:                          # the words at once; the AI's music searches alongside, as soon
            first = ex.submit(_openverse_get, {**base, "q": q or "instrumental", "page": page})   # as they come (Openverse is slow)
            alts = []
            if q:
                from .stockfind import searches_for
                alts = [ex.submit(_openverse_get, {**base, "q": a, "page": page}) for a in searches_for(q, "music")[1:]]
            add(first.result(), 2)
            for f in alts:
                try:
                    add(f.result(), 1.5)
                except MusicSourceError:
                    pass                                              # (one refused: the others stand)
        words = [w for w in re.split(r"[^\w]+", q.lower()) if len(w) > 2][:3]
        if len(out) < n and len(words) > 1 and page == 1:
            for w in words:
                add(_openverse_get({**base, "q": w}), 1)
                if len(out) >= n * 2:
                    break
        for t in out:                                                 # more of the words in its title and tags: first
            hay = (t["title"] + " " + " ".join(t["tags"])).lower()
            t["_w"] += sum(0.5 for w in words if w in hay)
        out.sort(key=lambda t: -t.pop("_w"))
        if not ASKS_VOCAL.search(q):                                  # background music: no singing (unless that is all there is)
            plain = [t for t in out if not _has_singing(t)]
            out = plain if len(plain) >= min(5, len(out)) else out
        return listen_rank(q, out)[:n]

    return _cached(("openverse", q.lower(), n, page), run)


# ------------------------------------------------------------------------------------------------ Epidemic Sound
def _es_headers(key: str) -> dict:
    return {"Authorization": f"Bearer {key.strip()}", "Accept": "application/json", "User-Agent": UA, "x-partner-user-id": "firstcut-user"}


def _es_get(key: str, path: str, params: dict | None = None) -> dict:
    if not key or not str(key).strip():
        raise MusicSourceError("connect Epidemic Sound first (Settings › Music library)")
    try:
        r = requests.get(EPIDEMIC + path, params=params or {}, headers=_es_headers(key), timeout=25)
    except requests.RequestException as e:
        raise MusicSourceError("couldn't reach Epidemic Sound: check the internet connection") from e
    if r.status_code in (401, 403):
        raise MusicSourceError("Epidemic Sound didn't accept that key: make one in the Epidemic Sound Developer Portal (API keys) and paste it again")
    if r.status_code == 429:
        raise MusicSourceError("Epidemic Sound is limiting requests right now: try again in a minute")
    if not r.ok:
        raise MusicSourceError(f"Epidemic Sound didn't answer (HTTP {r.status_code})")
    try:
        return r.json()
    except ValueError as e:
        raise MusicSourceError("Epidemic Sound sent something unexpected") from e


def _es_track(t: dict) -> dict:
    artists = ", ".join(str(a.get("name") if isinstance(a, dict) else a) for a in (t.get("mainArtists") or [])) or "Epidemic Sound"
    title = " ".join(str(t.get("title") or "Untitled").split())[:120]
    tags = [str(m.get("name") or m.get("id") or "") for m in (t.get("moods") or []) if isinstance(m, dict)] + \
           [str(g.get("name") or g.get("id") or "") for g in (t.get("genres") or []) if isinstance(g, dict)]
    from urllib.parse import quote
    return {"source": "epidemic", "id": str(t.get("id")), "title": title, "artist": artists, "dur": float(t.get("length") or 0),
            "preview": "", "license": "Epidemic Sound", "license_url": "https://www.epidemicsound.com/licensing/", "credit": "",
            "page": "https://www.epidemicsound.com/music/search/?term=" + quote(f"{title} {artists}"), "tags": [x for x in tags if x][:12],
            "provider": "epidemic", "bpm": t.get("bpm"), "preview_only": bool(t.get("isPreviewOnly"))}


def epidemic_status(key: str) -> dict:
    """Does this key work? (A one-song search: an API key isn't a user, so there is no account name to show.)"""
    try:
        _es_get(key, "/v0/tracks/search", {"term": "piano", "limit": 1})
    except MusicSourceError as e:
        return {"ok": False, "error": str(e), "name": None}
    return {"ok": True, "error": None, "name": "Epidemic Sound API key"}


def epidemic_search(key: str, q: str, n: int = 30) -> list[dict]:
    q = " ".join(str(q or "").split())[:120]
    n = max(1, min(60, int(n)))

    def run():
        d = _es_get(key, "/v0/tracks/search", {"term": q or "cinematic", "limit": n})
        return [_es_track(t) for t in (d.get("tracks") or []) if isinstance(t, dict) and t.get("id")]

    return _cached(("epidemic", hash(key), q.lower(), n), run)        # (the key itself isn't kept in the cache's keys)


def epidemic_preview(key: str, track_id: str) -> str:
    """A short-lived link to hear the song (Epidemic's /stream: AAC/HLS, plays in the app's page)."""
    if not re.fullmatch(r"[\w-]{1,80}", str(track_id or "")):
        raise MusicSourceError("which song?")
    d = _es_get(key, f"/v0/tracks/{track_id}/stream")
    url = str(d.get("url") or "")
    if not url.startswith("https://"):
        raise MusicSourceError("Epidemic Sound didn't give a preview for that song")
    return url


# ------------------------------------------------------------------------------------------------ fetching into the library
def _name(track: dict) -> str:
    name = _BAD_NAME.sub(" ", f"{track.get('artist') or 'Unknown'} - {track.get('title') or 'Untitled'}")
    return re.sub(r"\s+", " ", name).strip().strip(".")[:120] or "Song"


def _probe_audio(path: Path) -> float:
    """The song's length in seconds if it is real audio, else 0."""
    try:
        info = ffmpeg.probe(path)
    except Exception:  # noqa: BLE001
        return 0.0
    if ffmpeg.audio_stream(info) is None:
        return 0.0
    return float((info.get("format") or {}).get("duration") or 0)


def fetch(track: dict, dest_dir: Path, key: str | None = None, licensed: bool = False) -> Path:
    """The song, downloaded once into dest_dir/<source>/<Artist - Title>.mp3 with its credit (sidecar + tags). Already there: the same
    file, no download. A failed or not-audio download leaves nothing behind."""
    src = str(track.get("source") or "")
    if src not in ("openverse", "epidemic"):
        raise MusicSourceError("that song isn't from a music source FirstCut knows")
    if src == "epidemic" and not licensed:
        raise MusicSourceError("Epidemic Sound's free developer key is for trying the API: songs downloaded with it aren't licensed for videos you "
                               "publish (Epidemic's API Developer License). Download this song on epidemicsound.com with your subscription and drop "
                               "it on FirstCut, or use a key from a paid Epidemic Sound API plan")
    folder = Path(dest_dir) / src
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{_name(track)}.mp3"
    side = out.with_suffix(".json")
    if out.exists() and side.exists() and _probe_audio(out) > 0:
        return out
    if src == "openverse":
        url = str(track.get("preview") or "")
        if not url.startswith(("https://", "http://")):
            raise MusicSourceError("that song has no file to download")
    else:
        d = _es_get(key or "", f"/v0/tracks/{track.get('id')}/download", {"format": "mp3", "quality": "high"})
        url = str(d.get("url") or "")
    part = out.with_name(out.name + ".part")
    raw = out.with_name(out.name + ".download")
    try:
        try:
            with requests.get(url, headers={"User-Agent": UA}, timeout=(20, 180), stream=True) as r:
                if not r.ok:
                    raise MusicSourceError(f"the download didn't work (HTTP {r.status_code})")
                size = 0
                with open(raw, "wb") as fh:
                    for chunk in r.iter_content(256 * 1024):
                        size += len(chunk)
                        if size > 200 * 1024 * 1024:
                            raise MusicSourceError("that file is far too big to be a song")
                        fh.write(chunk)
        except requests.RequestException as e:
            raise MusicSourceError("the download stopped: check the internet connection and try again") from e
        if raw.stat().st_size < 30 * 1024 or _probe_audio(raw) < 10:
            raise MusicSourceError("what came back isn't a song (it may have been taken down)")
        meta = {"title": track.get("title") or "", "artist": track.get("artist") or "", "comment": track.get("credit") or "",
                "firstcut_credit": track.get("credit") or "", "firstcut_license": track.get("license") or "",
                "firstcut_source": f"{src}:{track.get('id')}"}
        tags = [x for k, v in meta.items() for x in ("-metadata", f"{k}={v}")]
        with open(raw, "rb") as fh:
            head = fh.read(3)
        mp3 = (track.get("filetype") or "").lower().startswith("mp3") or head in (b"ID3", b"\xff\xfb", b"\xff\xf3")
        codec = ["-c:a", "copy"] if mp3 else ["-c:a", "libmp3lame", "-b:a", "192k"]
        subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-i", str(raw), "-map", "0:a:0", "-vn", *codec, *tags, "-id3v2_version", "3",
                        "-f", "mp3", str(part)], check=True, capture_output=True, timeout=600)
        if _probe_audio(part) < 10:
            raise MusicSourceError("the song couldn't be saved")
        part.replace(out)
        side.write_text(json.dumps({"source": src, "id": track.get("id"), "title": track.get("title"), "artist": track.get("artist"),
                                    "license": track.get("license"), "license_url": track.get("license_url"), "credit": track.get("credit") or "",
                                    "page": track.get("page"), "provider": track.get("provider"), "fetched_at": int(time.time())}, indent=1) + "\n")
        log.info(f"fetched {out.name} from {src} ({track.get('license')})")
        return out
    except subprocess.CalledProcessError as e:
        out.unlink(missing_ok=True)
        side.unlink(missing_ok=True)
        raise MusicSourceError("the song couldn't be converted to MP3") from e
    except BaseException:
        out.unlink(missing_ok=True)
        side.unlink(missing_ok=True)
        raise
    finally:
        part.unlink(missing_ok=True)
        raw.unlink(missing_ok=True)


def credit_for(path) -> str | None:
    """The credit line of a song fetched from an online source ("" when its licence needs none), from its sidecar or - for a copy, say in a
    project's Music folder - from the file's own tags. None for your own music."""
    p = Path(str(path or ""))
    for side in (p.with_suffix(".json"), p.with_name(p.name + ".json")):
        try:
            d = json.loads(side.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and d.get("source") in ("openverse", "epidemic"):
            return str(d.get("credit") or "")
    if not p.is_file():
        return None
    try:
        info = ffmpeg.probe(p)
    except Exception:  # noqa: BLE001
        return None
    tags = {str(k).lower(): str(v) for k, v in ((info.get("format") or {}).get("tags") or {}).items()}
    if "firstcut_source" in tags or "firstcut_credit" in tags:
        return tags.get("firstcut_credit", "")
    return None


def license_for(path) -> str | None:
    """The licence name of a fetched song (sidecar or tags), None for your own music."""
    p = Path(str(path or ""))
    for side in (p.with_suffix(".json"), p.with_name(p.name + ".json")):
        try:
            d = json.loads(side.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and d.get("source") in ("openverse", "epidemic"):
            return str(d.get("license") or "")
    if not p.is_file():
        return None
    try:
        tags = {str(k).lower(): str(v) for k, v in ((ffmpeg.probe(p).get("format") or {}).get("tags") or {}).items()}
    except Exception:  # noqa: BLE001
        return None
    return tags.get("firstcut_license") if "firstcut_source" in tags else None
