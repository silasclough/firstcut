"""Find my music downloads: songs you already downloaded from a licensed music library (Artlist, Soundstripe, Musicbed, Epidemic Sound,
PremiumBeat, Motion Array, Envato, Storyblocks, Pond5) are found in Downloads (or any folder you add) and offered for your music library.

How a file is told apart (nothing is guessed from the sound, only from what the file says about itself):
    sure     the library's name is in the file's tags (copyright "Licensed for video by Artlist.io", album artist, publisher, comment...),
             in its file name, in its folder's name, or in the name of the zip it came in
    maybe    no library named, but it looks like one ("Artist - Title", exported from Pro Tools): offered, not ticked
    never    ripped from YouTube or a downloader site (container brand "dash", "ytdown", "youtube"...) are not licensed: they are only
             counted, never offered. Voice memos and generated voices are counted too.

Scanning reads only folder listings and each file's tags (ffprobe, 4 at a time, remembered by path + size + date so a second scan of a
big Downloads folder is instant). Nothing is changed by a scan.

Importing COPIES (never moves or deletes the original) into <library folder>/<Source>/, keeps the file name (a counter on a clash),
unpacks songs from zips, and writes a small `<song>.json` next to each copy (source, licence line, where it came from) so the video's
Notes/credits.txt can say "Song, Artist (licensed via Artlist)". A song is remembered by a fingerprint (sha1 of its first MB + its size):
the same song is never offered twice, nor imported twice. Songs the person unticked and chose "Don't offer again" for are remembered
(declined.json) and never come back, not even in "Import new" (import_new: the one-click version, which takes only the sure songs that
are not in the library yet and not declined, and never a maybe or a rip).

    scan(folders, ...)  -> {found, maybe, skipped: {youtube, recordings, other, duplicates, already}, partial, denied, scanned, ...}
    import_items(items, dest_root, state_dir) -> {imported: [...], failed: [...], already: [...]}
    credit_line_for(path) -> "Title, Artist (licensed via Artlist)" | None
    Finder: the app's one background scan (start / state), with a time budget and a clear message when macOS has not allowed the folder.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..core import ffmpeg
from ..core.log import get_logger

log = get_logger("clipper.musicfind")

AUDIO_EXT = {".mp3", ".wav", ".aif", ".aiff", ".m4a", ".flac"}
SITES = [("Artlist", r"art-?list"), ("Soundstripe", r"sound-?stripe"), ("Musicbed", r"musicbed"), ("Epidemic Sound", r"epidemic[ _-]?sound"),
         ("PremiumBeat", r"premium[ _-]?beat"), ("Motion Array", r"motion[ _-]?array"), ("Envato", r"envato"),
         ("Storyblocks", r"(?:story|audio)blocks"), ("Pond5", r"pond-?5")]
SITE_RX = [(name, re.compile(rx, re.I)) for name, rx in SITES]
RIP_RX = re.compile(r"ytdown|youtube|youtu\.be|yt1s|y2mate|savefrom|ytmp3|spotdl|spotifydown|snaptube|tubidy|mp3juice|\byt-?dlp?\b", re.I)
RECORDING_RX = re.compile(r"^(new recording|voice memo|audio recording|recording|untitled)[ _\d()-]*$|elevenlabs", re.I)
TAG_KEYS = ("copyright", "album_artist", "publisher", "comment", "label", "organization", "encoded_by", "album", "artist", "title",
            "description", "composer", "grouping", "url")
SKIP_DIRS = {"node_modules", "__macosx", ".git"}
SKIP_SUFFIX = (".app", ".photoslibrary", ".imovielibrary", ".fcpbundle", ".logicx", ".download")   # (".download": Safari's unfinished downloads)
DEFAULT_LIBRARY = Path.home() / "Music" / "FirstCut Library"
CACHE_MAX = 30000
MAX_ZIP_MEMBERS = 200


def _safe(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", str(name or "")).strip(" .")[:120] or "song"


def _site_in(text: str) -> str | None:
    for name, rx in SITE_RX:
        if rx.search(text):
            return name
    return None


# ------------------------------------------------------------------------------------------------------------------- classification
def classify(path: str, tags: dict, folders: str = "", container: str = "") -> dict:
    """What a file says it is: {bucket: found | maybe | youtube | recordings | other, source, why, license_line, reason}.
    `tags` are the file's tags (lower-case keys), `folders` the names of the folders above it (and a zip's name), `container` a zip's name."""
    name = Path(path).name
    stem = Path(path).stem
    tags = {str(k).lower(): str(v) for k, v in (tags or {}).items()}
    tag_text = " ".join(tags.get(k, "") for k in TAG_KEYS)
    if tags.get("major_brand", "").strip().lower() == "dash" or RIP_RX.search(name) or RIP_RX.search(tag_text) or RIP_RX.search(folders) or RIP_RX.search(container):
        return {"bucket": "youtube", "why": "ripped from YouTube or a downloader: not licensed"}
    if RECORDING_RX.search(stem) or RECORDING_RX.search(tags.get("encoded_by", "")):
        return {"bucket": "recordings", "why": "a voice recording or generated voice"}
    # the site named in a tag (the strongest sign), then the file name, then the folder or zip it came in
    for key in ("copyright", "album_artist", "publisher", "label", "organization", "comment", "encoded_by", "album", "url", "description", "grouping"):
        site = _site_in(tags.get(key, ""))
        if site:
            line = tags.get(key, "").strip()
            return {"bucket": "found", "source": site, "why": f"its {key.replace('_', ' ')} tag says “{line[:80]}”",
                    "license_line": line[:160] if key == "copyright" or re.search(r"licen[cs]ed", line, re.I) else tags.get("copyright", "")[:160]}
    site = _site_in(name)
    if site:
        return {"bucket": "found", "source": site, "why": f"the file name says {site}", "license_line": tags.get("copyright", "")[:160]}
    site = _site_in(container) or _site_in(folders)
    if site:
        return {"bucket": "found", "source": site, "why": f"it came in a folder called {site}", "license_line": ""}
    if "pro tools" in tags.get("encoded_by", "").lower() and " - " in name:
        return {"bucket": "maybe", "source": "", "why": "", "license_line": "",
                "reason": "Named “Artist - Title” and exported from Pro Tools, like many library downloads, but nothing in it says where it came from."}
    return {"bucket": "other", "why": "nothing says it came from a music library"}


# ------------------------------------------------------------------------------------------------------------------- the tag cache
class _Cache:
    def __init__(self, file: Path | None):
        self.file, self.lock, self.data, self.dirty = file, threading.Lock(), {}, False
        if file and file.is_file():
            try:
                self.data = json.loads(file.read_text())
            except (OSError, ValueError):
                self.data = {}

    def get(self, key):
        with self.lock:
            return self.data.get(key)

    def put(self, key, value):
        with self.lock:
            self.data[key] = value
            self.dirty = True

    def save(self):
        if not (self.file and self.dirty):
            return
        with self.lock:
            if len(self.data) > CACHE_MAX:
                self.data = dict(list(self.data.items())[-CACHE_MAX // 2:])
            tmp = self.file.with_name(self.file.name + ".part")
            try:
                self.file.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(json.dumps(self.data))
                tmp.replace(self.file)
                self.dirty = False
            except OSError as e:
                log.info(f"music finder: could not keep its cache ({e})")


def probe_tags(path: Path, run=subprocess.run) -> dict:
    """{"tags": {...lower-case...}, "dur": seconds} from ffprobe (empty when the file cannot be read)."""
    try:
        r = run([ffmpeg.bin_path("ffprobe"), "-v", "error", "-show_entries", "format=duration:format_tags", "-of", "json", str(path)],
                capture_output=True, text=True, timeout=25)
        f = (json.loads(r.stdout or "{}").get("format") or {})
    except Exception:  # noqa: BLE001 - unreadable: no tags, and it will count as "other"
        return {"tags": {}, "dur": 0.0, "bad": True}
    try:
        dur = float(f.get("duration") or 0.0)
    except (TypeError, ValueError):
        dur = 0.0
    return {"tags": {str(k).lower(): str(v) for k, v in (f.get("tags") or {}).items()}, "dur": round(dur, 2)}


def fingerprint(path: Path) -> str:
    """sha1 of the first MB + the size: the same song under another name or in another folder is recognised."""
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        h.update(fh.read(1_000_000))
    return h.hexdigest()[:20] + f"-{path.stat().st_size}"


def _imported(state_dir: Path | None) -> dict:
    try:
        return json.loads((Path(state_dir) / "imported.json").read_text()) if state_dir else {}
    except (OSError, ValueError):
        return {}


def _json_file(state_dir: Path | None, name: str) -> dict:
    try:
        return json.loads((Path(state_dir) / name).read_text()) if state_dir else {}
    except (OSError, ValueError):
        return {}


def _write_json(state_dir: Path | None, name: str, data: dict) -> None:
    if not state_dir:
        return
    p = Path(state_dir) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    tmp.write_text(json.dumps(data))
    tmp.replace(p)


def item_key(it: dict) -> str:
    """What is remembered about a song across scans: its fingerprint, or (a song inside a zip) the zip's path + the member."""
    return str(it.get("fp") or f"{it.get('path')}|{it.get('member')}")


def declined(state_dir: Path | None) -> dict:
    return _json_file(state_dir, "declined.json")


def decline(items: list[dict], state_dir: Path | None) -> int:
    """"Don't offer again": these songs never come back in a scan or in Import new."""
    d = declined(state_dir)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    for it in items:
        d[item_key(it)] = {"name": str(it.get("name") or ""), "at": now}
    _write_json(state_dir, "declined.json", d)
    return len(d)


def undecline(state_dir: Path | None) -> None:
    _write_json(state_dir, "declined.json", {})


def last_imported(state_dir: Path | None) -> str:
    return str(_json_file(state_dir, "meta.json").get("last_imported") or "")


def _remember(state_dir: Path | None, fp: str, dest: str) -> None:
    if not state_dir:
        return
    d = _imported(state_dir)
    d[fp] = dest
    p = Path(state_dir) / "imported.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    tmp.write_text(json.dumps(d))
    tmp.replace(p)


# ------------------------------------------------------------------------------------------------------------------- the scan
def _walk(root: Path, max_depth: int, skip_roots: list[Path], stop, note_denied):
    """(file, folder names above it) for every file down to max_depth; folder listings only."""
    def rec(d: Path, depth: int, names: str):
        if stop():
            return
        try:
            with os.scandir(d) as it:
                entries = sorted(it, key=lambda e: e.name.lower())
        except PermissionError:
            note_denied(d)
            return
        except OSError:
            return
        for e in entries:
            if stop():
                return
            n = e.name
            if n.startswith("."):
                continue
            try:
                isdir = e.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if isdir:
                low = n.lower()
                if depth >= max_depth or low in SKIP_DIRS or low.endswith(SKIP_SUFFIX) or any(Path(e.path) == s or s in Path(e.path).parents for s in skip_roots):
                    continue
                rec(Path(e.path), depth + 1, names + "/" + n)
            else:
                yield_list.append((Path(e.path), names))
    yield_list: list = []
    rec(root, 0, root.name)
    return yield_list


def scan(folders=None, *, budget_s: float = 60.0, state_dir: Path | None = None, library_roots=(), progress=None, workers: int = 4,
         max_depth: int = 4, run=subprocess.run) -> dict:
    """Look for licensed songs. `progress(scanned, found)` is called as it goes. Stops listing/probing after `budget_s` seconds (partial)."""
    t0 = time.monotonic()
    roots = [Path(f).expanduser() for f in (folders or [Path.home() / "Downloads"])]
    skip = [Path(r).expanduser() for r in library_roots if r]
    cache = _Cache(Path(state_dir) / "scan-cache.json" if state_dir else None)
    done = _imported(state_dir)
    no = declined(state_dir)
    out = {"found": [], "maybe": [], "skipped": {"youtube": 0, "recordings": 0, "other": 0, "duplicates": 0, "already": 0, "declined": 0},
           "partial": False, "denied": [], "scanned": 0, "folders": [str(r) for r in roots]}
    lock = threading.Lock()
    seen_fp: dict[str, bool] = {}

    def over() -> bool:
        return time.monotonic() - t0 > budget_s

    def note_denied(d):
        out["denied"].append(str(d))

    def add(item: dict, fp: str | None):
        with lock:
            if fp and fp in done:
                out["skipped"]["already"] += 1
                return
            if item_key(item) in no:
                out["skipped"]["declined"] += 1
                return
            if fp and fp in seen_fp:
                out["skipped"]["duplicates"] += 1
                return
            if fp:
                seen_fp[fp] = True
            out["found" if item["confidence"] == "sure" else "maybe"].append(item)

    def one(path: Path, folders_txt: str):
        if over():
            out["partial"] = True
            return
        try:
            st = path.stat()
        except OSError:
            return
        key = f"{path}|{st.st_size}|{int(st.st_mtime)}"
        info = cache.get(key)
        if info is None:
            info = probe_tags(path, run)
            if not info.get("bad"):
                cache.put(key, info)
        c = classify(str(path), info.get("tags") or {}, folders_txt)
        with lock:
            out["scanned"] += 1
            n = out["scanned"]
        if progress and n % 10 == 0:
            progress(n, len(out["found"]))
        if c["bucket"] in ("youtube", "recordings", "other"):
            with lock:
                out["skipped"][c["bucket"]] += 1
            return
        tags = info.get("tags") or {}
        try:
            fp = fingerprint(path)
        except OSError:
            return
        add({"id": hashlib.sha1(f"{path}".encode()).hexdigest()[:16], "path": str(path), "member": None, "name": path.name, "folder": str(path.parent),
             "size": st.st_size, "duration": info.get("dur") or 0.0, "title": tags.get("title") or path.stem, "artist": tags.get("artist") or tags.get("album_artist") or "",
             "source": c.get("source") or "", "why": c.get("why") or "", "reason": c.get("reason") or "", "license_line": c.get("license_line") or "",
             "confidence": "sure" if c["bucket"] == "found" else "maybe", "fp": fp, "from_zip": False}, fp)

    def zipped(path: Path, folders_txt: str):
        try:
            with zipfile.ZipFile(path) as zf:
                members = [i for i in zf.infolist() if not i.is_dir() and Path(i.filename).suffix.lower() in AUDIO_EXT
                           and "__macosx" not in i.filename.lower() and not Path(i.filename).name.startswith(".")][:MAX_ZIP_MEMBERS]
        except (OSError, zipfile.BadZipFile, RuntimeError):
            return
        for i in members:
            if over():
                out["partial"] = True
                return
            c = classify(i.filename, {}, folders_txt, container=path.name)
            with lock:
                out["scanned"] += 1
            if c["bucket"] != "found":
                with lock:
                    out["skipped"]["youtube" if c["bucket"] == "youtube" else "other"] += 1
                continue
            stem = Path(i.filename).stem
            add({"id": hashlib.sha1(f"{path}|{i.filename}".encode()).hexdigest()[:16], "path": str(path), "member": i.filename, "name": Path(i.filename).name,
                 "folder": str(path.parent), "size": i.file_size, "duration": 0.0, "title": stem, "artist": "", "source": c["source"], "why": c["why"] + f" (inside {path.name})",
                 "reason": "", "license_line": "", "confidence": "sure", "fp": None, "from_zip": True}, None)

    files: list = []
    for root in roots:
        if not root.is_dir():
            continue
        files += _walk(root, max_depth, skip, over, note_denied)
    audio = [(p, n) for p, n in files if p.suffix.lower() in AUDIO_EXT]
    zips = [(p, n) for p, n in files if p.suffix.lower() == ".zip"]
    with ThreadPoolExecutor(max(1, workers), thread_name_prefix="musicfind") as pool:
        list(pool.map(lambda a: one(*a), audio))
    for p, n in zips:
        if over():
            out["partial"] = True
            break
        zipped(p, n)
    cache.save()
    if over():
        out["partial"] = True
    out["found"].sort(key=lambda x: (x["source"], x["artist"].lower(), x["title"].lower()))
    out["maybe"].sort(key=lambda x: (x["artist"].lower(), x["title"].lower()))
    out["took_s"] = round(time.monotonic() - t0, 2)
    if progress:
        progress(out["scanned"], len(out["found"]))
    return out


# ------------------------------------------------------------------------------------------------------------------- the import
def credit_line_for(path) -> str | None:
    """"Title, Artist (licensed via Artlist)" for a song that came in through the music finder (from its sidecar), else None."""
    p = Path(str(path or ""))
    for side in (p.with_name(p.name + ".json"), p.with_suffix(".json")):
        try:
            d = json.loads(side.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and d.get("source") == "library" and d.get("credit_line"):
            return str(d["credit_line"])
    return None


def _free_name(folder: Path, name: str) -> Path:
    p = folder / name
    n = 2
    while p.exists() or p.with_name(p.name + ".json").exists():
        p = folder / f"{Path(name).stem} ({n}){Path(name).suffix}"
        n += 1
    return p


def import_items(items: list[dict], dest_root: Path, state_dir: Path | None = None) -> dict:
    """COPY the songs into <dest_root>/<Source>/ (zip members are unpacked), with a sidecar each. Originals are never touched."""
    dest_root = Path(dest_root).expanduser()
    done = _imported(state_dir)
    res = {"imported": [], "failed": [], "already": []}
    zips: dict[str, zipfile.ZipFile] = {}
    try:
        for it in items:
            src, member = Path(str(it.get("path") or "")), it.get("member")
            title, artist, site = str(it.get("title") or src.stem), str(it.get("artist") or ""), str(it.get("source") or "")
            try:
                fp = it.get("fp") or (fingerprint(src) if not member else None)
                if fp and fp in done:
                    res["already"].append(str(it.get("name") or src.name))
                    continue
                folder = dest_root / _safe(site or "Other")
                folder.mkdir(parents=True, exist_ok=True)
                name = _safe(Path(member).name if member else src.name)
                dst = _free_name(folder, name)
                tmp = dst.with_name(dst.name + ".part")
                if member:
                    zf = zips.get(str(src)) or zips.setdefault(str(src), zipfile.ZipFile(src))
                    with zf.open(member) as r, open(tmp, "wb") as w:
                        shutil.copyfileobj(r, w)
                else:
                    shutil.copy2(src, tmp)
                tmp.replace(dst)
                credit = f"{title}, {artist} (licensed via {site})" if artist and site else f"{title} (licensed via {site})" if site else title
                dst.with_name(dst.name + ".json").write_text(json.dumps({
                    "source": "library", "provider": site, "title": title, "artist": artist, "license_line": str(it.get("license_line") or ""),
                    "credit": "", "credit_line": credit if site else "", "from": str(src) + (f"!{member}" if member else ""),
                    "imported": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
                if fp:
                    _remember(state_dir, fp, str(dst))
                    try:
                        done[fp] = str(dst)                          # (the same bytes again, in this batch: already imported)
                    except TypeError:
                        pass
                m = _json_file(state_dir, "meta.json")
                m["last_imported"] = time.strftime("%Y-%m-%d %H:%M:%S")
                _write_json(state_dir, "meta.json", m)
                res["imported"].append({"path": str(dst), "title": title, "artist": artist, "source": site, "id": it.get("id")})
            except Exception as e:  # noqa: BLE001 - one bad file never stops the rest
                res["failed"].append({"name": str(it.get("name") or src.name), "error": f"{type(e).__name__}: {e}"[:160]})
    finally:
        for z in zips.values():
            try:
                z.close()
            except Exception:  # noqa: BLE001
                pass
    return res


def import_new(result: dict, dest_root: Path, state_dir: Path | None = None) -> dict:
    """One click: every SURE song of a scan (already not in the library, not declined: the scan left those out) is copied in. A maybe is
    never included, nor a rip (those are not in `found`). -> import_items' result + {sources: {Artlist: 7}, maybes: n}."""
    items = list(result.get("found") or [])
    res = import_items(items, dest_root, state_dir) if items else {"imported": [], "failed": [], "already": []}
    srcs: dict[str, int] = {}
    for i in res["imported"]:
        srcs[i["source"] or "Other"] = srcs.get(i["source"] or "Other", 0) + 1
    res["sources"], res["maybes"] = srcs, len(result.get("maybe") or [])
    return res


# ------------------------------------------------------------------------------------------------------------------- the app's one scan
class Finder:
    """One background scan at a time, with a state the page can poll."""

    def __init__(self, state_dir: Path | None, library_roots=lambda: (), budget_s: float = 90.0):
        self.state_dir, self.library_roots, self.budget_s = state_dir, library_roots, budget_s
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.result: dict | None = None
        self.started = 0.0
        self.scanned = 0
        self.error = ""
        self.items: dict[str, dict] = {}                 # id -> item of the last scan: the only things the page may ask to play or import
        self.after_result: dict | None = None             # what `after` returned (e.g. the result of Import new)

    def start(self, folders=None, budget_s: float | None = None, after=None) -> bool:
        """Scan in the background. `after(result)` runs in the same thread when the scan is done (to allow playback of the found songs, or
        to import them: Import new); what it returns is in state()['after']."""
        with self.lock:
            if self.thread and self.thread.is_alive():
                return False
            self.started, self.scanned, self.error, self.result, self.after_result = time.time(), 0, "", None, None
            self.thread = threading.Thread(target=self._run, args=(folders, budget_s or self.budget_s, after), name="music-finder", daemon=True)
            self.thread.start()
            return True

    def _run(self, folders, budget, after=None):
        try:
            def prog(n, f):
                self.scanned = n
            r = scan(folders, budget_s=budget, state_dir=self.state_dir, library_roots=self.library_roots(), progress=prog)
            with self.lock:
                self.result = r
                self.items = {i["id"]: i for i in r["found"] + r["maybe"]}
            if not r["denied"] or r["scanned"]:
                m = _json_file(self.state_dir, "meta.json")
                m["last_scan"] = time.strftime("%Y-%m-%d %H:%M:%S")
                _write_json(self.state_dir, "meta.json", m)
            if after:
                self.after_result = after(r)
        except Exception as e:  # noqa: BLE001 - said on the page
            log.exception("music finder failed")
            self.error = f"{type(e).__name__}: {e}"[:200]

    def drop(self, ids) -> None:
        """Songs just imported or declined are no longer 'new': the count on the Import new button goes down at once."""
        ids = {i for i in ids if i}
        with self.lock:
            r = self.result
            if r:
                r["found"] = [i for i in r["found"] if i["id"] not in ids]
                r["maybe"] = [i for i in r["maybe"] if i["id"] not in ids]
            for i in ids:
                self.items.pop(i, None)

    def running(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    def state(self, items: bool = True) -> dict:
        running = self.running()
        elapsed = time.time() - self.started if self.started else 0
        r = self.result
        out = {"running": running, "scanned": self.scanned, "error": self.error, "at": self.started or None, "after": self.after_result,
               "waiting": bool(running and self.scanned == 0 and elapsed > 6),           # (stuck in the folder: macOS is asking whether FirstCut may read it)
               "done": bool(r is not None)}
        if r is not None:
            out.update({"skipped": r["skipped"], "partial": r["partial"], "denied": r["denied"], "folders": r["folders"], "took_s": r.get("took_s"),
                        "n_found": len(r["found"]), "n_maybe": len(r["maybe"])})
            if items:
                out.update({"found": r["found"], "maybe": r["maybe"]})
        return out

    def new_summary(self) -> dict:
        """For the Music nav badge and the Import new button: how many sure songs the last scan found that are not in the library yet (and not
        declined), and when the library last took songs from here."""
        r = self.result
        meta = _json_file(self.state_dir, "meta.json")
        base = {"running": self.running(), "scanned_before": bool(meta.get("last_scan")), "last_imported": meta.get("last_imported") or "",
                "at": self.started or None}
        if not r:
            return {**base, "n": 0, "sources": {}, "n_maybe": 0}
        src: dict[str, int] = {}
        for i in r["found"]:
            src[i["source"]] = src.get(i["source"], 0) + 1
        return {**base, "n": len(r["found"]), "sources": src, "n_maybe": len(r["maybe"])}
