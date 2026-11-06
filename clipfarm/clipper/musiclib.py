"""Your music library: the songs you downloaded (Artlist, Soundstripe, your own...) in one or more folders, catalogued once on this Mac so
the editor can pick a song that fits, start a short clip at the song's best part, and you can search it in words.

Per song (one JSON per song in <dir>/tracks/<id>.json, made once, redone when the file changes):
    duration, a rough loudness, tempo (BPM, from the onset rhythm), an energy curve (0-1, every 0.5 s), `best` = where the most
    energetic stretch starts (the drop), whether it has a proper ending (fades or rings out), title/artist from the file's tags (else
    the file name), and - when the CLAP music model is on this Mac (Setup downloads MODEL into HF_HOME) - an embedding of the song
    for searching in words ("upbeat corporate", "gentle piano").
Without the model, search reads the names (file, folders, tags) and turns mood words into energy and tempo.

Config: clipper.music_lib.roots (folders), clipper.music_lib.dir (the catalogue; default <root>/data/music_lib).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
import threading
import time
from pathlib import Path

from ..core import ffmpeg
from ..core.log import get_logger
from ..core.music import Track
from ..core.state import atomic_write_json

log = get_logger("clipper.musiclib")

# laion/larger_clap_music's text side is broken (2026-11-06: every phrase got the same vector, "calm piano" = "hardcore techno",
# under transformers 4.57 and 5.22 alike), so describing a song found nothing. clap-htsat-unfused tells them apart (0.81 vs 0.07)
# and ranks real songs right ("calm piano": the piano pieces first).
MODEL = "laion/clap-htsat-unfused"
MODEL_GB = 0.8
AUDIO_EXT = {".mp3", ".wav", ".aif", ".aiff", ".m4a", ".flac", ".ogg", ".aac"}
SR = 11025                                   # analysis sample rate: enough for rhythm and energy
HOP_S = 0.5                                  # energy curve resolution
SKIP = re.compile(r"^\.|^__macosx$|stems?$|^sfx$|sound ?effects", re.I)
VERSION = 3                                  # bump to re-analyse every song after the analysis changes

# Mood words -> what the music should sound like, when there is no model to listen for us: (energy 0-1, tempo range)
MOOD_SHAPE = {
    "calm": (0.25, (60, 100)), "chill": (0.3, (70, 105)), "gentle": (0.2, (60, 100)), "soft": (0.2, (60, 100)), "ambient": (0.15, (50, 100)),
    "piano": (0.25, (60, 110)), "emotional": (0.35, (60, 100)), "sad": (0.25, (55, 90)), "cinematic": (0.5, (70, 120)),
    "inspiring": (0.55, (90, 125)), "corporate": (0.55, (95, 125)), "happy": (0.65, (100, 135)), "upbeat": (0.75, (105, 140)),
    "fun": (0.7, (100, 140)), "funny": (0.6, (95, 135)), "energetic": (0.85, (115, 160)), "driving": (0.8, (110, 150)),
    "epic": (0.85, (80, 140)), "hype": (0.9, (120, 170)), "suspense": (0.45, (60, 110)), "adventure": (0.7, (95, 140)),
}


def track_id(path: Path) -> str:
    return hashlib.sha1(str(Path(path).expanduser().resolve()).encode()).hexdigest()[:16]


def _decode(path: Path, sr: int = SR, start: float = 0.0, dur: float | None = None):
    """Mono float32 samples of a song (numpy), via ffmpeg."""
    import numpy as np
    cmd = [ffmpeg.bin_path("ffmpeg"), "-v", "error", "-nostdin"] + (["-ss", f"{start:.2f}"] if start else []) + ["-i", str(path)] + \
        (["-t", f"{dur:.2f}"] if dur else []) + ["-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    r = subprocess.run(cmd, capture_output=True, timeout=600)
    if r.returncode or not r.stdout:
        raise RuntimeError((r.stderr or b"").decode(errors="replace")[:200] or "no audio")
    return np.frombuffer(r.stdout, dtype="<i2").astype("float32") / 32768.0


def analyse(path: Path) -> dict:
    """Tempo, energy curve, best part, ending and loudness of one song (a few seconds for a 3-minute song)."""
    import numpy as np
    x = _decode(path)
    dur = len(x) / SR
    hop = int(SR * HOP_S)
    n = max(1, len(x) // hop)
    rms = np.sqrt(np.array([float(np.mean(x[k * hop:(k + 1) * hop] ** 2)) for k in range(n)]) + 1e-12)
    loud_db = round(float(20 * np.log10(np.sqrt(float(np.mean(x ** 2))) + 1e-9)), 1)
    top = float(np.percentile(rms, 98)) or 1e-9
    energy = np.clip(rms / top, 0, 1)
    # tempo: spectral flux onsets, autocorrelated over 60-180 BPM
    bpm = 0.0
    fr, fh = 1024, 256
    if len(x) > fr * 8:
        win = np.hanning(fr).astype("float32")
        frames = np.lib.stride_tricks.sliding_window_view(x, fr)[::fh] * win
        mag = np.abs(np.fft.rfft(frames, axis=1))
        flux = np.maximum(0, np.diff(np.log1p(mag), axis=0)).sum(axis=1)
        flux = flux - np.convolve(flux, np.ones(16) / 16, mode="same")
        flux = np.maximum(flux, 0)
        fps = SR / fh
        ac = np.correlate(flux, flux, mode="full")[len(flux) - 1:]
        lags = np.arange(len(ac))
        lo, hi = int(fps * 60 / 180), int(fps * 60 / 60)
        if hi < len(ac) and lo > 0:
            cand = lags[lo:hi]
            score = ac[lo:hi] * (1.0 + 0.15 * np.exp(-((60 * fps / cand - 118) / 40) ** 2))     # a little preference for 90-150
            lag = int(cand[int(np.argmax(score))])
            if 2 * lag < len(ac) and ac[2 * lag] > 0.9 * ac[lag] and 60 * fps / lag > 150:      # (a double-time reading: halve it)
                lag *= 2
            bpm = round(60 * fps / lag, 1)
    # best part: where a loud stretch (about 8 s) starts, rewarded for coming after something quieter (the drop)
    best = 0.0
    w = max(2, int(8 / HOP_S))
    if n > w + 2:
        csum = np.concatenate([[0.0], np.cumsum(energy)])
        mean_ahead = (csum[w:] - csum[:-w]) / w
        before = np.array([float(np.mean(energy[max(0, k - int(4 / HOP_S)):k])) if k else float(energy[0]) for k in range(len(mean_ahead))])
        score = mean_ahead - 0.4 * before
        k = int(np.argmax(score))
        best = round(min(k * HOP_S, max(0.0, dur - 6)), 2)
    tail = energy[-max(1, int(2.5 / HOP_S)):]
    ending = bool(len(energy) > 8 and float(np.mean(tail)) < 0.35 * float(np.median(energy) or 1))
    step = max(1, int(np.ceil(n / 240)))                                  # (stored at 0.5 s up to 2 minutes, coarser beyond)
    curve = [round(float(np.max(energy[i:i + step])), 3) for i in range(0, n, step)]
    return {"dur": round(dur, 2), "loud_db": loud_db, "bpm": bpm, "best": best, "ending": ending,
            "energy": curve, "energy_step": round(HOP_S * step, 3), "avg_energy": round(float(np.mean(energy)), 3)}


def _tags(path: Path) -> dict:
    try:
        info = ffmpeg.probe(path)
    except Exception:  # noqa: BLE001
        return {}
    tags = {str(k).lower(): str(v) for k, v in ((info.get("format") or {}).get("tags") or {}).items()}
    return {"title": tags.get("title", ""), "artist": tags.get("artist") or tags.get("album_artist", ""),
            "genre": tags.get("genre", ""), "comment": tags.get("comment", "")[:200]}


class ClapEmbedder:
    """The CLAP music model: songs and words in one space. Only when it is already on this Mac (Setup downloads it)."""

    def __init__(self):
        from transformers import ClapModel, ClapProcessor
        import torch
        self.torch = torch
        self.proc = ClapProcessor.from_pretrained(MODEL, local_files_only=True)
        self.model = ClapModel.from_pretrained(MODEL, local_files_only=True).eval()
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.dev)

    @staticmethod
    def available() -> bool:
        try:
            import transformers  # noqa: F401
            from huggingface_hub import try_to_load_from_cache
            return isinstance(try_to_load_from_cache(MODEL, "config.json"), str)
        except Exception:  # noqa: BLE001
            return False

    def _norm(self, v):
        v = v.detach().float().cpu().numpy()
        return v / (((v ** 2).sum(axis=-1, keepdims=True)) ** 0.5 + 1e-9)

    def audio(self, path: Path, start: float, dur: float = 20.0):
        x = _decode(path, sr=48000, start=start, dur=dur)
        with self.torch.no_grad():
            try:                                                           # (newer transformers: audio=, older: audios=)
                inp = self.proc(audio=[x], sampling_rate=48000, return_tensors="pt")
            except TypeError:
                inp = self.proc(audios=[x], sampling_rate=48000, return_tensors="pt")
            inp = {k: v for k, v in inp.items() if k in ("input_features", "is_longer", "attention_mask")}
            inp = {k: v.to(self.dev) for k, v in inp.items()}
            out = self.model.get_audio_features(**inp)
        return self._norm(getattr(out, "pooler_output", out))[0]

    def text(self, texts: list[str]):
        with self.torch.no_grad():
            inp = self.proc(text=texts, return_tensors="pt", padding=True).to(self.dev)
            out = self.model.get_text_features(**inp)
        return self._norm(getattr(out, "pooler_output", out))



def _stem(w: str) -> str:
    """A rough English stem for matching names ("rising" -> "ris", "drums" -> "drum", "calming" -> "calm")."""
    for suf in ("ings", "ing", "ers", "er", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w

class MusicLibrary:
    def __init__(self, cfg, embedder=None):
        self.cfg = cfg
        c = cfg.get("clipper.music_lib") or {}
        self.roots = [Path(os.path.expanduser(str(r))) for r in (c.get("roots") or [])]
        d = Path(os.path.expanduser(str(c.get("dir") or "data/music_lib")))
        self.dir = d if d.is_absolute() else Path(cfg.root) / d
        self._emb = embedder
        self._emb_tried = embedder is not None
        self._idx = None
        self._idx_stamp = None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.status: dict = {"running": False, "stage": "", "done": 0, "total": 0, "error": None}

    # -- where things are ------------------------------------------------------------------
    def enabled(self) -> bool:
        return bool(self.roots)

    def mounted(self) -> list[Path]:
        return [r for r in self.roots if r.is_dir()]

    @staticmethod
    def under(path: str, root) -> bool:
        return str(path).startswith(str(root).rstrip("/") + "/")

    def _file(self, i: str) -> Path:
        return self.dir / "tracks" / f"{i}.json"

    def load(self, i: str) -> dict | None:
        try:
            return json.loads(self._file(i).read_text())
        except (OSError, ValueError):
            return None

    def save(self, rec: dict) -> None:
        self._file(rec["id"]).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self._file(rec["id"]), rec)
        self._idx = None

    def records(self) -> list[dict]:
        d = self.dir / "tracks"
        stamp = (d.stat().st_mtime, len(list(d.iterdir()))) if d.is_dir() else None
        if self._idx is not None and stamp == self._idx_stamp:
            return self._idx
        out = []
        for p in sorted(d.glob("*.json")) if d.is_dir() else []:
            try:
                r = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if not r.get("error") and not r.get("extra"):
                out.append(r)
        self._idx, self._idx_stamp = out, stamp
        return out

    def get(self, i: str) -> dict | None:
        r = self.load(i)
        return r if r and not r.get("error") else None

    def embedder(self):
        if not self._emb_tried:
            self._emb_tried = True
            if ClapEmbedder.available():
                try:
                    self._emb = ClapEmbedder()
                except Exception as e:  # noqa: BLE001 - search falls back to the names
                    log.warning(f"the music model didn't load: {e}")
        return self._emb

    # -- cataloguing ------------------------------------------------------------------------
    def scan(self) -> list[dict]:
        out, seen = [], set()
        for root in self.mounted():
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = sorted(d for d in dirnames if not SKIP.search(d))
                for f in sorted(filenames):
                    p = Path(dirpath) / f
                    if f.startswith(".") or p.suffix.lower() not in AUDIO_EXT:
                        continue
                    try:
                        st = p.stat()
                    except OSError:
                        continue
                    if st.st_size < 20_000:
                        continue
                    i = track_id(p)
                    if i in seen:
                        continue
                    seen.add(i)
                    out.append({"id": i, "path": str(p), "rel": str(p.relative_to(root.parent)), "size": st.st_size, "mtime": int(st.st_mtime)})
        return out

    def _stale(self, f: dict) -> bool:
        r = self.load(f["id"])
        return not r or r.get("size") != f["size"] or r.get("mtime") != f["mtime"] or r.get("v") != VERSION or \
            (bool(r.get("error")) and time.time() - float(r.get("at") or 0) > 86400) or \
            (not r.get("error") and r.get("emb") is None and self.embedder() is not None)

    def _record(self, f: dict) -> dict:
        p = Path(f["path"])
        a = analyse(p)
        tags = _tags(p)
        rec = {**f, **a, "v": VERSION, "at": time.time(), "title": tags.get("title") or p.stem, "artist": tags.get("artist", ""),
               "genre": tags.get("genre", ""), "tags": tags.get("comment", ""),
               "words": " ".join(re.split(r"[\s_./-]+", (f.get("rel") or p.name).lower()) + [tags.get("genre", "").lower(), tags.get("comment", "").lower()]).strip()}
        emb = self.embedder()
        if emb is not None:
            try:
                rec["emb"] = [round(float(v), 5) for v in emb.audio(p, max(0.0, a["best"] - 2), 20.0)]
            except Exception as e:  # noqa: BLE001 - the song is still usable, just not searchable by sound
                log.warning(f"music model on {p.name}: {e}")
                rec["emb"] = None
        else:
            rec["emb"] = None
        return rec

    def index(self, deadline: float | None = None, progress=None) -> dict:
        """Catalogue new and changed songs (resumable: each one is saved as it is done) and forget songs that are gone from a plugged-in
        folder."""
        found = self.scan()
        live = {f["id"] for f in found}
        mounted = self.mounted()
        forgotten = 0
        for r in list(self.records()):
            if r["id"] not in live and any(self.under(r.get("path", ""), m) for m in mounted):
                self._file(r["id"]).unlink(missing_ok=True)
                forgotten += 1
        todo = [f for f in found if self._stale(f)]
        done = failed = 0
        for k, f in enumerate(todo):
            if self.stop.is_set() or (deadline and time.time() > deadline):
                break
            if progress:
                progress(k, len(todo), Path(f["path"]).name)
            try:
                self.save(self._record(f))
                done += 1
            except Exception as e:  # noqa: BLE001 - one bad file never stops the rest
                log.warning(f"music library: {f['path']}: {e}")
                self.save({**f, "error": str(e)[:200], "v": VERSION, "at": time.time()})
                failed += 1
        self._idx = None
        return {"new": done, "failed": failed, "forgotten": forgotten, "left": len(todo) - done - failed, "tracks": len(self.records())}

    def start(self, hours: float | None = None) -> bool:
        if self.thread and self.thread.is_alive():
            return False
        self.stop.clear()
        deadline = time.time() + hours * 3600 if hours else None

        def prog(k, n, what):
            self.status.update(done=k, total=n, stage=f"Listening to {what}")

        def run():
            self.status.update(running=True, error=None, started=time.time(), stage="Looking for songs", done=0, total=0)
            try:
                r = self.index(deadline, progress=prog)
                self.status.update(last=r, stage="Up to date" if not r["left"] else "Stopped")
            except Exception as e:  # noqa: BLE001
                log.exception("music library run failed")
                self.status.update(error=str(e)[:300], stage="Failed")
            finally:
                self.status.update(running=False, finished=time.time())
        self.thread = threading.Thread(target=run, name="music-lib", daemon=True)
        self.thread.start()
        return True

    def stats(self) -> dict:
        recs = self.records()
        return {"tracks": sum(1 for r in recs if any(self.under(r["path"], root) for root in self.roots)),
                "roots": [{"path": str(r), "mounted": r.is_dir(), "tracks": sum(1 for x in recs if self.under(x["path"], r))} for r in self.roots],
                "searchable": any(r.get("emb") for r in recs), "status": dict(self.status)}

    # -- one song, wherever it is (a song dropped onto a project) ------------------------------
    def info(self, path: str | Path) -> dict | None:
        """What the catalogue knows about this song; a song from outside the library is analysed now and kept (no model needed)."""
        p = Path(str(path)).expanduser()
        if not p.is_file():
            return None
        i = track_id(p)
        r = self.load(i)
        st = p.stat()
        if r and not r.get("error") and r.get("size") == st.st_size and r.get("mtime") == int(st.st_mtime) and r.get("v") == VERSION:
            return r
        try:
            rec = {"id": i, "path": str(p), "rel": p.name, "size": st.st_size, "mtime": int(st.st_mtime), **analyse(p), "v": VERSION,
                   "at": time.time(), "title": _tags(p).get("title") or p.stem, "artist": _tags(p).get("artist", ""), "emb": None,
                   "extra": not any(self.under(str(p), root) for root in self.roots)}
        except Exception as e:  # noqa: BLE001
            log.warning(f"couldn't listen to {p.name}: {e}")
            return None
        self.save(rec)
        return rec

    # -- searching and picking ------------------------------------------------------------------
    def _public(self, r: dict, score: float = 0.0) -> dict:
        e = r.get("energy") or []
        k = max(1, -(-len(e) // 120))
        return {"id": r["id"], "path": r["path"], "title": r.get("title") or Path(r["path"]).stem, "artist": r.get("artist", ""),
                "dur": r.get("dur", 0), "bpm": r.get("bpm", 0), "best": r.get("best", 0), "ending": r.get("ending", False),
                "energy": [max(e[i:i + k]) for i in range(0, len(e), k)], "energy_step": round(float(r.get("energy_step", HOP_S)) * k, 3),
                "rel": r.get("rel", ""), "score": round(score, 4)}

    def _shape_score(self, r: dict, words: list[str]) -> float:
        """How well a song's measured energy and tempo fit the mood words (no model)."""
        shapes = [MOOD_SHAPE[w] for w in words if w in MOOD_SHAPE]
        if not shapes:
            return 0.0
        e = float(r.get("avg_energy", 0.5))
        bpm = float(r.get("bpm") or 0)
        s = 0.0
        for want_e, (lo, hi) in shapes:
            s += 1.0 - min(1.0, abs(e - want_e) * 1.6)
            if bpm:
                s += 0.6 if lo <= bpm <= hi else 0.6 - min(0.6, min(abs(bpm - lo), abs(bpm - hi)) / 60)
        return s / len(shapes) / 1.6

    def search(self, text: str, n: int = 40) -> list[dict]:   # noqa: C901
        """Songs that sound like the words, best first (all songs, newest first, with no words)."""
        recs = [r for r in self.records() if any(self.under(r["path"], root) for root in self.roots)]
        text = " ".join((text or "").split())[:200]
        if not text:
            return [self._public(r) for r in sorted(recs, key=lambda r: -float(r.get("mtime") or 0))[:n]]
        words = [w for w in re.split(r"[^a-z0-9]+", text.lower()) if len(w) > 1]
        stems = [_stem(w) for w in words]
        q = None
        emb = self.embedder() if any(r.get("emb") for r in recs) else None
        if emb is not None:
            try:
                q = emb.text([text])[0]
            except Exception as e:  # noqa: BLE001
                log.warning(f"music search without the model: {e}")
        scored = []
        for r in recs:
            hay = r.get("words") or ""
            name = sum(1 for w, st in zip(words, stems) if w in hay or (len(st) > 2 and st in hay)) / max(1, len(words))   # ("rising" finds "Rise")
            if q is not None and r.get("emb"):
                s = float(sum(a * b for a, b in zip(q, r["emb"]))) + 0.08 * name
            else:
                s = 0.6 * name + 0.4 * self._shape_score(r, words)
            scored.append((s, r))
        scored.sort(key=lambda x: -x[0])
        return [self._public(r, s) for s, r in scored[:n]]

    def pick(self, mood_or_text: str = "", length_s: float = 0.0, seed: str = "", exclude: set[str] | None = None) -> Track | None:
        """One song for a video: among the best few for the words (or any, with none), long enough for `length_s` when possible (a shorter
        one loops), not in `exclude` (file names), and the least used lately - so videos don't all get the same song."""
        exclude = exclude or set()
        hits = [h for h in self.search(mood_or_text, n=60) if Path(h["path"]).name not in exclude] or self.search(mood_or_text, n=60)
        if not hits:
            return None
        if length_s:
            long_enough = [h for h in hits if h["dur"] >= length_s * 0.9 or h["dur"] >= 150]
            hits = long_enough or hits
        top = hits[:6] if mood_or_text.strip() else hits
        usage_f = self.dir / "usage.json"
        try:
            usage = json.loads(usage_f.read_text())
        except (OSError, ValueError):
            usage = {}
        low = min(usage.get(h["id"], 0) for h in top)
        pool = [h for h in top if usage.get(h["id"], 0) == low]
        h = random.Random(seed or None).choice(pool)
        usage[h["id"]] = usage.get(h["id"], 0) + 1
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(usage_f, usage)
        except OSError:
            pass
        from .musicsources import credit_for, license_for
        credit = credit_for(h["path"])                                 # a song fetched from Openverse / Epidemic: its credit line comes along
        return Track(path=Path(h["path"]), title=h["title"], artist=h["artist"] or "(your music)",
                     license=(license_for(h["path"]) or "") if credit is not None else "your music library", credit=credit or "")


_LIBS: dict = {}


def for_cfg(cfg) -> MusicLibrary | None:
    """The music library this config points at (shared, so the model loads once), or None when there isn't one."""
    c = cfg.get("clipper.music_lib") or {}
    if not c.get("roots"):
        return None
    key = (str(c.get("dir") or ""), tuple(str(r) for r in c.get("roots") or []), str(getattr(cfg, "root", "")))
    lib = _LIBS.get(key)
    if lib is None:
        lib = _LIBS[key] = MusicLibrary(cfg)
    return lib
