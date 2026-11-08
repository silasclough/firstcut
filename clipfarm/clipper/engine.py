"""Clipper engine: long video -> N vertical, captioned, re-paced short clips.

    ingest      yt-dlp (YouTube / podcast / stream URL) or an uploaded file
    transcribe  word timestamps: local faster-whisper (default) or OpenAI whisper-1 if a key is set
    moments     LLM over the transcript in chunks (OpenAI -> local Ollama -> heuristic), see moments.py
    render      per clip: dead-air cuts (re-pacing), face-tracked 9:16 crop or blurred fit,
                word-by-word captions, hook title card, loudness normalisation; mp4 + jpg + srt + caption

Every step is cached in the job folder, so a re-run (or a restart mid-job) picks up where it left off,
and re-styling a clip only re-renders that clip.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import threading
import time
from pathlib import Path

from ..core import ffmpeg
from ..core.captions import render_captions
from ..core.config import Config
from ..core.llm import LLMRouter
from ..core.log import get_logger
from ..core.state import atomic_write_json
from . import live
from . import moments as M
from . import reframe as R
from .store import Store

log = get_logger("clipper")

CAPTION_STYLES = {
    "bold": {"label": "Bold yellow"},
    "pop": {"label": "Pop green", "highlight_color": "#39FF14", "font_size": 104, "max_words": 2, "max_chars": 14},
    "clean": {"label": "Clean", "uppercase": False, "font_size": 78, "highlight_color": "#FFFFFF",
              "stroke_width": 7, "max_words": 4, "max_chars": 24},
    "fire": {"label": "Red karaoke", "highlight_color": "#FF3B3B", "font_size": 96},
    "sky": {"label": "Sky blue", "highlight_color": "#4FC3FF", "font_size": 92, "max_words": 3},
    "none": {"label": "No captions"},
}
MIX = ["bold", "pop", "clean", "fire", "sky"]
LAYOUTS = {"auto": "Auto (wide: shows more of the scene)", "wide": "Wide window (everyone in frame)",
           "track": "Follow faces (tight)", "fit": "Whole frame + blurred background", "center": "Center crop"}
FRAMING_V = 3         # 1 = tight face-following crop; 2 = wide window / whole frame; 3 = follows whoever is TALKING
PACES = {"natural": {"gap": 0.75, "speed": 1.0}, "tight": {"gap": 0.40, "speed": 1.0},
         "fast": {"gap": 0.35, "speed": 1.08}, "slow": {"gap": 1.1, "speed": 1.0}}       # (slow: the app's "Take it slow": room to breathe)


def _cap_groups(cap_words, ccfg):
    from .polish import caption_groups
    return caption_groups(cap_words, ccfg)


class Cancelled(Exception):
    pass


def default_options(ccfg: dict) -> dict:
    return {"clips": int(ccfg.get("default_clips", 10)), "min_len": float(ccfg.get("min_len", 20)),
            "max_len": float(ccfg.get("max_len", 55)), "layout": "auto", "caption_style": "mix",
            "pace": "tight", "edit_style": "standard", "hook_title": True, "language": ccfg.get("language", "en"), "source_type": "auto",
            "mode": "clips", "music": "", "fcp": False}


# ----------------------------------------------------------------------------- pacing
END_PAD = 0.10        # speech recognizers time the end of a word ~100 ms early: every cut keeps this much extra


def repace(words: list[dict], start: float, end: float, gap_max: float, pad: float = 0.12,
           end_pad: float = END_PAD, style: dict | None = None) -> list[tuple[float, float]]:
    """Keep-intervals (absolute source seconds) inside [start, end] with silences longer than gap_max cut.
    `end_pad` is extra room after the last word of every stretch so the end of the word is never clipped.
    With a `style` (styles_profile.py) a long silence is cut down by a multiplier with a floor instead of to nothing, sentence and idea
    boundaries keep a minimum pause, and gap_max only says which pace the person asked for (speechtrim.py)."""
    if style and (style.get("pace") or {}).get("gap_shrink_over_s") is not None:
        from .speechtrim import repace_style
        return repace_style(words, start, end, style, gap_max, pad, end_pad)
    ws = [w for w in words if w["end"] > start and w["start"] < end]
    if not ws:
        return [(start, end)]
    out = [[max(start, ws[0]["start"] - pad), None]]
    for a, b in zip(ws, ws[1:]):
        if b["start"] - a["end"] > gap_max:
            out[-1][1] = a["end"] + pad + end_pad
            out.append([b["start"] - pad * 0.7, None])
    out[-1][1] = min(end + 0.6 + end_pad, ws[-1]["end"] + pad + end_pad + 0.45)     # let the last word ring out
    merged: list[tuple[float, float]] = []
    for a, b in out:
        if merged and a <= merged[-1][1] + 0.05:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        elif b - a > 0.05:
            merged.append((a, b))
    return merged


def remapper(intervals: list[tuple[float, float]], speed: float = 1.0):
    """Absolute source time -> output time (None if it falls after the end). Intervals normally run forward in time; for an
    edit that reorders them (a sentence moved to the front as the hook) a time maps through the interval that contains it."""
    if any(intervals[i][1] > intervals[i + 1][0] + 0.01 for i in range(len(intervals) - 1)):
        starts, acc = [], 0.0
        for a, b in intervals:
            starts.append(acc)
            acc += b - a

        def g(t: float):
            for (a, b), o in zip(intervals, starts):
                if a - 0.05 <= t <= b + 0.05:
                    return (o + max(0.0, t - a)) / speed
            return None
        return g

    def f(t: float):
        acc = 0.0
        for a, b in intervals:
            if t < a:
                return acc / speed
            if t <= b:
                return (acc + t - a) / speed
            acc += b - a
        return None
    return f


def remap_words(words: list[dict], intervals, speed: float) -> list[dict]:
    f = remapper(intervals, speed)
    out = []
    for w in words:
        if not any(a - 0.05 <= w["start"] and w["end"] <= b + 0.05 for a, b in intervals):
            continue
        s, e = f(w["start"]), f(w["end"])
        if s is None or e is None:
            continue
        out.append({"text": w["text"], "start": round(s, 3), "end": round(max(e, s + 0.05), 3)})
    return out


# ----------------------------------------------------------------------------- engine


class Engine:
    def __init__(self, cfg: Config, store: Store):
        self.cfg = cfg
        self.store = store
        self.ccfg = cfg.get("clipper", {}) or {}
        llm_cfg = dict(cfg.get("llm", {}) or {})
        # a 14B model on an Intel CPU needs ~10 min per transcript chunk; wait for it instead of giving up
        llm_cfg["ollama"] = {**(llm_cfg.get("ollama") or {}),
                             "timeout_s": float(self.ccfg.get("local_timeout_s", 2400))}
        if self.ccfg.get("ollama_model"):
            llm_cfg["ollama"].update({"model": self.ccfg["ollama_model"], "keep_alive": "5m", "fallback_models": []})
        if self.ccfg.get("openai_model"):
            llm_cfg["openai"] = {**(llm_cfg.get("openai") or {}), "model": self.ccfg["openai_model"],
                                 "reasoning_effort": self.ccfg.get("openai_reasoning_effort", "low")}
        self.llm_cfg = llm_cfg
        self.llm = LLMRouter(llm_cfg, chain=list(self.ccfg.get("llm_chain", ["openai", "ollama"])))
        self.models_dir = cfg.root / "data" / "models"

    def router_for_now(self) -> tuple[LLMRouter, str]:
        """Use the M2's Ollama (faster, 32 GB) inside its allowed window when it answers and isn't
        mid-render; otherwise this machine's own Ollama. Returns (router, where)."""
        from ..core import sharing
        r = self.ccfg.get("remote_ollama") or {}
        chain = list(self.ccfg.get("llm_chain", ["openai", "ollama"]))
        where = "local"
        if r.get("base_url"):
            start, end = (r.get("window") or ["17:00", "08:00"])[:2]
            if not sharing.in_window(self.cfg.now(), start, end):
                where = f"local (M2 allowed {start}-{end})"
            elif sharing.m2_busy(self.cfg.root):
                where = "local (M2 busy rendering)"
            else:
                remote = {"base_url": r["base_url"], "model": r.get("model", "qwen3:14b"),
                          "fallback_models": r.get("fallback_models", []), "keep_alive": r.get("keep_alive", "60s"),
                          "timeout_s": r.get("timeout_s", 600), "vision_model": r.get("vision_model"),
                          "num_ctx": r.get("num_ctx")}
                remote = {k: v for k, v in remote.items() if v is not None}
                cfg = {**self.llm_cfg, "ollama_remote": remote}
                # inside the window the M2 goes first (free); OpenAI, then this machine, are the backups
                router = LLMRouter(cfg, chain=["ollama_remote"] + [n for n in chain if n != "ollama_remote"])
                ok, why = dict((n, (o, w)) for n, o, w in router.status()).get("ollama_remote", (False, ""))
                if ok:
                    return router, "M2"
                where = f"local (M2 unreachable: {why})"
        # name the machine/service that will actually answer first
        first = next((n for n, ok, _ in self.llm.status() if ok), None)
        label = {"openai": "OpenAI", "ollama": "local Ollama"}.get(first, "keyword fallback")
        reason = where.partition(" (")[2].rstrip(")")
        return self.llm, f"{label} ({reason})" if reason else label

    def creative_router(self):
        """The AI for writing that needs the best model (scripts, commentary, edits): the posting server's OpenAI over SSH
        when configured and reachable, else whatever router_for_now() gives."""
        from ..core.llm import Layered, RemoteJson
        local, _where = self.router_for_now()
        from .auto import _target
        tgt = _target(self.cfg)                                  # the posting server: clipper.auto.push_host, else handoff.target (the M2 has only that)
        host = tgt[0] if tgt else None
        if host and self.cfg.get("clipper.remote_llm.enabled", True) and not os.environ.get("OPENAI_API_KEY"):
            tasks = set(self.cfg.get("clipper.remote_llm.tasks", ["original_script", "original_critic", "clip_commentary", "clip_edit",
                                                                    "asis_text", "story_edit", "newsreel_script", "newsreel_critic"]))
            return Layered(RemoteJson(host, timeout_s=float(self.cfg.get("clipper.remote_llm.timeout_s", 300))), local, tasks)
        return local

    # --------------------------------------------------------------- status helpers
    def status(self) -> dict:
        router, where = self.router_for_now()
        llm = [{"name": "M2 ollama" if n == "ollama_remote" else n, "ok": ok, "why": why}
               for n, ok, why in router.status()]
        return {"llm": llm, "ai_on": where, "openai_key": bool(os.environ.get("OPENAI_API_KEY")),
                "transcriber": self._transcriber(), "whisper_model": self.ccfg.get("whisper_model", "base.en"),
                "caption_styles": {k: v["label"] for k, v in CAPTION_STYLES.items()}, "layouts": LAYOUTS,
                "paces": list(PACES)}

    def _transcriber(self) -> str:
        want = self.ccfg.get("transcriber", "auto")
        if want == "openai" or (want == "auto" and os.environ.get("OPENAI_API_KEY") and self.ccfg.get("openai_whisper", False)):
            return "openai"
        return "local"

    def _progress(self, job_id: str, status: str, stage: str, frac: float) -> None:
        job = self.store.get(job_id)
        if job.get("cancel"):
            raise Cancelled()
        self.store.update(job_id, status=status, stage=stage, progress=round(max(0.0, min(1.0, frac)), 4))
        live.stage(self.store, job_id, stage, status)                       # the page's activity log (live.py)

    # --------------------------------------------------------------- pipeline
    def process(self, job_id: str) -> dict:
        d = self.store.dir(job_id)
        try:
            src = self._ingest(job_id, d)
            try:                                                     # the live view: a strip of the video while it is transcribed
                dur = ffmpeg.duration(src)
                if not (self.store.get(job_id).get("source") or {}).get("url"):
                    live.emit(self.store, job_id, "source", title=self.store.get(job_id).get("title"), duration=round(dur, 1))
                live.filmstrip(self.store, job_id, src, dur, of=self.store.get(job_id).get("title") or "the video")
            except Exception:  # noqa: BLE001
                pass
            profile = self._source_profile(job_id, src)
            if self.store.get(job_id)["options"].get("mode") not in ("asis", "polish", "long"):
                self._refuse_still_source(job_id, src)
            words = self._transcribe(job_id, d, src)
            if self.store.get(job_id)["options"].get("mode") == "asis":
                return self._as_is(job_id, d, src, words)
            if self.store.get(job_id)["options"].get("mode") == "recap":      # a long video -> a few fast vertical recap reels (recap.py)
                from . import recap
                recap.run(self, job_id, d, src, words)
                return self.store.get(job_id)
            if self.store.get(job_id)["options"].get("mode") in ("polish", "long"):
                from . import polish
                try:
                    clip = polish.run(self, job_id, src, words)      # whole video: captions + music + trims
                except ffmpeg.FFmpegError as e:                      # a stalled picture render: once more without the frosted-glass blur, then without graphics
                    log.warning(f"the final render failed ({str(e)[:160]}); trying again without the glass blur")
                    self.store.update(job_id, options={**self.store.get(job_id)["options"], "no_glass": True})
                    try:
                        clip = polish.run(self, job_id, src, words)
                    except ffmpeg.FFmpegError as e2:
                        log.warning(f"still failing ({str(e2)[:160]}); trying again without the on-screen graphics")
                        self.store.update(job_id, options={**self.store.get(job_id)["options"], "no_glass": True, "gfx_cards": [], "graphics": "title"})
                        clip = polish.run(self, job_id, src, words)
                self.store.update(job_id, status="done", progress=1.0,
                                  stage=f"Done - {clip['trimmed_s']:.0f}s of dead space cut"
                                        + (", Final Cut project ready" if clip.get("fcp") else ""))
                self._archive(job_id)
                try:
                    polish.archive_bundle(self.cfg, self.store, job_id)
                except Exception as e:  # noqa: BLE001
                    log.warning(f"archiving the FCP project failed: {e}")
                return self.store.get(job_id)
            moments = self._moments(job_id, d, src, words, profile)
            if not moments and int(self.store.get(job_id)["options"].get("clips", 10)) <= 0:
                self.store.update(job_id, status="done", progress=1.0, stage="Nothing in this video was good enough to clip")
                return self.store.get(job_id)
            usable = ffmpeg.readable_seconds(src)
            claimed = ffmpeg.duration(src)
            if usable < claimed - 3:                         # the file is cut off part-way (a damaged / unfinished export)
                info = self.store.get(job_id).get("info") or {}
                self.store.update(job_id, info={**info, "damaged": {"usable_s": round(usable, 1), "claimed_s": round(claimed, 1)}})
                moments = [m for m in moments if m.end <= usable + 0.2]
                if not moments:
                    raise RuntimeError(f"this video file is damaged: its header says {claimed:.0f} s but only the first "
                                       f"{usable:.0f} s can be read (an interrupted export or upload). Please export / upload it again.")
                log.warning(f"{job_id}: only the first {usable:.0f} of {claimed:.0f} s of the video can be read")
            if not moments:
                raise RuntimeError("no usable moments found" + ("" if profile == "glasses" else
                                                                 " (is there speech in this video?)"))
            job = self.store.get(job_id)
            opts = job["options"]
            made, skipped = 0, []
            for i, m in enumerate(moments):
                self._progress(job_id, "rendering", f"Rendering clip {i + 1} of {len(moments)}",
                               0.62 + 0.38 * i / len(moments))
                live.emit(self.store, job_id, "render", index=i + 1, of=len(moments), title=m.title, start=round(m.start, 1), end=round(m.end, 1))
                live.later(live.shot, self.store, job_id, src, (m.start + m.end) / 2, of=f"clip {i + 1}", what=m.title)
                style = opts.get("caption_style", "mix")
                if style == "mix":
                    style = self._caption_look(job, m, words, i)
                try:
                    self.render_clip(job_id, i + 1, m, words, src, layout=opts.get("layout", "auto"), style=style,
                                     pace=opts.get("pace", "tight"), hook_title=bool(opts.get("hook_title", True)),
                                     edit_style=("auto" if (job.get("source") or {}).get("auto") and opts.get("edit_style", "standard") == "standard" else opts.get("edit_style", "standard")),
                                     profile=profile, music=opts.get("music") or "",
                                     commentary=bool((job.get("source") or {}).get("auto")
                                                     and self.cfg.get("clipper.commentary.enabled", True)))
                    made += 1
                    c = next((c for c in self.store.get(job_id).get("clips") or [] if c.get("index") == i + 1), None)
                    if c:                                                   # the live view: the finished clip
                        live.emit(self.store, job_id, "clip", index=i + 1, title=c.get("title") or m.title, thumb=c.get("thumb"),
                                  duration=c.get("duration"), v=c.get("rendered_at"))
                except Cancelled:
                    raise
                except Exception as e:  # noqa: BLE001 - one bad clip must not sink the others
                    log.warning(f"clip {i + 1} of {job_id} failed and was skipped: {e}")
                    skipped.append(i + 1)
                    d_ = self.store.dir(job_id)
                    for junk in d_.glob(f"clip_{i + 1:02d}*"):
                        shutil.rmtree(junk, ignore_errors=True) if junk.is_dir() else junk.unlink(missing_ok=True)
            if not made:
                raise RuntimeError(f"none of the {len(moments)} clips could be rendered")
            self.store.update(job_id, status="done", progress=1.0,
                              stage=f"{made} clips ready" + (f" ({len(skipped)} skipped after errors)" if skipped else ""))
            self._archive(job_id)
            return self.store.get(job_id)
        except Cancelled:
            return self.store.update(job_id, status="cancelled", stage="Cancelled")
        except Exception as e:  # noqa: BLE001 - surfaced in the GUI
            log.exception(f"clipper job {job_id} failed")
            return self.store.update(job_id, status="failed", stage="Failed", error=f"{type(e).__name__}: {e}"[:800])

    # --------------------------------------------------------------- ingest
    def _ingest(self, job_id: str, d: Path) -> Path:
        existing = next((p for p in d.glob("source.*") if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov", ".m4v")),
                        None)
        if existing:
            return existing
        parts_dir = d / "parts"
        parts = sorted(p for p in parts_dir.glob("part_*") if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov", ".m4v")) \
            if parts_dir.exists() else []
        if parts:                                           # several uploads -> one timeline
            from .combine import combine
            self._progress(job_id, "joining", f"Joining {len(parts)} videos" if len(parts) > 1 else "Preparing the video", 0.02)
            meta = combine(parts, d / "source.mp4",
                           progress=lambda msg: self._progress(job_id, "joining", msg, 0.05))
            shutil.rmtree(parts_dir, ignore_errors=True)
            info = self.store.get(job_id).get("info") or {}
            self.store.update(job_id, info={**info, "files": meta["files"], "join_mode": meta["mode"]})
            return d / "source.mp4"
        job = self.store.get(job_id)
        url = job["source"].get("url")
        if not url:
            raise RuntimeError("no source video (upload missing)")
        self._progress(job_id, "downloading", "Downloading video", 0.0)
        import yt_dlp

        max_min = float(self.ccfg.get("max_source_minutes", 240))

        seen = {"source": False, "t": 0.0}

        def hook(h):
            if h.get("status") == "downloading":
                tot = h.get("total_bytes") or h.get("total_bytes_estimate") or 0
                try:                                                     # the live view: what is coming down, and how fast
                    inf = h.get("info_dict") or {}
                    if not seen["source"] and inf:
                        seen["source"] = True
                        live.emit(self.store, job_id, "source", title=inf.get("title"), uploader=inf.get("uploader") or inf.get("channel"),
                                  duration=inf.get("duration"), thumb=inf.get("thumbnail"), site=inf.get("extractor_key"))
                    if time.time() - seen["t"] > 0.8:
                        seen["t"] = time.time()
                        live.emit(self.store, job_id, "download", got=h.get("downloaded_bytes") or 0, total=tot, speed=h.get("speed"),
                                  eta=h.get("eta"))
                except Exception:  # noqa: BLE001
                    pass
                if tot:
                    self._progress(job_id, "downloading", f"Downloading video ({h['downloaded_bytes'] / tot:.0%})",
                                   0.15 * h["downloaded_bytes"] / tot)

        opts = {"outtmpl": str(d / "source.%(ext)s"), "format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]/b",
                "merge_output_format": "mp4", "quiet": True, "no_warnings": True, "noplaylist": True,
                "progress_hooks": [hook], "retries": 5, "fragment_retries": 5,
                "match_filter": lambda info, *, incomplete=False: (
                    f"longer than {max_min:.0f} min" if (info.get("duration") or 0) > max_min * 60 else None)}
        cookies = self.ccfg.get("cookies_file")                  # optional: a cookies.txt export, if YouTube blocks downloads
        if cookies and Path(cookies).expanduser().is_file():
            opts["cookiefile"] = str(Path(cookies).expanduser())
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
        if info and info.get("entries"):
            info = info["entries"][0]
        meta = {"title": (info or {}).get("title"), "uploader": (info or {}).get("uploader"),
                "duration": (info or {}).get("duration"), "webpage_url": (info or {}).get("webpage_url") or url}
        src = next((p for p in d.glob("source.*") if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov")), None)
        if not src:
            raise RuntimeError(f"download produced no video (longer than {max_min:.0f} min, or not a video URL?)")
        self.store.update(job_id, title=meta["title"] or job["title"], info=meta)
        return src

    # --------------------------------------------------------------- source type
    def _source_profile(self, job_id: str, src: Path) -> str:
        """'glasses' when the user picked Meta glasses, or (on auto) when the file's metadata says so."""
        from . import glasses as G
        job = self.store.get(job_id)
        want = job["options"].get("source_type", "auto")
        files = (job.get("info") or {}).get("files") or []
        if len(files) > 1:                                  # joined uploads: judge by the original files
            hits = [f for f in files if f.get("glasses")]
            det = {"glasses": bool(hits), "shape_match": all(f.get("shape_match") for f in files),
                   "dims": files[0].get("dims"),
                   "evidence": f"{hits[0]['evidence']} in {len(hits)} of {len(files)} files" if hits else None}
        else:
            try:
                det = G.detect(src)
            except Exception as e:  # noqa: BLE001
                det = {"glasses": False, "evidence": None, "shape_match": False, "error": str(e)[:200]}
        profile = "glasses" if want == "glasses" or (want == "auto" and det["glasses"]) else "standard"
        how = ("selected" if want == "glasses" else f"detected from metadata ({det['evidence']})") \
            if profile == "glasses" else ("selected regular video" if want == "standard" else "regular video")
        info = self.store.get(job_id).get("info") or {}
        self.store.update(job_id, info={**info, "profile": profile, "profile_how": how, "device": det})
        return profile

    # --------------------------------------------------------------- transcribe
    def _transcribe(self, job_id: str, d: Path, src: Path) -> list[dict]:
        cache = d / "transcript.json"
        if cache.exists():
            return json.loads(cache.read_text())["words"]
        total = ffmpeg.duration(src)
        info = self.store.get(job_id).get("info") or {}
        self.store.update(job_id, info={**info, "duration": round(total, 1)})
        lang = (self.store.get(job_id)["options"].get("language") or "en").lower()
        words: list[dict] = []
        backend = self._transcriber()
        if (self.store.get(job_id)["options"].get("project") or self.ccfg.get("parakeet_for_clips", False)) \
                and lang in ("en", "auto"):                   # AI-video projects (and clips, if enabled): Parakeet, sharper word times
            from . import parakeet
            if parakeet.binary(self.cfg):
                try:
                    self._progress(job_id, "transcribing", "Transcribing with Parakeet", 0.2)
                    words = parakeet.transcribe(src, self.cfg)
                    backend = "parakeet"
                except Exception as e:  # noqa: BLE001 - Whisper below still works
                    log.warning(f"parakeet failed ({e}); using Whisper")
                    words = []
        if backend == "parakeet" and words:
            pass
        elif backend == "openai":
            from .openai_audio import transcribe_openai
            words = transcribe_openai(src, total, lang, lambda f: self._progress(
                job_id, "transcribing", f"Transcribing with OpenAI ({f:.0%})", 0.15 + 0.35 * f))
        else:
            backend = "local" if backend == "parakeet" else backend
            from ..core import whisper
            wcfg = {"backend": "auto", "model": self.ccfg.get("whisper_model", "base.en"),
                    "compute_type": self.ccfg.get("whisper_compute", "int8")}
            if lang != "en" and wcfg["model"].endswith(".en"):
                wcfg["model"] = wcfg["model"][:-3]
            step = float(self.ccfg.get("transcribe_chunk_s", 300))
            n = max(1, math.ceil(total / step))
            for k in range(n):
                self._progress(job_id, "transcribing", f"Transcribing ({k / n:.0%})", 0.15 + 0.35 * k / n)
                a, b = k * step, min(total, (k + 1) * step)
                tr = whisper.transcribe(src, wcfg, a, b, language=None if lang == "auto" else lang)
                part = [w for w in whisper.all_words(tr) if w["start"] >= a - 0.01]
                words += part
                live.words(self.store, job_id, part, of=f"part {k + 1} of {n}")
        if backend in ("parakeet", "openai"):                                # (Whisper's parts went to the live view as they came)
            live.words(self.store, job_id, words, of=self.store.get(job_id).get("title") or "the video")
        atomic_write_json(cache, {"backend": backend, "duration": total, "words": words})
        return words

    # --------------------------------------------------------------- moments
    def _refuse_still_source(self, job_id: str, src: Path) -> None:
        """A "video" that is one picture (a story read over cover art) only makes clips of a still with captions: those got 5-20 views. Say so now, before the transcription."""
        if not self.cfg.get("clipper.source_check.still", True):
            return
        from . import sourcecheck as SC
        r = SC.probe_source(src)
        info = self.store.get(job_id).get("info") or {}
        self.store.update(job_id, info={**info, "still_probe": r})
        if r["still"]:
            raise RuntimeError(f"this video is one still picture: {r['still_windows']} of the {r['windows']} places looked at do not move. "
                               "A clip of it would be a picture with captions, which viewers swipe away from. Skipped.")

    def _screen_stills(self, job_id: str, src: Path, found: list) -> list:
        """Before moments are picked: a candidate whose stretch of the source does not move is dropped, so the pick falls on the next best moment instead of a still."""
        if not self.cfg.get("clipper.source_check.still", True) or not found:
            return found
        from . import sourcecheck as SC
        kept, dropped = SC.screen_moments(src, found, int(self.cfg.get("clipper.source_check.screen_limit", 30)))
        for m, why in dropped:
            log.info(f"{job_id}: moment '{m.title}' {m.start:.0f}-{m.end:.0f} s dropped before rendering: {why}")
            live.emit(self.store, job_id, "dropped", title=m.title, start=round(m.start, 1), end=round(m.end, 1), why=why)
        if not kept:
            raise RuntimeError("every moment found is a still picture (nothing on screen moves): skipped")
        return kept

    def _strip_plan(self, d: Path, index: int, src: Path, span0: float, span1: float, words: list[dict]) -> dict:
        """What to cut away from this stretch of the source (see sourcecheck.crop_plan). Read once per stretch and kept next to the clip; any trouble means "nothing"."""
        from . import sourcecheck as SC
        key = f"{span0:.0f}-{span1:.0f}"
        cache = d / f"clip_{index:02d}_srccheck.json"
        try:
            c = json.loads(cache.read_text())
            if c.get("key") == key:
                return c["plan"]
        except Exception:  # noqa: BLE001 - no cache yet
            pass
        try:
            found = SC.burned_in(src, span0, span1, " ".join(w["text"] for w in words if span0 - 1 <= w["start"] <= span1 + 1))
        except Exception as e:  # noqa: BLE001 - a check is never a reason to lose a clip
            log.debug(f"source text check failed: {e}")
            found = None
        plan = SC.crop_plan(found)
        if found is not None:
            try:
                atomic_write_json(cache, {"key": key, "found": found, "plan": plan})
            except Exception:  # noqa: BLE001
                pass
        return plan

    def _moments(self, job_id: str, d: Path, src: Path, words: list[dict], profile: str = "standard") -> list[M.Moment]:
        job = self.store.get(job_id)
        opts = job["options"]
        key = hashlib.sha1(json.dumps([opts.get("clips"), opts.get("min_len"), opts.get("max_len"), profile,
                                       opts.get("prompt", ""), M.RUBRIC_V]).encode()).hexdigest()[:10]
        brief = (opts.get("prompt") or "").strip()
        cache = d / "moments.json"
        if cache.exists():
            c = json.loads(cache.read_text())
            if c.get("key") == key:
                return [M.Moment(**m) for m in c["moments"]]
        if (job.get("source") or {}).get("kind") == "twitch_clip":          # a Twitch clip is already the moment: keep all of it (a breath before, half a second after)
            picked = self._whole_clip_moment(job, words, src)
            atomic_write_json(cache, {"key": key, "moments": [m.to_dict() for m in picked]})
            return picked
        if profile == "glasses":
            picked = self._glasses_moments(job_id, src, words)
            atomic_write_json(cache, {"key": key, "moments": [m.to_dict() for m in picked]})
            return picked
        self._progress(job_id, "finding", "Finding the best moments", 0.5)
        sents = M.sentences_from_words(words)
        if not sents:
            return []
        n, lo, hi = int(opts.get("clips", 10)), float(opts.get("min_len", 20)), float(opts.get("max_len", 55))
        adaptive = n <= 0                               # 0 = as many GOOD moments as the video really has (never padded)
        cap = int(self.ccfg.get("max_clips_per_video", 25))
        blocks = len(M.chunks(sents))
        per_chunk = 3 if adaptive else max(2, min(4, math.ceil(n * 1.5 / blocks) + 1))
        if adaptive:
            n = cap
        from .combine import boundary_note
        context = (f"Video: {job.get('title') or ''}\n" +
                   boundary_note((job.get("info") or {}).get("files") or [])).strip()
        if brief:
            context += (f"\nTHE USER'S BRIEF - what they are looking for in this video (weigh this heavily when choosing "
                        f"moments, titles and captions): {brief}")
        router, where = self.router_for_now()
        info = self.store.get(job_id).get("info") or {}
        self.store.update(job_id, info={**info, "ai_on": where})
        found = M.llm_moments(router, sents, per_chunk, lo, hi, context, progress=lambda i, t: self._progress(
            job_id, "finding", f"Finding the best moments on {where.split(' (')[0]} ({i}/{t})", 0.5 + 0.12 * i / t))
        for fm in sorted(found, key=lambda m: m.start)[:60]:                # the live view: every candidate the AI found
            live.emit(self.store, job_id, "moment", title=fm.title, hook=getattr(fm, "hook", "") or "", start=round(fm.start, 1),
                      end=round(fm.end, 1), score=round(float(fm.score), 1),
                      quote=" ".join(x.text for x in sents[fm.first:fm.last + 1])[:260])
        found = self._screen_stills(job_id, src, found)
        src_info = job.get("source") or {}
        sb = float(self.ccfg.get("story_bias", 0.6))
        if sb > 0:                                               # clips that tell something that HAPPENED did best on the channel: a push, not a filter
            from .discover import story_score
            for fm in found:
                said = " ".join(s.text for s in sents[fm.first:fm.last + 1])
                fm.score = round(max(0.0, min(10.0, fm.score + 1.5 * sb * story_score(fm.title, fm.hook, said))), 2)
        if src_info.get("auto"):                                 # nudge the AI's scores toward what you tend to like
            from .taste import Taste
            tst = Taste(self.cfg.path("state"))
            for fm in found:
                said = " ".join(s.text for s in sents[fm.first:fm.last + 1])
                fm.score = round(max(0.0, min(10.0, fm.score + tst.moment_bonus(
                    fm.title, getattr(fm, "hook", "") or "", fm.end - fm.start, fm.score, src_info, said))), 2)
        picked = M.pick(found, n)
        if adaptive:                                    # only moments that clear the bar - a short or dull video gives few
            bar = float(self.ccfg.get("adaptive_min_score", 6.5))
            picked = [m for m in picked if m.score >= bar]
        if src_info.get("auto") and not adaptive:
            # Quality over quantity: consider half as many again, then render only the best n - dropping the ones the
            # model is sure you'd delete (apart from a small share kept so it can still learn it was wrong).
            import random
            pool = M.pick(found, math.ceil(n * 1.5))
            keep = []
            for fm in pool:
                said = " ".join(s.text for s in sents[fm.first:fm.last + 1])
                p = tst.moment_prob(fm.title, getattr(fm, "hook", "") or "", fm.end - fm.start, fm.score, src_info, said)
                if p is not None and p < float(self.ccfg.get("min_predicted", 0.12)) and random.random() > 0.15:
                    continue
                keep.append(fm)
            picked = keep[:n] if len(keep) >= max(1, n // 2) else picked
        if adaptive and src_info.get("auto"):
            import random
            keep = []
            for fm in picked:
                said = " ".join(s.text for s in sents[fm.first:fm.last + 1])
                p = tst.moment_prob(fm.title, getattr(fm, "hook", "") or "", fm.end - fm.start, fm.score, src_info, said)
                if p is not None and p < float(self.ccfg.get("min_predicted", 0.12)) and random.random() > 0.15:
                    continue                            # it's confident you'd delete this one - don't spend a render on it
                keep.append(fm)
            picked = keep
        if len(picked) < n and not adaptive:                 # top up from the heuristic
            from ..core.whisper import load_audio
            loud = M.loudness_per_second(load_audio(src, sr=4000), 4000)
            picked = M.top_up(picked, M.heuristic_moments(sents, loud, lo, hi), n)
        self._progress(job_id, "finding", "Checking that every clip ends on a complete thought", 0.62)
        picked = M.complete_endings(M.llm_fix_endings(router, picked, sents, hi), sents, hi)
        if self.ccfg.get("hook_first_starts", True):                   # a clip opens on a hook, says something, and is titled truthfully
            self._progress(job_id, "finding", "Checking that every clip starts on a hook", 0.64)
            picked = M.llm_fix_starts(router, picked, sents, lo)
            picked = M.fix_starts_heuristic(picked, sents, lo)
        picked = M.retitle_weak(router, picked, sents)                   # one more ask for a title that states the claim, before the score is lowered
        picked = M.demote_weak_titles(picked)
        picked = M.llm_rehooks(router, picked, sents)                   # long clips: one marked sentence where the stake is restated (a marker for punch-ins)
        live.emit(self.store, job_id, "pick", moments=[{"title": m.title, "start": round(m.start, 1), "end": round(m.end, 1),
                                                        "score": round(float(m.score), 1)} for m in picked])
        atomic_write_json(cache, {"key": key, "moments": [m.to_dict() for m in picked]})
        return picked

    def _hook_card(self, text: str, png: Path, W: int, H: int) -> None:
        """Draw the hook title of a clip into `png`. The one place the card is made (WP11 swaps it for the themed, face-aware card)."""
        from . import placement as PL
        sb = PL.safe_box(W, H)                                                   # the card fits inside what the platforms leave free (no buttons over it, no cut-off sides)
        title_card(text, png, int((sb["x1"] - sb["x0"]) / 0.80), self.cfg.get("captions", {}).get("fonts", []))

    @staticmethod
    def _hook_xy(W: int, H: int) -> tuple[str, int]:
        """-> (ffmpeg x expression, y pixels) of the hook title: centred on the safe box, its top just inside the safe box."""
        from . import placement as PL
        sb = PL.safe_box(W, H)
        return f"{(sb['x0'] + sb['x1']) // 2}-w/2", max(int(H * 0.11), int(sb["y0"] + 0.008 * H))

    def _caption_look(self, job: dict, m: M.Moment, words: list[dict], i: int) -> str:
        """The caption look for clip i of a batch on 'mix': karaoke looks (one to three words) for teaching, phrase looks (three to five) for stories,
        emotion, comedy and interviews; streamer and podcast sources are always phrase. Without a reason to choose, the old rotation."""
        if not self.ccfg.get("caption_by_mood", True):
            return MIX[i % len(MIX)]
        from . import clipcraft as CC
        said = " ".join(w["text"] for w in words if m.start <= float(w["start"]) < m.end)
        src = job.get("source") or {}
        ans = None
        if str(src.get("kind") or "") not in CC.PHRASE_SOURCES and self.ccfg.get("caption_mood_ai", True):
            try:
                from . import editstyle
                ans = editstyle.caption_mood(self.router_for_now()[0], src, m.title, said)
            except Exception as e:  # noqa: BLE001 - the keywords below decide
                log.debug(f"caption mood: {e}")
        return CC.caption_style_for(CC.caption_kind(src, said, ans), i)

    def _whole_clip_moment(self, job: dict, words: list[dict], src: Path) -> list[M.Moment]:
        """Not part of FirstCut."""
        return []

    def _glasses_moments(self, job_id: str, src: Path, words: list[dict]) -> list[M.Moment]:
        """POV footage: a vision model looks at timestamped contact sheets (+ speech + signals)."""
        from ..core.whisper import load_audio
        from . import glasses as G
        job = self.store.get(job_id)
        opts = job["options"]
        gcfg = self.ccfg.get("glasses", {}) or {}
        duration = ffmpeg.duration(src)
        lo = min(float(opts.get("min_len", 20)), float(gcfg.get("min_len", 8)))
        hi = min(float(opts.get("max_len", 55)), float(gcfg.get("max_len", 40)))
        want = int(opts.get("clips", 10))
        room = max(1, math.ceil(duration / float(gcfg.get("seconds_per_clip", 30))))
        n = room if want <= 0 else min(want, room)                       # 0 = as many as the footage supports
        brief = (opts.get("prompt") or "").strip()
        self._progress(job_id, "finding", "Looking at the footage (motion, blur, sound)", 0.5)
        sig = G.visual_signals(src)
        loud = M.loudness_per_second(load_audio(src, sr=4000), 4000)
        router, where = self.vision_router()
        info = self.store.get(job_id).get("info") or {}
        self.store.update(job_id, info={**info, "ai_on": where})
        found = G.vision_moments(router, src, duration, words, sig, loud, n, lo, hi, (job.get("title") or "") + (f"\nUser's brief: {brief}" if brief else ""),
                                 files=(job.get("info") or {}).get("files") or [],
                                 progress=lambda i, t: self._progress(
                                     job_id, "finding", f"Watching the footage with {where.split(' (')[0]} ({i}/{t})",
                                     0.53 + 0.09 * i / t))
        picked = M.pick(found, n)
        if not picked:            # vision unavailable/failed: fall back. If it ran and chose fewer, trust it.
            picked = M.pick(G.heuristic_moments(sig, loud, words, duration, lo, hi), max(1, min(n, 3)))
        return picked

    def vision_router(self) -> tuple[LLMRouter, str]:
        """Vision needs an image-capable model: OpenAI, or an Ollama vision model if configured
        (clipper.remote_ollama.vision_model on the M2 at night / llm.ollama.vision_model)."""
        router, where = self.router_for_now()
        chain = [n for n in router.chain
                 if n == "openai" or (router.cfg.get(n) or {}).get("vision_model")]
        if not chain:
            return LLMRouter(router.cfg, chain=[]), "keyword fallback (no vision model)"
        r = LLMRouter(router.cfg, chain=chain)
        first = next((n for n, ok, _ in r.status() if ok), None)
        label = {"openai": "OpenAI", "ollama_remote": "M2", "ollama": "local Ollama"}.get(first, "keyword fallback")
        return r, label

    # --------------------------------------------------------------- render
    def _commentary_lines(self, job_id: str, index: int, m: M.Moment, words: list[dict], intervals, speed: float,
                          work: Path) -> list[dict]:
        return self._commentary(job_id, index, m, words, intervals, speed, work)[0]

    def _commentary(self, job_id: str, index: int, m: M.Moment, words: list[dict], intervals, speed: float,
                    work: Path) -> tuple[list[dict], list[tuple[float, float]]]:
        """Write + voice ONE plain line for this cut, in a real silence of the clip. The silence is kept open (a stretch the pacing would have
        trimmed is left in) so the line never talks over anything the clip says, and a line never goes near the end (a new sentence there
        would be cut off). -> ([{"at": output seconds, "wav": path, "dur": seconds, "text": ...}], the intervals with the silence opened)."""
        from ..core import voice
        from . import commentary as CM
        ccfg = self.cfg.get("clipper.commentary", {}) or {}
        if not ccfg.get("enabled", True):
            return [], list(intervals)
        mw = [w for w in words if m.start - 0.05 <= w["start"] and w["end"] <= m.end + 0.05]
        sents = M.sentences_from_words(mw)
        f0 = remapper(intervals, speed)
        total0 = sum(b - a for a, b in intervals) / speed
        starts = sorted(w["start"] for w in words)

        def next_start(e: float):
            return next((x for x in starts if x > e + 0.02), None)
        room: dict[int, float] = {}                                       # how long a silence can be held open after each sentence
        for _i, _e, _g in CM.pauses(sents):
            at = f0(_e)
            nxt = next_start(_e)
            if at is None or nxt is None or at > total0 - 3.0:
                continue
            room[_i] = min(nxt - _e - 0.3, 3.6)
        room = {k: v for k, v in room.items() if v >= 1.4}
        router = self.creative_router()
        lines = CM.plan(router, sents, m.title, int(ccfg.get("max_lines", 1)), room)
        lines = [l for l in lines if l.idx in room]
        if not lines:
            return [], list(intervals)
        try:
            wavs = voice.speak_checked(self.cfg, [l.text for l in lines], work / "commentary", role="commentator")
        except Exception as e:  # noqa: BLE001 - a clip without commentary is still a good clip
            log.warning(f"commentary voice unavailable for {job_id}#{index}: {e}")
            return [], list(intervals)
        iv = [list(x) for x in intervals]
        kept = []
        for l, w in zip(lines, wavs):
            dur = ffmpeg.duration(w)
            need = dur + 0.45
            e = l.source_time
            k = next((i for i, (a, b) in enumerate(iv) if a - 0.01 <= e <= b + 0.05), None)
            nxt = next_start(e)
            if k is None or nxt is None:
                continue
            a, b = iv[k]
            if b - e < need:                                              # open the silence up (it is only silence: nothing is lost)
                new_b = min(e + need, nxt - 0.15)
                if new_b - e < need - 0.01:
                    continue
                iv[k][1] = new_b
            kept.append((l, w, dur))
        new_iv = [(a, b) for a, b in iv]
        f = remapper(new_iv, speed)
        total = sum(b - a for a, b in new_iv) / speed
        out = []
        for l, w, dur in kept:
            at = f(l.source_time)
            if at is None or at + dur > total - 0.8:
                continue                                                  # too near the end: it would be cut off
            out.append({"at": float(at), "wav": str(w), "dur": dur, "text": l.text})
        return out, new_iv

    def clean_voice(self, d: Path, src: Path, seek: float, dur: float, inputs: list[str], graph: list[str]) -> tuple[list[str], list[str]]:
        """A short clip's speech, cleaned when its sound is weak or bad (soundcheck, cached in the job folder) or the setting is Always:
        the cleaned copy becomes one more input and the clip's sound ([0:a]) is read from it instead. Untouched otherwise."""
        try:
            from . import soundcheck, voiceclean as VC
            if not VC.available(self.cfg) or VC.mode("short", self.cfg) == "never":
                return inputs, graph
            gf = d / "sound_grades.json"
            try:
                g = soundcheck.Grades(json.loads(gf.read_text()))
            except (OSError, ValueError):
                g = soundcheck.Grades({})
            grade = g.span(str(src), seek, seek + dur)["grade"]
            if g.dirty:
                gf.write_text(json.dumps(g.data))
            if not VC.wanted(grade, "short", self.cfg):
                return inputs, graph
            wav, start = VC.cleaned_range(src, seek, seek + dur, d / "clean", cfg=self.cfg)
            k = inputs.count("-i")
            inputs = inputs + ["-ss", f"{max(0.0, seek - start):.3f}", "-i", str(wav)]
            return inputs, [x.replace("[0:a]", f"[{k}:a]") for x in graph]
        except Exception as e:  # noqa: BLE001 - the original sound is used instead
            log.warning(f"voice cleanup skipped for {Path(src).name}: {e}")
            return inputs, graph

    def render_clip(self, job_id: str, index: int, m: M.Moment, words: list[dict], src: Path, layout: str = "auto",
                    style: str = "bold", pace: str = "tight", hook_title: bool = True, profile: str = "standard",
                    music: str = "", segments: list[tuple[float, float]] | None = None, commentary: bool = False,
                    out_dir: Path | None = None, record: bool = True, edit_style: str = "standard", song: str | None = None) -> dict:
        """out_dir / record=False: render into a scratch folder and return the clip without touching the job (used to cut the
        building blocks of a compilation).
        segments: an explicit edit - several (start, end) source ranges in order (cold opens, skipping
        the middle of a story...). Dead air inside each range is still trimmed per `pace`."""
        from .glasses import _display_dims
        d = Path(out_dir) if out_dir else self.store.dir(job_id)
        d.mkdir(parents=True, exist_ok=True)
        vcfg = dict(self.cfg.get("video", {}) or {})
        W, H, fps = int(vcfg.get("width", 1080)), int(vcfg.get("height", 1920)), int(vcfg.get("fps", 30))
        p = PACES.get(pace, PACES["tight"])
        speed = float(p["speed"])
        if profile == "glasses":
            # POV footage: ambient sound and action carry it - only trim pauses when someone is really talking
            talk = [w for w in words if m.start <= w["start"] < m.end]
            dense = len(talk) / max(1.0, m.end - m.start) >= 1.2
            intervals = repace(words, m.start, m.end, max(0.6, float(p["gap"]))) if dense and pace != "natural" \
                else [(m.start, m.end)]
            speed = 1.0
        elif segments:
            intervals = [iv for a, b in segments for iv in repace(words, a, b, float(p["gap"]))]
        else:
            intervals = repace(words, m.start, m.end, float(p["gap"]))
        if self.cfg.get("clipper.cut_fillers", True) and profile != "glasses":
            from .localproj import cut_fillers                   # shared with the AI editor: "um", stutters and a phrase said twice lose their first take
            intervals = cut_fillers(intervals, words) or intervals
        from . import negation
        intervals, kept_not = negation.keep_negations(intervals, words)   # no cut flips a sentence ("did not like" -> "did like")
        intervals = intervals or [(m.start, m.end)]
        if profile != "glasses" and not segments and self.cfg.get("clipper.clip_edges", True):     # the clip starts at its first word and ends just after its last (WP8)
            from . import clipcraft as CC
            intervals = CC.trim_lead(intervals, words, CC.LEAD_S)
            tail_words = [w for w in words if intervals[0][0] <= float(w["start"]) < intervals[-1][1]]
            loop = sum(b - a for a, b in intervals) <= CC.LOOP_UNDER_S and CC.could_loop(" ".join(w["text"] for w in tail_words[:8]), " ".join(w["text"] for w in tail_words[max(8, len(tail_words) - 8):]))
            intervals = CC.end_after_last_word(intervals, words, CC.TAIL_S, loop)
        for n_ in kept_not:
            log.info(f"meaning: {n_}")
        for s_ in negation.lost_negations(intervals, words):
            log.warning(f"meaning check: a kept sentence lost its negation: {s_[:160]}")
        cm_lines: list[dict] = []
        from . import commentary as CM
        if commentary and not CM.allowed((self.store.get(job_id) or {}).get("source") if record else None, CM.talk_density(words, m.start, m.end),
                                         float(self.cfg.get("clipper.commentary.max_words_per_s", CM.MAX_WORDS_PER_S))):
            commentary = False                                    # podcasts, interviews and anything with a lot of talking: no AI voice-over
        if commentary and ffmpeg.audio_stream(ffmpeg.probe(src)) is not None:
            cm_lines, intervals = self._commentary(job_id, index, m, words, intervals, speed, d / f"clip_{index:02d}_voice")
        out_len = sum(b - a for a, b in intervals) / speed
        span0, span1 = min(a for a, _ in intervals), max(b for _, b in intervals)
        ordered = all(intervals[i][1] <= intervals[i + 1][0] + 0.01 for i in range(len(intervals) - 1))

        info = ffmpeg.probe(src)
        vs = ffmpeg.video_stream(info) or {}
        sw, sh = _display_dims(vs)                          # ffmpeg auto-rotates; crop in displayed pixels
        sw, sh = sw or 1920, sh or 1080
        has_audio = ffmpeg.audio_stream(info) is not None

        # ---- the source's own promo bars, name tags and captions: cropped out of the picture here, so the finished clip never has them (the gate stays as the last look)
        crop_pre, vertical, plan = "", sw <= int(sh * W / H) // 2 * 2, None
        if record and profile != "glasses" and self.cfg.get("clipper.source_check.strips", True):
            from . import sourcecheck as SC
            plan = self._strip_plan(d, index, src, span0, span1, words)
            crop_pre, sh, _ycut = SC.crop_filter(sh, plan)
            if not plan["own_captions"]:                          # the source's captions sit in the middle of the picture: ours would be a second set on top
                style = "none"
            if crop_pre:
                log.info(f"{job_id} clip {index}: cropped {plan['top']:.0%} off the top and {plan['bottom']:.0%} off the bottom (the source's own text)")

        # ---- framing
        faces_measured, face_first_s = False, None
        mode = "glasses" if profile == "glasses" and layout in ("auto", "track") else layout
        xexpr = None
        fit_spans: list[tuple[float, float]] = []
        crop_w = int(sh * W / H) // 2 * 2
        if vertical or sw <= crop_w:                         # already vertical
            mode = "center"
        wide_w = int(min(sw, sh * float(self.ccfg.get("wide_ratio", 1.0)))) // 2 * 2     # the "wide window": a square by default
        if mode in ("auto", "track", "wide"):
            rate = float(self.ccfg.get("speaker_rate", 6.0))          # fast enough to see lips move
            if not ordered and span1 - span0 > 300:
                mode = "fit"                                          # (a reordered edit spanning minutes: just show the whole frame)
            samples, _n = R.detect_faces(src, span0, span1 - span0, self.models_dir, rate, mouths=not ordered) \
                if mode in ("auto", "track", "wide") else ([], 0)
            if _n and ordered:
                from . import clipcraft as CC
                faces_measured, face_first_s = True, CC.first_face_time(samples, rate)         # (for the opening check at the end)
            cov = R.face_coverage(samples)
            f = remapper(intervals, speed)
            wide = mode in ("auto", "wide") and wide_w > crop_w
            small_cam = wide and R.typical_face_width(samples) < float(self.ccfg.get("small_face", 0.13))
            fcam = None
            if small_cam and mode == "auto":
                top_h = int(round(H * float(self.ccfg.get("facecam_split", 0.38)) / 2)) * 2
                fcam = R.facecam_box(samples, sw, sh, W / top_h) if bool(self.ccfg.get("facecam_layout", True)) and not crop_pre else None     # (the face-cam box is in uncropped pixels)
                mode = "facecam" if fcam else "fit"                  # gameplay with a corner facecam: the face big on top, the game below (else the whole frame)
            elif cov >= float(self.ccfg.get("min_face_coverage", 0.35)) or (mode in ("track", "wide") and cov > 0):
                win = wide_w if wide else crop_w
                if ordered:
                    env = R.audio_envelope(src, span0, span1 - span0, rate) if has_audio else None
                    path = R.speaker_path(samples, rate, win / sw, env=env)         # follow whoever is talking...
                    speaker_led = path is not None
                    if path is None:                                                  # ...unless nobody's lips move (no signal)
                        path = (R.group_path if wide else R.crop_path)(samples, rate, win / sw)
                    xexpr = R.x_expression(path, sw, win, lambda t: f(span0 + t))
                else:                                                                 # reordered edit: one steady framing
                    speaker_led = True
                    cxs = sorted(max(s_, key=lambda f_: f_.w).cx for s_ in samples if s_)
                    xexpr = str(int(round(min(sw - win, max(0.0, cxs[len(cxs) // 2] * sw - win / 2)))))
                crop_w = win
                mode = "wide" if wide else "track"
                spans = list(R.faceless_spans(samples, rate)) if ordered else []     # slides / b-roll: show the whole frame
                if wide and not speaker_led:
                    spans += R.too_wide_spans(samples, rate, win / sw)          # people far apart: whole frame too
                for a, b in sorted(spans):
                    oa, ob = f(span0 + a), f(span0 + b)
                    if oa is not None and (ob or out_len) - oa > 0.3:
                        fit_spans.append((oa, ob if ob is not None else out_len))
            else:
                ap = None
                if mode in ("auto", "track", "wide") and bool(self.ccfg.get("action_crop", True)) and sw > wide_w + 40 and not small_cam:
                    ap = R.action_path(src, span0, span1 - span0, wide_w / sw)         # no faces: follow what moves (hands, a pan, a craft) instead of a small picture on blur
                if ap and ap["concentrated"]:
                    xexpr = R.x_expression(ap["path"], sw, wide_w, lambda t: f(span0 + t))
                    crop_w = wide_w
                    mode = "wide"
                else:
                    mode = "fit"
        fit_f = (f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=24:4,eq=brightness=-0.08[bg];"
                 f"[fb]scale={W}:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1")
        wide_f = (f"split=2[wbg][wfg];[wbg]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=24:4,"
                  f"eq=brightness=-0.08[wbgb];[wfg]crop={crop_w}:{sh}:'{xexpr}':0,scale={W}:-2[wfgs];"
                  f"[wbgb][wfgs]overlay=(W-w)/2:(H-h)/2,setsar=1")
        if mode == "wide" and fit_spans:
            enable = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in fit_spans)
            layout_f = (f"[vc]split=3[tk][fa][fb];[tk]{wide_f}[trk];[fa]{fit_f}[fit];"
                        f"[trk][fit]overlay=0:0:enable='{enable}',fps={fps}[base]")
            cap_y = 0.79
        elif mode == "wide":
            layout_f = f"[vc]{wide_f},fps={fps}[base]"
            cap_y = 0.79
        elif mode == "track" and fit_spans:
            enable = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in fit_spans)
            layout_f = (f"[vc]split=3[tk][fa][fb];[tk]crop={crop_w}:{sh}:'{xexpr}':0,scale={W}:{H},setsar=1[trk];"
                        f"[fa]{fit_f}[fit];[trk][fit]overlay=0:0:enable='{enable}',fps={fps}[base]")
            cap_y = 0.72
        elif mode == "track":
            layout_f = f"[vc]crop={crop_w}:{sh}:'{xexpr}':0,scale={W}:{H},setsar=1,fps={fps}[base]"
            cap_y = 0.70
        elif mode == "facecam":
            top_h = int(round(H * float(self.ccfg.get("facecam_split", 0.38)) / 2)) * 2
            bot_h = H - top_h
            gw = min(sw, int(sh * W / bot_h) // 2 * 2)                                # the game fills the lower panel (a centred slice, away from the face-cam)
            gx = int(min(sw - gw, max(0, sw * (0.5 + (0.06 if fcam["cx"] < 0.5 else -0.06)) - gw / 2)))
            layout_f = (f"[vc]split=2[fa][fb];[fa]crop={fcam['w']}:{fcam['h']}:{fcam['x']}:{fcam['y']},scale={W}:{top_h},setsar=1[ftop];"
                        f"[fb]crop={gw}:{sh}:{gx}:0,scale={W}:{bot_h},setsar=1[fbot];[ftop][fbot]vstack,fps={fps}[base]")
            cap_y = 0.80
        elif mode == "glasses":
            # head-mounted: steady it, then fill 9:16 from 3:4 with a 5% zoom that hides the stabiliser's edges
            layout_f = (f"[vc]deshake=rx=32:ry=32:edge=mirror,scale={W}:{H}:force_original_aspect_ratio=increase,"
                        f"scale=trunc(iw*1.05/2)*2:trunc(ih*1.05/2)*2,crop={W}:{H},setsar=1,fps={fps}[base]")
            cap_y = 0.74
        elif mode == "center":
            layout_f = f"[vc]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,fps={fps}[base]"
            cap_y = 0.70
        else:
            layout_f = f"[vc]split=2[fa][fb];[fa]{fit_f},fps={fps}[base]"
            cap_y = 0.76
            mode = "fit"

        # ---- cut + pace
        seek = max(0.0, span0 - 1.0)
        parts, labels = [], []
        for i, (a, b) in enumerate(intervals):
            parts.append(f"[0:v]trim=start={a - seek:.3f}:end={b - seek:.3f},setpts=PTS-STARTPTS{crop_pre}[v{i}]")
            labels.append(f"[v{i}]")
            if has_audio:
                parts.append(f"[0:a]atrim=start={a - seek:.3f}:end={b - seek:.3f},asetpts=PTS-STARTPTS[a{i}]")
                labels.append(f"[a{i}]")
        concat = "".join(labels) + f"concat=n={len(intervals)}:v=1:a={1 if has_audio else 0}" + \
            ("[vc0][ac0]" if has_audio else "[vc0]")
        hdr = str(vs.get("color_transfer", "")) in ("arib-std-b67", "smpte2084")
        tone = ",colorspace=iall=bt2020:itrc=bt2020-10:all=bt709:format=yuv420p:fast=1" if hdr else ""
        if not hdr:                                                        # log footage (S-Log3...) -> normal colour, the same way
            from . import logcolor as LC
            lut = LC.vf(src, self.cfg)
            tone = f",{lut}" if lut else tone
        graph = parts + [concat, f"[vc0]setpts=PTS/{speed:.4f}{tone}[vc]"]   # iPhone HDR -> normal (SDR) colour
        if has_audio:
            graph.append(f"[ac0]atempo={speed:.4f}[ac]" if speed != 1.0 else "[ac0]anull[ac]")
        graph.append(layout_f)

        # ---- captions + hook title
        stem = f"clip_{index:02d}"
        work = d / f"{stem}_work"
        shutil.rmtree(work, ignore_errors=True)            # a restyle must not reuse the old caption images
        work.mkdir()
        inputs = ["-ss", f"{seek:.3f}", "-t", f"{span1 - seek + 0.5:.3f}", "-i", str(src)]
        cur = "base"
        cap_words = remap_words(words, intervals, speed)
        ccfg_used: dict = {}
        srt_text = ""
        if style != "none" and len(cap_words) >= (3 if profile == "glasses" else 1):
            ccfg = {**(self.cfg.get("captions", {}) or {}), **{k: v for k, v in CAPTION_STYLES.get(style, {}).items()
                                                               if k != "label"}, "center_y": cap_y}
            caps = render_captions(cap_words, out_len, work, ccfg, vcfg)
            ccfg_used = ccfg
            srt_text = Path(caps["srt"]).read_text()
            inputs += ["-f", "concat", "-safe", "0", "-i", str(caps["ffconcat"])]
            graph.append(f"[1:v]format=rgba[cap];[{cur}][cap]overlay=0:{caps['y']}:format=auto:eof_action=repeat[wc]")
            cur = "wc"
        if hook_title and m.title:
            png = work / "title.png"
            self._hook_card(m.title, png, W, H)
            hx, hy = self._hook_xy(W, H)
            idx = 2 if cur == "wc" else 1
            inputs += ["-loop", "1", "-t", f"{min(out_len, float(self.ccfg.get('title_seconds', 3.2))):.2f}", "-i", str(png)]
            graph.append(f"[{idx}:v]format=rgba,fade=t=out:st={max(0.1, float(self.ccfg.get('title_seconds', 3.2)) - 0.35):.2f}"
                         f":d=0.35:alpha=1[tt];[{cur}][tt]overlay={hx}:{hy}:eof_action=pass:format=auto[wt]")
            cur = "wt"
        graph.append(f"[{cur}]format=yuv420p[vout]")
        lufs = self.cfg.get("video.loudness_lufs", -14)
        ac_label = "ac"
        if has_audio and cm_lines:                          # a voiced aside: the clip's sound ducks while it speaks
            if cm_lines:
                duck = "+".join(f"between(t,{c['at'] - 0.08:.2f},{c['at'] + c['dur'] + 0.2:.2f})" for c in cm_lines)
                from . import soundmix as SMX
                under = 10 ** (SMX.DEFAULTS["bed_under_speech_lu"] / 20.0)     # (the clip stays audible under the line: 20 LU down, as the music bed ducks)
                graph.append(f"[ac]volume={under:.3f}:eval=frame:enable='{duck}'[acd]")
                mix_in = "[acd]"
                for k, c in enumerate(cm_lines):
                    inputs += ["-i", c["wav"]]
                    ix = inputs.count("-i") - 1
                    ms = int(c["at"] * 1000)
                    graph.append(f"[{ix}:a]aresample=48000,aformat=channel_layouts=stereo,adelay={ms}|{ms},volume=1.5[cm{k}]")
                    mix_in += f"[cm{k}]"
                graph.append(f"{mix_in}amix=inputs={len(cm_lines) + 1}:normalize=0:dropout_transition=0,"
                             f"atrim=duration={out_len:.3f}[ac2]")
                ac_label = "ac2"
        track, music_start, mood = None, 0.0, ""
        if song is None:                                    # a song chosen for the whole batch (dropped on it, or added afterwards)
            try:
                song = str((self.store.get(job_id).get("options") or {}).get("song") or "")
            except (KeyError, OSError, ValueError):
                song = ""
        if song:
            from . import music as MU
            track, mood = MU.song(song), "your song"
            if track is None:
                log.warning(f"the chosen song isn't there any more: {song}")
        if not track and music and music != "your song":    # "auto" = the clip's own mood, else a mood name
            from . import music as MU
            mood = (m.mood if music == "auto" else music) or ("upbeat" if profile == "glasses" else "chill")
            track = MU.pick(self.cfg, mood=mood, seed=f"{job_id}:{index}", length_s=out_len)
        if track:
            from . import music as MU
            midx = inputs.count("-i")
            music_start = MU.clip_start(self.cfg, track, out_len, f"{job_id}:{index}")      # (from just before the song's best part)
            inputs += ["-stream_loop", "-1", "-ss", f"{music_start:.2f}",
                       "-i", str(track.path)]
            graph += MU.mix_filter(midx, out_len, MU.levels(self.cfg, cap_words, out_len),
                                   ac_label if has_audio else None, "aout", lufs)
        elif has_audio:
            graph.append(f"[{ac_label}]aresample=48000,aformat=channel_layouts=stereo,loudnorm=I={lufs}:TP=-1.5:LRA=11,"
                         f"aresample=48000[aout]")
        audio_out = has_audio or bool(track)
        if has_audio:                                       # noisy, far-away speech: the cleaned copy (voiceclean.py) for this clip
            inputs, graph = self.clean_voice(d, src, seek, span1 - seek + 0.5, inputs, graph)

        out = d / f"{stem}.mp4"
        part = d / f"{stem}.part.mp4"
        base_cmd = [ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", *inputs, "-filter_complex", ";".join(graph),
                    "-map", "[vout]"] + (["-map", "[aout]"] if audio_out else []) + ["-t", f"{out_len:.3f}"]
        tail_args = ["-metadata", f"title={m.title}", "-metadata", f"comment=clipper {job_id} #{index}",
                     "-movflags", "+faststart", str(part)]
        t0 = time.monotonic()
        for attempt, vc in enumerate((vcfg, ffmpeg.software_cfg(vcfg))):
            cmd = base_cmd + ffmpeg.video_encode_args(vc) + (ffmpeg.audio_encode_args(vc) if audio_out else []) + tail_args
            try:
                ffmpeg.run(cmd, timeout=float(vcfg.get("render_timeout_s", 1800)), what=f"clip {index}")
            except ffmpeg.FFmpegError:
                if attempt == 1:                           # even the software encoder failed: a real problem
                    raise
                log.warning(f"clip {index} of {job_id}: the hardware encoder failed; redoing it in software")
                continue
            if ffmpeg.has_video(part):
                break
            log.warning(f"clip {index} of {job_id} came out empty with the hardware encoder; redoing it in software")
        else:
            raise RuntimeError(f"clip {index} rendered empty")
        part.replace(out)
        styled = None
        if edit_style != "standard" and mode != "asis" and profile not in ("asis", "glasses"):
            from . import editstyle
            src_meta = (self.store.get(job_id) or {}).get("source") if record else None
            styled = editstyle.apply(self, job_id, out, cap_words, out_len, edit_style, W, H, fps, hook_title, source=src_meta, title=m.title, cap_y=cap_y if style != "none" else None,
                                     hook_end=(min(out_len, float(self.ccfg.get("title_seconds", 3.2))) if hook_title and m.title else 0.0))
        qa_res = None
        if record:                                              # the same look at the finished file the AI editor gets: dead audio, loudness, picture that never changes
            try:
                from . import qa as QA
                r = QA.check(out, None)
                qa_res = {"lufs": r["lufs"], "issues": [i for i in r["issues"] if i["level"] != "info"]}
            except Exception as e:  # noqa: BLE001 - a check is never a reason to lose a clip
                log.debug(f"quality check of {stem} failed: {e}")
            try:
                from . import opening                           # a picture that never moves, an empty opening, another channel's burned-in promo
                found = opening.check(out, " ".join(w["text"] for w in cap_words))
                qa_res = {**(qa_res or {"lufs": None}), "issues": ((qa_res or {}).get("issues") or []) + found}
                if self.cfg.get("clipper.clip_edges", True) and profile != "glasses" and not segments:
                    from . import clipcraft as CC                  # the first second: late speech, no face (warnings, never a rejection)
                    hs = self.ccfg.get("hook") or {}
                    qa_res["issues"] += CC.hook_warnings(cap_words, out_len, face_first_s, faces_measured, float(hs.get("speech_by_s", 0.5)), float(hs.get("face_by_s", 1.0)))
                qa_res["issues"] = sorted(qa_res["issues"], key=lambda i: i["level"] != "bad")[:6]       # (the problems first: the gate reads these)
            except Exception as e:  # noqa: BLE001 - a check is never a reason to lose a clip
                log.debug(f"quality check of {stem} failed: {e}")
        thumb = d / f"{stem}.jpg"
        ffmpeg.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-ss", f"{min(1.5, out_len / 3):.2f}", "-i", str(out),
                    "-frames:v", "1", "-vf", "scale=360:-2", "-q:v", "4", str(thumb)], what="thumbnail")
        prev_name, prev_v = None, 0
        if record and self.cfg.get("clipper.feed.previews", True):      # small fast-loading copy for the swipe feed
            from . import preview
            if preview.make(out, d / f"{stem}.preview.mp4"):
                prev_name = f"{stem}.preview.mp4"
                prev_v = 2
        if srt_text:
            (d / f"{stem}.srt").write_text(srt_text)
        tags = " ".join("#" + re.sub(r"[^\w]", "", t) for t in m.hashtags if re.sub(r"[^\w]", "", t))
        post = (m.caption + ("\n\n" + tags if tags else "")).strip()
        if styled and styled.get("credits"):
            post += "\n\n" + "\n".join(styled["credits"])
        if track:                                          # CC BY requires the credit; keep it with the clip
            post += "\n\n" + track.credit
        (d / f"{stem}.txt").write_text(post + "\n")
        clip = {"index": index, "file": out.name, "thumb": thumb.name, "preview": prev_name, "preview_v": prev_v, "framing_v": FRAMING_V, "srt": f"{stem}.srt" if srt_text else None,
                "title": m.title, "hook": m.hook, "caption": post, "score": round(m.score, 1), "reason": m.reason,
                **({"rubric": m.rubric} if getattr(m, "rubric", None) else {}), **({"rehook_at": m.rehook_at} if getattr(m, "rehook_at", None) else {}),
                "picked_by": m.source, "source_start": round(m.start, 2), "source_end": round(m.end, 2),
                "duration": round(out_len, 2), "cuts": len(intervals) - 1, "layout": mode, "fit_spans": len(fit_spans), "caption_style": style,
                "pace": pace, "edit_style": edit_style, "styled": styled, "qa": qa_res, **({"source_fix": plan} if plan and (crop_pre or not plan["own_captions"]) else {}), **({"style_done": __import__("clipfarm.clipper.feed", fromlist=["STYLE_V"]).STYLE_V} if edit_style != "standard" else {}), "hook_title": hook_title, "profile": profile, "moment": m.to_dict(),
                "music": ({"title": track.title, "artist": track.artist, "license": track.license, "mood": mood,
                           "file": track.path.name, "path": str(track.path), "start": music_start} if track else None),
                "rendered_at": int(time.time()), "word_end_pad": END_PAD,
                "commentary": [{"text": c["text"], "at": round(c["at"], 2)} for c in cm_lines],
                "transcript": " ".join(w["text"] for w in cap_words)[:1800],
                "render_s": round(time.monotonic() - t0, 1),
                "edit": {"intervals": [[round(a, 3), round(b, 3)] for a, b in intervals], "speed": speed,
                         "width": W, "height": H, "src_w": sw, "src_h": sh, "layout": mode,
                         "captions": [list(g) for g in _cap_groups(cap_words, ccfg_used)] if srt_text else [],
                         "music_file": str(track.path) if track else None, "music_start": music_start,
                         "hook": m.title if (hook_title and m.title) else None, "cap_y": cap_y}}
        if not record:
            return clip
        old = next((c for c in self.store.get(job_id)["clips"] if c["index"] == index), None) or {}
        for k in ("review", "feed"):                               # a re-render keeps what happened to the clip so far
            if k in old:
                clip[k] = old[k]
        self.store.set_clip(job_id, clip)
        return clip

    def edit_clip(self, job_id: str, index: int, instruction: str = "", revert: bool = False) -> dict:
        """Apply a plain-words instruction to a finished clip ("put this sentence first as the hook, cut that one") by letting
        the AI choose the new order of sentences, then re-render it from the original footage. revert=True restores the first cut."""
        from . import edit as E
        d = self.store.dir(job_id)
        job = self.store.get(job_id)
        clip = next(c for c in job["clips"] if c["index"] == index)
        words = json.loads((d / "transcript.json").read_text())["words"]
        src = next(p for p in d.glob("source.*") if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov", ".m4v"))
        profile = clip.get("profile") or (job.get("info") or {}).get("profile", "standard")
        music = (clip.get("music") or {}).get("mood", "") if clip.get("music") else ""
        orig = clip.get("orig") or {"moment": clip["moment"], "title": clip["title"]}
        if revert:
            m = M.Moment(**orig["moment"])
            m.title = orig["title"]
            segs, edited, ranges = None, None, None
        else:
            m = M.Moment(**clip["moment"])
            sents = M.sentences_from_words(words)
            win = E.window(sents, clip.get("source_start", m.start), clip.get("source_end", m.end))
            current = [tuple(r) for r in (clip.get("edit_ranges") or [])] or \
                [(s.idx, s.idx) for s in win if m.start - 0.05 <= s.start and s.end <= m.end + 0.05]
            current = ([(current[0][0], current[-1][1])] if not clip.get("edit_ranges") and current else current)
            router = self.creative_router()
            plan = E.plan(router, win, m.start, m.end, current, instruction, clip["title"])
            if plan.problem and not plan.summary:
                self.store.set_clip(job_id, {**clip, "restyling": False,
                                             "edit_note": {"instruction": instruction, "result": plan.problem, "ok": False}})
                return clip
            segs, ranges = plan.segments, plan.ranges
            m.start, m.end = min(a for a, _ in segs), max(b for _, b in segs)
            if plan.title:
                m.title = plan.title
            edited = {"instruction": instruction, "summary": plan.summary or "Edited", "ok": True, "at": int(time.time())}
        out = self.render_clip(job_id, index, m, words, src, clip["layout"], clip["caption_style"], clip["pace"],
                               bool(clip.get("hook_title", True)), profile=profile, music=music, segments=segs, edit_style=clip.get("edit_style", "standard"),
                               commentary=bool(clip.get("commentary")))
        with self.store._lock:
            cur = next(c for c in self.store.get(job_id)["clips"] if c["index"] == index)
            keep = {k: clip[k] for k in ("post_next",) if k in clip}
            self.store.set_clip(job_id, {**cur, **keep, "orig": orig, "edit_ranges": ranges, "edited": edited, "restyling": False,
                                         "edit_note": None})
        self._archive(job_id, index)
        return out

    def _as_is(self, job_id: str, d: Path, src: Path, words: list[dict]) -> dict:
        """A video you edited yourself: no editing at all. The AI only writes the title and post text from what is said,
        and it goes to the posting line (first in line, since it's your own original work)."""
        job = self.store.get(job_id)
        opts = job["options"]
        text = " ".join(w["text"] for w in words)[:5000]
        title, caption, tags = job.get("title") or src.stem, "", []
        try:
            router = self.creative_router()
            brief = (opts.get("prompt") or "").strip()
            res = router.complete_json(
                "asis_text", "You write the title and post text for a finished short video. Title: max 70 characters, a real hook, no "
                "clickbait lies. Caption: 1-2 lines. 4-5 hashtags. Base it ONLY on what is said (if nothing is said, on the user's notes).",
                f"Transcript: {text or '(no speech)'}\nUser's notes: {brief or '(none)'}\nFile name: {src.stem}",
                {"type": "object", "properties": {"title": {"type": "string"}, "caption": {"type": "string"},
                                                  "hashtags": {"type": "array", "items": {"type": "string"}}},
                 "required": ["title", "caption", "hashtags"]})
            title = str(res.data.get("title") or title).strip()[:90]
            caption = str(res.data.get("caption") or "").strip()
            tags = [t for t in res.data.get("hashtags", []) if re.sub(r"[^\w]", "", t)][:5]
        except Exception as e:  # noqa: BLE001 - the file name is a fine title if the AI is unavailable
            log.warning(f"as-is title for {job_id}: {e}")
        out, part = d / "clip_01.mp4", d / "clip_01.part.mp4"
        self._progress(job_id, "rendering", "Preparing your video", 0.5)
        ffmpeg.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-i", str(src), "-map", "0:v:0", "-map", "0:a?", "-c", "copy",
                    "-movflags", "+faststart", str(part)], timeout=1800, what="copy your video")
        if not ffmpeg.has_video(part):
            ffmpeg.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-i", str(src), *ffmpeg.video_encode_args(dict(self.cfg.get("video", {}) or {})),
                        *ffmpeg.audio_encode_args(dict(self.cfg.get("video", {}) or {})), "-movflags", "+faststart", str(part)],
                       timeout=7200, what="re-encode your video")
        part.replace(out)
        dur = ffmpeg.duration(out)
        ffmpeg.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-ss", f"{min(1.5, dur / 3):.2f}", "-i", str(out), "-frames:v", "1",
                    "-vf", "scale=360:-2", "-q:v", "4", str(d / "clip_01.jpg")], what="thumbnail")
        from . import preview
        prev = preview.make(out, d / "clip_01.preview.mp4")
        post = (caption + ("\n\n" + " ".join("#" + re.sub(r"[^\w]", "", t) for t in tags) if tags else "")).strip()
        (d / "clip_01.txt").write_text(post + "\n")
        clip = {"index": 1, "file": "clip_01.mp4", "thumb": "clip_01.jpg", "preview": "clip_01.preview.mp4" if prev else None,
                "preview_v": 2, "srt": None, "title": title, "hook": "", "caption": post, "score": None, "reason": "your own edit",
                "picked_by": "as-is", "source_start": 0.0, "source_end": round(dur, 2), "duration": round(dur, 2), "cuts": 0,
                "layout": "original", "caption_style": "none", "pace": "natural", "hook_title": False, "profile": "asis", "polish": True,
                "own_upload": True, "transcript": text[:1800], "rendered_at": int(time.time())}
        self.store.set_clip(job_id, clip)
        self.store.update(job_id, status="done", progress=1.0, stage="Ready - going to the front of the posting line", title=title)
        self._archive(job_id)
        return self.store.get(job_id)

    def edit_whole_video(self, job_id: str, instruction: str = "", revert: bool = False) -> dict:
        """The same plain-words edit for a whole / long video (cuts, or a part moved to the front), re-made from the original."""
        from . import edit as E
        from . import polish
        d = self.store.dir(job_id)
        job = self.store.get(job_id)
        clip = job["clips"][0]
        words = json.loads((d / "transcript.json").read_text())["words"]
        src = next(p for p in d.glob("source.*") if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov", ".m4v"))
        orig = clip.get("orig") or {"title": clip["title"]}
        override, ranges, edited = None, None, None
        if not revert:
            sents = M.sentences_from_words(words)
            if len(sents) > 600:
                self.store.set_clip(job_id, {**clip, "restyling": False, "edit_note": {
                    "instruction": instruction, "ok": False,
                    "result": "This video is too long to edit by typing. Use the prompt box and 'One long video' mode, or cut it in Final Cut."}})
                return clip
            current = [tuple(r) for r in (clip.get("edit_ranges") or [])]
            if not current and (d / "plan.json").exists():          # first edit of a long video: start from what the AI kept, not the whole footage
                iv = json.loads((d / "plan.json").read_text()).get("intervals") or []
                run: list[int] = []
                for st in sents:
                    if any(a <= (st.start + st.end) / 2 <= b for a, b in iv):
                        run.append(st.idx)
                    elif run:
                        current.append((run[0], run[-1]))
                        run = []
                if run:
                    current.append((run[0], run[-1]))
            current = current or [(0, sents[-1].idx)]
            router = self.creative_router()
            plan = E.plan(router, sents, 0.0, ffmpeg.duration(src), current, instruction, clip["title"], max_s=36000.0)
            if plan.problem and not plan.summary:
                self.store.set_clip(job_id, {**clip, "restyling": False,
                                             "edit_note": {"instruction": instruction, "result": plan.problem, "ok": False}})
                return clip
            gap = float(PACES.get(job["options"].get("pace") or "natural", PACES["natural"])["gap"])
            override = [iv for a, b in plan.segments for iv in repace(words, a, b, gap)]
            ranges = plan.ranges
            edited = {"instruction": instruction, "summary": plan.summary or "Edited", "ok": True, "at": int(time.time())}
        try:
            polish.run(self, job_id, src, words, override=override)
        finally:                                              # polish.run leaves the job "rendering"; an edit is not a new job, so it is done again
            if self.store.get(job_id).get("clips"):
                self.store.update(job_id, status="done", progress=1.0, error=None,
                                  stage="Done - edited" + (", Final Cut project ready" if clip.get("fcp") else ""))
        with self.store._lock:
            cur = self.store.get(job_id)["clips"][0]
            self.store.set_clip(job_id, {**cur, "orig": orig, "edit_ranges": ranges, "edited": edited, "restyling": False,
                                         **{k: clip[k] for k in ("post_next",) if k in clip}, "edit_note": None})
        try:
            polish.archive_bundle(self.cfg, self.store, job_id)
        except Exception as e:  # noqa: BLE001
            log.warning(f"archiving the FCP project failed: {e}")
        return self.store.get(job_id)["clips"][0]

    def restyle(self, job_id: str, index: int, layout: str, style: str, pace: str, hook_title: bool,
                title: str | None = None, music: str | None = None, commentary: bool | None = None, edit_style: str | None = None) -> dict:
        d = self.store.dir(job_id)
        job = self.store.get(job_id)
        clip = next(c for c in job["clips"] if c["index"] == index)
        m = M.Moment(**clip["moment"])
        if title is not None:
            m.title = title.strip()[:70]
        words = json.loads((d / "transcript.json").read_text())["words"]
        src = next(p for p in d.glob("source.*") if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov", ".m4v"))
        profile = clip.get("profile") or (job.get("info") or {}).get("profile", "standard")
        if music is None:                                   # keep whatever the clip had
            music = (clip.get("music") or {}).get("mood", "") if clip.get("music") else ""
        out = self.render_clip(job_id, index, m, words, src, layout, style, pace, hook_title, profile=profile,
                               music=music, commentary=bool(clip.get("commentary")) if commentary is None else commentary,
                               edit_style=edit_style if edit_style is not None else clip.get("edit_style", "standard"))
        self._archive(job_id, index)
        return out

    def resong(self, job_id: str, song: str) -> dict:
        """Give a finished batch of clips a new song (or none, with song=""): every clip is rendered again with the same cut, captions and
        framing - nothing re-cut, no AI - and the new song under it (from just before its best part). Returns at once; the job shows
        progress ("Adding the song: clip 2 of 6") and is done again when every clip has it."""
        from . import music as MU
        job = self.store.get(job_id)
        if job.get("status") not in ("done", "failed", "cancelled") or not job.get("clips"):
            raise ValueError("these clips are still being made" if job.get("clips") else "there are no clips to add a song to yet")
        song = str(song or "").strip()
        if song and MU.song(song) is None:
            raise ValueError("that song isn't an audio file on this Mac")
        opts = {**job["options"], "song": song}
        if not song:
            opts["music"] = ""
        self.store.update(job_id, options=opts, status="rendering", progress=0.0, error=None,
                          stage="Adding the song" if song else "Taking the music out")

        def run():
            clips = sorted(self.store.get(job_id)["clips"], key=lambda c: c["index"])
            d = self.store.dir(job_id)
            done = 0
            try:
                words = json.loads((d / "transcript.json").read_text())["words"]
                src = next(p for p in d.glob("source.*") if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov", ".m4v"))
                if song:
                    live.stage(self.store, job_id, f"Song: {Path(song).stem}")
                for k, clip in enumerate(clips):
                    self.store.update(job_id, progress=k / len(clips),
                                      stage=f"{'Adding the song' if song else 'Taking the music out'}: clip {k + 1} of {len(clips)}")
                    m = M.Moment(**clip["moment"])
                    m.title = clip.get("title") or m.title
                    segs = [tuple(x) for x in ((clip.get("edit") or {}).get("intervals") or [])] if clip.get("edited") else None
                    profile = clip.get("profile") or (self.store.get(job_id).get("info") or {}).get("profile", "standard")
                    self.render_clip(job_id, clip["index"], m, words, src, clip.get("layout", "auto"), clip.get("caption_style", "bold"),
                                     clip.get("pace", "tight"), bool(clip.get("hook_title", True)), profile=profile,
                                     music="your song" if song else "", segments=segs, commentary=bool(clip.get("commentary")),
                                     edit_style=clip.get("edit_style", "standard"), song=song)
                    with self.store._lock:                           # (render_clip writes a fresh clip: keep what happened to it so far)
                        cur = next(c for c in self.store.get(job_id)["clips"] if c["index"] == clip["index"])
                        keep = {k2: clip[k2] for k2 in ("orig", "edit_ranges", "edited", "keep", "post_next", "app_saved") if k2 in clip}
                        self.store.set_clip(job_id, {**cur, **keep})
                    done += 1
                self.store.update(job_id, status="done", progress=1.0,
                                  stage=f"{len(clips)} clips with {'the new song' if song else 'no music'}")
            except Exception as e:  # noqa: BLE001 - the clips that are done keep their new song; the rest keep the old one
                log.exception(f"adding the song to {job_id} failed")
                self.store.update(job_id, status="done" if done else "failed", progress=1.0, error=None if done else str(e)[:400],
                                  stage=(f"The song is on {done} of {len(clips)} clips: {e}"[:200] if done else "Adding the song failed"))
            try:
                self._archive(job_id)
            except Exception:  # noqa: BLE001
                pass
        threading.Thread(target=run, name=f"resong-{job_id}", daemon=True).start()
        return {"ok": True, "clips": len(job["clips"])}

    def rerender_saved(self, job_id: str) -> dict:
        """Render a finished batch's clips again from what was saved (storage.py took the videos off the disk to free space): the same cuts,
        captions, framing and song, nothing re-cut and no AI. A link's video is downloaded again first. Returns at once; the job shows
        progress and is done again when every clip is back."""
        job = self.store.get(job_id)
        if job.get("status") in ("queued", "downloading", "transcribing", "finding", "rendering", "joining"):
            raise ValueError("these clips are still being made")
        if not job.get("clips"):
            raise ValueError("there are no saved clips to render again")
        self.store.update(job_id, status="rendering", progress=0.0, error=None, stage="Rendering the clips again")

        def run():
            made = 0
            clips = sorted(self.store.get(job_id)["clips"], key=lambda c: c["index"])
            d = self.store.dir(job_id)
            try:
                words = json.loads((d / "transcript.json").read_text())["words"]
                src = self._ingest(job_id, d)                            # (a download that was removed comes down again)
                for k, clip in enumerate(clips):
                    self._progress(job_id, "rendering", f"Rendering clip {k + 1} of {len(clips)} again", 0.05 + 0.9 * k / len(clips))
                    m = M.Moment(**clip["moment"])
                    m.title = clip.get("title") or m.title
                    segs = [tuple(x) for x in ((clip.get("edit") or {}).get("intervals") or [])] if clip.get("edited") else None
                    profile = clip.get("profile") or (self.store.get(job_id).get("info") or {}).get("profile", "standard")
                    music = (clip.get("music") or {}).get("mood", "") if clip.get("music") else ""
                    song = (self.store.get(job_id).get("options") or {}).get("song") or ""
                    self.render_clip(job_id, clip["index"], m, words, src, clip.get("layout", "auto"), clip.get("caption_style", "bold"),
                                     clip.get("pace", "tight"), bool(clip.get("hook_title", True)), profile=profile,
                                     music="your song" if song else music, segments=segs, commentary=bool(clip.get("commentary")),
                                     edit_style=clip.get("edit_style", "standard"), song=song or None)
                    with self.store._lock:                           # (render_clip writes a fresh clip: keep what happened to it so far)
                        cur = next(c for c in self.store.get(job_id)["clips"] if c["index"] == clip["index"])
                        keep = {k2: clip[k2] for k2 in ("orig", "edit_ranges", "edited", "keep", "post_next", "app_saved") if k2 in clip}
                        self.store.set_clip(job_id, {**cur, **keep})
                    made += 1
                self.store.update(job_id, status="done", progress=1.0, stage=f"{len(clips)} clips", archived=None)
            except Cancelled:
                self.store.update(job_id, status="done" if made else "cancelled", stage="Stopped")
            except Exception as e:  # noqa: BLE001 - the clips that came back stay; the page says what went wrong
                log.exception(f"rendering {job_id} again failed")
                self.store.update(job_id, status="done" if made else "failed", progress=1.0, error=None if made else str(e)[:400],
                                  stage=(f"{made} of {len(clips)} clips are back: {e}"[:200] if made else "Rendering again failed"))
        threading.Thread(target=run, name=f"rerender-{job_id}", daemon=True).start()
        return {"ok": True, "clips": len(job["clips"])}

    def _archive(self, job_id: str, index: int | None = None) -> None:
        """Copy finished clips to the archive drive (clipper.library_dir) in a titled folder."""
        from . import library
        try:
            if index is None:
                library.export(self.cfg, self.store, job_id)
            else:
                library.update_clip(self.cfg, self.store, job_id, index)
        except Exception as e:  # noqa: BLE001 - the clips are safe in the working folder; retried later
            log.warning(f"archive copy of {job_id} failed: {e}")
            self.store.update(job_id, library_pending=True)


def title_card(text: str, out: Path, width: int, fonts: list[str]) -> None:
    """White rounded card with bold black text (the on-screen hook)."""
    from PIL import Image, ImageDraw, ImageFont

    font_path = next((f for f in fonts if Path(f).exists()), None)
    size = 64
    font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default(size=size)
    max_w = int(width * 0.80)
    draw = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    words, lines, cur = text.split(), [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if draw.textlength(test, font=font) <= max_w - 60 or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    lines = lines[:3]
    lh = int(size * 1.18)
    tw = int(max(draw.textlength(ln, font=font) for ln in lines)) + 64
    th = lh * len(lines) + 44
    img = Image.new("RGBA", (tw + 16, th + 16), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    dr.rounded_rectangle((8, 12, tw + 8, th + 12), radius=26, fill=(0, 0, 0, 90))      # soft shadow
    dr.rounded_rectangle((4, 4, tw + 4, th + 4), radius=26, fill=(255, 255, 255, 255))
    for i, ln in enumerate(lines):
        lw = dr.textlength(ln, font=font)
        dr.text((4 + (tw - lw) / 2, 4 + 22 + i * lh), ln, font=font, fill=(10, 10, 10, 255))
    img.save(out)
