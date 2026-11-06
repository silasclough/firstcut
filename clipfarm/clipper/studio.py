"""FirstCut, the app (built from app/VideoBuilder; it was called Video Builder until 2026-11-05): its own home folder, settings, the "how it edits" memory for each kind of video, the AI
sign-ins and the first-run setup. The app starts the Clipper through `python -m clipfarm studio`, which reads settings.json, writes the
engine settings (overlay.yaml) and serves the app page; nothing here is used by the personal Clipfarm machines.

    <home> = VIDEOBUILDER_HOME, default ~/Library/Application Support/FirstCut (or .../Video Builder on a Mac that had the first version)
        settings.json       appearance, which AI decides the edit, where things go, defaults per kind of video, footage library folders
        styles/short.json   {"points": [...], "history": [...]}  what the editor follows for short clips ("long.json" for long videos)
        overlay.yaml        the engine settings made from settings.json (CLIPFARM_OVERLAY)
        runtime/            the app's Python (made by the app itself), Parakeet's own Python, Ollama when it isn't installed already
        claude/             the Claude login used for editing decisions (CLAUDE_CONFIG_DIR), kept apart from any other
        data/, logs/        the engine's records: projects, clip jobs, the ChatGPT sign-in

Long videos, short clips and the footage catalog all live in the folders the person picks (settings "places"); every project gets a
folder of its own in the "videos" folder, and Final Cut projects point at the original footage where it is.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from pathlib import Path

from ..core.log import get_logger

log = get_logger("clipper.studio")
KINDS = ("short", "long")
OWN_OLLAMA_PORT = 11545                                     # our own Ollama, when the Mac has none running on 11434
OLLAMA_URL = "https://github.com/ollama/ollama/releases/latest/download/ollama-darwin.tgz"
PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
CLAUDE_INSTALL = "https://claude.ai/install.sh"
MAX_POINTS = 15


APP_NAME = "FirstCut"
WANTED = "setup-wanted"
MUSIC_MODEL, MUSIC_GB = "laion/clap-htsat-unfused", 0.6                 # the music search model (clipper/musiclib.py uses the same)
CLAUDE_MODELS = {"opus": "Opus (best, uses the most)", "sonnet": "Sonnet (balanced)", "haiku": "Haiku (fastest, uses the least)"}                                            # Setup was started and hasn't finished every part
OLD_NAME = "Video Builder"                                          # (its name until 2026-11-05)


def home() -> Path:
    if os.environ.get("VIDEOBUILDER_HOME"):
        return Path(os.environ["VIDEOBUILDER_HOME"]).expanduser()
    support = Path.home() / "Library" / "Application Support"
    old = support / OLD_NAME                                         # the first name: its Python and models stay where they are
    return old if not (support / APP_NAME).exists() and (old / "settings.json").exists() else support / APP_NAME


def ram_gb() -> float:
    try:
        return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout.strip()) / 2 ** 30
    except (OSError, ValueError, subprocess.SubprocessError):
        return 8.0


def local_models(ram: float) -> dict:
    """The local models that fit this much memory: the editing model (only used without an account, or as the fallback) and the picture
    model (always: it looks at B-roll)."""
    if ram >= 60:
        return {"text": "qwen3:32b", "text_gb": 20.0, "vision": "qwen3-vl:8b", "vision_gb": 6.1}
    if ram >= 30:
        return {"text": "qwen3:14b", "text_gb": 9.3, "vision": "qwen3-vl:8b", "vision_gb": 6.1}
    if ram >= 15:
        return {"text": "qwen3:8b", "text_gb": 5.2, "vision": "qwen3-vl:4b", "vision_gb": 3.3}
    return {"text": "qwen3:4b", "text_gb": 2.5, "vision": "qwen3-vl:2b", "vision_gb": 1.9}


def mac_info() -> dict:
    def sysctl(k):
        try:
            return subprocess.run(["sysctl", "-n", k], capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    chip = sysctl("machdep.cpu.brand_string") or "Apple silicon"
    try:
        free = shutil.disk_usage(Path.home()).free / 1e9
    except OSError:
        free = 0.0
    return {"chip": chip, "ram_gb": round(ram_gb()), "disk_free_gb": round(free), "arm64": os.uname().machine == "arm64"}


# ---------------------------------------------------------------------------------------------- settings
def defaults() -> dict:
    h = home()
    return {
        "appearance": "system",                                       # system | light | dark
        "updates": "auto",                                            # auto (download, restart when idle) | ask | off (updates.swift)
        "ai": {"use": "chatgpt", "fallback": True, "local_model": None, "claude_edit": "sonnet", "claude_quick": "haiku", "chatgpt_model": "",
               "backup": "auto",                                     # when `use` runs out of usage: "claude" | "chatgpt" | "local" | "" (wait)
                                                                     # | "auto" (the other account if connected, else this Mac)
               "look": "account"},                                   # who describes the footage: "account" (fast, cheap: the default) or "local" (free, private)
        "places": {"videos": str(Path.home() / "Movies" / APP_NAME), "models": str(h / "Models"),
                   "catalog": str(h / "Catalog"), "work": str(Path.home() / "Library" / "Caches" / APP_NAME)},
        "defaults": {"short": {"min_len": 20, "max_len": 35, "caption_style": "bold", "layout": "auto", "music": "", "clips": 8, "pace": "auto", "voices": "auto"},
                     "long": {"long_minutes": 0, "caption_style": "none", "library": True, "music": "", "graphics": "", "gfx_style": "auto", "stock": False,
                              "pace": "auto", "log": "convert", "voices": "auto"}},   # voices: "auto" | "always" | "never" (voiceclean)  # pace: "auto" | "fast" | "slow"; log: "convert" (log footage -> normal
                                                                        # colour in what FirstCut renders, never in Final Cut) | "leave"
        "library": {"folders": [], "when": "open", "recent": {}, "look": "", "picked": [],    # when: "open" | "night" | "off"
                    "describe_with": "account"},                     # who describes the catalogue's shots: "account" (default) | "local"
        "music": {"folders": [], "favorites": []},                    # your music (e.g. downloaded from Artlist / Soundstripe); hearted songs
        "setup": {"done": False},
        "mcp": {"enabled": False},                                      # Settings › Connect an AI: let AI apps use FirstCut (mcp_server.py)
        "glossary": [],                                               # names and words spelled the person's way (glossary.py)
    }


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


class Settings:
    def __init__(self, root: Path | None = None):
        self.home = Path(root or home())
        self.home.mkdir(parents=True, exist_ok=True)
        self.file = self.home / "settings.json"
        self.lock = threading.RLock()

    def get(self) -> dict:
        try:
            saved = json.loads(self.file.read_text())
        except (OSError, ValueError):
            saved = {}
        return _merge(defaults(), saved)

    def update(self, changes: dict) -> dict:
        """Merge `changes` into the saved settings (only known keys; values checked) and return the whole result."""
        with self.lock:
            cur = self.get()
            new = _merge(cur, clean(changes))
            rec = new["library"].get("recent") or {}
            if len(rec) > 100:                                            # the newest 100 (the merge would otherwise keep them all)
                new["library"]["recent"] = dict(sorted(rec.items(), key=lambda kv: kv[1])[-100:])
            tmp = self.file.with_suffix(".tmp")
            tmp.write_text(json.dumps(new, indent=1))
            tmp.replace(self.file)
            return new


def _section(ch: dict, key: str) -> dict:
    v = ch.get(key)
    if v in (None, "", {}):
        return {}
    if not isinstance(v, dict):
        raise ValueError(f"{key} must be a set of settings, not {type(v).__name__}")
    return v


def library_folders(given) -> list[str]:
    """Footage folders ("sources" on Find footage) as full paths, each once."""
    if isinstance(given, str):
        given = [given]
    if not isinstance(given, (list, tuple)):
        raise ValueError("folders must be a list of folders")
    out: list[str] = []
    for f in given:
        if not isinstance(f, str) or not f.strip():
            continue
        p = Path(f.strip()).expanduser()
        if p.is_absolute():                                       # (a relative folder means nothing to the catalog: left out)
            out.append(os.path.normpath(str(p)))
    return list(dict.fromkeys(out))[:50]                     # (a folder inside another is fine: the catalog reads each file once)


FAV_MAX = 500
FAV_SOURCES = ("library", "openverse", "epidemic")


def music_favorites(given) -> list[dict]:
    """Hearted songs (the Music page's Favourites), newest first: small records of the song, enough to show, play and download it again.
    Each song once (by source + id); anything malformed is left out."""
    if not isinstance(given, list):
        raise ValueError("favorites must be a list of songs")
    out, seen = [], set()
    for x in given:
        if not isinstance(x, dict) or x.get("source") not in FAV_SOURCES:
            continue
        tid = str(x.get("id") or "").strip()
        if not re.fullmatch(r"[\w.:-]{1,80}", tid) or (x["source"], tid) in seen:
            continue
        try:
            dur = float(x.get("dur") or 0)
        except (TypeError, ValueError):
            dur = 0.0
        dur = round(max(0.0, min(36000.0, dur)), 1) if math.isfinite(dur) else 0.0
        rec = {"source": x["source"], "id": tid, "title": " ".join(str(x.get("title") or "Untitled").split())[:120],
               "artist": " ".join(str(x.get("artist") or "").split())[:80], "dur": dur,
               "license": str(x.get("license") or "")[:60], "credit": str(x.get("credit") or "")[:400]}
        for k in ("preview", "page", "license_url"):                         # web links only
            v = str(x.get(k) or "")
            rec[k] = v[:1000] if v.startswith("https://") else ""
        path = str(x.get("path") or "")
        rec["path"] = os.path.normpath(path)[:1000] if x["source"] == "library" and path.startswith("/") else ""
        seen.add((x["source"], tid))
        out.append(rec)
        if len(out) >= FAV_MAX:
            break
    return out


PACE_CHOICES = ("auto", "fast", "slow")
# The app's pace -> the engine's (engine.PACES): automatic keeps what each kind of video does (short clips cut tight; a long video cuts
# tighter when it is a promo or an ad), fast trims every pause to a beat (short clips also speed the talking up a little), slow leaves
# room to breathe. In long videos the pace also sets how often B-roll cuts in (localproj.PACE_BROLL).
ENGINE_PACE = {"long": {"auto": "natural", "fast": "fast", "slow": "slow"}, "short": {"auto": "tight", "fast": "fast", "slow": "natural"}}


BACKUPS = ("auto", "claude", "chatgpt", "local", "")


def clean(ch: dict) -> dict:
    """Only what the page may change, in the shape the engine expects. Wrong shapes are a ValueError (the page gets a 400)."""
    if not isinstance(ch, dict):
        raise ValueError("settings must be a set of settings")
    out: dict = {}
    if ch.get("appearance") in ("system", "light", "dark"):
        out["appearance"] = ch["appearance"]
    if ch.get("updates") in ("auto", "ask", "off"):
        out["updates"] = ch["updates"]
    ai = _section(ch, "ai")
    if ai:
        out["ai"] = {}
        if ai.get("use") in ("chatgpt", "claude", "local"):
            out["ai"]["use"] = ai["use"]
        if "fallback" in ai:
            out["ai"]["fallback"] = bool(ai["fallback"])
        if "local_model" in ai:
            m = str(ai.get("local_model") or "").strip()
            out["ai"]["local_model"] = m if re.fullmatch(r"[a-z0-9][a-z0-9._:/-]{0,80}", m) else None
        if ai.get("claude_edit") in CLAUDE_MODELS:                       # which Claude makes the editing decisions...
            out["ai"]["claude_edit"] = ai["claude_edit"]
        if ai.get("back_to") in ("chatgpt", "claude", ""):                # switched to this Mac when the plan ran out: offer the way back
            out["ai"]["back_to"] = ai["back_to"]
        if ai.get("backup") in BACKUPS:                                   # who takes over when the primary runs out of usage
            out["ai"]["backup"] = ai["backup"]
        if ai.get("look") in ("local", "account"):                        # describing the footage: this Mac or the account
            out["ai"]["look"] = ai["look"]
        if ai.get("claude_quick") in ("haiku", "sonnet"):                 # ...and the small jobs (titles, picking B-roll)
            out["ai"]["claude_quick"] = ai["claude_quick"]
        if "chatgpt_model" in ai:                                         # "" = the app's choice
            m = str(ai.get("chatgpt_model") or "").strip()
            out["ai"]["chatgpt_model"] = m if re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,60}", m) else ""
    mcp = _section(ch, "mcp")
    if mcp and "enabled" in mcp:
        out["mcp"] = {"enabled": bool(mcp["enabled"])}
    places = _section(ch, "places")
    if places:
        out["places"] = {k: os.path.normpath(str(Path(str(v)).expanduser())) for k, v in places.items() if k in ("videos", "models", "catalog", "work")
                         and isinstance(v, str) and v.strip() and Path(v.strip()).expanduser().is_absolute()}
    dfl = _section(ch, "defaults")
    if dfl:
        out["defaults"] = {}
        for kind in KINDS:
            d = dfl.get(kind) or {}
            if not isinstance(d, dict) or not d:
                continue
            o: dict = {}
            for k in ("min_len", "max_len", "long_minutes", "clips"):
                if k in d:
                    try:
                        o[k] = max(0.0, min(600.0, float(d[k])))
                    except (TypeError, ValueError):
                        pass
            for k in ("caption_style", "layout", "music", "graphics", "gfx_style"):
                if k in d:
                    o[k] = str(d[k] or "")[:20]
            if "log" in d:
                o["log"] = "leave" if d["log"] == "leave" else "convert"
            if "voices" in d:                                            # Clean up voices: automatically / always / never
                o["voices"] = d["voices"] if d["voices"] in ("auto", "always", "never") else "auto"
            if "graphics" in o and o["graphics"] not in ("", "title", "all"):
                o["graphics"] = ""
            if "gfx_style" in o:
                from .graphics import THEMES
                o["gfx_style"] = o["gfx_style"] if o["gfx_style"] in THEMES else "auto"
            if "pace" in d:
                o["pace"] = d["pace"] if d["pace"] in PACE_CHOICES else "auto"
            if "library" in d:
                o["library"] = bool(d["library"])
            if "stock" in d:
                o["stock"] = bool(d["stock"])
            out["defaults"][kind] = o
    mus = _section(ch, "music")
    if mus and "folders" in mus:
        out["music"] = {"folders": library_folders(mus.get("folders"))}
    if mus and "favorites" in mus:
        out.setdefault("music", {})["favorites"] = music_favorites(mus.get("favorites"))
    lib = _section(ch, "library")
    if lib:
        out["library"] = {}
        if "folders" in lib:
            out["library"]["folders"] = library_folders(lib["folders"])
        if lib.get("describe_with") in ("local", "account"):
            out["library"]["describe_with"] = lib["describe_with"]
        if lib.get("when") in ("open", "night", "off"):                   # when it catalogues by itself: whenever the app is open,
            out["library"]["when"] = lib["when"]                           # 7 pm to 7 am, or only when asked
        if isinstance(lib.get("picked"), list):                       # shots picked lately in Find footage (shown on Home), newest first
            keep = []
            for x in lib["picked"][:24]:
                if isinstance(x, dict) and re.fullmatch(r"[0-9a-f]{8,64}", str(x.get("id") or "")):
                    try:
                        keep.append({"id": str(x["id"]), "start": round(float(x.get("start") or 0), 2), "end": round(float(x.get("end") or 0), 2),
                                     "k": int(x.get("k") or 0), "what": str(x.get("what") or "")[:120], "rel": str(x.get("rel") or "")[:300],
                                     "path": str(x.get("path") or "")[:1000], "dur": round(float(x.get("dur") or 0), 2),
                                     "step": float(x.get("step") or 0), "frames": int(x.get("frames") or 0), "kind": "video"})
                    except (TypeError, ValueError):
                        pass
            out["library"]["picked"] = keep
        if isinstance(lib.get("look"), str):                         # where Find footage looked last ("" = every source)
            out["library"]["look"] = os.path.normpath(lib["look"]) if lib["look"].startswith("/") else ""
        if isinstance(lib.get("recent"), dict):                      # when each source was last searched (newest first in the menu)
            out["library"]["recent"] = {str(k): float(v) for k, v in list(lib["recent"].items())[-100:]
                                        if isinstance(k, str) and k.startswith("/") and isinstance(v, (int, float)) and not isinstance(v, bool)
                                        and math.isfinite(v)}                  # (NaN would break the page's JSON)
    if "setup" in ch:
        out["setup"] = {"done": bool(_section(ch, "setup").get("done"))}
    if "glossary" in ch:                                                  # Settings › How it edits › Names and words
        from .glossary import clean as _gclean
        out["glossary"] = _gclean(ch["glossary"])
    return out


# ---------------------------------------------------------------------------------------------- the engine's settings
def claude_bin() -> str | None:
    """The Claude Code command on this Mac: its own installer's place, Homebrew, or the Claude desktop app's copy."""
    import glob
    for spec in ("~/.local/bin/claude", "~/.claude/local/claude", "/opt/homebrew/bin/claude", "/usr/local/bin/claude",
                 "~/Library/Application Support/Claude/claude-code/*/claude",
                 "~/Library/Application Support/Claude/claude-code/*/*/claude.app/Contents/MacOS/claude"):
        found = sorted(glob.glob(os.path.expanduser(spec)))
        if found and os.access(found[-1], os.X_OK):
            return found[-1]
    return shutil.which("claude")


def ollama_url(s: dict) -> str:
    """The Ollama this app talks to: one already running on this Mac (the Ollama app), else its own on OWN_OLLAMA_PORT."""
    return "http://127.0.0.1:11434" if _answers("http://127.0.0.1:11434") else f"http://127.0.0.1:{OWN_OLLAMA_PORT}"


def _answers(url: str) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(url + "/api/tags", timeout=2):
            return True
    except Exception:  # noqa: BLE001
        return False


def overlay(s: dict, root: Path) -> dict:
    """The engine settings for these app settings (written to overlay.yaml before the engine starts)."""
    pl = {k: Path(v) for k, v in s["places"].items()}
    m = local_models(ram_gb())
    text = s["ai"].get("local_model") or m["text"]
    use = s["ai"]["use"]
    backup = s["ai"].get("backup", "auto")
    fallback = use == "local" or backup == "local" or (backup == "auto" and bool(s["ai"].get("fallback", True)))
    look_here = use == "local" or s["ai"].get("look", "account") != "account"  # describing footage: the account unless the person chose this Mac
    url = ollama_url(s)
    folders = [f for f in s["library"]["folders"] if f]
    return {
        "timezone": local_zone(),
        "clipper": {
            "app": "studio", "background": False, "data_dir": str(pl["work"] / "clips"), "parakeet_for_clips": True,
            "log_footage": "leave" if (s["defaults"].get("long") or {}).get("log") == "leave" else "convert",   # (logcolor.py)
            "feed": {"local": False}, "auto": {"enabled": False}, "transcriber": "local", "llm_chain": ["ollama"],
            "ollama_model": text,
            "music_lib": {"roots": [f for f in (s.get("music") or {}).get("folders", []) if f], "dir": str(pl["catalog"] / "Music")},
            "ai_videos": {"local_projects": True, "data_dir": str(root / "data" / "ai_videos"), "out_dir": str(pl["videos"]),
                          "default_rules": False, "openai": {"enabled": False},
                          # both accounts are usable whenever they are signed in: `primary` first, `backup` when it runs out (localai)
                          "primary": use, "backup": backup,
                          "chatgpt_plan": {"enabled": True, **({"model": s["ai"]["chatgpt_model"]} if s["ai"].get("chatgpt_model") else {})},
                          "vision": "ollama" if look_here else "", "fallback": "ollama" if fallback else "", "ollama_model": text, "vision_model": m["vision"],
                          "parakeet_bin": str(root / "runtime" / "parakeet" / "bin" / "parakeet-mlx")},
        },
        "llm": {
            "claude": {"enabled": True, "bin": claude_bin() or "claude", "config_dir": str(root / "claude"), "tasks": [],
                       "skip_tasks": ["broll_describe", "photo_describe"] if look_here else [],
                       "models": {"script": s["ai"].get("claude_edit") or "sonnet", "default": s["ai"].get("claude_edit") or "sonnet",
                                  "cheap": s["ai"].get("claude_quick") or "haiku"},
                       "efforts": {"script": "medium", "default": "low", "cheap": "low"},
                       "cheap_tasks": ["broll_pick", "broll_queries", "video_title", "video_kind", "video_length", "hook_text", "key_points", "music_vibe",
                                       "broll_suitable", "clip_title", "style_merge", "broll_describe", "photo_describe", "gfx_place",
                                       "catalog_describe", "stock_queries", "video_chapters"],
                       "fallback": fallback, "fallback_on_limit": fallback},
            "ollama": {"base_url": url, "model": text, "vision_model": m["vision"], "fallback_models": []},
            "provider_chain": ["ollama"],
        },
        "catalog": {"roots": folders, "dir": str(pl["catalog"]),
                    "window": {"open": "always", "night": ["19:00", "07:00"]}.get(s["library"].get("when", "open")),
                    "describe_with": "account" if use != "local" and s["library"].get("describe_with") == "account" else "local",
                    "vision_model": m["vision"], "ollama_url": url},
        "paths": {"work": str(pl["work"] / "work"), "logs": str(root / "logs")},
    }


def local_zone() -> str:
    """This Mac's time zone name (the night window and dates follow it)."""
    try:
        return os.readlink("/etc/localtime").split("zoneinfo/", 1)[1]
    except (OSError, IndexError):
        return "UTC"


def write_overlay(root: Path | None = None) -> Path:
    import yaml
    root = Path(root or home())
    s = Settings(root).get()
    p = root / "overlay.yaml"
    p.write_text("# Made by FirstCut from settings.json: change the settings in the app, not here.\n"
                 + yaml.safe_dump(overlay(s, root), sort_keys=False))
    return p


def environment(root: Path | None = None) -> dict:
    """Environment for the engine and everything it starts: models land in the chosen folder, ffmpeg and our tools are found first."""
    root = Path(root or home())
    s = Settings(root).get()
    models = Path(s["places"]["models"])
    env = {"HF_HOME": str(models / "huggingface"), "OLLAMA_MODELS": str(models / "ollama"), "VIDEOBUILDER_HOME": str(root)}
    extra = [str(root / "runtime" / "bin"), os.environ.get("VIDEOBUILDER_BIN", "")]
    env["PATH"] = os.pathsep.join([p for p in extra if p] + [os.environ.get("PATH", "/usr/bin:/bin"), "/opt/homebrew/bin", "/usr/local/bin"])
    return env


# ---------------------------------------------------------------------------------------------- how it edits (per kind)
STYLE_SYSTEM = (
    "You keep a video editor's instructions as a short list of key points for {kind}. Rewrite the whole list after taking in what the "
    "person just wrote. Keep only concrete editing instructions (hook, pacing, length, captions, music, B-roll, framing, what to cut, "
    "tone), each one short imperative sentence under 110 characters, in plain words. Merge duplicates. When the new text contradicts an "
    "old point, keep the new one and drop the old. Leave out praise, chit-chat, personal details, and anything about other kinds of video. "
    "At most {max} points. Never invent a preference the person did not state. If the text is a skill or prompt written for Claude or ChatGPT, "
    "take only the editing preferences from it.")
STYLE_SCHEMA = {"type": "object", "properties": {"points": {"type": "array", "items": {"type": "string"}}}, "required": ["points"]}
KIND_WORDS = {"short": "short vertical clips (Reels, Shorts, TikTok) cut from a longer video",
              "long": "long-form videos edited from interviews and footage (YouTube, client and event videos)"}


class Styles:
    """What the editor follows, per kind of video: a short list of points the person can read, add to (in their own words, which the AI
    boils down) and remove from. Short clips never see the long-form list and the other way round."""

    def __init__(self, root: Path | None = None):
        self.dir = Path(root or home()) / "styles"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def _file(self, kind: str) -> Path:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        return self.dir / f"{kind}.json"

    def get(self, kind: str) -> dict:
        f = self._file(kind)                                     # (an unknown kind is an error, not an empty list)
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            d = {}
        return {"points": [str(p) for p in d.get("points") or []], "history": d.get("history") or [], "updated": d.get("updated"),
                "added": d.get("added") or []}

    def _save(self, kind: str, points: list[str], before: list[str], added: list[str]) -> dict:
        d = self.get(kind)
        hist = ([before] + d["history"])[:20]
        out = {"points": points, "history": hist, "updated": int(time.time()), "added": added}
        tmp = self._file(kind).with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=1))
        tmp.replace(self._file(kind))
        return self.get(kind)

    def add(self, kind: str, text: str, ai) -> dict:
        text = " ".join(str(text or "").split())[:20000]
        if not text:
            raise ValueError("write how you like your videos first")
        with self.lock:
            cur = self.get(kind)["points"]
            prompt = ("Current points:\n" + ("\n".join(f"- {p}" for p in cur) or "(none yet)")
                      + f"\n\nWhat the person just wrote:\n\"\"\"\n{text}\n\"\"\"\n\nReturn the whole new list.")
            res = ai.complete_json("style_merge", STYLE_SYSTEM.format(kind=KIND_WORDS[kind], max=MAX_POINTS), prompt, STYLE_SCHEMA)
            pts = tidy_points(res.data.get("points"))
            if not pts and cur:
                return self.get(kind)
            return self._save(kind, pts, cur, [p for p in pts if p not in cur])

    def remove(self, kind: str, index: int) -> dict:
        with self.lock:
            cur = self.get(kind)["points"]
            if not 0 <= index < len(cur):
                raise ValueError("no such point")
            return self._save(kind, cur[:index] + cur[index + 1:], cur, [])

    def undo(self, kind: str) -> dict:
        with self.lock:
            d = self.get(kind)
            if not d["history"]:
                return d
            out = {"points": d["history"][0], "history": d["history"][1:], "updated": int(time.time()), "added": []}
            self._file(kind).write_text(json.dumps(out, indent=1))
            return self.get(kind)

    def brief(self, kind: str) -> str:
        pts = self.get(kind)["points"]
        return ("How this person wants their videos edited (follow every point):\n" + "\n".join(f"- {p}" for p in pts)) if pts else ""


def tidy_points(raw) -> list[str]:
    if isinstance(raw, str):                                          # (one point given as text, not a list)
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    out, seen = [], set()
    for p in raw:
        if isinstance(p, bool) or not isinstance(p, (str, int, float)):   # (None and lists are no point; a number is kept as text)
            continue
        p = " ".join(str(p).split()).strip(" -•*")
        if not p:
            continue
        if len(p) > 160:
            p = p[:157].rsplit(" ", 1)[0] + "…"
        k = p.lower()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out[:MAX_POINTS]


def skill_text(name: str, data: bytes) -> str:
    """The text of an imported skill (Claude, ChatGPT) or style guide: a SKILL.md / .md / .txt file, or a .zip of a skill folder (its SKILL.md and other text
    files). Frontmatter and code blocks are dropped; what is left is read by the AI like anything the person writes."""
    texts: list[str] = []
    if name.lower().endswith(".zip") or data[:2] == b"PK":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                names = sorted(n for n in z.namelist() if n.lower().endswith((".md", ".txt")) and "__MACOSX" not in n)
                names.sort(key=lambda n: (not n.lower().endswith("skill.md"), n))
                left = 400_000                                           # read at most this much text: a tiny zip can unpack to gigabytes
                for n in names[:12]:
                    with z.open(n) as fh:
                        chunk = fh.read(min(left, 200_000))
                    texts.append(chunk.decode("utf-8", errors="replace"))
                    left -= len(chunk)
                    if left <= 0:
                        break
        except zipfile.BadZipFile as e:
            raise ValueError("that file isn't a readable zip: export the skill again, or import its SKILL.md") from e
    else:
        texts.append(data[:400_000].decode("utf-8", errors="replace"))
    out = []
    for t in texts:
        t = re.sub(r"\A---\n.*?\n---\n", "", t, flags=re.S)             # YAML frontmatter
        t = re.sub(r"```.*?```", "", t, flags=re.S)                      # code blocks: not editing preferences
        out.append(t.strip())
    return "\n\n".join(x for x in out if x)[:20000]


# ---------------------------------------------------------------------------------------------- AI sign-ins
class ClaudeLogin:
    """Sign the person's Claude plan in for editing decisions: install Claude Code if this Mac has none (its official installer), then
    `claude auth login` with this app's own login folder; the browser does the rest."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root or home())
        self.dir = self.root / "claude"
        self.state = {"status": "idle", "error": None}
        self.thread: threading.Thread | None = None

    def env(self) -> dict:
        return {**os.environ, "CLAUDE_CONFIG_DIR": str(self.dir)}

    def status(self) -> dict:
        exe = claude_bin()
        if self.state["status"] in ("installing", "signing in"):
            return dict(self.state)
        if not exe:
            return {"status": "not installed", "error": self.state.get("error")}
        try:
            r = subprocess.run([exe, "auth", "status", "--json"], capture_output=True, text=True, timeout=20, env=self.env())
            d = json.loads(r.stdout or "{}")
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            return {"status": "error", "error": str(e)[:200]}
        if d.get("loggedIn"):
            acct = d.get("account") if isinstance(d.get("account"), dict) else {}
            return {"status": "connected", "email": d.get("email") or acct.get("email"), "plan": d.get("subscriptionType") or d.get("plan")}
        return {"status": "disconnected", "error": self.state.get("error")}

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return

        def run():
            try:
                exe = claude_bin()
                if not exe:
                    self.state = {"status": "installing", "error": None}
                    r = subprocess.run(["/bin/bash", "-c", f"curl -fsSL {CLAUDE_INSTALL} | bash"], capture_output=True, text=True, timeout=900)
                    exe = claude_bin()
                    if not exe:
                        raise RuntimeError("Claude Code didn't install: " + (r.stderr or r.stdout)[-200:])
                self.dir.mkdir(parents=True, exist_ok=True)
                self.state = {"status": "signing in", "error": None}
                subprocess.run([exe, "auth", "login", "--claudeai"], capture_output=True, text=True, timeout=900, env=self.env())
                self.state = {"status": "idle", "error": None}
            except Exception as e:  # noqa: BLE001
                log.warning(f"Claude sign-in failed: {e}")
                self.state = {"status": "idle", "error": str(e)[:300]}
        self.thread = threading.Thread(target=run, name="claude-login", daemon=True)
        self.thread.start()

    def logout(self) -> None:
        exe = claude_bin()
        if exe:
            subprocess.run([exe, "auth", "logout"], capture_output=True, timeout=60, env=self.env())


# ---------------------------------------------------------------------------------------------- first-run setup
class Setup:
    """Downloads what this Mac needs, once, each part with its own progress. Safe to run again: finished parts are skipped.

        speech      Parakeet (its own small Python, made with the bundled `uv`) and its model
        ollama      the local model runner: the Ollama app if it is running, else our own copy (runtime/ollama)
        vision      the picture model that describes B-roll (always local)
        local       the editing model, only when the editing AI is "this Mac" or the fallback is on
        search      the footage search model (and its Python packages), only when there is a footage library
    """

    def __init__(self, root: Path | None = None):
        self.root = Path(root or home())
        self.runtime = self.root / "runtime"
        self.steps: dict[str, dict] = {}
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()

    def plan(self) -> list[dict]:
        s = Settings(self.root).get()
        m = local_models(ram_gb())
        text = s["ai"].get("local_model") or m["text"]
        want_local = s["ai"]["use"] == "local" or s["ai"].get("backup") == "local" or (s["ai"].get("backup", "auto") == "auto" and s["ai"].get("fallback"))
        lib = bool([f for f in s["library"]["folders"] if f])
        mus = bool([f for f in (s.get("music") or {}).get("folders", []) if f])
        steps = [
            {"id": "speech", "what": "Speech to text", "model": "Parakeet", "gb": 1.2, "why": "Writes down every word with its timing, on this Mac."},
            {"id": "ollama", "what": "Local model runner", "model": "Ollama", "gb": 0.2, "why": "Runs the picture model (and the local editing model)."},
            {"id": "vision", "what": "Looking at footage", "model": m["vision"], "gb": m["vision_gb"], "why": "Describes what each B-roll shot shows."},
            {"id": "local", "what": "Local editing AI", "model": text, "gb": m["text_gb"], "why": "Makes the editing decisions on this Mac.",
             "skip": None if want_local else "only needed for “On this Mac only” or the fallback"},
            {"id": "search", "what": "Footage search", "model": "SigLIP 2", "gb": 1.9, "why": "Lets you search your footage library in plain words.",
             "skip": None if lib else "only needed with a footage library"},
            {"id": "music", "what": "Music search", "model": "CLAP", "gb": MUSIC_GB, "why": "Lets you find songs in your music by describing them.",
             "skip": None if mus else "only needed with a music library"},
            {"id": "voices", "what": "Voice cleanup", "model": "DeepFilterNet", "gb": VOICE_GB, "why": "Takes room noise and echo out of noisy recordings, on this Mac.",
             "skip": None if any((s["defaults"].get(k) or {}).get("voices", "auto") != "never" for k in ("long", "short")) else "Clean up voices is set to Never"},
        ]
        for st in steps:
            cur = self.steps.get(st["id"]) or {}
            if st.get("skip"):
                st.update(state="skipped", pct=0.0, error=None)
            elif cur.get("state") in ("running", "failed"):
                st.update(state=cur["state"], pct=cur.get("pct", 0.0), error=cur.get("error"), stage=cur.get("stage"))
            elif self._done(st["id"], s):
                st.update(state="done", pct=1.0, error=None)
            else:
                st.update(state="waiting", pct=0.0, error=None)
        return steps

    # -- what is already there
    def _done(self, sid: str, s: dict) -> bool:
        m = local_models(ram_gb())
        if sid == "speech":
            return (self.runtime / "parakeet" / "bin" / "parakeet-mlx").exists() and (self.runtime / "parakeet" / ".model-ready").exists()
        if sid == "ollama":
            return _answers("http://127.0.0.1:11434") or (self.runtime / "ollama" / "ollama").exists()
        if sid in ("vision", "local"):
            name = m["vision"] if sid == "vision" else (s["ai"].get("local_model") or m["text"])
            return name in self._pulled()
        if sid == "search":
            try:
                import transformers  # noqa: F401
            except ImportError:
                return False
            return (Path(s["places"]["models"]) / "huggingface" / ".siglip-ready").exists()
        if sid == "music":
            try:
                import transformers  # noqa: F401
            except ImportError:
                return False
            return (Path(s["places"]["models"]) / "huggingface" / ".clap-ready").exists()
        if sid == "voices":
            exe = self.runtime / "bin" / "deep-filter"
            return exe.is_file() and os.access(exe, os.X_OK)
        return False

    def _pulled(self) -> set[str]:
        import urllib.request
        url = ollama_url(Settings(self.root).get())
        try:
            with urllib.request.urlopen(url + "/api/tags", timeout=3) as r:
                names = [m["name"] for m in json.loads(r.read()).get("models", [])]
        except Exception:  # noqa: BLE001
            return set()
        return set(names) | {n.removesuffix(":latest") for n in names}

    # -- running it
    def start(self) -> bool:
        if self.thread and self.thread.is_alive():
            return False
        gone = missing_drive(Settings(self.root).get()["places"]["models"])
        if gone:
            raise ValueError(f"the drive “{gone}” with your AI models isn't plugged in: plug it in, or choose another folder in Settings › Where things go")
        (self.root / WANTED).parent.mkdir(parents=True, exist_ok=True)
        (self.root / WANTED).write_text(str(time.time()))               # quit before it finished: the next launch carries on
        self.thread = threading.Thread(target=self._run, name="setup", daemon=True)
        self.thread.start()
        return True

    def running(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    def resume(self) -> bool:
        """At launch: downloads that were going when the app quit carry on (a part-downloaded model continues where it stopped)."""
        if not (self.root / WANTED).exists() or self.running():
            return False
        if all(st["state"] in ("done", "skipped") for st in self.plan()):
            (self.root / WANTED).unlink(missing_ok=True)
            return False
        try:
            return self.start()
        except ValueError as e:                                          # the models drive isn't plugged in: the page says so
            log.warning(f"setup not resumed: {e}")
            return False

    def _set(self, sid: str, **kw) -> None:
        with self.lock:
            self.steps[sid] = {**self.steps.get(sid, {}), **kw}

    def _run(self) -> None:
        from . import net
        for st in self.plan():
            if st["state"] in ("done", "skipped"):
                continue
            for attempt in range(3):
                while not net.online(max_age=10):                         # offline: wait for the internet, then carry on by itself
                    self._set(st["id"], state="running", stage="waiting for the internet", error=None)
                    time.sleep(10)
                self._set(st["id"], state="running", pct=0.0, error=None, stage="")
                try:
                    getattr(self, f"_do_{st['id']}")(st)
                    self._set(st["id"], state="done", pct=1.0)
                    break
                except Exception as e:  # noqa: BLE001 - the next parts still install; this one says what went wrong
                    if attempt < 2 and net.looks_offline(e):              # the connection dropped: wait for it and go again
                        net.changed()
                        continue
                    log.exception(f"setup step {st['id']} failed")
                    self._set(st["id"], state="failed", error=str(e)[:300])
                    break
        if all(st["state"] in ("done", "skipped") for st in self.plan()):
            (self.root / WANTED).unlink(missing_ok=True)

    def _uv(self) -> str:
        exe = os.environ.get("VIDEOBUILDER_UV") or shutil.which("uv")
        if not exe:
            raise RuntimeError("the app's installer (uv) is missing: reinstall FirstCut")
        return exe

    def _do_speech(self, st: dict) -> None:
        env_dir = self.runtime / "parakeet"
        if not (env_dir / "bin" / "parakeet-mlx").exists():
            self._set("speech", stage="installing Parakeet")
            shutil.rmtree(env_dir, ignore_errors=True)                    # (half made when the app quit: start it again)
            subprocess.run([self._uv(), "venv", "--python", "3.12", str(env_dir)], check=True, capture_output=True, timeout=900)
            subprocess.run([self._uv(), "pip", "install", "--python", str(env_dir / "bin" / "python"), "parakeet-mlx"],
                           check=True, capture_output=True, timeout=1800)
        self._set("speech", pct=0.3, stage="downloading the speech model")
        self._hf_download(PARAKEET_MODEL, "speech", 0.3, 1.0, python=str(env_dir / "bin" / "python"))
        (env_dir / ".model-ready").write_text(PARAKEET_MODEL)

    def _hf_download(self, repo: str, sid: str, lo: float, hi: float, python: str | None = None) -> None:
        """Download a Hugging Face model into the chosen models folder (HF_HOME), with progress from the folder's size."""
        s = Settings(self.root).get()
        hf = Path(s["places"]["models"]) / "huggingface"
        hf.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "HF_HOME": str(hf), "HF_HUB_DISABLE_TELEMETRY": "1"}
        code = f"from huggingface_hub import snapshot_download; snapshot_download({repo!r})"
        proc = subprocess.Popen([python or sys.executable, "-c", code], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        expect = float(st_gb(sid)) * 1e9
        mine = hf / "hub" / ("models--" + repo.replace("/", "--"))           # (only this model: the folder holds the others too)
        while proc.poll() is None:
            got = sum(f.stat().st_size for f in mine.rglob("*") if f.is_file()) if mine.is_dir() else 0
            self._set(sid, pct=lo + (hi - lo) * min(0.99, got / max(1.0, expect)))
            time.sleep(2)
        if proc.returncode:
            raise RuntimeError("download failed: " + (proc.stderr.read() or b"").decode(errors="replace")[-200:])

    def _do_ollama(self, st: dict) -> None:
        if _answers("http://127.0.0.1:11434"):
            return
        dest = self.runtime / "ollama"
        if not (dest / "ollama").exists():
            import urllib.request
            dest.mkdir(parents=True, exist_ok=True)
            tgz = dest / "ollama.tgz.part"
            with urllib.request.urlopen(OLLAMA_URL, timeout=60) as r, tgz.open("wb") as fh:
                total = int(r.headers.get("Content-Length") or 0)
                got = 0
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    fh.write(b)
                    got += len(b)
                    if total:
                        self._set("ollama", pct=0.9 * got / total)
            with tarfile.open(tgz) as t:
                t.extractall(dest, filter="data")
            tgz.unlink(missing_ok=True)
            exe = next((p for p in dest.rglob("ollama") if p.is_file()), None)
            if exe is None:
                raise RuntimeError("the Ollama download had no ollama program in it")
            if exe != dest / "ollama":
                (dest / "ollama").symlink_to(exe)
            (dest / "ollama").chmod(0o755)
        OllamaRunner(self.root).ensure()

    def _pull(self, sid: str, name: str) -> None:
        import urllib.request
        OllamaRunner(self.root).ensure()
        url = ollama_url(Settings(self.root).get())
        req = urllib.request.Request(url + "/api/pull", data=json.dumps({"name": name, "stream": True}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            for line in r:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("error"):
                    raise RuntimeError(d["error"])
                if d.get("total"):
                    self._set(sid, pct=min(0.99, float(d.get("completed") or 0) / float(d["total"])), stage=d.get("status"))

    def _do_vision(self, st: dict) -> None:
        self._pull("vision", st["model"])

    def _do_local(self, st: dict) -> None:
        self._pull("local", st["model"])

    def _do_voices(self, st: dict) -> None:
        """DeepFilterNet's program for Apple silicon (voiceclean.URL, about 28 MB, model inside) into runtime/bin. No checksum is
        published, so it must be exactly the release's size and arm64 code, and must start, before it is kept."""
        import urllib.request
        from .voiceclean import SIZE, URL
        dest = self.runtime / "bin" / "deep-filter"
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name("deep-filter.part")
        with urllib.request.urlopen(URL, timeout=60) as r, part.open("wb") as fh:
            got = 0
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                fh.write(b)
                got += len(b)
                self._set("voices", pct=0.95 * got / SIZE, stage="downloading DeepFilterNet")
        head = part.read_bytes()[:8]
        if part.stat().st_size != SIZE or head[:4] != bytes.fromhex("cffaedfe") or head[4:8] != bytes.fromhex("0c000001"):
            part.unlink(missing_ok=True)
            raise RuntimeError("the voice cleanup download wasn't the expected program; try Run setup again later")
        part.chmod(0o755)
        r = subprocess.run([str(part), "--version"], capture_output=True, timeout=60)
        if r.returncode:
            part.unlink(missing_ok=True)
            raise RuntimeError("the voice cleanup program didn't start on this Mac")
        part.replace(dest)

    def _do_music(self, st: dict) -> None:
        try:
            import transformers  # noqa: F401
        except ImportError:
            self._set("music", stage="installing the search engine (PyTorch)")
            subprocess.run([self._uv(), "pip", "install", "--python", sys.executable, "torch", "transformers>=4.56", "sentencepiece", "protobuf"],
                           check=True, capture_output=True, timeout=3600)
            import importlib
            importlib.invalidate_caches()
        self._set("music", pct=0.3, stage="downloading the music search model")
        try:
            from .musiclib import MODEL
        except ImportError:
            MODEL = MUSIC_MODEL
        self._hf_download(MODEL, "music", 0.3, 1.0)
        s = Settings(self.root).get()
        (Path(s["places"]["models"]) / "huggingface" / ".clap-ready").write_text(MODEL)

    def _do_search(self, st: dict) -> None:
        try:
            import transformers  # noqa: F401
        except ImportError:
            self._set("search", stage="installing the search engine (PyTorch)")
            subprocess.run([self._uv(), "pip", "install", "--python", sys.executable, "torch", "transformers>=4.56", "sentencepiece", "protobuf"],
                           check=True, capture_output=True, timeout=3600)
            import importlib
            importlib.invalidate_caches()                        # the running engine sees the new packages without a restart
        self._set("search", pct=0.4, stage="downloading the search model")
        from .catalog import MODEL
        self._hf_download(MODEL, "search", 0.4, 1.0)
        s = Settings(self.root).get()
        (Path(s["places"]["models"]) / "huggingface" / ".siglip-ready").write_text(MODEL)


VOICE_GB = 0.03                           # DeepFilterNet's program, model inside (voiceclean.SIZE)


def st_gb(sid: str) -> float:
    m = local_models(ram_gb())
    return {"speech": 1.25, "search": 1.5, "music": MUSIC_GB, "vision": m["vision_gb"], "local": m["text_gb"], "voices": VOICE_GB}.get(sid, 1.0)


class OllamaRunner:
    """Our own `ollama serve` (when the Mac has no Ollama running): models in the chosen models folder, on its own port, stopped with
    the app."""
    proc: subprocess.Popen | None = None

    def __init__(self, root: Path | None = None):
        self.root = Path(root or home())

    def ensure(self) -> str:
        if _answers("http://127.0.0.1:11434"):
            return "http://127.0.0.1:11434"
        url = f"http://127.0.0.1:{OWN_OLLAMA_PORT}"
        if _answers(url):
            return url
        exe = self.root / "runtime" / "ollama" / "ollama"
        if not exe.exists():
            raise RuntimeError("Ollama isn't installed yet (Setup does it)")
        s = Settings(self.root).get()
        models = Path(s["places"]["models"]) / "ollama"
        models.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "OLLAMA_HOST": f"127.0.0.1:{OWN_OLLAMA_PORT}", "OLLAMA_MODELS": str(models)}
        logf = (self.root / "logs")
        logf.mkdir(parents=True, exist_ok=True)
        OllamaRunner.proc = subprocess.Popen([str(exe), "serve"], env=env, stdout=(logf / "ollama.log").open("ab"), stderr=subprocess.STDOUT)
        for _ in range(60):
            if _answers(url):
                return url
            time.sleep(0.5)
        raise RuntimeError("Ollama didn't start (see logs/ollama.log)")

    @classmethod
    def stop(cls) -> None:
        if cls.proc and cls.proc.poll() is None:
            cls.proc.terminate()


# ---------------------------------------------------------------------------------------------- project folders
EDITORS = [  # (id, name, app name patterns, how it takes our project: "opens" the .fcpxml itself, or "imports" it from its menu)
    ("fcp", "Final Cut Pro", ("Final Cut Pro*.app",), "opens"),
    ("resolve", "DaVinci Resolve", ("DaVinci Resolve*.app", "DaVinci Resolve*/DaVinci Resolve*.app"), "imports"),
]


def editors() -> list[dict]:
    """The video editors on this Mac that can take a FirstCut project (the Final Cut XML), Final Cut first."""
    import glob
    out = []
    for eid, name, pats, how in EDITORS:
        for base in ("/Applications", str(Path.home() / "Applications")):
            hits = sorted(h for pat in pats for h in glob.glob(os.path.join(base, pat)))
            if hits:
                out.append({"id": eid, "name": name, "app": hits[0], "how": how})
                break
    return out


# ---------------------------------------------------------------------------------------------- API keys (in the Mac's Keychain)
KEY_SERVICE = "FirstCut API keys"
KEYS = {"pixabay": "PIXABAY_API_KEY", "pexels": "PEXELS_API_KEY", "epidemic": None}   # (the stock keys reach the engine as its environment)


def key_get(name: str) -> str:
    """A key the person saved (Settings › Online sources), from the login Keychain; "" when there is none. Never logged."""
    if name not in KEYS or sys.platform != "darwin":
        return ""
    r = subprocess.run(["/usr/bin/security", "find-generic-password", "-a", name, "-s", KEY_SERVICE, "-w"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def key_set(name: str, value: str) -> None:
    """Save (or with "" remove) a key in the Keychain, and give the engine the stock ones straight away."""
    if name not in KEYS:
        raise ValueError("unknown key")
    value = (value or "").strip()
    if value and not re.fullmatch(r"[A-Za-z0-9_\-.]{8,200}", value):
        raise ValueError("that doesn't look like a key: copy it again from the site (letters and numbers only)")
    if value:
        r = subprocess.run(["/usr/bin/security", "add-generic-password", "-U", "-a", name, "-s", KEY_SERVICE, "-w", value], capture_output=True, text=True)
        if r.returncode:
            raise ValueError("the Mac's Keychain didn't take it: unlock the Keychain and try again")
    else:
        subprocess.run(["/usr/bin/security", "delete-generic-password", "-a", name, "-s", KEY_SERVICE], capture_output=True, text=True)
    if KEYS[name]:
        if value:
            os.environ[KEYS[name]] = value
        else:
            os.environ.pop(KEYS[name], None)


def keys_into_env() -> None:
    """At engine start: the stock-footage keys from the Keychain into the engine's environment (clipfarm/stock.py reads them there)."""
    for name, env in KEYS.items():
        if env and not os.environ.get(env):
            v = key_get(name)
            if v:
                os.environ[env] = v


def missing_drive(path: str | Path) -> str | None:
    """The drive's name when `path` is on an external drive that isn't plugged in (/Volumes/<name> is not there), else None."""
    parts = Path(str(path)).expanduser().parts
    if len(parts) > 2 and parts[1] == "Volumes" and not Path("/Volumes", parts[2]).is_dir():
        return parts[2]
    return None


def project_folder(videos: str | Path, title: str) -> Path:
    """A new folder of its own for one project in the videos folder ("Title", else "Title (2)", ...)."""
    from .library import safe
    base = Path(videos).expanduser()
    name = safe(title or "Untitled", 70) or "Untitled"
    p, n = base / name, 2
    while p.exists():
        p = base / f"{safe(title or 'Untitled', 64)} ({n})"
        n += 1
    p.mkdir(parents=True)
    return p


def link_raw(folder: Path, sources: list[str | Path]) -> int:
    """`Raw footage/` in a project folder: links to the original files (Final Cut's project already points at them where they are; these
    are for finding them). Nothing is copied."""
    raw = Path(folder) / "Raw footage"
    n = 0
    for src in sources:
        src = Path(src)
        if not src.exists():
            continue
        raw.mkdir(exist_ok=True)
        dest, k = raw / src.name, 2
        while dest.exists() or dest.is_symlink():
            if dest.is_symlink() and os.readlink(dest) == str(src):
                break
            dest = raw / f"{src.stem} ({k}){src.suffix}"
            k += 1
        else:
            os.symlink(src, dest)
            n += 1
    return n
