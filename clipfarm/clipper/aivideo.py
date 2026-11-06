"""AI videos: long-form projects (talking-head / screen footage -> one finished video, plus a Final Cut Pro project).

You upload raw footage and type notes; the AI transcribes it, picks the takes, cuts dead air, adds captions and music, and
makes a Final Cut project you can keep editing. Everything about one video is a *project*:

    <ai_videos_dir>/<2026-10-12 Title>/
        Raw footage/source.mp4         your footage joined in the order shot
        Output/<Title> v1.mp4 (+ .srt) every finished version (v2 after you ask for a change ...)
        Final Cut/<Title>/             the Final Cut Pro project (import the .fcpxml)
        Notes/                         brief.txt, plan.md (what was kept and why), transcript.txt, style_rules.md

The working copy stays on the machine that renders (data/clipper/<job>); the archive above is copied to the big drive.
If that drive is on another machine (clipper.ai_videos.host, default the posting server) it is sent over SSH, and only when
the drive is really mounted. Until then the project simply says "waiting for the drive".

Style rules (data/ai_videos/style_rules.md) are the editor's memory: after you react to a video, "Save what you learned"
turns your feedback into short rules that every later edit reads first.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from ..core.log import get_logger
from .library import drive_ready, safe

log = get_logger("clipper.aivideo")
DEFAULT_DIR = "~/Movies/FirstCut"
RULES_MAX = 12000                                    # characters of style rules given to the editor (whole rules only)
LEARNED_MAX = 400                                    # one learned rule at most this long (cut at a word, never mid-word)

RULE_SYSTEM = (
    "You maintain a video editor's style rules. The creator reacted to a finished video. Turn their reaction into 1-3 short, "
    "specific, testable rules for future edits (e.g. 'Cut every pause longer than 0.4 s', 'Open with the strongest sentence "
    "as a hook before any intro'). One line each, imperative, no names or one-off details, no duplicates of the existing "
    "rules. If the reaction only praises, return an empty list.")
RULE_SCHEMA = {"type": "object", "properties": {"rules": {"type": "array", "items": {"type": "string"}}},
               "required": ["rules"], "additionalProperties": False}
CHANGE_RULE = (
    " This time the reaction is a CHANGE the creator asked for on one video. Only turn it into a rule if it shows a general preference "
    "for future videos (for example 'the music is always too loud', 'hooks should ask a question'). A request about this one video - its "
    "length, a particular line, sentence, moment or song - is a one-off: return an empty list.")


# What the editor has been taught so far (from the creator's own feedback). They sit in the same rules file the editor reads before
# EVERY edit, next to anything learned later, and can be changed on the page. Each one says when it applies, so an ad's rules don't
# leak into a vlog.
DEFAULT_RULES = [
    "ADVERTISEMENTS: an ad is not an interview and not a YouTube video. Make it mostly B-roll (store, products, people, the experience) with the "
    "guest's lines running underneath as voiceover; change the shot every 1.5 to 3 seconds; show the speaker's face only when a line needs it.",
    "ADVERTISEMENTS: every spoken line must be a selling point or proof (what makes the business different, benefits, expertise, trust, results). "
    "No anecdotes or stories about individual products or customers. Say the point, then move on.",
    "ADVERTISEMENTS: finish on a COMPLETE closing line, then a closing B-roll shot with an end card (logo, business name, short call to action). "
    "Never end mid-sentence.",
    "ADVERTISEMENTS: use energetic, driving music (about 110 to 140 BPM, upbeat or epic), never a chill or thinking track, and cut on the beat.",
    "INTERVIEWS: use only what the person being interviewed says. Never use the interviewer's questions or prompts as content.",
    "Never change the colour of the footage. Footage shot in log stays flat as shot; the creator grades it in Final Cut. The finished video is a "
    "preview of the edit, not a grade.",
    "On-screen graphics are lower thirds: only as wide as their text, bottom-left above the captions (the hook card too), never across the "
    "middle of the picture or over the subject's face.",
    "Every graphic that appears in the finished video must also exist in the Final Cut project as an editable title, and each clip's audio must stay "
    "with its own clip and stay in sync.",
    "Finish loud enough: about -12 LUFS for an ad, -14 for a story. Check the finished file (picture and sound the same length, no frozen picture) "
    "BEFORE handing it over, and hand over nothing that is unfinished.",
    "A multicam shoot uses ALL its camera angles: switch at natural moments every 9 to 12 seconds (not more often: too many cuts feels frantic), show the guest, and keep the sound continuous "
    "from the best recording (the recorder if there is one; otherwise the camera whose own audio is measurably the cleanest, used for EVERY shot, "
    "never the sound of whichever camera is showing).",
    "THE TWELVE-SECOND RULE: never hold the same picture longer than 12 seconds without a change (the other camera, B-roll, or a photo). Don't cut "
    "more often than that: the creator found 5 s and 9 s far too fast. Put B-roll and photos only in RELEVANT spots, where what is shown matches "
    "what is being said, and never end a video mid-sentence: the closing point of the script is always heard in full.",
    "FOLLOW THE CREATOR'S SCRIPT: when the notes lay out the video (an outline or numbered steps), that outline is a REQUIREMENT. Tell every step, in "
    "that order, with the guest's own words, and check afterwards that each step is there. Report any step the footage cannot support.",
    "PHOTOS in a B-roll folder are used, every one of them, as animated stills: a slow Ken Burns zoom or pan, or a tilted "
    "polaroid over a soft blurred copy of itself (no film-strip border: the creator dislikes it). Keep the photo's own colours. They also go into "
    "the Final Cut project as clips.",
    "Never use the same line twice in one video (the hook included): a sentence that is the opening hook is not repeated later on.",
    "Always put a copy of the Final Cut XML (.fcpxml) in the root of the folder the creator gave access to, next to the Output folder.",
]


RETIRED_RULES = ["THE FIVE-SECOND RULE", "THE NINE-SECOND RULE", "every 6 to 9 seconds", "switch at natural moments every 3 to 6", "switch at natural moments every 5 to 12", "film-strip border",
                 "PHOTOS in a B-roll folder are used, every one of them, as animated stills: a slow Ken Burns zoom or pan, a film",
                 "Hook text top-left"]                # each is removed ONCE (rules_state.json), and never from a rule that is current


def _norm(t: str) -> str:
    return re.sub(r"\W+", " ", str(t).lower()).strip()


def _rule_body(line: str) -> str:
    return line.strip().lstrip("-*").strip()


def _rule_key(rule: str) -> str:
    import hashlib
    return hashlib.sha1(_norm(rule).encode()).hexdigest()[:12]


def _clip(rule: str, limit: int = LEARNED_MAX) -> str:
    """A rule on one line, at most `limit` characters, cut at a word (never mid-word)."""
    r = " ".join(str(rule).split()).strip(" -*")
    if len(r) <= limit:
        return r
    return r[:limit].rsplit(" ", 1)[0].rstrip(",;:") + " …"


def root_dir(cfg) -> Path:
    return Path(cfg.get("clipper.ai_videos.dir", DEFAULT_DIR))


def drive_host(cfg) -> str | None:
    """user@host of the machine whose disk holds the archive; None = the drive is on this machine."""
    if cfg.get("clipper.ai_videos.local", False):
        return None
    host = cfg.get("clipper.ai_videos.host")
    if not host:
        from .auto import _target
        tgt = _target(cfg)                                  # the posting server (set on the M2; absent on the server itself)
        host = tgt[0] if tgt else None
    return host or None



def _kind(v) -> str:
    """The kind of video from the form: ad / promo / story / business / youtube, else 'auto' (the AI reads the notes)."""
    from .localproj import KINDS
    return v if v in KINDS else "auto"


def _graphics(v) -> str:
    """'all' every kind of graphic | 'title' the animated title card | '' none."""
    return "all" if v == "all" else ("title" if v in (True, "title", "1", "true", "on") else "")


class Projects:
    def __init__(self, cfg, store, engine=None):
        self.cfg, self.store, self.engine = cfg, store, engine
        self.dir = cfg.root / cfg.get("clipper.ai_videos.data_dir", "data/ai_videos")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.sync_default_rules()                         # what the editor has already been taught (each default once, whole)

    # ------------------------------------------------------------------ storage
    def _file(self, pid: str) -> Path:
        return self.dir / pid / "project.json"

    def get(self, pid: str) -> dict:
        p = self._file(pid)
        if not p.exists():
            raise KeyError(pid)
        with self.lock:
            return json.loads(p.read_text())

    def _save(self, proj: dict) -> dict:
        with self.lock:
            p = self._file(proj["id"])
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(proj, indent=1))
            tmp.replace(p)
        return proj

    def update(self, pid: str, **fields) -> dict:
        with self.lock:
            proj = self.get(pid)
            proj.update(fields)
            proj["updated"] = time.time()
            return self._save(proj)

    def create_local(self, folder: str, title: str, notes: str, opts: dict, default_options: dict, extra_audio: list[str] | None = None,
                     out_dir: str | None = None) -> dict:
        """A project that works IN PLACE on a folder of this computer: no upload, no copy. With `out_dir` the results (and the working
        files) go to <out_dir>/<title>/ and the footage folder is only read."""
        p = Path(str(folder)).expanduser()
        if not p.is_dir():
            raise ValueError(f"{p} is not a folder on this computer")
        title = " ".join((title or "").split())[:90] or p.name
        home = None
        if out_dir and str(out_dir).strip():
            o = Path(str(out_dir).strip()).expanduser()
            if not o.is_absolute():
                raise ValueError(f"{o} is not a full folder path")
            taken = set()                                                 # every video gets a folder of its own: never one that is on disk
            for i in self.ids():                                          # already (even empty: a project waiting to start made it) or
                try:                                                      # that another project points to
                    taken.add(str(Path(self.get(i).get("home") or "")))
                except (KeyError, OSError, ValueError):
                    pass
            home = o / safe(title, 70)
            n = 2
            while home.exists() or str(home) in taken:
                home = o / f"{safe(title, 64)} ({n})"
                n += 1
            home.mkdir(parents=True)
        options = {**default_options, "mode": "polish", "fcp": True, "music": "" if opts.get("music") is None else str(opts.get("music")),         # no music unless asked: the library is all one well-known catalogue and ruins a video
                   
                   "caption_style": opts.get("caption_style") or "clean", "pace": opts.get("pace") or "natural",
                   "project": True, "clips": 0, "graphics": "all", "transcriber": opts.get("transcriber") or "parakeet"}
        job = self.store.create({"local": str(p)}, options)
        self.store.update(job["id"], status="draft", stage="Ready - press Make the video", title=title)
        pid = f"ai-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
        proj = {"id": pid, "mode": "local", "folder": str(p), "title": title, "notes": (notes or "").strip()[:3000] or _read_notes(p),
                "job": job["id"], "created": time.time(), "updated": time.time(), "files": [], "version": 0, "versions": [],
                "archive": {"state": "local", "path": str(home or p)}, "extra_audio": _paths(extra_audio), "home": str(home) if home else None}
        return self._save(proj)

    def prepare_local(self, pid: str, body: dict) -> dict:
        """Take the form's settings for a local project (before it is run)."""
        proj = self.get(pid)
        job = self.store.get(proj["job"])
        if "notes" in body or "title" in body:
            proj = self.set_notes(pid, str(body.get("notes", proj["notes"])), body.get("title"))
        if "extra_audio" in body:
            proj = self.update(pid, extra_audio=_paths(body["extra_audio"]))
        opts = dict(job["options"])
        if body.get("music") is not None:
            m = body["music"]
            opts["music"] = "auto" if m is True else ("" if m in (False, None, "", "off", "none") else str(m)[:20])
        for k in ("caption_style", "pace", "transcriber"):
            if body.get(k):
                opts[k] = str(body[k])
        if "library" in body:                                            # B-roll from the footage catalog (when this Mac has one)
            opts["library"] = bool(body["library"])
        if "stock" in body:                                              # stock B-roll (Pixabay / Pexels) for what your own footage doesn't show
            opts["stock"] = bool(body["stock"])
        if "kind" in body:                                               # ad / promo / story; "auto" = the AI reads the notes
            opts["kind"] = _kind(body["kind"])
        if "graphics" in body:
            g = body["graphics"]
            opts["graphics"] = _graphics(g)
        if "gfx_style" in body:                                          # the look the person picked ("auto": matched to the footage)
            from .graphics import THEMES
            opts["gfx_style"] = body["gfx_style"] if body["gfx_style"] in THEMES else "auto"
        if "long_minutes" in body:
            try:
                opts["long_minutes"] = max(0.0, min(600.0, float(body.get("long_minutes") or 0)))
            except (TypeError, ValueError):
                opts["long_minutes"] = 0.0
        if "song" in body:                                               # a song dropped on the project: this one, as it is
            song = str(body.get("song") or "").strip()
            if song:
                from ..core.music import AUDIO_EXT
                sp = Path(song).expanduser()
                if not sp.is_absolute() or not sp.is_file() or sp.suffix.lower() not in AUDIO_EXT | {".aif"}:
                    raise ValueError("that song isn't an audio file on this Mac (mp3, wav, aiff, m4a...)")
                opts["song"] = str(sp)
                if not opts.get("music"):                                 # (a song you chose means: music, please)
                    opts["music"] = "auto"
            else:
                opts.pop("song", None)
        if "keep_cut" in body:                                           # "Add B-roll to a video": that video stays as it is, B-roll goes over it
            keep = bool(body.get("keep_cut"))
            aroll = str(body.get("aroll") or "").strip()
            if keep:
                if not aroll or Path(aroll).name != aroll or aroll.startswith(".") or not (Path(proj["folder"]) / aroll).is_file():
                    raise ValueError("choose the video to add B-roll to (it must be in the footage folder)")
                opts["keep_cut"], opts["aroll"] = True, aroll
            else:
                opts.pop("keep_cut", None)
                opts.pop("aroll", None)
        opts["prompt"] = self.brief(proj)
        self.store.update(job["id"], options=opts, title=proj["title"])
        return self.get(pid)

    def delete(self, pid: str) -> None:
        """Remove the project and its working files here. The copy on the drive is never touched."""
        proj = self.get(pid)
        try:
            self.store.delete(proj["job"])
        except KeyError:
            pass
        shutil.rmtree(self.dir / pid, ignore_errors=True)

    def ids(self) -> list[str]:
        return sorted((p.parent.name for p in self.dir.glob("*/project.json")), reverse=True)

    def list(self) -> list[dict]:
        return [self.view(self.get(i)) for i in self.ids()]

    def view(self, proj: dict) -> dict:
        """The project joined with its render job, for the page."""
        out = dict(proj)
        try:
            job = self.store.get(proj["job"])
        except KeyError:
            out["job_state"] = None
            return out
        clip = (job.get("clips") or [None])[0]
        out["options"] = {k: job["options"].get(k) for k in ("music", "caption_style", "pace", "long_minutes", "graphics", "gfx_style", "kind", "transcriber", "library", "keep_cut", "aroll", "song")}
        if proj.get("mode") == "local":                                   # what the page can offer: back to the first cut / render the saved edit again
            out["saved_to"] = proj.get("home") or proj["folder"]
            work = Path(out["saved_to"]) / ".aivideo"
            out["can_revert"] = (work / "plan_v1.json").exists()
            try:
                plan = json.loads((work / "plan.json").read_text())
                out["pending_render"] = bool(plan.get("pending_render"))
                out["awaiting_approval"] = bool(plan.get("awaiting_approval"))
                out["script_map"] = plan.get("script_map") if out["awaiting_approval"] else None
                if out["script_map"]:                                      # how long it will be, from the lines on the review screen
                    out["review"] = {"length_s": round(sum(ln.get("end", 0) - ln.get("start", 0) for st_ in out["script_map"] for ps in st_["passages"] for ln in ps["lines"]), 1),
                                     "target_s": plan.get("target_s"), "edited": bool(plan.get("flow_edited"))}
            except (OSError, ValueError, KeyError, TypeError):             # (a plan saved before the lines carried their times)
                out["pending_render"] = False
                out["awaiting_approval"], out["script_map"] = False, None
        out["job_state"] = {"status": job["status"], "stage": job.get("stage"), "progress": job.get("progress", 0.0), "detail": job.get("detail") or "",
                            "error": job.get("error"), "title": job.get("title"), "started": (job.get("run") or {}).get("at")}
        if clip:
            out["clip"] = {k: v for k, v in clip.items() if k not in ("edit", "moment", "transcript")}
        return out

    # ------------------------------------------------------------------ style rules
    def rules_path(self) -> Path:
        return self.dir / "style_rules.md"

    def rules(self) -> str:
        p = self.rules_path()
        return p.read_text().strip() if p.exists() else ""

    def _rules_state_path(self) -> Path:
        return self.dir / "rules_state.json"

    def sync_default_rules(self) -> None:
        """Bring the rules file up to date with what the editor has been taught, ONCE per rule:
          - a taught rule that an older version cut off mid-word (it kept only 220 characters) is put back whole;
          - a retired rule is removed once - and never a line that is one of the current rules;
          - a taught rule is added the first time only: if you rewrite or delete it on the page, it stays the way you left it.
        Not in the app (clipper.ai_videos.default_rules: false): there the rules are the person's own long-form points."""
        if self.cfg.get("clipper.ai_videos.default_rules", True) is False:
            return
        with self.lock:
            sp = self._rules_state_path()
            try:
                state = json.loads(sp.read_text())
            except (OSError, ValueError):
                state = {}
            offered, retired = set(state.get("offered") or []), set(state.get("retired") or [])
            lines = self.rules().splitlines()
            before = list(lines)
            current = {_norm(r) for r in DEFAULT_RULES}
            for k, ln in enumerate(lines):                               # 1. put cut-off rules back whole
                nb = _norm(_rule_body(ln))
                if len(nb) >= 40 and nb not in current:
                    whole = next((r for r in DEFAULT_RULES if _norm(r).startswith(nb)), None)
                    if whole:
                        lines[k] = f"- {whole}"
            for frag in RETIRED_RULES:                                   # 2. retire replaced rules, once each
                if frag in retired:
                    continue
                lines = [ln for ln in lines if frag.lower() not in ln.lower() or _norm(_rule_body(ln)) in current]
                retired.add(frag)
            have = {_norm(_rule_body(ln)) for ln in lines}
            for r in DEFAULT_RULES:                                      # 3. a rule never offered before is added (once)
                if _rule_key(r) in offered:
                    continue
                if _norm(r) not in have:
                    if not any(ln.strip() for ln in lines):
                        lines = ["# Editing style rules (learned from your feedback)", ""]
                    lines.append(f"- {r}")
                offered.add(_rule_key(r))
            if lines != before:
                self.rules_path().write_text("\n".join(lines).rstrip() + "\n")
            tmp = sp.with_suffix(".tmp")
            tmp.write_text(json.dumps({"offered": sorted(offered), "retired": sorted(retired)}, indent=1))
            tmp.replace(sp)

    def add_rules(self, new: list[str], limit: int = LEARNED_MAX) -> list[str]:
        """Add rules that aren't there yet (each on one line, whole - a long one is cut at a word, never mid-word)."""
        with self.lock:
            have = self.rules()
            have_l = {_norm(_rule_body(line)) for line in have.splitlines()}
            add = []
            for r in new:
                r = _clip(r, limit)
                if r and _norm(r) not in have_l:
                    add.append(r)
                    have_l.add(_norm(r))
            if add:
                body = (have + "\n" if have else "# Editing style rules (learned from your feedback)\n\n") + "\n".join(f"- {r}" for r in add) + "\n"
                self.rules_path().write_text(body)
            return add

    def rules_for_brief(self, budget: int = RULES_MAX) -> str:
        """The rules the editing AI reads, as WHOLE lines: every taught rule, then as many learned ones as fit - the newest first, so an
        old learned rule is the first to make room (the taught ones are never pushed out)."""
        lines = [ln for ln in self.rules().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        current = {_norm(r) for r in DEFAULT_RULES}
        core = [ln for ln in lines if _norm(_rule_body(ln)) in current]
        size = sum(len(ln) + 1 for ln in core)
        chosen: set[int] = set()
        for k in reversed(range(len(lines))):
            ln = lines[k]
            if _norm(_rule_body(ln)) in current:
                continue
            if size + len(ln) + 1 > budget:
                break
            chosen.add(k)
            size += len(ln) + 1
        return "\n".join(ln for k, ln in enumerate(lines) if ln in core or k in chosen)

    def _ai(self, openai=None):
        """The AI that turns feedback into rules (`openai` = an AI Videos OpenAI connection to use as the fallback). On a computer that edits folders in place (local projects): the M2's Ollama when it is
        allowed and free (clipper.remote_ollama, 17:00-08:00), else the AI Videos OpenAI key - never this Mac's own local model.
        Elsewhere: the server's best AI, as before."""
        if self.cfg.get("clipper.ai_videos.local_projects", False):
            from ..core.llm import Layered, LLMRouter
            from .localai import LocalAI
            ai = openai or LocalAI(self.cfg)
            ai = ai if openai or ai.available()[0] else None
            m2 = None
            try:
                router, where = self.engine.router_for_now() if self.engine else (None, "")
                if where == "M2":                                         # only the M2: no falling through to this Mac's Ollama
                    m2 = LLMRouter(router.cfg, chain=["ollama_remote"])
            except Exception as e:  # noqa: BLE001 - the M2 is a bonus
                log.warning(f"the M2 isn't available for rule-learning ({e})")
            if m2 and ai:
                return Layered(m2, ai)                                    # the M2 first; OpenAI if it can't answer
            return m2 or ai
        return self.engine.creative_router() if self.engine else None

    def learn(self, feedback: str, title: str = "", fallback: bool = True, ai=None, change: bool = False) -> list[str]:
        """Feedback in plain words -> style rules the next edits will follow. (fallback: keep the words as a rule if the AI can't turn them
        into one - for typed feedback yes, for the changes you ask for no. change=True: the words are a change asked for on one video, which
        only becomes a rule if it says something general.)"""
        feedback = " ".join(str(feedback or "").split())[:1200]
        if not feedback:
            return []
        rules: list[str] = []
        try:
            router = self._ai(openai=ai)
            if router is not None:
                said = f"The creator asked for this change to a finished video: {feedback}" if change else f"The creator said: {feedback}"
                res = router.complete_json(
                    "ai_style_rule", RULE_SYSTEM + (CHANGE_RULE if change else ""),
                    f"Video: {title}\nExisting rules:\n{self.rules_for_brief() or '(none yet)'}\n\n{said}",
                    RULE_SCHEMA)
                rules = [str(r) for r in (res.data.get("rules") or [])][:3]
        except Exception as e:  # noqa: BLE001 - never lose the feedback: store it as written
            log.warning(f"turning feedback into rules failed ({e})")
            rules = [feedback] if fallback else []
        return self.add_rules(rules)

    def brief(self, proj: dict) -> str:
        """What the editing AI reads: your notes for this video, then the style rules learned so far."""
        parts = []
        if (proj.get("notes") or "").strip():
            parts.append(proj["notes"].strip())
        rules = self.rules_for_brief()
        if rules:
            parts.append("STYLE RULES you must follow (learned from this creator's feedback):\n" + rules)
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ lifecycle
    def create(self, title: str, notes: str, names: list[str], opts: dict, default_options: dict,
               auto_make: dict | None = None) -> dict:
        names = [Path(str(n)).name for n in names][:200]
        if not names:
            raise ValueError("add at least one video file")
        bad = [n for n in names if Path(n).suffix.lower() not in (".mp4", ".mov", ".mkv", ".webm", ".m4v")]
        if bad:
            raise ValueError(f"not a video file: {bad[0]}")
        title = " ".join((title or "").split())[:90] or Path(names[0]).stem
        options = {**default_options, "mode": "long", "fcp": True, "music": str(opts.get("music") or ""),
                   "caption_style": opts.get("caption_style") or "clean", "pace": opts.get("pace") or "natural",
                   "project": True, "transcriber": "ai_video", "clips": 0}
        job = self.store.create({"files": names}, options)
        self.store.update(job["id"], status="uploading", stage=f"Uploading 0 of {len(names)}", title=title)
        pid = f"ai-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
        proj = {"id": pid, "title": title, "notes": (notes or "").strip()[:3000], "job": job["id"], "created": time.time(),
                "updated": time.time(), "files": names, "version": 0, "versions": [], "archive": {"state": "none"}}
        if auto_make:                                       # "start when the upload finishes": no page or uploader needs to stay open
            proj["auto_make"] = {k: auto_make[k] for k in ("music", "caption_style", "pace", "long_minutes", "graphics", "fcp", "notes", "kind", "transcriber")
                                 if k in auto_make}
        return self._save(proj)

    # ------------------------------------------------------------------ resumable uploads
    def by_job(self, job_id: str) -> str | None:
        return next((p for p in self.ids() if self.get(p)["job"] == job_id), None)

    def uploads(self, pid: str) -> dict:
        """Which files have fully arrived: [{index, name, size|None}] (a file is whole only once it is renamed from .tmp)."""
        proj = self.get(pid)
        pd = self.store.dir(proj["job"]) / "parts"
        have = {}
        for p in pd.glob("part_*") if pd.exists() else []:
            if p.suffix == ".tmp":
                continue
            try:
                have[int(p.name[5:8])] = p.stat().st_size
            except (ValueError, OSError):
                pass
        status = self.store.get(proj["job"])["status"]
        return {"status": status, "files": [{"index": i, "name": n, "size": have.get(i)} for i, n in enumerate(proj["files"])]}

    def resume(self, pid: str) -> dict:
        """Let an interrupted upload carry on: only the files that never fully arrived need sending."""
        proj = self.get(pid)
        job = self.store.get(proj["job"])
        if job["status"] in ("failed", "uploading") and not (self.store.dir(job["id"]) / "source.mp4").exists():
            self.store.update(job["id"], status="uploading", stage="Resuming the upload", error=None)
        return self.uploads(pid)

    def uploaded(self, job_id: str) -> None:
        """The last file arrived. If the project was created with auto_make, start it now."""
        pid = self.by_job(job_id)
        if not pid:
            return
        body = self.get(pid).get("auto_make")
        if body:
            try:
                self.update(pid, auto_make=None)
                self.make(pid, body)
            except Exception as e:  # noqa: BLE001
                log.warning(f"auto-start of {pid} failed: {e}")

    def set_notes(self, pid: str, notes: str, title: str | None = None) -> dict:
        fields = {"notes": (notes or "").strip()[:3000]}
        if title:
            fields["title"] = " ".join(title.split())[:90]
        return self.update(pid, **fields)

    def make(self, pid: str, body: dict) -> dict:
        """Start (or redo from scratch) the video: transcribe -> pick the takes -> cut -> captions + music -> Final Cut project."""
        proj = self.get(pid)
        job = self.store.get(proj["job"])
        if job["status"] not in ("draft", "done", "failed", "cancelled"):
            raise ValueError("this video is still being made")
        if "notes" in body or "title" in body:
            proj = self.set_notes(pid, str(body.get("notes", proj["notes"])), body.get("title"))
        opts = dict(job["options"])
        if body.get("music") is not None:
            m = body["music"]
            opts["music"] = "auto" if m is True else ("" if m in (False, None, "", "off") else str(m)[:20])
        if body.get("caption_style"):
            opts["caption_style"] = str(body["caption_style"])
        if body.get("pace"):
            opts["pace"] = str(body["pace"])
        if "graphics" in body:
            opts["graphics"] = _graphics(body["graphics"])
        if "kind" in body:
            opts["kind"] = _kind(body["kind"])
        if body.get("transcriber") in ("parakeet", "openai", "ai_video"):
            opts["transcriber"] = body["transcriber"]
        if "fcp" in body:
            opts["fcp"] = str(body["fcp"]).lower() in ("1", "true", "yes", "on")
        try:
            opts["long_minutes"] = max(0.0, min(120.0, float(body.get("long_minutes") or 0)))
        except (TypeError, ValueError):
            opts["long_minutes"] = 0.0
        opts["prompt"] = self.brief(proj)
        opts["mode"], opts["project"] = "long", True
        if job["status"] in ("done", "failed", "cancelled"):            # a full redo starts from the original footage again
            d = self.store.dir(job["id"])
            for f in [*d.glob("clip_*"), d / "fcp", d / "polish_work"]:
                shutil.rmtree(f, ignore_errors=True) if f.is_dir() else f.unlink(missing_ok=True)
            self.store.update(job["id"], clips=[])
        self.store.update(job["id"], options=opts, status="queued", stage="Queued", error=None, cancel=False, title=proj["title"])
        return self.get(pid)

    def finished(self, pid: str) -> None:
        """A render of this project's job just completed: record the version (so each one is kept in the archive)."""
        proj = self.get(pid)
        try:
            clip = self.store.get(proj["job"])["clips"][0]
        except (KeyError, IndexError):
            return
        stamp = clip.get("rendered_at")
        if not stamp or any(v.get("rendered_at") == stamp for v in proj["versions"]):
            return
        n = proj["version"] + 1
        note = (clip.get("edited") or {}).get("instruction") or ("first cut" if n == 1 else "re-made")
        self.update(pid, version=n, versions=[*proj["versions"], {"n": n, "rendered_at": stamp, "note": note,
                                                                   "duration": clip.get("duration"), "archived": False}])

    # ------------------------------------------------------------------ the archive on the big drive
    def _remote_mounted(self, host: str, mount: str) -> bool:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host,
               "python3 -c 'import os,sys;m=sys.argv[1];sys.exit(0 if (os.path.ismount(m) if m.startswith(\"/Volumes/\") else os.path.exists(m)) else 1)' " + _q(mount)]
        try:
            return subprocess.run(cmd, capture_output=True, timeout=60).returncode == 0
        except (subprocess.SubprocessError, OSError):
            return False

    def _mount_of(self, path: Path) -> str:
        p = path.parts
        return str(Path(*p[:3])) if len(p) >= 3 and p[1] == "Volumes" else str(path.parent)

    def folder_name(self, proj: dict) -> str:
        return f"{datetime.fromtimestamp(proj['created']):%Y-%m-%d} {safe(proj['title'], 70)}"

    def pending(self) -> list[str]:
        """Projects whose latest version isn't on the drive yet."""
        out = []
        for pid in self.ids():
            try:
                proj = self.get(pid)
                if proj.get("mode") == "local":
                    continue                                          # these live in the creator's own folder; nothing to copy
                job = self.store.get(proj["job"])
            except KeyError:
                continue
            if job["status"] == "done" and (job.get("clips") or []) and any(not v.get("archived") for v in proj["versions"]):
                out.append(pid)
        return out

    def archive(self, pid: str) -> str:
        """Copy the project to the drive. Returns 'done' | 'waiting' | 'nothing'."""
        self.finished(pid)
        proj = self.get(pid)
        job = self.store.get(proj["job"])
        if proj.get("mode") == "local":
            return "done"                                             # already in the folder it came from; never copy the footage
        if job["status"] != "done" or not job.get("clips"):
            return "nothing"
        root, host = root_dir(self.cfg), drive_host(self.cfg)
        mount = self._mount_of(root)
        ok = self._remote_mounted(host, mount) if host else drive_ready(Path(mount))
        if not ok:
            self.update(pid, archive={**proj["archive"], "state": "waiting", "why": f"the drive ({mount}) isn't mounted"})
            return "waiting"
        clip = job["clips"][0]
        d = self.store.dir(job["id"])
        name = self.folder_name(proj)
        dest = str(root / name)
        todo = [v for v in proj["versions"] if not v.get("archived")]
        with tempfile.TemporaryDirectory(prefix="aivideo-") as tmp:
            stage = Path(tmp)
            self._stage(stage, proj, job, clip, d, todo, first=not proj["archive"].get("raw"))
            err = self._ship(stage, dest, host)
        if err:
            self.update(pid, archive={**proj["archive"], "state": "waiting", "why": err[:200]})
            return "waiting"
        versions = [{**v, "archived": True} if not v.get("archived") else v for v in proj["versions"]]
        self.update(pid, versions=versions, archive={"state": "done", "path": dest, "at": time.time(), "raw": True,
                                                     "host": host or "this Mac"})
        log.info(f"{pid}: archived to {dest}" + (f" on {host}" if host else ""))
        return "done"

    def _stage(self, stage: Path, proj: dict, job: dict, clip: dict, d: Path, todo: list[dict], first: bool) -> None:
        """Symlink what must be copied into the shape it will have on the drive (tar -h follows the links)."""
        t = safe(proj["title"], 70)
        out = stage / "Output"
        out.mkdir()
        latest = max(proj["versions"], key=lambda v: v["n"]) if proj["versions"] else {"n": 1}
        for v in todo:                                      # only the newest render still exists on disk; older ones were archived before
            if v["n"] != latest["n"]:
                continue
            if clip.get("file") and (d / clip["file"]).exists():
                (out / f"{t} v{v['n']}.mp4").symlink_to(d / clip["file"])
            if clip.get("srt") and (d / clip["srt"]).exists():
                (out / f"{t} v{v['n']}.srt").symlink_to(d / clip["srt"])
        if first and (d / "source.mp4").exists():
            (stage / "Raw footage").mkdir()
            (stage / "Raw footage" / "source.mp4").symlink_to(d / "source.mp4")
        meta = clip.get("fcp") or {}
        bundle = d / "fcp" / meta.get("folder", "") if meta.get("folder") else None
        if bundle and bundle.is_dir():
            (stage / "Final Cut").mkdir()
            (stage / "Final Cut" / bundle.name).symlink_to(bundle)
        notes = stage / "Notes"
        notes.mkdir()
        (notes / "brief.txt").write_text((proj.get("notes") or "") + "\n")
        (notes / "plan.md").write_text(self.plan_md(proj, job, clip, d))
        if (d / "graphics_check.md").exists():
            shutil.copy2(d / "graphics_check.md", notes / "graphics_check.md")
        if (d / "transcript.json").exists():
            words = json.loads((d / "transcript.json").read_text()).get("words") or []
            (notes / "transcript.txt").write_text(" ".join(w["text"].strip() for w in words) + "\n")
        if self.rules():
            (notes / "style_rules.md").write_text(self.rules() + "\n")

    def plan_md(self, proj: dict, job: dict, clip: dict, d: Path) -> str:
        lines = [f"# {proj['title']}", "", f"Made {datetime.fromtimestamp(proj['created']):%Y-%m-%d %H:%M}; version {proj['version']}.", ""]
        if proj.get("notes"):
            lines += ["## Your notes", proj["notes"], ""]
        story = clip.get("story") or {}
        if story.get("summary"):
            lines += ["## The story the edit follows", story["summary"], ""]
        if story.get("files"):
            lines += ["## Kept from each file"] + [f"- {f['name']}: {f['kept_s']} s" for f in story["files"]] + [""]
        plan = d / "plan.json"
        if plan.exists():
            iv = json.loads(plan.read_text()).get("intervals") or []
            lines += ["## Cut list (source seconds)"] + [f"{i + 1}. {a:.1f} - {b:.1f}  ({b - a:.1f} s)" for i, (a, b) in enumerate(iv)] + [""]
            cw = json.loads(plan.read_text()).get("cutaways") or []
            if cw:
                lines += ["## B-roll cutaways (timeline seconds)"] + [f"- {c['at']:.1f} s for {c['length']:.1f} s: {c['what']}" for c in cw] + [""]
            ang = json.loads(plan.read_text()).get("angles") or []
            if ang:
                lines += ["## Camera angles (shared-timeline seconds)"] + [f"- {a['start']:.1f} - {a['end']:.1f}: {a['angle']}" for a in ang] + [""]
        if clip.get("music"):
            m = clip["music"]
            lines += ["## Music", f"{m.get('title')} - {m.get('artist')} ({m.get('license')})", ""]
        if proj["versions"]:
            lines += ["## Versions"] + [f"- v{v['n']}: {v['note']}" for v in proj["versions"]] + [""]
        return "\n".join(lines)

    def _ship(self, stage: Path, dest: str, host: str | None) -> str | None:
        """Copy the staged tree into `dest` (never replacing files that are already there that we didn't re-send). None = ok."""
        try:
            if not host:
                Path(dest).mkdir(parents=True, exist_ok=True)
                for p in stage.rglob("*"):
                    rel = p.relative_to(stage)
                    if p.is_dir() and not p.is_symlink():
                        (Path(dest) / rel).mkdir(parents=True, exist_ok=True)
                    elif p.is_file():
                        (Path(dest) / rel).parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, Path(dest) / rel)
                    elif p.is_symlink():
                        target = p.resolve()
                        (Path(dest) / rel).parent.mkdir(parents=True, exist_ok=True)
                        if target.is_dir():
                            shutil.copytree(target, Path(dest) / rel, dirs_exist_ok=True)
                        else:
                            shutil.copy2(target, Path(dest) / rel)
                return None
            send = subprocess.Popen(["nice", "-n", "15", "tar", "-C", str(stage), "-chf", "-", "."], stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            recv = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host,
                                   f"mkdir -p {_q(dest)} && nice -n 15 tar -C {_q(dest)} -xf -"],
                                  stdin=send.stdout, capture_output=True, timeout=6 * 3600)
            send.stdout.close()
            rc = send.wait()
            if recv.returncode or rc:
                return (recv.stderr or send.stderr.read() or b"copy failed").decode(errors="replace").strip()
            return None
        except (subprocess.SubprocessError, OSError) as e:
            return f"{type(e).__name__}: {e}"


def _paths(v) -> list[str]:
    """A list of paths from a list or from text with one per line."""
    items = v.splitlines() if isinstance(v, str) else list(v or [])
    return [str(x).strip().strip('"') for x in items if str(x).strip()][:20]


def _read_notes(folder: Path) -> str:
    from .localproj import read_notes
    return read_notes(folder)


def _q(s: str) -> str:
    import shlex
    return shlex.quote(s)
