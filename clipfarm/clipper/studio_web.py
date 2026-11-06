"""The app's own API (/api/app/...): what the FirstCut page (web/studio.html) needs beyond the long-form projects (/api/ai/...), the
clip jobs (/api/jobs/...) and the footage catalog (/api/library/...) the Clipper already serves. Only used when clipper.app = "studio".

    GET  /api/app                      everything the page shows: settings, the Mac, which AI and whether it is ready, the two "how it
                                       edits" lists, setup progress, the library, every long video and batch of clips
    GET  /api/app/ping | busy          the app shell: is the engine up / what would quitting stop
    POST /api/app/settings             change settings (the engine restarts itself when that needs it, once nothing is being made)
    POST /api/app/places               move the models / catalog to another folder, or change the videos / working-files folder
    POST /api/app/ai/<chatgpt|claude>/<connect|disconnect>
    POST /api/app/styles/<short|long>/<add|remove|undo|import>
    POST /api/app/setup/start
    POST /api/app/long | short         make a long video from a folder / clips from a video or a link
    POST /api/app/short/<job>/keep | remake
    POST /api/app/pick | open          (in a browser, without the app: a folder picker and Finder via macOS itself)
    GET  /api/app/music/online?source=&q=&page=      Openverse / Epidemic Sound songs (server.py)
    POST /api/app/music/favorite {track, on}         heart a song (settings music.favorites)
    POST /api/app/music/download {track}             a song into ~/Downloads (+ "<name> - credit.txt")
    GET  /api/app/stock/search?q=&kinds=&page=       stock videos/photos (stockfind.py; server.py)
    POST /api/app/stock/<add|use|download> {id}      into the app's "Footage library/Stock" folder (a footage source) / ~/Downloads
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from ..core.log import get_logger
from . import studio as SD

log = get_logger("clipper.studio_web")
DICTATION_WORDS = [(r"\b(?:be|bee|b)[ -]?rolls?\b", "B-roll"), (r"\bfinal ?cut\b", "Final Cut"),
                   (r"\bf ?c ?p ?x ?m ?l\b", "FCPXML"), (r"\bj ?cuts?\b", "J-cut"), (r"\bl ?cuts?\b", "L-cut")]


def link_title(url: str) -> str | None:
    """A video link's title without downloading anything (yt-dlp reads the page only)."""
    import yt_dlp
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True, "socket_timeout": 15}) as y:
        info = y.extract_info(url, download=False, process=False)
    return " ".join(str((info or {}).get("title") or "").split()) or None
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".mkv", ".webm", ".mts", ".avi")
RESTART = 75                                             # exit code the app shell answers by starting the engine again


class StudioAPI:
    def __init__(self, cfg, store, worker, projects, runner, chatgpt_plan, default_options, clean_options):
        self.cfg, self.store, self.worker, self.projects, self.runner = cfg, store, worker, projects, runner
        self.chatgpt = chatgpt_plan
        self.default_options, self.clean_options = default_options, clean_options
        self.root = SD.home()
        self.settings = SD.Settings(self.root)
        self.styles = SD.Styles(self.root)
        self.setup = SD.Setup(self.root)
        self.claude = SD.ClaudeLogin(self.root)
        self.mac = SD.mac_info()
        self.moving = ""
        self.restart_wanted = False
        self._cache: dict = {}
        self.lock = threading.Lock()
        self.sync_long_style()
        from .transcript import Transcripts
        self.transcripts = Transcripts(self.root / "data" / "transcripts", cfg, lambda: self.settings.get()["places"]["videos"])
        self.transcripts.resume()                                         # a transcript that was being written when the app quit
        self.setup.resume()                                               # downloads that were going when the app quit
        threading.Thread(target=self._watch, name="studio-watch", daemon=True).start()

    # ------------------------------------------------------------------ helpers
    def cached(self, key: str, secs: float, fn):
        hit = self._cache.get(key)
        if hit and time.time() - hit[0] < secs:
            return hit[1]
        val = fn()
        self._cache[key] = (time.time(), val)
        return val

    def busy(self) -> str:
        if self.runner and self.runner.running:
            return "a video is being made"
        if self.worker.busy_with:
            return "clips are being made"
        cat = self.runner.library() if self.runner else None
        if cat is not None and cat.status.get("running"):
            return "your footage library is being catalogued"
        if self.setup.running():
            return "downloads are running"
        tx = getattr(self, "transcripts", None)
        if tx is not None and tx.thread and tx.thread.is_alive():
            return "a transcript is being written"
        return ""

    def _watch(self) -> None:
        """Every few seconds: save finished clips into their project folders; restart the engine for new settings once nothing runs."""
        from . import net
        while True:
            time.sleep(3)
            net.online()                                                  # (at most every 15 s; the page reads net.peek())
            try:
                for j in self.short_jobs():
                    self.save_clips(j)
            except Exception:  # noqa: BLE001
                log.exception("saving clips failed")
            cat = self.runner.catalog if self.runner else None
            if cat is not None and not getattr(cat, "looping", False) and time.time() - getattr(self, "_ticked", 0) > 300:
                self._ticked = time.time()
                try:
                    cat.tick(busy=bool(self.runner.running or self.worker.busy_with))
                except Exception:  # noqa: BLE001
                    log.exception("catalog tick failed")
            if self.restart_wanted and not self.busy():
                log.info("restarting the engine for new settings")
                SD.OllamaRunner.stop()
                os._exit(RESTART)

    def want_restart(self) -> None:
        SD.write_overlay(self.root)
        self.restart_wanted = True

    # ------------------------------------------------------------------ what the page shows
    def ai_state(self, s: dict) -> dict:
        """Who makes the decisions: the primary account, or the backup while the primary is out of usage (localai.read_trips), with
        both accounts' connection and usage for Settings › AI and the sidebar."""
        from .localai import read_trips
        primary = s["ai"]["use"]
        m = SD.local_models(SD.ram_gb())
        model = s["ai"].get("local_model") or m["text"]
        chat = self.chatgpt.status()
        claude = self.cached("claude", 8, self.claude.status)
        pulled = self.cached("pulled", 5, self.setup._pulled)
        local = {"model": model, "ready": model in pulled}
        conn = {"chatgpt": chat.get("status") == "connected", "claude": claude.get("status") == "connected", "local": local["ready"]}
        choice = s["ai"].get("backup", "auto")
        other = {"claude": "chatgpt", "chatgpt": "claude"}.get(primary, "")
        backup = (other if other and conn[other] else "local" if primary != "local" and local["ready"] else "") if choice == "auto" else choice
        backup = "" if backup == primary else backup
        out = read_trips(self.root / "data" / "ai_videos").get(primary)
        use = backup if out and backup and conn.get(backup) else primary      # the account answering now
        ready = conn[use]
        name = {"chatgpt": "ChatGPT", "claude": "Claude", "local": f"this Mac ({model})"}[use]
        label = f"{name}, connected" if ready and use != "local" else name if ready else f"{name}, not {'downloaded' if use == 'local' else 'connected'}"
        brain = {"claude": SD.CLAUDE_MODELS.get(s["ai"].get("claude_edit") or "sonnet", "").split(" ")[0],
                 "chatgpt": s["ai"].get("chatgpt_model") or chat.get("model") or "", "local": model}[use]
        from ..core import llm as _llm
        speed = {**_llm.LOCAL_SPEED, "history": list(_llm.LOCAL_SPEEDS)} if _llm.LOCAL_SPEED else None
        return {"local_speed": speed,                                    # this Mac's model: tokens a second, last answers
                "use": use, "ready": ready, "label": label, "name": name if use != "local" else "This Mac", "model": brain,
                "usage": self.usage(use), "chatgpt": chat, "claude": claude, "local": local,
                "primary": primary, "backup": backup, "backup_choice": choice, "connected": conn,
                "out": {"acct": primary, "until": out["until"], "why": out.get("why") or "out of usage"} if out else None,
                "usages": {k: self.usage(k) for k in ("claude", "chatgpt") if conn[k]},
                "back_to": s["ai"].get("back_to") if primary == "local" else None,       # (settings from before the switch-over: switched
                "back_usage": self.usage(s["ai"]["back_to"]) if primary == "local" and s["ai"].get("back_to") else None}   # to this Mac by hand)

    def usage(self, use: str) -> dict | None:
        """How much of the plan's 5-hour and weekly allowance is used, as of its last answer (a window that has reset since reads 0)."""
        try:
            if use == "claude":
                u = json.loads((self.root / "claude" / "usage.json").read_text())
            elif use == "chatgpt" and self.chatgpt:
                u = self.chatgpt.usage()
            else:
                return None
        except (OSError, ValueError):
            return None
        if not isinstance(u, dict):
            return None
        now = time.time()
        for k in ("five_hour", "week"):
            w = u.get(k)
            if isinstance(w, dict) and w.get("resets") and now > float(w["resets"]):
                u[k] = {"used": 0.0, "resets": 0, "reset": True}
        return u

    def state(self) -> dict:
        s = self.settings.get()
        cat = self.runner.library() if self.runner else None
        lib = None
        if cat is not None:
            st = cat.stats()
            lib = {k: st.get(k) for k in ("videos", "photos", "described", "hours", "roots", "status")}
        elif s["library"]["folders"]:
            lib = {"videos": 0, "photos": 0, "described": 0, "hours": 0, "status": {}, "roots": [{"path": f, "mounted": Path(f).is_dir()} for f in s["library"]["folders"]]}
        return {
            "settings": s, "mac": self.mac, "models": SD.local_models(SD.ram_gb()), "ai": self.ai_state(s),
            "styles": {k: self.style_view(k) for k in SD.KINDS},
            "setup": {"running": self.setup.running(), "steps": self.cached("plan", 2, self.setup.plan)},
            "library": lib, "projects": self.projects.list(), "shorts": [self.short_view(j) for j in self.short_jobs()],
            "sizes": self.cached("sizes", 300, lambda: self.sizes(s)), "moving": self.moving, "restarting": self.restart_wanted,
            "music": self.cached("music", 3, lambda: (self.musiclib().stats() if self.musiclib() else None)),
            "keys": self.cached("keys", 30, self.keys_state),
            "stock_folder": str(self.stock_folder()),                    # (Find footage tells it apart from your own folders)
            "transcripts": self.transcripts.list()[:30],                 # Long form › Get the transcript
            "mcp": self.cached("mcp", 10, self.mcp_state),               # Settings › Connect an AI
            "online": self.online_state(s),                              # offline: what still works
            "missing": {k: SD.missing_drive(v) for k, v in s["places"].items() if SD.missing_drive(v)},
        }

    def online_state(self, s: dict) -> dict:
        """{"online", "local_editing", "local_looking"}: offline, editing goes on only with this Mac's model downloaded."""
        from . import net
        on = net.peek()
        if on:
            return {"online": True}
        plan = {st["id"]: st for st in self.cached("plan", 2, self.setup.plan)}
        return {"online": False, "local_editing": (plan.get("local") or {}).get("state") == "done",
                "local_looking": (plan.get("vision") or {}).get("state") == "done"}

    def mcp_state(self) -> dict:
        """Settings › Connect an AI: the launcher an AI app runs, and whether Claude Desktop already knows it."""
        from . import mcp_server as MCP
        return {"launcher": MCP.launcher_path(), "claude_desktop": MCP.claude_desktop_state(), "codex": MCP.codex_state(), "tools": len(MCP.TOOLS)}

    def style_view(self, kind: str) -> dict:
        d = self.styles.get(kind)
        return {"points": d["points"], "updated": d["updated"], "added": d["added"], "can_undo": bool(d["history"])}

    def sizes(self, s: dict) -> dict:
        def gb(p):
            p = Path(p)
            if not p.is_dir():
                return None
            try:
                out = subprocess.run(["du", "-sk", str(p)], capture_output=True, text=True, timeout=30).stdout.split()
                return round(int(out[0]) * 1024 / 1e9, 1)
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                return None
        return {k: gb(v) for k, v in s["places"].items()}

    # ------------------------------------------------------------------ "how it edits"
    def sync_long_style(self) -> None:
        """Long videos read their rules from the Projects rules file (aivideo.Projects.brief): keep it the long-form points."""
        pts = self.styles.get("long")["points"]
        try:
            self.projects.rules_path().write_text("\n".join(f"- {p}" for p in pts) + ("\n" if pts else ""))
        except OSError:
            pass

    def style_action(self, kind: str, what: str, b: dict) -> dict:
        from .localai import LocalAI
        if what == "add":
            self.styles.add(kind, str(b.get("text") or ""), LocalAI(self.cfg))
        elif what == "import":
            data = base64.b64decode(str(b.get("data") or ""), validate=False)
            text = SD.skill_text(str(b.get("name") or "skill.md"), data)
            if not text.strip():
                raise ValueError("that file has no text in it")
            self.styles.add(kind, "Here is a skill, prompt or style guide the person uses for their videos (from Claude, ChatGPT or their own notes):\n" + text, LocalAI(self.cfg))
        elif what == "remove":
            self.styles.remove(kind, int(b.get("index", -1)))
        elif what == "undo":
            self.styles.undo(kind)
        else:
            raise KeyError(what)
        if kind == "long":
            self.sync_long_style()
        return self.style_view(kind)

    # ------------------------------------------------------------------ long videos
    def picked_folder(self, clips) -> str:
        """Shots picked in Find footage, as a footage folder of links to the original files (nothing copied; the links stay with the
        app, so the project can be made again later)."""
        if not isinstance(clips, list) or not clips:
            raise ValueError("pick some shots first")
        from .localproj import VIDEO_EXT as VX
        files = []
        for c in clips[:400]:
            f = Path(str(c or "")).expanduser()
            if not f.is_absolute() or not f.is_file() or f.suffix.lower() not in VX:
                raise ValueError(f"{f.name or c} isn't a video on this Mac (is its drive plugged in?)")
            files.append(f)
        d = self.root / "Picked" / time.strftime("%Y%m%d-%H%M%S")
        n = 2
        while d.exists():
            d = d.with_name(d.name.split(" (")[0] + f" ({n})")
            n += 1
        d.mkdir(parents=True)
        used = set()
        for f in dict.fromkeys(files):
            name = f.name if f.name.lower() not in used else f"{f.parent.name} - {f.name}"
            used.add(name.lower())
            os.symlink(f, d / name)
        return str(d)

    def broll_folder(self, video: str, broll: str = "") -> tuple[str, str]:
        """"Add B-roll to a video": the person's own video (kept exactly as it is) and, if they chose one, a folder of B-roll, as one
        footage folder of links. Returns (folder, the A-roll's name in it)."""
        from .localproj import VIDEO_EXT as VX
        v = Path(str(video or "")).expanduser()
        if not v.is_absolute() or not v.is_file() or v.suffix.lower() not in VX:
            raise ValueError("choose your video (a .mov or .mp4 on this Mac)")
        extra = []
        if broll:
            bf = Path(str(broll)).expanduser()
            if not bf.is_absolute() or not bf.is_dir():
                raise ValueError("choose a folder of B-roll (or leave it out to use your footage library)")
            extra = [f for f in sorted(bf.rglob("*")) if f.is_file() and f.suffix.lower() in VX and not f.name.startswith(".") and f.resolve() != v.resolve()]
        folder = Path(self.picked_folder([str(v), *map(str, extra[:1500])]))
        aroll = next(f.name for f in folder.iterdir() if f.is_symlink() and Path(os.readlink(f)) == v)
        return str(folder), aroll

    def make_long(self, b: dict) -> dict:
        if b.get("mode") == "broll":                                       # keep their cut; only add B-roll (and graphics, captions if asked)
            folder, aroll = self.broll_folder(b.get("video"), b.get("broll_folder") or "")
            b = {**b, "source": folder, "keep_cut": True, "aroll": aroll, "title": str(b.get("title") or Path(str(b.get("video"))).stem)}
        elif b.get("clips"):                                               # shots picked in Find footage instead of a folder
            b = {**b, "source": self.picked_folder(b["clips"]), "title": str(b.get("title") or "Picked shots")}
        src = str(b.get("source") or "").strip()
        if not src or not Path(src).expanduser().is_dir():
            raise ValueError("choose the folder with your footage")
        if not Path(src).expanduser().is_absolute():
            raise ValueError("choose the folder with your footage (a full path)")
        s = self.settings.get()
        given = b.get("options") or {}
        if not isinstance(given, dict):
            raise ValueError("options must be a set of options")
        o = {**s["defaults"]["long"], **given}
        try:
            minutes = max(0.0, min(600.0, float(o.get("long_minutes") or 0)))
        except (TypeError, ValueError):
            raise ValueError("the length must be a number of minutes") from None
        self.check_drive(s["places"]["videos"], "the folder where finished videos go")
        body = {"notes": str(b.get("notes") or ""), "music": str(o.get("music") or ""), "caption_style": str(o.get("caption_style") or "none"),
                "pace": SD.ENGINE_PACE["long"].get(str(o.get("pace") or "auto"), "natural"), "long_minutes": minutes, "graphics": str(o.get("graphics") or ""), "gfx_style": str(o.get("gfx_style") or "auto"),
                **({"keep_cut": True, "aroll": str(b["aroll"])} if b.get("keep_cut") else {}),
                "transcriber": "parakeet", "kind": "auto", "fcp": True, "library": bool(o.get("library", True)), "stock": bool(o.get("stock"))}
        proj = self.projects.create_local(src, str(b.get("title") or ""), str(b.get("notes") or ""), {}, self.default_options(self.cfg.get("clipper", {}) or {}),
                                          out_dir=s["places"]["videos"])
        if b.get("song") or b.get("track"):                              # their own song for this video (copied into the project)
            body["song"] = self.song_path(b, Path(proj["home"]) if proj.get("home") else None)
        self.projects.prepare_local(proj["id"], body)
        self.runner.start(proj["id"], "make", body=body)
        return {"id": proj["id"]}

    # ------------------------------------------------------------------ short clips
    def short_jobs(self) -> list[dict]:
        return [j for j in self.store.list() if (j.get("source") or {}).get("app") == "studio"]

    def short_view(self, j: dict) -> dict:
        clips = [{k: c.get(k) for k in ("index", "file", "thumb", "duration", "title", "rendered_at", "keep", "srt", "score")} for c in j.get("clips") or []]
        return {"id": j["id"], "title": j.get("title"), "status": j.get("status"), "stage": j.get("stage"), "progress": j.get("progress", 0.0),
                "error": j.get("error"), "created": j.get("created"), "updated": j.get("updated"), "home": j.get("app_home"), "clips": clips,
                "options": {k: (j.get("options") or {}).get(k) for k in ("min_len", "max_len", "caption_style", "layout", "music", "song")}}

    def short_options(self, s: dict, given: dict, notes: str) -> dict:
        o = {**s["defaults"]["short"], **(given or {})}
        if o.get("len"):
            try:
                a, b2 = (float(x) for x in str(o["len"]).split("-"))
                o["min_len"], o["max_len"] = a, b2
            except ValueError:
                pass
        raw = {"clips": int(o.get("clips") or 8), "min_len": o.get("min_len", 20), "max_len": o.get("max_len", 35), "layout": o.get("layout") or "auto",
               "caption_style": o.get("caption_style") or "bold", "pace": SD.ENGINE_PACE["short"].get(str(o.get("pace") or "auto"), "tight"),
               "mode": "clips", "language": "en"}
        opts = self.clean_options(raw, self.default_options(self.cfg.get("clipper", {}) or {}))
        opts["music"] = str(o.get("music") or "")
        opts["prompt"] = "\n\n".join(x for x in (notes.strip(), self.styles.brief("short")) if x)[:4000]
        return opts

    def make_short(self, b: dict) -> dict:
        src = str(b.get("source") or "").strip()
        is_url = bool(re.match(r"^https?://\S+$", src))
        p = Path(src).expanduser().resolve() if src and not is_url else Path(src)   # (the links point at the real file)
        if not is_url and not (p.is_file() and p.suffix.lower() in VIDEO_EXT):
            raise ValueError("choose a video on this Mac, or paste a link")
        if not is_url:
            from ..core import ffmpeg as FF
            try:
                info = FF.probe(p)
            except PermissionError:
                raise                                                     # (said plainly below)
            except Exception:  # noqa: BLE001 - the engine says what's wrong with it
                info = None
            if info is not None and FF.audio_stream(info) is None:
                raise ValueError("that video has no sound: short clips are found from what is said, so it needs its audio")
        s = self.settings.get()
        notes = str(b.get("notes") or "")
        opts = self.short_options(s, b.get("options") or {}, notes)
        title = p.stem if not is_url else "Clips from a link"
        home = SD.project_folder(s["places"]["videos"], title)
        if b.get("song") or b.get("track"):                              # every clip with their song
            opts["song"] = self.song_path(b, home)
        source = {"app": "studio", "notes": notes[:3000]}
        source.update({"url": src} if is_url else {"filename": p.name, "path": str(p)})        # ("local" would mean a long-form project)
        job = self.store.create(source, opts, status="draft")
        if not is_url:
            os.symlink(p, self.store.dir(job["id"]) / ("source" + p.suffix.lower()))         # read where it is: nothing is copied
            SD.link_raw(home, [p])
        self.store.update(job["id"], title=title, app_home=str(home), status="queued")
        if is_url:                                                        # the video's own title, in a second or two (the sidebar shows it)
            threading.Thread(target=self._name_link, args=(job["id"], src), name="link-title", daemon=True).start()
        self.worker.wake.set()
        return {"id": job["id"]}

    def _name_link(self, jid: str, url: str) -> None:
        try:
            title = link_title(url)
        except Exception as e:  # noqa: BLE001 - the download names it later anyway
            log.info(f"no title for {url}: {e}")
            return
        title = " ".join(str(title or "").split())[:120]
        if title and self.store.get(jid).get("title", "").startswith("Clips from a link"):
            self.store.update(jid, title=title)

    def remake_short(self, jid: str, b: dict) -> dict:
        j = self.store.get(jid)
        if (j.get("source") or {}).get("app") != "studio":
            raise KeyError(jid)
        if j.get("status") in ("queued", "downloading", "transcribing", "finding", "rendering", "joining"):
            raise ValueError("these clips are still being made")
        s = self.settings.get()
        prev = {"min_len": j["options"].get("min_len"), "max_len": j["options"].get("max_len"), "caption_style": j["options"].get("caption_style"),
                "layout": j["options"].get("layout"), "music": j["options"].get("music")}
        opts = self.short_options(s, {**prev, **(b.get("options") or {})}, (j.get("source") or {}).get("notes") or "")
        home = Path(j.get("app_home") or "")
        if (home / "Clips").is_dir():                                     # the last version stays, out of the way
            old, n = home / f"Clips (before {time.strftime('%H.%M')})", 2
            while old.exists():
                old, n = home / f"Clips (before {time.strftime('%H.%M')}, {n})", n + 1
            (home / "Clips").rename(old)
        d = self.store.dir(jid)
        for f in d.glob("clip_*"):
            f.unlink(missing_ok=True)
        self.store.update(jid, options=opts, status="queued", stage="Queued", progress=0.0, error=None, clips=[], app_saved={})
        self.worker.wake.set()
        return {"id": jid}

    def save_clips(self, j: dict) -> None:
        """A finished batch: every clip into <project>/Clips (as it is now: a clip changed in words is copied again); clips you untick
        are in Clips/Not kept."""
        if j.get("status") != "done" or not j.get("app_home"):
            return
        home = Path(j["app_home"])
        if home.name.startswith("Clips from a link") and j.get("title") and not j.get("title", "").startswith("Clips from a link") \
                and not (home / "Clips").exists():                    # a link: its folder is named after the video once the title is known
            try:
                new = SD.project_folder(home.parent, j["title"])
                new.rmdir()
                home.rename(new)
                home = new
                self.store.update(j["id"], app_home=str(home))
            except OSError as e:
                log.warning(f"naming the clips folder after the video failed: {e}")
        d = self.store.dir(j["id"])
        saved = dict(j.get("app_saved") or {})
        changed = False
        first = min((int(c["index"]) for c in j.get("clips") or []), default=0)     # (the engine counts clips from 1, the page from 0)
        for c in j.get("clips") or []:
            key = str(c["index"])
            stamp = str(c.get("rendered_at") or c.get("file"))
            keep = c.get("keep") is not False
            want_dir = home / "Clips" if keep else home / "Clips" / "Not kept"
            name = f"{int(c['index']) - first + 1:02d} {_safe(c.get('title') or 'clip')}"
            prev = saved.get(key) or {}
            if prev.get("stamp") == stamp and prev.get("dir") == str(want_dir):
                continue
            for old in (prev.get("files") or []):
                Path(old).unlink(missing_ok=True)
            want_dir.mkdir(parents=True, exist_ok=True)
            files = []
            for src_name, ext in ((c.get("file"), ".mp4"), (c.get("srt"), ".srt")):
                if src_name and (d / src_name).exists():
                    dest = want_dir / f"{name}{ext}"
                    shutil.copy2(d / src_name, dest)
                    files.append(str(dest))
            saved[key] = {"stamp": stamp, "dir": str(want_dir), "files": files}
            changed = True
        if changed:
            self.store.update(j["id"], app_saved=saved)

    def keep_clip(self, jid: str, b: dict) -> dict:
        j = self.store.get(jid)
        idx = int(b.get("index", -1))
        clip = next((c for c in j.get("clips") or [] if c["index"] == idx), None)
        if clip is None:
            raise KeyError(jid)
        self.store.set_clip(jid, {**clip, "keep": bool(b.get("keep"))})
        self.save_clips(self.store.get(jid))
        return {"ok": True}

    # ------------------------------------------------------------------ places
    def change_place(self, b: dict) -> dict:
        key, new = str(b.get("key") or ""), Path(str(b.get("path") or "")).expanduser()
        if key not in ("videos", "models", "catalog", "work") or not new.is_absolute():
            raise ValueError("choose a folder")
        s = self.settings.get()
        old = Path(s["places"][key])
        self.check_drive(new, "that folder")
        msg = ""
        if key in ("models", "catalog") and old.is_dir() and any(old.iterdir()) and old.resolve() != new.resolve():
            if self.busy():
                raise ValueError(f"wait until {self.busy()} finishes, then move it")
            if new.resolve().is_relative_to(old.resolve()):
                raise ValueError("choose a folder outside the one it is in now (it can't move into itself)")
            self.make_dir(new)
            self.moving = f"Moving {key} to {new}…"

            def move():
                try:
                    for item in old.iterdir():
                        target = new / item.name
                        if not target.exists():
                            shutil.move(str(item), str(target))
                    self.moving = f"Moved to {new}."
                    self.settings.update({"places": {key: str(new)}})
                    self.want_restart()
                except Exception as e:  # noqa: BLE001
                    self.moving = f"Moving stopped: {e}"
            threading.Thread(target=move, name="move-place", daemon=True).start()
            msg = self.moving
        else:
            self.make_dir(new)
            self.settings.update({"places": {key: str(new)}})
            self.want_restart()
            msg = "Changed."
        return {"settings": self.settings.get(), "message": msg}

    # ------------------------------------------------------------------ graphics styles (the long-form style picker)
    GFX_SAMPLES = {
        "lower": {"kind": "lower", "text": "Jordan Lee", "role": "Program director", "len": 4.0},
        "point": {"kind": "point", "text": "Students run the campus radio station", "len": 4.0},
        "stat": {"kind": "stat", "value": "92%", "label": "of graduates are hired within six months", "len": 4.0},
    }

    def gfx_styles(self) -> list[dict]:
        from .graphics import STYLE_NAMES, THEMES
        return [{"id": k, "name": v} for k, v in STYLE_NAMES.items() if k in THEMES]

    def gfx_backdrop(self) -> Path:
        """The scene the style samples sit on: a flat, softly out-of-focus interview set (a person on the right, a bookshelf and a window
        behind), mid-bright so light and dark looks both read, and the person's face shows that cards keep clear of it."""
        out = Path(self.settings.get()["places"]["work"]) / "gfx" / "backdrop-v1.jpg"
        if out.exists():
            return out
        from PIL import Image, ImageDraw, ImageFilter
        W, H = 1280, 720
        im = Image.new("RGB", (W, H), (56, 66, 78))                       # the wall
        d = ImageDraw.Draw(im)
        d.rectangle((70, 70, 420, 470), fill=(118, 138, 156))             # a window, daylight behind it
        d.rectangle((240, 70, 252, 470), fill=(70, 82, 96))
        d.rectangle((70, 262, 420, 274), fill=(70, 82, 96))
        d.rectangle((0, 560, W, H), fill=(44, 50, 58))                    # the floor
        shelf_x = 930                                                     # a bookshelf on the far right, behind the person
        d.rectangle((shelf_x, 110, W - 40, 560), fill=(74, 58, 46))
        books = [(150, 92, 74), (196, 160, 92), (86, 112, 134), (120, 136, 96), (172, 84, 70), (220, 206, 176), (96, 88, 120)]
        for row, y in enumerate((130, 270, 410)):
            x = shelf_x + 18
            k = row * 3
            while x < W - 70:
                w = 16 + (k * 7) % 14
                h = 92 + (k * 11) % 26
                d.rectangle((x, y + 120 - h, x + w, y + 120), fill=books[k % len(books)])
                x += w + 4
                k += 1
            d.rectangle((shelf_x, y + 120, W - 40, y + 130), fill=(60, 46, 36))
        d.ellipse((540, 380, 640, 560), fill=(64, 104, 70))               # a plant
        d.rectangle((570, 520, 610, 560), fill=(150, 112, 84))
        im = im.filter(ImageFilter.GaussianBlur(6))                       # the background out of focus, like a real interview
        d = ImageDraw.Draw(im)
        cx = 780                                                          # the person (sharp): shoulders, neck, head
        d.rounded_rectangle((cx - 190, 470, cx + 190, H + 40), radius=120, fill=(42, 54, 72))
        d.rectangle((cx - 34, 400, cx + 34, 480), fill=(196, 150, 120))
        d.ellipse((cx - 92, 200, cx + 92, 420), fill=(214, 168, 136))
        d.chord((cx - 98, 186, cx + 98, 360), 180, 360, fill=(52, 38, 30))   # hair
        out.parent.mkdir(parents=True, exist_ok=True)
        im.save(out, quality=90)
        return out

    def gfx_preview(self, theme: str, kind: str, src: str = "") -> Path:
        """One sample card in one look, over the placeholder scene (frosted looks blur what is under them)."""
        from PIL import Image, ImageFilter
        from . import gfxcards as G
        from .graphics import THEMES
        kind = kind.removesuffix(".jpg")
        if theme not in THEMES or kind not in self.GFX_SAMPLES:
            raise KeyError("no such style")
        bg_file = self.gfx_backdrop()
        import hashlib
        key = hashlib.sha1(f"{theme}:{kind}:backdrop-v1:5".encode()).hexdigest()[:20]
        out = Path(self.settings.get()["places"]["work"]) / "gfx" / f"{key}.jpg"
        if out.exists():
            return out
        W, H = 1280, 720
        bg = Image.open(bg_file).convert("RGBA").resize((W, H))
        spec = {**self.GFX_SAMPLES[kind], "style": "wipe", "at": 0.0}
        look = G.Look(theme, W, H)
        look.avoid = [(660, 180, 900, 480)]                              # the placeholder person's head: cards keep clear, as in a real render
        card = G.frame_at(G.build_scene(kind, look, spec, spec["len"]), 1.6).convert("RGBA")
        if card.size != (W, H):
            card = card.resize((W, H))
        if look.glass:                                                    # (the render frosts the picture under glass with ffmpeg)
            frosted = bg.filter(ImageFilter.GaussianBlur(22))
            bg.paste(frosted, (0, 0), card.split()[3].point(lambda a: 255 if a > 8 else 0))
        bg.alpha_composite(card)
        out.parent.mkdir(parents=True, exist_ok=True)
        bg.convert("RGB").resize((640, 360), Image.LANCZOS).save(out, quality=86)
        return out

    # ------------------------------------------------------------------ a project's right-click menu
    def item_folder(self, kind: str, iid: str) -> tuple[dict, Path | None]:
        """The project and its own folder in the videos folder (never the footage it was made from)."""
        if kind == "long":
            proj = self.projects.get(iid)
            home = Path(proj["home"]) if proj.get("home") else None
            if home and proj.get("folder") and home.resolve() == Path(proj["folder"]).resolve():
                home = None
        else:
            proj = self.store.get(iid)
            if (proj.get("source") or {}).get("app") != "studio":
                raise KeyError(iid)
            home = Path(proj["app_home"]) if proj.get("app_home") else None
        videos = Path(self.settings.get()["places"]["videos"]).expanduser().resolve()
        if home and not str(home.resolve()).startswith(str(videos) + os.sep):  # only a folder FirstCut made for it
            home = None
        return proj, home

    def item_action(self, kind: str, iid: str, what: str, b: dict) -> dict:
        proj, home = self.item_folder(kind, iid)
        job = self.store.get(proj["job"]) if kind == "long" else proj
        busy = job.get("status") in ("queued", "downloading", "transcribing", "finding", "rendering", "joining", "running")
        if what == "folder":
            return {"folder": str(home) if home and home.exists() else None}
        if what == "open-in" and kind == "long":                          # the project in another editor (Final Cut opens it itself)
            ed = next((e for e in SD.editors() if e["id"] == b.get("editor")), None)
            if not ed:
                raise ValueError("that editor isn't on this Mac")
            xml = sorted(home.glob("*.fcpxml"), key=lambda x: x.stat().st_mtime) if home and home.is_dir() else []
            if not xml:
                raise ValueError("there's no Final Cut project for this video yet")
            if ed["how"] == "opens":
                subprocess.run(["open", "-a", ed["app"], str(xml[-1])], capture_output=True, timeout=30)
                return {"ok": True}
            subprocess.run(["open", "-a", ed["app"]], capture_output=True, timeout=30)       # (Resolve imports it from its own menu)
            subprocess.run(["open", "-R", str(xml[-1])], capture_output=True, timeout=30)
            return {"ok": True, "note": f"{ed['name']} is opening. Choose File › Import › Timeline… and pick the file that is selected in Finder: "
                                        f"{xml[-1].name}"}
        if what == "song":                                                 # a new song for a finished video / batch ("" = no music)
            if busy:
                raise ValueError("it is still being made: wait for it to finish")
            song = self.song_path(b, home) if (b.get("song") or b.get("track")) else ""
            if kind == "long":
                self.runner.start(iid, "music", song=song)
            else:
                self.worker.engine.resong(iid, song)
                self.worker.wake.set()
            return {"ok": True, "song": song}
        if what == "rename":
            title = " ".join(str(b.get("title") or "").split())[:90]
            if not title:
                raise ValueError("give it a name")
            if kind == "long":
                self.projects.update(iid, title=title)
            else:
                self.store.update(iid, title=title)
            return {"ok": True, "title": title}
        if what == "remove":                                              # from FirstCut's lists; the files stay (or were just trashed)
            if busy:
                raise ValueError("it is still being made: stop it first")
            if kind == "long":
                self.projects.delete(iid)
            else:
                self.store.delete(iid)
            return {"ok": True}
        raise KeyError(what)

    # ------------------------------------------------------------------ dictation (the composer's microphone)
    def dictate(self, audio: bytes, ctype: str = "") -> dict:
        """What was said, as text: the recording from the page, transcribed by Parakeet on this Mac (nothing leaves it)."""
        from . import parakeet
        ext = ".mp4" if "mp4" in ctype or "m4a" in ctype else ".webm" if "webm" in ctype else ".wav" if "wav" in ctype else ".audio"
        work = Path(self.settings.get()["places"]["work"]) / "dictation"
        work.mkdir(parents=True, exist_ok=True)
        f = work / f"{int(time.time() * 1000)}{ext}"
        f.write_bytes(audio)
        try:
            words = parakeet.transcribe(f, self.cfg, timeout=600)
        except RuntimeError as e:
            raise RuntimeError("dictation needs the speech model: Settings › General › Run setup again" if "isn't installed" in str(e) else f"couldn't hear that: {e}") from None
        finally:
            f.unlink(missing_ok=True)
        text = " ".join(str(w.get("text") or w.get("word") or "").strip() for w in words).strip()
        text = re.sub(r"\s+([,.!?;:])", r"\1", " ".join(text.split()))
        for wrong, right in DICTATION_WORDS:                              # an editor's words, spelled the way editors write them
            text = re.sub(wrong, right, text, flags=re.I)
        return {"text": text}

    # ------------------------------------------------------------------ the music library (songs the person licensed: Artlist, Soundstripe...)
    def keys_state(self) -> dict:
        """Which online sources are connected (never the keys themselves)."""
        out = {k: bool(SD.key_get(k)) for k in SD.KEYS}
        if out.get("epidemic"):
            out["epidemic_status"] = self.cached("epidemic", 600, self._epidemic_status)
        return out

    def _epidemic_status(self) -> dict:
        try:
            from . import musicsources as MS
            return MS.epidemic_status(SD.key_get("epidemic"))
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e)[:200], "name": None}

    def smart_search(self) -> None:
        """Stock search, done better (stockfind / musicsources "better matches"): the editing AI writes extra searches (a cheap task),
        SigLIP 2 (Find footage's model) ranks pictures by what they show, CLAP ranks songs by how they sound. Each only when it is
        already on this Mac: a search never downloads a model."""
        if getattr(self, "_smart", False):
            return
        self._smart = True
        from . import musicsources as MS
        from . import stockfind as SF
        try:
            from .localai import LocalAI
            SF.configure(rewrite=SF.ai_rewriter(LocalAI(self.cfg)))
        except Exception as e:  # noqa: BLE001 - the words alone
            log.info(f"stock search without the AI's extra searches: {e}")

        def cached(model: str) -> bool:
            try:
                from huggingface_hub import try_to_load_from_cache
                return isinstance(try_to_load_from_cache(model, "config.json"), str)
            except Exception:  # noqa: BLE001
                return False

        def pictures():
            from .catalog import MODEL, SigLIP
            cat = self.runner.library() if self.runner else None
            if cat is not None and getattr(cat, "_embedder", None) is not None:
                return cat._embedder                                    # (Find footage's, already loaded)
            if not cached(MODEL):
                return None
            if getattr(self, "_siglip", None) is None:
                self._siglip = cat.embedder() if cat is not None else SigLIP(MODEL)
            return self._siglip

        def sounds():
            from . import musiclib
            lib = musiclib.for_cfg(self.cfg)
            if lib is not None and lib.embedder() is not None:
                return lib.embedder()
            if not musiclib.ClapEmbedder.available():
                return None
            if getattr(self, "_clap", None) is None:
                self._clap = musiclib.ClapEmbedder()
            return self._clap
        SF.configure(embedder=pictures)
        MS.configure(clap=sounds)

    def music_online(self, source: str, q: str, page: int = 1) -> dict:
        """Songs from Openverse (free, Creative Commons) or the person's Epidemic Sound account, to hear and use. `more`: another page
        is worth asking for (the Music page's "More songs")."""
        from . import musicsources as MS
        q = " ".join(str(q or "").split())[:200]
        page = max(1, min(12, int(page or 1)))
        if source == "openverse":
            self.smart_search()
            tracks = MS.openverse_search(q or "background music", n=40, page=page)
            return {"tracks": tracks, "page": page, "more": page < 12 and len(tracks) >= 10}
        if source == "epidemic":
            key = SD.key_get("epidemic")
            if not key:
                raise ValueError("connect your Epidemic Sound account first: Settings › Online sources")
            return {"tracks": MS.epidemic_search(key, q or "background", n=40) if page == 1 else [], "page": page, "more": False}
        raise KeyError(source)

    def music_preview(self, source: str, tid: str) -> dict:
        """A link to hear an Epidemic Sound song (short-lived: asked for when the person presses play)."""
        from . import musicsources as MS
        if source != "epidemic":
            raise KeyError(source)
        key = SD.key_get("epidemic")
        if not key:
            raise ValueError("connect your Epidemic Sound account first: Settings › Online sources")
        return {"url": MS.epidemic_preview(key, tid)}

    def music_fetch(self, track: dict) -> dict:
        """An online song, downloaded once into the app's own music folder (which joins the music library), ready to use."""
        from . import musicsources as MS
        if not isinstance(track, dict) or track.get("source") not in ("openverse", "epidemic"):
            raise ValueError("which song?")
        dest = self.root / "Music library" / "Online"
        dest.mkdir(parents=True, exist_ok=True)
        s = self.settings.get()
        folders = list((s.get("music") or {}).get("folders", []))
        if str(dest) not in folders:                                       # the online folder is part of the library from now on
            s = self.settings.update({"music": {"folders": folders + [str(dest)]}})
            self.apply_music(s)
            SD.write_overlay(self.root)
        path = MS.fetch(track, dest, SD.key_get("epidemic") if track["source"] == "epidemic" else None)
        ml = self.musiclib()
        if ml:
            ml.start()
        return {"path": str(path), "name": str(track.get("title") or Path(path).stem), "credit": MS.credit_for(path)}

    def music_favorite(self, b: dict) -> dict:
        """Heart / unheart a song (the Music page's Favourites): newest first, each song once."""
        t = b.get("track")
        if not isinstance(t, dict) or not str(t.get("id") or ""):
            raise ValueError("which song?")
        src = t.get("source") or "library"
        src = "library" if src == "mine" else src
        if src not in SD.FAV_SOURCES:
            raise ValueError("which song?")
        rec = {**t, "source": src}
        if src == "library" and not rec.get("path"):
            ml = self.musiclib()
            got = ml.get(str(t["id"])) if ml and re.fullmatch(r"[0-9a-f]{8,64}", str(t["id"])) else None
            rec["path"] = (got or {}).get("path") or ""
        with self.settings.lock:
            favs = [f for f in (self.settings.get().get("music") or {}).get("favorites", []) if (f.get("source"), f.get("id")) != (src, str(t["id"]))]
            if b.get("on", True):
                favs = [rec, *favs]
            s = self.settings.update({"music": {"favorites": favs}})
        return {"favorites": s["music"]["favorites"]}

    @staticmethod
    def downloads_dir() -> Path:
        d = Path.home() / "Downloads"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def to_downloads(self, src: Path, name: str, credit: str = "", move: bool = False) -> Path:
        """A copy in ~/Downloads under a readable name (never over another file: "Name (2)"), with "<name> - credit.txt" next to it when
        the licence asks for a credit. The same file downloaded twice is the same download."""
        from .library import safe
        d, src = self.downloads_dir(), Path(src)
        stem = safe(name, 100) or src.stem
        dst, k = d / f"{stem}{src.suffix}", 2
        while dst.exists() and not (dst.is_file() and dst.stat().st_size == src.stat().st_size):
            dst, k = d / f"{stem} ({k}){src.suffix}", k + 1
        if not dst.exists():
            part = dst.with_name(dst.name + ".part")
            if move:
                shutil.move(str(src), part)
            else:
                shutil.copy2(src, part)
            part.replace(dst)
        if credit:
            (d / f"{dst.stem} - credit.txt").write_text(credit.strip() + "\n")
        return dst

    def music_download(self, b: dict) -> dict:
        """A song into ~/Downloads, like any music site's download button: an online song is fetched (once, into the app's own music
        folder, so it's in your music too), your own songs are copied."""
        t = b.get("track")
        if not isinstance(t, dict):
            raise ValueError("which song?")
        if t.get("source") in ("openverse", "epidemic"):
            from . import musicsources as MS
            try:
                got = self.music_fetch(t)
            except MS.MusicSourceError as e:
                raise ValueError(str(e)) from None
            src, credit = Path(got["path"]), got.get("credit") or ""
            text = "\n".join(x for x in (credit, f"Licence: {t.get('license')}" if t.get("license") else "", str(t.get("page") or "")) if x)
        else:
            if t.get("id") and re.fullmatch(r"[0-9a-f]{8,64}", str(t.get("id"))) and self.musiclib():
                src = self.music_file(str(t["id"]))
            else:                                                            # a favourite whose song is no longer in the library
                src = Path(str(t.get("path") or ""))
                folders = [Path(f) for f in (self.settings.get().get("music") or {}).get("folders", [])]
                if not (src.is_absolute() and src.is_file() and any(src.is_relative_to(f) for f in folders)):
                    raise ValueError("that song isn't in your music any more")
            text = ""
        name = " - ".join(x for x in (str(t.get("artist") or "").strip(), str(t.get("title") or "").strip()) if x) or src.stem
        dst = self.to_downloads(src, name, text)
        return {"path": str(dst), "credit": text or None}

    # ------------------------------------------------------------------ stock footage (Find footage › Stock): Pixabay, Pexels, Openverse
    def stock_search(self, q: str, kinds: str = "", page: int = 1) -> dict:
        from . import stockfind as SF
        want = tuple(k for k in ("video", "photo") if k in str(kinds or "video,photo").split(",")) or ("video", "photo")
        page = max(1, min(20, int(page or 1)))
        self.smart_search()
        found = SF.search(q, want, n=30, page=page)
        return {"results": found, "sources": SF.sources(), "page": page, "more": len(found) >= 12}

    def stock_folder(self) -> Path:
        return self.root / "Footage library" / "Stock"

    def _stock_fetch(self, b: dict, dest: Path):
        """(the result the page means, its file on this Mac)"""
        import requests
        from . import stockfind as SF
        it = SF.known((b.get("item") or {}).get("id") if isinstance(b.get("item"), dict) else b.get("id"))
        if not it:
            raise ValueError("search again first: that result is from an earlier search")
        have = self.stock_folder() / it["source"] / SF.name_for(it)          # added to your B-roll before: no second download
        if have.is_file() and have.stat().st_size >= 1_000:
            return it, have
        self.make_dir(dest)
        try:
            return it, SF.fetch(it, dest)
        except requests.RequestException as e:
            raise ValueError(f"couldn't download it from {it['source'].capitalize()} ({type(e).__name__}): check the internet connection") from None

    def stock_in_library(self) -> dict:
        """The stock folder is one of the footage sources (from the first shot added), and is catalogued first: a few seconds after the
        last shot added, so adding several shots starts one run."""
        dest = str(self.stock_folder())
        s = self.settings.get()
        folders = s["library"]["folders"]
        if dest not in folders:
            if len(folders) >= 50:
                raise ValueError("that's 50 folders already: remove one in Settings › Footage library first")
            s = self.settings.update({"library": {"folders": folders + [dest]}})
            self.apply_library(s)
            SD.write_overlay(self.root)
        cat = self.runner.library() if self.runner else None
        if cat is not None:
            old = getattr(self, "_stock_timer", None)
            if old:
                old.cancel()
            self._stock_timer = threading.Timer(3.0, lambda: cat.first([dest]))
            self._stock_timer.daemon = True
            self._stock_timer.start()
        return s

    def stock_action(self, what: str, b: dict) -> dict:
        if what in ("add", "use"):                                          # "Add to my B-roll" / "Use in a long video"
            it, path = self._stock_fetch(b, self.stock_folder())
            s = self.stock_in_library()
            return {"path": str(path), "credit": it.get("credit") or None, "settings": s}
        if what == "download":                                              # into ~/Downloads (not your B-roll)
            import tempfile
            tmp = Path(tempfile.mkdtemp(prefix="stock-", dir=self.root))
            try:
                it, path = self._stock_fetch(b, tmp)
                text = "\n".join(x for x in (it.get("credit"), f"Licence: {it.get('license')}" if it.get("license") else "", it.get("page")) if x)
                from . import stockfind as SF
                name = Path(SF.name_for(it)).stem
                dst = self.to_downloads(path, name, text, move=tmp in path.parents)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            return {"path": str(dst), "credit": it.get("credit") or None}
        raise KeyError(what)

    def musiclib(self):
        """The music library (the engine's own, shared: the music model loads once), or None without music folders."""
        if not (self.settings.get().get("music") or {}).get("folders"):
            return None
        try:
            from .musiclib import for_cfg
        except ImportError:                                                   # (an engine without it: no library)
            return None
        return for_cfg(self.cfg)

    def apply_music(self, s: dict) -> None:
        """The engine follows the music folders now (the next video, and the library's catalogue)."""
        folders = [f for f in (s.get("music") or {}).get("folders", []) if f]
        data = getattr(self.cfg, "data", None)
        if isinstance(data, dict):
            data.setdefault("clipper", {})["music_lib"] = {"roots": folders, "dir": str(Path(s["places"]["catalog"]) / "Music")}
        self._musiclib = None

    def music_action(self, what: str, b: dict) -> dict:
        s = self.settings.get()
        folders = list((s.get("music") or {}).get("folders", []))
        path = os.path.normpath(str(Path(str(b.get("path") or "")).expanduser())) if b.get("path") else ""
        if what == "add":
            if not path.startswith("/") or not Path(path).is_dir():
                self.check_drive(path, "that folder")
                raise ValueError("choose the folder with your music")
            s = self.settings.update({"music": {"folders": folders + [path] if path not in folders else folders}})
            self.apply_music(s)
            SD.write_overlay(self.root)
            ml = self.musiclib()
            if ml:
                ml.start()
            return {"settings": s}
        if what == "remove":
            s = self.settings.update({"music": {"folders": [f for f in folders if f != path]}})
            self.apply_music(s)
            SD.write_overlay(self.root)
            return {"settings": s}
        if what == "catalog":
            ml = self.musiclib()
            if not ml:
                raise ValueError("add a music folder first")
            ml.start()
            return {"ok": True}
        raise KeyError(what)

    def music_search(self, q: str, n: int = 60) -> dict:
        ml = self.musiclib()
        return {"tracks": ml.search(" ".join(str(q or "").split())[:200], n=max(1, min(200, int(n)))) if ml else []}

    def music_file(self, tid: str) -> Path:
        ml = self.musiclib()
        t = ml.get(tid) if ml and re.fullmatch(r"[0-9a-f]{8,64}", tid or "") else None
        if not t:
            raise KeyError("no such song")
        return Path(t["path"])

    def song_path(self, b: dict, home: Path | None) -> str:
        """The song to use: a track from the library (by id) or an audio file on this Mac, copied into the project's Music folder, so the
        project keeps it (Final Cut points at that copy, and it stays if the download is deleted)."""
        from ..core.music import AUDIO_EXT
        src = self.music_file(str(b["track"])) if b.get("track") else Path(str(b.get("song") or "")).expanduser()
        if not src.is_absolute() or not src.is_file() or src.suffix.lower() not in AUDIO_EXT:
            raise ValueError("choose a song (an audio file like .mp3, .wav or .m4a)")
        if not home:
            return str(src)
        dst = home / "Music" / src.name
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        return str(dst)

    # ------------------------------------------------------------------ footage sources (Find footage, Settings › Footage library)
    def apply_library(self, s: dict) -> None:
        """The catalog follows the footage folders now (made on first use when the app started without any)."""
        folders = s["library"]["folders"]
        cat = self.runner.catalog if self.runner else None
        if cat is None:
            if not folders or not self.runner:
                return
            from .catalog import Catalog
            cat = Catalog(self.cfg)
            cat.busy = lambda: bool(self.runner.running or self.worker.busy_with)
            self.runner.catalog = cat
        cat.roots = [Path(f) for f in folders]
        cat.window = {"open": "always", "night": ["19:00", "07:00"]}.get(s["library"].get("when", "open"))   # (studio.overlay's rule)
        want = "account" if s["ai"]["use"] != "local" and s["library"].get("describe_with") == "account" else "local"
        if getattr(cat, "describe_with", "local") != want:               # who describes changed: a run that is going starts again with
            cat.describe_with, cat._vision = want, None                  # the new describer (after the file in hand), not half-and-half
            if getattr(cat, "thread", None) and cat.thread.is_alive():
                def restart():
                    cat.stop.set()
                    cat.thread.join(timeout=600)
                    cat.start(hours=None, night=True)
                threading.Thread(target=restart, name="catalog-restart", daemon=True).start()

    def shot_search(self, pid: str, k: int, q: str) -> dict:
        """The Replace sheet's search: this project's unused B-roll by its words, then the footage library by how it looks."""
        if not self.runner:
            raise KeyError(pid)
        return self.runner.shot_search(pid, k, q)

    def music_wave(self, source: str, track_id: str) -> dict:
        """A song's waveform for the Music page's scrub bar: {"peaks": [0..1] x 160, "dur": seconds}. Your songs by id, Openverse songs by
        the id of a recent search result (the link comes from that result, never from the page); measured once, then remembered."""
        import subprocess
        import numpy as np
        from ..core import ffmpeg
        key = f"{source}:{track_id}"
        cache = self.__dict__.setdefault("_waves", {})
        if key in cache:
            return cache[key]
        if source in ("", "mine", "library"):
            src = str(self.music_file(track_id))
        elif source == "openverse":
            from . import musicsources as MS
            src = MS.known_preview("openverse", track_id) or ""
            if not src.startswith("https://"):
                raise KeyError("search for the song again first")
        else:
            raise KeyError("no waveform for that source")
        tmp = None
        if src.startswith("https://"):                                   # (the app's ffmpeg has no certificate store: fetch it here)
            import tempfile
            import requests
            g = requests.get(src, timeout=60, headers={"User-Agent": "FirstCut/1.0"})
            g.raise_for_status()
            f = tempfile.NamedTemporaryFile(prefix="wave-", suffix=".mp3", delete=False)
            f.write(g.content)
            f.close()
            src = tmp = f.name
        try:
            r = subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-i", src, "-ac", "1", "-ar", "2000", "-f", "f32le", "-"],
                               capture_output=True, timeout=90)
        finally:
            if tmp:
                Path(tmp).unlink(missing_ok=True)
        y = np.abs(np.frombuffer(r.stdout, dtype=np.float32))
        if len(y) < 200:
            raise KeyError("that song couldn't be read")
        n = 160
        peaks = np.array([b.max() for b in np.array_split(y, n)], dtype=float)
        peaks = peaks / (np.percentile(peaks, 98) or 1.0)
        out = {"peaks": [round(float(min(1.0, p)), 3) for p in peaks], "dur": round(len(y) / 2000.0, 2)}
        if len(cache) > 200:
            cache.pop(next(iter(cache)))
        cache[key] = out
        return out

    def library_new(self) -> dict:
        """How many videos/photos each source has that aren't catalogued yet. Worked out in the background (a walk of the folders), kept
        for 10 minutes: the Find page asks, and offers to catalogue them."""
        cat = self.runner.library() if self.runner else None
        if cat is None:
            return {"checking": False, "roots": []}
        cache = getattr(self, "_new", None)
        if cache and time.time() - cache["at"] < 600:
            return {"checking": False, "roots": cache["roots"]}
        if not getattr(self, "_new_thread", None) or not self._new_thread.is_alive():
            def count():
                try:
                    found = cat.scan()
                    todo = cat.todo(found)
                    roots = [{"path": str(r), "new": sum(1 for f in todo if cat.under(f["path"], r))} for r in cat.roots if Path(r).is_dir()]
                    self._new = {"at": time.time(), "roots": roots}
                except Exception:  # noqa: BLE001
                    log.exception("counting new footage failed")
            self._new_thread = threading.Thread(target=count, name="library-new", daemon=True)
            self._new_thread.start()
        return {"checking": True, "roots": (cache or {}).get("roots", [])}

    def library_action(self, what: str, b: dict) -> dict:
        path = os.path.normpath(str(Path(str(b.get("path") or "")).expanduser())) if b.get("path") else ""
        s = self.settings.get()
        if what == "add":                                                    # a new source: catalogued first, before anything else
            if not path.startswith("/") or not Path(path).is_dir():
                self.check_drive(path, "that folder")
                raise ValueError("choose a folder of videos")
            folders = s["library"]["folders"]
            if path not in folders and len(folders) >= 50:
                raise ValueError("that's 50 folders already: remove one in Settings › Footage library first")
            recent = {**s["library"].get("recent", {}), path: time.time()}
            s = self.settings.update({"library": {"folders": folders + [path] if path not in folders else folders, "recent": recent, "look": path}})
            self.apply_library(s)
            SD.write_overlay(self.root)
            cat = self.runner.library()
            if cat is not None and b.get("now", True):
                cat.first([path])
            return {"settings": s}
        if what == "picked":                                                 # a shot picked in Find footage: remembered for Home
            shot = b.get("shot") if isinstance(b.get("shot"), dict) else None
            if not shot:
                raise ValueError("which shot?")
            key = (shot.get("id"), round(float(shot.get("start") or 0), 1))
            rest_ = [x for x in s["library"].get("picked", []) if (x.get("id"), round(float(x.get("start") or 0), 1)) != key]
            self.settings.update({"library": {"picked": [shot, *rest_][:24]}})
            return {"ok": True}
        if what == "new":                                                    # videos added to the sources since they were catalogued
            return self.library_new()
        if what == "use":                                                    # looked in: first in the menu, and chosen, next time
            if not path:
                return {"settings": self.settings.update({"library": {"look": ""}})}
            if path not in s["library"]["folders"]:
                raise KeyError(path)
            return {"settings": self.settings.update({"library": {"look": path, "recent": {**s["library"].get("recent", {}), path: time.time()}}})}
        if what == "catalog":                                                # "Catalogue it now"
            cat = self.runner.library() if self.runner else None
            if cat is None or path not in s["library"]["folders"]:
                raise KeyError(path)
            self.check_drive(path, "that folder")
            cat.first([path])
            self._new = None
            return {"ok": True}
        raise KeyError(what)

    def check_drive(self, path, what: str) -> None:
        gone = SD.missing_drive(path)
        if gone:
            raise ValueError(f"the drive “{gone}” for {what} isn't plugged in")

    @staticmethod
    def make_dir(p: Path) -> None:
        try:
            p.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ValueError(f"FirstCut can't make that folder ({e.strerror}): choose another one") from None

    # ------------------------------------------------------------------ the routes
    def get(self, rest: list[str]) -> tuple[int, dict]:
        if rest == []:
            return 200, self.state()
        if rest == ["ping"]:
            return 200, {"ok": True}
        if rest == ["busy"]:
            return 200, {"what": self.busy()}
        if len(rest) in (2, 3) and rest[0] == "live" and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", rest[1]):   # the live view (live.py)
            from . import live
            if not self.store.dir(rest[1]).is_dir():
                return 404, {"error": "no such job"}
            after = int(rest[2]) if len(rest) == 3 and rest[2].isdigit() else 0
            return 200, live.since(self.store, rest[1], after)
        if len(rest) in (2, 3) and rest[:2] == ["library", "live"]:           # Find footage's live panel: what the catalogue did since <after>
            cat = self.runner.library() if self.runner else None
            after = int(rest[2]) if len(rest) == 3 and rest[2].isdigit() else 0
            return 200, (cat.live_since(after) if cat is not None and hasattr(cat, "live_since")
                         else {"events": [], "last": 0, "status": {"running": False}})
        if rest == ["library", "new"]:                                    # new footage in the sources (Find footage offers to catalogue it)
            return 200, self.library_new()
        if rest == ["gfx-styles"]:                                        # the looks for the style picker
            return 200, {"styles": self.gfx_styles()}
        if rest == ["keys"]:
            return 200, {"keys": self.keys_state()}
        if rest == ["music"]:                                             # the music library: folders, how far its catalogue is
            ml = self.musiclib()
            return 200, ml.stats() if ml else {"tracks": 0, "roots": [], "status": {}}
        if len(rest) == 2 and rest[0] == "transcripts":                   # one transcript, with its words
            try:
                return 200, self.transcripts.get(rest[1])
            except ValueError as e:
                return 404, {"error": str(e)}
        if len(rest) in (3, 5) and rest[0] == "long" and rest[2] == "shots" and self.runner:    # the B-roll strip under a finished video
            try:
                if len(rest) == 3:
                    return 200, self.runner.shots(rest[1])
                if rest[4] == "choices" and rest[3].isdigit():
                    return 200, self.runner.shot_choices(rest[1], int(rest[3]))
            except ValueError as e:
                return 409, {"error": str(e)}
            except KeyError:
                return 404, {"error": "no such video"}
        if rest == ["editors"]:                                           # Final Cut, DaVinci Resolve...: whichever this Mac has
            return 200, {"editors": self.cached("editors", 60, SD.editors)}
        if rest == ["ai", "chatgpt", "models"]:                           # the models this ChatGPT plan offers (Settings › AI)
            try:
                return 200, {"models": self.cached("gpt-models", 600, self.chatgpt.models) if self.chatgpt else []}
            except Exception as e:  # noqa: BLE001 - not connected / offline: the menu keeps the app's choice
                return 200, {"models": [], "error": str(e)[:200]}
        return 404, {"error": "not found"}

    def post(self, rest: list[str], b: dict) -> tuple[int, dict]:
        if rest == ["settings"]:
            before = self.settings.get()
            after = self.settings.update(b)
            if any(before.get(k) != after.get(k) for k in ("ai", "places")):
                self.want_restart()
            elif before.get("music") != after.get("music"):                  # music folders: the same, at once
                self.apply_music(after)
                SD.write_overlay(self.root)
            elif before["library"] != after["library"]:                      # footage folders change at once, no restart (a long
                self.apply_library(after)                                    # catalogue run would otherwise hold the restart for hours)
                SD.write_overlay(self.root)
            return 200, {"settings": after}
        if len(rest) == 2 and rest[0] == "library":
            return 200, self.library_action(rest[1], b)
        if rest == ["music", "fetch"]:
            return 200, self.music_fetch(b.get("track"))
        if rest == ["music", "favorite"]:
            return 200, self.music_favorite(b)
        if rest == ["music", "download"]:
            return 200, self.music_download(b)
        if len(rest) == 2 and rest[0] == "stock":
            return 200, self.stock_action(rest[1], b)
        if len(rest) == 4 and rest[0] == "long" and rest[2] == "shots" and rest[3].isdigit():    # swap one B-roll shot: {choice}
            # (choice: a pool number, "speaker", or "lib:<id>:<start>:<end>" from the search box: that library shot joins the project)
            if not self.runner:
                return 404, {"error": "not found"}
            if self.runner.running:
                raise ValueError("a video is being made right now - swap the shot when it's done")
            pid, k, choice = rest[1], int(rest[3]), b.get("choice")
            self.runner.check_replace(pid, k, choice)                       # (a wrong choice is said at once, not as a failed render)
            self.runner.start(pid, "replace_shot", k=k, choice=choice)
            return 200, {"ok": True}
        if rest == ["transcripts"]:                                       # Get the transcript: {src: a path or a link}
            return 200, {"id": self.transcripts.start(str(b.get("src") or ""))}
        if len(rest) == 3 and rest[0] == "transcripts" and rest[2] == "remove":
            self.transcripts.remove(rest[1])
            return 200, {"ok": True}
        if len(rest) == 2 and rest[0] == "music":
            return 200, self.music_action(rest[1], b)
        if rest == ["keys"]:                                               # a key for Pixabay / Pexels / Epidemic Sound (into the Keychain)
            name = str(b.get("name") or "")
            SD.key_set(name, str(b.get("value") or ""))
            self._cache.pop("keys", None)
            self._cache.pop("epidemic", None)
            return 200, {"keys": self.keys_state()}
        if len(rest) == 4 and rest[0] == "item" and rest[1] in ("long", "short"):
            return 200, self.item_action(rest[1], rest[2], rest[3], b)
        if rest == ["places"]:
            return 200, self.change_place(b)
        if len(rest) == 3 and rest[0] == "ai" and rest[1] in ("chatgpt", "claude") and rest[2] in ("connect", "disconnect"):
            if rest[1] == "chatgpt":
                if rest[2] == "connect":
                    threading.Thread(target=self._chatgpt_connect, name="chatgpt-connect", daemon=True).start()
                else:
                    self.chatgpt.disconnect()
            else:
                if rest[2] == "connect":
                    self.claude.start()
                    self._use_when_ready("claude")                       # (primary only when the current one isn't a connected account)
                else:
                    self.claude.logout()
                self._cache.pop("claude", None)
            return 200, {"ok": True}
        if len(rest) == 3 and rest[0] == "styles" and rest[1] in SD.KINDS:
            return 200, self.style_action(rest[1], rest[2], b)
        if rest == ["mcp", "claude-desktop"]:                              # Settings › Connect an AI › Add to Claude Desktop
            from . import mcp_server as MCP
            self._cache.pop("mcp", None)
            return 200, MCP.add_to_claude_desktop(MCP.launcher_path())
        if rest == ["mcp", "codex"]:                                       # ...› Add to Codex (~/.codex/config.toml)
            from . import mcp_server as MCP
            self._cache.pop("mcp", None)
            return 200, MCP.add_to_codex(MCP.launcher_path())
        if rest == ["mcp", "test"]:                                        # ...› Test the connection (as an AI app would)
            from . import mcp_server as MCP
            return 200, MCP.test_launcher(MCP.launcher_path())
        if rest == ["setup", "start"]:
            return 200, {"started": self.setup.start()}
        if rest == ["long"]:
            return 201, self.make_long(b)
        if rest == ["short"]:
            return 201, self.make_short(b)
        if len(rest) == 3 and rest[0] == "short" and rest[2] == "keep":
            return 200, self.keep_clip(rest[1], b)
        if len(rest) == 3 and rest[0] == "short" and rest[2] == "remake":
            return 200, self.remake_short(rest[1], b)
        if rest == ["restart"]:                                          # vb restart: once nothing is being made
            if self.busy() and not b.get("force"):
                raise ValueError(f"{self.busy()}; restarting now would stop it")
            self.restart_wanted = True
            return 200, {"ok": True}
        if rest == ["pick"]:
            return 200, {"path": _osascript_pick(str(b.get("kind") or "folder"), str(b.get("prompt") or ""), str(b.get("start") or ""))}
        if rest == ["open"]:
            p = Path(str(b.get("path") or "")).expanduser()
            if b.get("nearest") and p.is_absolute():                     # Show in Finder: a folder made only when the clips are done
                while not p.exists() and p.parent != p:                  # opens the project folder until then
                    p = p.parent
            if not p.exists() or str(p) == "/":
                return 404, {"error": "that isn't there any more"}
            subprocess.run(["open", "-R", str(p)] if b.get("reveal") else ["open", str(p)], capture_output=True, timeout=30)
            return 200, {"ok": True}
        return 404, {"error": "not found"}

    def _chatgpt_connect(self) -> None:
        try:
            self.chatgpt.connect()
            self._maybe_primary("chatgpt")
        except Exception as e:  # noqa: BLE001 - status() shows the error
            log.warning(f"ChatGPT sign-in failed: {e}")

    def _maybe_primary(self, acct: str) -> None:
        """A newly connected account becomes the primary only when the current primary isn't a connected account (both stay connected:
        the other one is the backup)."""
        cur = self.settings.get()["ai"]["use"]
        conn = {"chatgpt": lambda: self.chatgpt.status().get("status") == "connected",
                "claude": lambda: self.claude.status().get("status") == "connected"}
        if cur != acct and not (cur in conn and conn[cur]()):
            self.settings.update({"ai": {"use": acct}})
            self.want_restart()

    def _use_when_ready(self, use: str) -> None:
        def wait():
            for _ in range(180):
                time.sleep(5)
                if self.claude.status().get("status") == "connected":
                    self._cache.pop("claude", None)
                    self._maybe_primary(use)
                    return
        threading.Thread(target=wait, name="claude-wait", daemon=True).start()


def _safe(t: str) -> str:
    return " ".join(re.sub(r"[^\w .,'()&-]+", " ", t).split())[:60] or "clip"


def _osascript_pick(kind: str, prompt: str, start: str) -> str:
    """macOS's own chooser (in a browser, without the app). The prompt and the start folder go in as arguments, never as script text."""
    what = "choose folder" if kind == "folder" else "choose file"
    start = str(Path(start).expanduser()) if start and Path(start).expanduser().is_dir() else ""
    where = " default location (POSIX file (item 2 of argv))" if start else ""
    script = ["-e", "on run argv", "-e", f"return POSIX path of ({what} with prompt (item 1 of argv){where})", "-e", "end run"]
    r = subprocess.run(["osascript", *script, (prompt or "Choose")[:80], start], capture_output=True, text=True, timeout=600)
    return r.stdout.strip().rstrip("/") if r.returncode == 0 else ""
