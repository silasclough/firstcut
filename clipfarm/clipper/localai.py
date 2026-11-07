"""The AI for local AI-video projects: OpenAI, paid by you, no local models.

Calls go straight to OpenAI when this Mac has OPENAI_API_KEY (in .env). If it doesn't, they are relayed through the posting
server over SSH, which holds the key - only prompts, a few small frames (JPEG) and compressed audio travel, never your
footage, and the key never leaves the server.

It also keeps count of what was used, so the page can show what a video cost.

A second app on the same Mac (the work Video Builder, config/work.yaml) can keep everything local and on its own plan instead:
    clipper.ai_videos.openai.enabled: false   never OpenAI, never the ChatGPT plan, never the posting server
    clipper.ai_videos.vision: ollama          anything with pictures goes to the local vision model (llm.ollama.vision_model)
    clipper.ai_videos.fallback: ollama        when Claude is out of usage or missing, the local text model answers instead
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from datetime import date
from pathlib import Path

from ..core.llm import LLMResult, LLMUnavailable, OpenAILLM, RemoteJson
from ..core.log import get_logger
from .chatgpt_plan import ChatGPTPlan, ChatGPTPlanError

log = get_logger("clipper.localai")
PRICES = {                                   # USD per 1M tokens (input, output) - OpenAI's list prices
    "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1": (2.00, 8.00), "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00),
    "gpt-5.4": (2.50, 15.00),
}
WHISPER_PER_MIN = 0.006
# Simple steps that go to the cheap model (clipper.ai_videos.openai.cheap_model): describing what footage and photos show, short on-screen
# text, and small choices. Everything that decides the EDIT (the story, the best moments, the script, who is the interviewer, where B-roll
# goes, the camera angles, reading your change requests) stays on the main / careful models.
CHEAP_TASKS = ("broll_describe", "photo_describe", "video_chapters", "hook_text", "key_points", "video_length", "video_kind", "music_vibe", "ad_end",
               "broll_pick", "broll_suitable")
IMAGE_TOKENS = 1100                          # one 1024x768 frame sheet at high detail on a "mini" model, roughly
TRIES = 4
LOCAL_TRIES = 2                  # a local model that gave a bad answer twice gives the same again (each try can take minutes)
LOOK_TASKS = ("broll_describe", "photo_describe")    # "what is in this shot": the ones Settings › AI › Looking at footage moves
# What each AI step is, in the words the progress line uses ("Choosing camera angles · asking Claude · try 2 of 4").
TASK_WORDS = {
    "speaker_roles": "Working out who is speaking", "speaker_check": "Checking who is speaking", "broll_describe": "Looking at your footage",
    "photo_describe": "Looking at your photos", "catalog_describe": "Describing your footage library", "stock_queries": "Finding better stock searches", "broll_plan": "Choosing B-roll", "broll_pick": "Choosing B-roll", "broll_queries": "Choosing B-roll",
    "broll_suitable": "Choosing B-roll", "highlight_candidates": "Finding the best moments", "highlight_assemble": "Putting the story together",
    "montage_story": "Planning the montage", "video_title": "Naming the video", "video_kind": "Deciding what kind of video this is", "video_length": "Deciding how long it should be", "multicam_director": "Choosing camera angles",
    "graphics_plan": "Planning the graphics", "gfx_place": "Placing the graphics", "hook_text": "Writing the opening line", "music_vibe": "Choosing the music",
    "video_title": "Writing the title", "video_chapters": "Writing the chapters", "clip_title": "Writing the titles", "key_points": "Finding the key points", "story_edit": "Shaping the story",
    "edit_style": "Reading how you like it edited", "style_merge": "Remembering how you like it edited", "revise_request": "Reading your changes",
    "sfx_cues": "Choosing sound effects", "visual_pick": "Choosing pictures", "thumb_photo_pick": "Choosing the thumbnail",
    "clip_moments": "Finding clips", "clip_starts": "Finding where clips start", "clip_endings": "Finding where clips end", "clip_edit": "Editing clips",
    "glasses_moments": "Finding moments", "clip_commentary": "Writing commentary",
}


NAMES = {"claude": "Claude", "chatgpt": "ChatGPT", "local": "this Mac's model"}


def net_error(e) -> bool:
    from . import net
    return net.looks_offline(e)


class OutOfUsage(LLMUnavailable):
    """An account said it is out of usage (or isn't signed in): the backup takes over until `until`."""

    def __init__(self, acct: str, until: float, why: str = "out of usage"):
        self.acct, self.until, self.why = acct, float(until), why
        super().__init__(f"{NAMES.get(acct, acct)} is {why} until {clock(until)}")


def clock(t: float) -> str:
    """3:40 pm (today), or Fri 3:40 pm."""
    lt = time.localtime(t)
    s = time.strftime("%I:%M %p", lt).lstrip("0").lower()
    return s if t - time.time() < 20 * 3600 else time.strftime("%a ", lt) + s


def trips_file(data_dir) -> Path:
    return Path(data_dir) / "ai_trips.json"


def read_trips(data_dir) -> dict:
    """{"claude": {"until": t, "why": ...}, ...}: accounts that ran out and when they are back (past entries left out)."""
    try:
        d = json.loads(trips_file(data_dir).read_text())
    except (OSError, ValueError):
        return {}
    now = time.time()
    return {k: v for k, v in (d if isinstance(d, dict) else {}).items() if isinstance(v, dict) and float(v.get("until") or 0) > now}


def write_trip(data_dir, acct: str, until: float, why: str) -> None:
    try:
        d = read_trips(data_dir)
        d[acct] = {"until": float(until), "why": why, "at": time.time()}
        p = trips_file(data_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(d))
        tmp.replace(p)
    except OSError:
        pass


def reset_time(usage: dict | None, fallback_s: float = 1800.0) -> float:
    """When an account that just ran out has usage again: the reset of the window that is (nearly) full, else the soonest reset still
    ahead, else half an hour from now."""
    now = time.time()
    wins = [w for w in ((usage or {}).get(k) for k in ("five_hour", "week")) if isinstance(w, dict) and float(w.get("resets") or 0) > now]
    full = [float(w["resets"]) for w in wins if float(w.get("used") or 0) >= 0.95]
    if full:
        return max(full)
    if wins:
        return min(float(w["resets"]) for w in wins)
    return now + fallback_s


def task_words(task: str) -> str:
    return TASK_WORDS.get(task) or ("Writing the script" if task.startswith("flow_") else "Thinking")


KEY_ENV = "AIVIDEOS_OPENAI_API_KEY"          # the key YOU gave the AI Videos editor, kept apart from every other key so its usage is its own


def price(model: str) -> tuple[float, float]:
    return PRICES.get(model, PRICES["gpt-4.1-mini"])


def own_key(cfg) -> str | None:
    """The AI Videos key: from the environment, else read straight from this Mac's .env (so a newly added key works without a restart)."""
    k = os.environ.get(KEY_ENV)
    if k:
        return k.strip()
    try:
        for line in (Path(cfg.root) / ".env").read_text().splitlines():
            m = re.match(rf"^\s*{KEY_ENV}\s*=\s*(.+?)\s*$", line)
            if m:
                return m.group(1).strip().strip("'\"")
    except OSError:
        pass
    return None


class Ledger:
    """What the editor's AI has cost, by day and in total (saved in data/ai_videos/usage.json). Shown as a small number on the dashboard."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.Lock()

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {"days": {}}

    def add(self, source: str, model: str, d_in: int = 0, d_out: int = 0, images: int = 0, whisper_s: float = 0.0, usd: float = 0.0) -> None:
        with self.lock:
            d = self._load()
            day = d["days"].setdefault(date.today().isoformat(), {})
            e = day.setdefault(source, {"calls": 0, "in": 0, "out": 0, "images": 0, "whisper_s": 0.0, "usd": 0.0})
            e["calls"] += 1 if (d_in or d_out) else 0
            e["in"] += int(d_in)
            e["out"] += int(d_out)
            e["images"] += int(images)
            e["whisper_s"] = round(e["whisper_s"] + float(whisper_s), 1)
            e["usd"] = round(e["usd"] + float(usd), 5)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(d))
            tmp.replace(self.path)

    def summary(self) -> dict:
        d = self._load()
        today = d["days"].get(date.today().isoformat(), {})

        def tot(days):
            t = {"calls": 0, "in": 0, "out": 0, "images": 0, "whisper_s": 0.0, "usd": 0.0}
            for day in days:
                for src in day.values():
                    for k in t:
                        t[k] += src.get(k, 0)
            return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()}
        yours = lambda days: [{"yours": x["yours"]} for x in days if "yours" in x]            # noqa: E731
        return {"today": tot([today]), "total": tot(d["days"].values()), "since": min(d["days"]) if d["days"] else None,
                "yours_today": tot(yours([today])), "yours_total": tot(yours(list(d["days"].values())))}


def ledger_for(cfg) -> Ledger:
    return Ledger(Path(cfg.root) / cfg.get("clipper.ai_videos.data_dir", "data/ai_videos") / "usage.json")


class LocalAI:
    def __init__(self, cfg, host: str | None = None):
        self._cfg = cfg
        self.model = str(cfg.get("clipper.ai_videos.openai.model", "gpt-4.1-mini"))
        self.smart = str(cfg.get("clipper.ai_videos.openai.plan_model", "gpt-4.1"))          # the careful model: following the script, telling the speakers apart
        self.script_model = str(cfg.get("clipper.ai_videos.openai.script_model", "gpt-5.4"))
        self.script_effort = str(cfg.get("clipper.ai_videos.openai.script_reasoning_effort", "high"))
        self.anim_effort = str(cfg.get("clipper.ai_videos.openai.anim_reasoning_effort", "medium"))        # drawing an animation does not need the deepest thinking (the render and the picture check catch mistakes)
        self.smart_tasks = ("speaker_",)
        self.cheap = str(cfg.get("clipper.ai_videos.openai.cheap_model", "gpt-4.1-nano"))    # the simple steps: describing, short text, small choices
        self.cheap_tasks = tuple(cfg.get("clipper.ai_videos.openai.cheap_tasks", CHEAP_TASKS))
        self.language = str(cfg.get("clipper.language", "en"))
        if host is None:
            from .auto import _target
            tgt = _target(cfg)
            host = (cfg.get("clipper.ai_videos.openai.host") or (tgt[0] if tgt else None))
        self.host = host
        self.usage = {"calls": 0, "in": 0, "out": 0, "images": 0, "whisper_s": 0.0}
        self._lock = threading.Lock()
        self._doing: dict[int, dict] = {}                                # thread -> the AI call it is waiting on (now())
        self._tags: tuple[float, set] = (0.0, set())                     # the models this Mac's Ollama has (local_ready)
        self.offline_note = ""                                           # said once per run when it carries on offline
        self.key = own_key(cfg)
        self.ledger = ledger_for(cfg)
        self.chatgpt_plan = ChatGPTPlan(cfg)
        from .claude_cli import ClaudeCLI
        self.claude = ClaudeCLI(cfg)
        self.openai_on = bool(cfg.get("clipper.ai_videos.openai.enabled", True))
        self.chatgpt_on = bool(cfg.get("clipper.ai_videos.chatgpt_plan.enabled", self.openai_on))     # (the app: a ChatGPT sign-in without the API)
        if not self.openai_on:
            self.host, self.key = None, None
        self.vision_local = str(cfg.get("clipper.ai_videos.vision") or "") == "ollama"
        self.local_fallback = str(cfg.get("clipper.ai_videos.fallback") or "") == "ollama"
        # The app (FirstCut): both accounts can be connected. `primary` makes the editing decisions; when it runs out of usage the `backup`
        # ("claude" | "chatgpt" | "local" | "" = wait | "auto" = the other account if connected, else this Mac) takes the calls until the
        # primary's reset time, then the primary is used again by itself. Without `primary` (the posting server) the old order stands.
        self.primary = str(cfg.get("clipper.ai_videos.primary") or "")
        self.backup_choice = str(cfg.get("clipper.ai_videos.backup") if cfg.get("clipper.ai_videos.backup") is not None else "auto")
        self.data_dir = Path(cfg.root) / cfg.get("clipper.ai_videos.data_dir", "data/ai_videos")
        oc = cfg.get("llm.ollama") or {}
        self.ollama_cfg = {"base_url": oc.get("base_url", "http://localhost:11434"), "timeout_s": 900, "keep_alive": "10m",
                           "model": str(cfg.get("clipper.ai_videos.ollama_model") or oc.get("model") or "qwen3:14b"),
                           "vision_model": str(cfg.get("clipper.ai_videos.vision_model") or oc.get("vision_model") or "qwen3-vl:8b")}

    # -- where the calls go ------------------------------------------------------------
    def direct(self) -> bool:
        return self.openai_on and bool(self.key or os.environ.get("OPENAI_API_KEY"))

    def available(self) -> tuple[bool, str]:
        if self.primary:                                                 # the app: the primary, or the backup while it is out of usage
            acct, out = self.in_use()
            label = {"claude": "Claude (your plan)", "chatgpt": "ChatGPT plan", "local": "local AI on this Mac"}.get(acct, acct)
            if out and acct == self.primary:
                return False, f"{NAMES.get(acct, acct)} is {out.get('why') or 'out of usage'} until {clock(out['until'])}"
            if acct == "local":
                return (True, label) if self.local_ready() else (False, "this Mac's model isn't downloaded yet: Settings › AI")
            if self.connected(acct):
                return True, label + (f" ({NAMES[self.primary]} is out of usage until {clock(out['until'])})" if out else "")
            return False, f"{NAMES.get(acct, acct)} isn't connected: Settings › AI › Continue with {NAMES.get(acct, acct)}"
        if self.claude.available():
            return True, "Claude (your plan)"
        if self.chatgpt_on and self.chatgpt_plan.connected():
            return True, "ChatGPT plan"
        if not self.openai_on:
            why = self.claude.why_not() if self.claude.enabled else ""
            if self.local_fallback:
                return True, "local AI on this Mac" + (f" (Claude {why})" if why else " (Ollama)")
            if self.chatgpt_on:
                return False, "ChatGPT isn't signed in: Settings › AI › Continue with ChatGPT (or turn on the local fallback)"
            if self.claude.enabled:
                return False, f"Claude is {why or 'not available'} on this Mac and the local AI fallback is off (clipper.ai_videos.fallback: ollama)"
            return False, "no AI is set up for the editing decisions: Settings › AI"
        if self.chatgpt_plan.connected():
            return True, "ChatGPT plan"
        if self.key:
            return True, "OpenAI (your AI Videos key)"
        if self.direct():
            return True, "OpenAI (this Mac's key)"
        if self.host:
            return True, f"OpenAI via {self.host}"
        return False, "no OPENAI_API_KEY here and no server to relay through (set clipper.ai_videos.openai.host)"

    # -- what it is waiting on (the progress line) ---------------------------------------
    def _mark(self, task: str, via: str, attempt: int, of: int) -> None:
        with self._lock:
            d = self._doing.get(threading.get_ident())
            self._doing[threading.get_ident()] = {"task": task, "via": via, "try": attempt, "of": of,
                                                  "since": d["since"] if d and d["task"] == task else time.time()}

    def local_ready(self, images: bool = False) -> bool:
        """This Mac's model for the job (the picture model, or the editing model) is downloaded and Ollama answers."""
        import urllib.request
        if time.time() - self._tags[0] > 30:
            names: set = set()
            try:
                with urllib.request.urlopen(self.ollama_cfg["base_url"].rstrip("/") + "/api/tags", timeout=3) as r:
                    for m in json.loads(r.read()).get("models") or []:
                        n = str(m.get("name") or "")
                        names |= {n, n.removesuffix(":latest")}
            except Exception:  # noqa: BLE001 - Ollama not running / not installed: nothing here
                pass
            self._tags = (time.time(), names)
        want = self.ollama_cfg["vision_model" if images else "model"]
        return want in self._tags[1] or want.removesuffix(":latest") in self._tags[1]

    def _offline(self, task, system, prompt, schema, validate, images) -> LLMResult:
        if self.local_ready(images=bool(images)):
            if not self.offline_note:
                self.offline_note = "You're offline: this Mac's model is making the decisions until the internet is back."
                log.warning(f"{task}: offline, so this Mac's model answers")
            return self._ollama(task, system, prompt, schema, validate, images)
        raise LLMUnavailable("you're offline, and this Mac has no " + ("picture model" if images else "editing model") + " to use instead. "
                             "Connect to the internet, or download this Mac's model in Settings › AI (when you're back online) so editing "
                             "works offline too")

    def now(self) -> str:
        """The AI call this video is waiting on, in plain words, or "" ("Choosing camera angles · asking Claude · try 2 of 4 · 6 min").
        With several at once (describing shots in parallel) the longest-waiting one, and how many."""
        with self._lock:
            calls = list(self._doing.values())
        if not calls:
            return ""
        d = min(calls, key=lambda c: c["since"])
        mins = (time.time() - d["since"]) / 60
        bits = [task_words(d["task"]) + (f" ({len(calls)} at once)" if len(calls) > 1 else ""), "asking " + d["via"]]
        if d["try"] > 1:
            bits.append(f"try {d['try']} of {d['of']}")
        if mins >= 1:
            bits.append(f"{mins:.0f} min")
        return " · ".join(bits)

    # -- JSON answers ------------------------------------------------------------------
    def complete_json(self, task: str, system: str, prompt: str, schema: dict, validate=None, images=None, **kw) -> LLMResult:
        from . import glossary, net
        if task not in LOOK_TASKS and task != "catalog_describe":             # writing: names spelled the person's way
            system = system + glossary.prompt_note(glossary.terms(getattr(self, "_cfg", None)))
        try:
            if not net.online():                                         # offline: this Mac's model, if it has one
                return self._offline(task, system, prompt, schema, validate, images)
            try:
                return self._complete_json(task, system, prompt, schema, validate, images, **kw)
            except LLMUnavailable as e:                                  # the connection went in the middle: check, and carry on here
                if net.looks_offline(e):
                    net.changed()
                    if not net.online():
                        return self._offline(task, system, prompt, schema, validate, images)
                raise
        finally:
            with self._lock:
                self._doing.pop(threading.get_ident(), None)

    def _complete_json(self, task: str, system: str, prompt: str, schema: dict, validate=None, images=None, **kw) -> LLMResult:
        if self.primary:
            return self._switched(task, system, prompt, schema, validate, images, **kw)
        ok, why = self.available()
        if not ok:
            raise LLMUnavailable(why)
        if images and self.vision_local and task in LOOK_TASKS:                 # describing footage: on this Mac when the person chose that.
            return self._ollama(task, system, prompt, schema, validate, images)  # (Decisions that come with pictures, like which camera to show,
                                                                                # go to the editing AI: the local picture model can't make them.)
        last: Exception | None = None
        script_task = task.startswith("flow_")
        model = kw.get("model") or (self.script_model if script_task else self.smart if task.startswith(self.smart_tasks) else self.cheap if task in self.cheap_tasks else self.model)
        if self.claude.available() and self.claude.handles(task):               # the owner's Claude plan first: no per-call billing, and the better writer
            try:
                return self._claude_call(task, system, prompt, schema, validate, images, kw, switch=False)
            except OutOfUsage as e:                                             # out of usage / not signed in: the ChatGPT plan takes over
                last = e
            except LLMUnavailable as e:
                last = e
                if not self.claude.fallback:
                    raise
        if self.chatgpt_on and self.chatgpt_plan.connected():
            try:
                return self._chatgpt(task, system, prompt, schema, validate, images, model, script_task)
            except LLMUnavailable:
                if self.openai_on or not self.local_fallback:
                    raise
                log.warning(f"{task}: the ChatGPT plan didn't answer; the local AI takes it")
        if not self.openai_on:
            if self.local_fallback:
                return self._ollama(task, system, prompt, schema, validate, images)
            raise LLMUnavailable(f"{task}: Claude didn't answer ({last}) and nothing else is allowed here")
        for k in range(TRIES):
            self._mark(task, "OpenAI", k + 1, TRIES)
            try:
                if self.direct():
                    direct_cfg = {"model": model, "timeout_s": 300, "api_key": self.key}
                    if script_task:
                        direct_cfg["reasoning_effort"] = self.anim_effort if task == "flow_anim_design" else self.script_effort
                    res = OpenAILLM(direct_cfg).complete_json(system, prompt, schema, images=images)
                else:
                    res = RemoteJson(self.host, timeout_s=420).complete_json(task, system, prompt, schema, images=images, model=model)
                if validate:
                    validate(res.data)
                self._count(res, images, model)
                log.info(f"{task}: ok via {res.provider}/{res.model}", extra={"usage": res.usage})
                return res
            except Exception as e:  # noqa: BLE001 - network blips and rate limits: wait and try again
                last = e
                log.warning(f"{task}: attempt {k + 1} failed: {type(e).__name__}: {e}")
                time.sleep(min(30, 3 * (k + 1)))
        raise LLMUnavailable(f"{task}: OpenAI didn't answer after {TRIES} tries: {last}")

    def _claude_call(self, task, system, prompt, schema, validate, images, kw, switch: bool) -> LLMResult:
        """Claude with retries. OutOfUsage when the plan is out of usage or not signed in (it is also tripped, so other calls skip it),
        LLMUnavailable after TRIES bad answers."""
        from .claude_cli import ClaudeError
        cmodel = kw.get("claude_model") or self.claude.model_for(task)
        last: Exception | None = None
        for k in range(TRIES):
            self._mark(task, "Claude", k + 1, TRIES)
            try:
                data, used = self.claude.complete_json(system, prompt, schema, model=cmodel, images=images, effort=self.claude.effort_for(task))
                res = LLMResult(data, "claude_plan", used, {})
                if validate:
                    validate(res.data)
                with self._lock:
                    self.usage["calls"] += 1
                log.info(f"{task}: ok via Claude plan/{used}")
                return res
            except ClaudeError as e:
                last = e
                log.warning(f"{task}: Claude attempt {k + 1} failed: {e}")
                if re.search(r"usage limit|session limit|spend limit|rate limit|limit reached|hit your|out of (extra )?usage", str(e), re.I) and (
                        switch or self.claude.fallback_on_limit):
                    until = reset_time(self._usage_of("claude"))
                    self.claude.trip(max(1.0, (until - time.time()) / 60))
                    raise OutOfUsage("claude", until) from e
                if re.search(r"not logged in|/login|log in again|invalid api key|authenticat|unauthori[sz]ed|oauth token", str(e), re.I):
                    self.claude.trip(360, "not signed in")                  # asking again won't sign anyone in: the other one answers for 6 h
                    raise OutOfUsage("claude", time.time() + 6 * 3600, "not signed in") from e
            except Exception as e:  # noqa: BLE001 - a bad answer (validate): try again
                last = e
                log.warning(f"{task}: Claude attempt {k + 1} failed: {type(e).__name__}: {e}")
            if k + 1 < TRIES:
                time.sleep(min(20, 3 * (k + 1)))
        raise LLMUnavailable(f"{task}: Claude (the plan) didn't answer after {TRIES} tries: {last}")

    def _usage_of(self, acct: str) -> dict | None:
        try:
            if acct == "claude":
                return json.loads((Path(self.claude.config_dir).expanduser() / "usage.json").read_text()) if self.claude.config_dir else None
            if acct == "chatgpt":
                return self.chatgpt_plan.usage()
        except (OSError, ValueError, AttributeError):
            return None
        return None

    def connected(self, acct: str) -> bool:
        """Signed in and able to answer (out of usage or not)."""
        if acct == "claude":
            return bool(self.claude.enabled and self.claude.bin is not None and self.claude.signed_in())
        if acct == "chatgpt":
            try:
                return bool(self.chatgpt_on and self.chatgpt_plan.connected())
            except Exception:  # noqa: BLE001
                return False
        if acct == "local":
            return self.local_ready()
        return False

    def backup(self) -> str:
        """The backup, with "auto" worked out now: the other account if it is connected, else this Mac's model if it is downloaded."""
        b = self.backup_choice
        if b == "auto":
            other = {"claude": "chatgpt", "chatgpt": "claude"}.get(self.primary, "")
            b = other if other and self.connected(other) else "local" if self.primary != "local" and (self.local_fallback or self.local_ready()) else ""
        return "" if b == self.primary else b

    def in_use(self) -> tuple[str, dict | None]:
        """(the account answering now, the primary's out-of-usage record or None)."""
        out = read_trips(self.data_dir).get(self.primary)
        return (self.backup() or self.primary, out) if out else (self.primary, None)

    def _switched(self, task, system, prompt, schema, validate, images, **kw) -> LLMResult:
        """FirstCut: the primary account, or the backup while the primary is out of usage (until its reset; then the primary again)."""
        if images and self.vision_local and task in LOOK_TASKS:                 # describing footage: this Mac, when the person chose that
            return self._ollama(task, system, prompt, schema, validate, images)
        script_task = task.startswith("flow_")
        model = kw.get("model") or (self.script_model if script_task else self.smart if task.startswith(self.smart_tasks) else self.cheap if task in self.cheap_tasks else self.model)
        order = [self.primary] + ([self.backup()] if self.backup() else [])
        if self.local_fallback and "local" not in order:                       # "finish on this Mac": the last resort, so nothing stops halfway
            order.append("local")
        trips = read_trips(self.data_dir)
        outs: list[OutOfUsage] = []
        failed: list[Exception] = []
        for acct in order:
            if acct in trips:
                outs.append(OutOfUsage(acct, trips[acct]["until"], trips[acct].get("why") or "out of usage"))
                continue
            if acct != "local" and not self.connected(acct):
                if acct == self.primary:
                    outs.append(OutOfUsage(acct, time.time() + 300, "not signed in"))
                continue
            try:
                if acct == "claude":
                    if not self.claude.handles(task):
                        continue
                    res = self._claude_call(task, system, prompt, schema, validate, images, kw, switch=True)
                elif acct == "chatgpt":
                    res = self._chatgpt(task, system, prompt, schema, validate, images, model, script_task)
                else:
                    res = self._ollama(task, system, prompt, schema, validate, images)
            except OutOfUsage as e:
                if e.why == "out of usage":
                    write_trip(self.data_dir, e.acct, e.until, e.why)            # every video (and the sidebar) knows until when
                log.warning(f"{task}: {e}" + (f"; {NAMES.get(order[-1])} takes over" if acct != order[-1] else ""))
                outs.append(e)
                continue
            except LLMUnavailable as e:                                          # it failed (an expired sign-in, bad answers): the next one
                if net_error(e) or acct == order[-1]:                             # tries, unless it was the connection (offline handles that)
                    raise
                log.warning(f"{task}: {NAMES.get(acct, acct)} failed ({e}); {NAMES.get(order[order.index(acct) + 1])} takes it")
                failed.append(e)
                continue
            if acct != self.primary and outs:
                note = f"{NAMES[self.primary]} is {outs[0].why} until {clock(outs[0].until)}: {NAMES[acct]} is making the decisions meanwhile."
                if not self.offline_note:                                       # (the job's progress line shows it)
                    self.offline_note = note
                    log.info(note)
            return res
        if failed and not outs:
            raise failed[-1]
        if outs:
            first = outs[0]
            more = " (and the backup too)" if len(outs) > 1 else ""
            raise LLMUnavailable(f"{NAMES.get(first.acct, first.acct)} is {first.why} until {clock(first.until)}{more}. Pick a backup in "
                                 "Settings › AI (When it runs out of usage) to carry on meanwhile" if not self.backup() else
                                 f"{NAMES.get(first.acct, first.acct)} is {first.why} until {clock(first.until)}{more}")
        raise LLMUnavailable({"claude": "Claude isn't connected: Settings › AI › Continue with Claude",
                              "chatgpt": "ChatGPT isn't connected: Settings › AI › Continue with ChatGPT",
                              "local": "this Mac's model isn't downloaded yet: Settings › AI › Only this Mac › Download"}.get(self.primary, "no AI is set up"))

    def _chatgpt(self, task, system, prompt, schema, validate, images, model, script_task) -> LLMResult:
        """The ChatGPT plan, with retries; LLMUnavailable when it can't answer."""
        last: Exception | None = None
        # Never silently fall back to a metered API key after the creator selected their ChatGPT plan.
        for k in range(TRIES):                                              # a stream that ends early or times out is retried, as on the API route
            self._mark(task, "ChatGPT", k + 1, TRIES)
            try:
                data, used_model = self.chatgpt_plan.complete_json(
                    system, prompt, schema, model=model, images=images,
                    reasoning_effort=(self.anim_effort if task in ("flow_anim_design",) else self.script_effort) if script_task else None,
                )
                res = LLMResult(data, "chatgpt_plan", used_model, {})
                if validate:
                    validate(res.data)
                with self._lock:
                    self.usage["calls"] += 1
                log.info(f"{task}: ok via ChatGPT plan/{used_model}")
                return res
            except ChatGPTPlanError as e:
                last = e
                log.warning(f"{task}: ChatGPT plan attempt {k + 1} failed: {e}")
                if re.search(r"usage limit reached|rate limit|limit reached|too many requests", str(e), re.I):
                    raise OutOfUsage("chatgpt", reset_time(self._usage_of("chatgpt"))) from e
                if k + 1 < TRIES and not re.search(r"sign|auth|connect|credential|keychain|expired|denied", str(e), re.I):
                    time.sleep(min(20, 3 * (k + 1)))
                    continue
                raise LLMUnavailable(str(e)) from e
            except Exception as e:  # noqa: BLE001 - a bad answer (validate) or a network blip: try again
                last = e
                log.warning(f"{task}: ChatGPT plan attempt {k + 1} failed: {type(e).__name__}: {e}")
                if k + 1 >= TRIES:
                    raise LLMUnavailable(f"{task}: the ChatGPT plan didn't answer after {TRIES} tries: {e}") from e
                time.sleep(min(20, 3 * (k + 1)))
        raise LLMUnavailable(f"{task}: the ChatGPT plan didn't answer: {last}")

    def status(self) -> list[tuple[str, bool, str]]:
        """(name, ok, why) like LLMRouter.status(), for code written for the routers (Engine.status)."""
        ok, why = self.available()
        return [("ai", ok, why)]

    def _ollama(self, task: str, system: str, prompt: str, schema: dict, validate=None, images=None) -> LLMResult:
        """The local models (Ollama): the vision model for pictures, the text model when Claude can't answer."""
        from ..core.llm import OllamaLLM
        llm = OllamaLLM({**self.ollama_cfg, "num_ctx": 16384 if images else 32768})
        last: Exception | None = None
        for k in range(LOCAL_TRIES):
            self._mark(task, "this Mac's model", k + 1, LOCAL_TRIES)
            try:
                res = llm.complete_json(system, prompt, schema, images=images)
                if validate:
                    validate(res.data)
                with self._lock:
                    self.usage["calls"] += 1
                log.info(f"{task}: ok via local {res.model}")
                return res
            except Exception as e:  # noqa: BLE001 - a bad answer or a busy model: try again
                last = e
                log.warning(f"{task}: local attempt {k + 1} failed: {type(e).__name__}: {e}")
                if k + 1 < LOCAL_TRIES:
                    time.sleep(min(20, 3 * (k + 1)))
        raise LLMUnavailable(f"{task}: the local AI didn't answer after {LOCAL_TRIES} tries: {last}")

    def _count(self, res: LLMResult, images, model: str | None = None) -> None:
        u = res.usage or {}
        with self._lock:
            self.usage["calls"] += 1
            self.usage["in"] += int(u.get("input_tokens") or 0)
            self.usage["out"] += int(u.get("output_tokens") or 0)
            self.usage["images"] += len(images or [])
        pin, pout = price(model or self.model)
        with self._lock:
            self.usage["usd"] = self.usage.get("usd", 0.0) + int(u.get("input_tokens") or 0) * pin / 1e6 + int(u.get("output_tokens") or 0) * pout / 1e6
        try:
            self.ledger.add("yours" if self.key else "server", model or self.model, int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0),
                            len(images or []), 0.0, int(u.get("input_tokens") or 0) * pin / 1e6 + int(u.get("output_tokens") or 0) * pout / 1e6)
        except OSError:
            pass

    # -- speech to text ----------------------------------------------------------------
    def transcribe_mp3(self, mp3: Path, seconds: float) -> list[dict]:
        if not self.openai_on:                                           # (reached only when Parakeet could not run: say so at once)
            raise LLMUnavailable("Parakeet (the speech-to-text model) is not installed on this Mac or would not run, and OpenAI "
                                 "transcription is switched off here (clipper.ai_videos.parakeet_bin)")
        last: Exception | None = None
        for k in range(TRIES):
            try:
                if self.direct():
                    from .openai_audio import transcribe_mp3
                    words = transcribe_mp3(mp3, self.language, key=self.key)
                else:
                    import json
                    proc = subprocess.run(
                        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", self.host,
                         f"cd ~/clipfarm && .venv/bin/python -m clipfarm openai-transcribe --language {self.language}"],
                        input=Path(mp3).read_bytes(), capture_output=True, timeout=1500)
                    if proc.returncode:
                        raise RuntimeError((proc.stderr or b"")[-300:].decode(errors="replace"))
                    words = json.loads(proc.stdout.decode().strip().splitlines()[-1])
                with self._lock:
                    self.usage["whisper_s"] += seconds
                try:
                    self.ledger.add("yours" if self.key else "server", "whisper-1", whisper_s=seconds, usd=seconds / 60 * WHISPER_PER_MIN)
                except OSError:
                    pass
                return words
            except Exception as e:  # noqa: BLE001
                last = e
                log.warning(f"transcription attempt {k + 1} failed: {e}")
                time.sleep(min(30, 3 * (k + 1)))
        raise LLMUnavailable(f"transcription failed after {TRIES} tries: {last}")

    # -- what it cost ------------------------------------------------------------------
    def cost(self) -> float:
        pin, pout = price(self.model)
        u = self.usage
        if u.get("usd"):                                                  # each call was priced at the model it actually used
            return round(u["usd"] + u["whisper_s"] / 60 * WHISPER_PER_MIN, 4)
        tokens_in = u["in"] + (u["images"] * IMAGE_TOKENS if not u["in"] else 0)
        return round(tokens_in * pin / 1e6 + u["out"] * pout / 1e6 + u["whisper_s"] / 60 * WHISPER_PER_MIN, 4)

    # code written for the other routers asks for these
    def router_for_now(self):
        return self, "openai"

    def vision_router(self):
        return self, "openai"

    def creative_router(self):
        return self


def estimate_cost(model: str, talk_s: float, broll_files: int, revisions: int = 2, transcribe: str = "openai",
                  cheap_model: str = "gpt-4.1-nano") -> dict:
    """What making a video from this footage should cost (a forecast, within a factor of ~2)."""
    pin, pout = price(model)
    cin, cout = price(cheap_model)
    words = talk_s / 60 * 140
    chunks = max(1, int(words / 700) + 1)
    story_in, story_out = chunks * 1700, chunks * 420
    hl_in, hl_out = (chunks * 1700 + 3000, chunks * 600 + 400) if chunks > 4 else (0, 0)       # long footage: candidates, then the pick
    vis_in, vis_out = broll_files * (2 * IMAGE_TOKENS + 400), broll_files * 350
    misc_in, misc_out = 3500, 600                                                              # cutaways, music, hook
    rev_in, rev_out = revisions * 6000, revisions * 600
    tin = story_in + hl_in + vis_in + misc_in + rev_in
    tout = story_out + hl_out + vis_out + misc_out + rev_out
    llm = ((story_in + hl_in + rev_in) * pin + (story_out + hl_out + rev_out) * pout                  # the edit itself: main model
           + (vis_in + misc_in) * cin + (vis_out + misc_out) * cout) / 1e6                           # describing + short text: cheap model
    whisper = talk_s / 60 * WHISPER_PER_MIN if transcribe == "openai" else 0.0
    return {"llm_usd": round(llm, 3), "transcription_usd": round(whisper, 2), "total_usd": round(llm + whisper, 2),
            "model": model, "tokens_in": tin, "tokens_out": tout, "talk_hours": round(talk_s / 3600, 2)}
