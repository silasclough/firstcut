"""Claude through the Claude Code CLI (`claude -p`), signed in with the owner's Claude plan: no API key and no per-call billing, it counts against the plan's usage.

    ClaudeCLI(cfg).complete_json(system, prompt, schema, model="opus", effort="high", images=None) -> (data, model_name)

Runs in an empty temp folder (so no project CLAUDE.md or memory leaks into the prompt), with no tools, no MCP servers and no skills (each would add
tens of thousands of tokens to every call); pictures go inside the message (stream-json input), shrunk to about half a megapixel.
Config (llm.claude): enabled, models {script, default, cheap}, effort, timeout_s, fallback (use the other providers when Claude fails; default false), config_dir (a separate
login: `CLAUDE_CONFIG_DIR=<dir> claude` then /login, so the pipeline uses its own account and plan limits), bin (the `claude` command, or a
path / glob such as the desktop app's own copy: the newest match is used), skip_tasks (task prefixes that never go to Claude).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from ..core.log import get_logger

log = get_logger("clipper.claude_cli")


def _small(jpeg: bytes, pixels: int = 500_000) -> bytes:
    """A picture at about half a megapixel (a contact sheet is still easy to read; tokens grow with the pixels)."""
    try:
        import io
        from PIL import Image
        im = Image.open(io.BytesIO(jpeg))
        if im.width * im.height <= pixels:
            return jpeg
        k = (pixels / (im.width * im.height)) ** 0.5
        im = im.convert("RGB").resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)
        out = io.BytesIO()
        im.save(out, "JPEG", quality=82)
        return out.getvalue()
    except Exception:  # noqa: BLE001 - the original is still a picture
        return jpeg


class ClaudeError(RuntimeError):
    pass


class ClaudeCLI:
    def __init__(self, cfg, run=subprocess.run):
        c = cfg.get("llm.claude") or {}
        self.enabled = bool(c.get("enabled", False))
        self.models = {"script": "opus", "default": "sonnet", "cheap": "haiku", **(c.get("models") or {})}
        self.effort = str(c.get("effort", "high"))
        self.efforts = {"script": self.effort, "default": self.effort, "cheap": self.effort, **(c.get("efforts") or {})}             # thinking effort per class of task (it is most of the usage)
        self.timeout = int(c.get("timeout_s", 1200))
        self.fallback = bool(c.get("fallback", False))
        self.config_dir = str(c.get("config_dir") or "")                 # a separate Claude login for the pipeline (CLAUDE_CONFIG_DIR), so its usage never touches a personal account
        self.script_prefixes = tuple(c.get("script_tasks", ["flow_script_", "flow_longform_ideas", "flow_explainer_script"]))
        self.cheap_tasks = tuple(c.get("cheap_tasks", []))
        self.skip_prefixes = tuple(c.get("skip_tasks") or ())
        self.bin_spec = str(c.get("bin") or "claude")
        self.run = run
        self.only_prefixes = tuple(c.get("tasks") or ())
        self.tripped_until = 0.0
        self.trip_why = ""
        self.fallback_on_limit = bool(c.get("fallback_on_limit", True))

    @property
    def bin(self) -> str | None:
        """The claude command: on PATH, or a path / glob (the desktop app keeps its copy in a versioned folder: the newest one)."""
        spec = os.path.expanduser(self.bin_spec)
        if "/" not in spec:
            return shutil.which(spec)
        import glob
        found = sorted(glob.glob(spec), key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p)])
        return found[-1] if found and os.access(found[-1], os.X_OK) else None

    def signed_in(self) -> bool:
        """With a login of its own (config_dir): has anyone signed in there yet? /login writes the account into <dir>/.claude.json
        ("oauthAccount"); before that every call answers "Not logged in". Without a config_dir the normal login is assumed."""
        if not self.config_dir:
            return True
        try:
            return bool(json.loads((Path(self.config_dir).expanduser() / ".claude.json").read_text()).get("oauthAccount"))
        except (OSError, ValueError, AttributeError):
            return False

    def available(self) -> bool:
        return self.enabled and self.bin is not None and time.time() >= self.tripped_until and self.signed_in()

    def why_not(self) -> str:
        """Why Claude isn't answering right now, in a few words for the page ("" when it is)."""
        if not self.enabled:
            return "switched off"
        if self.bin is None:
            return "not installed on this Mac"
        if not self.signed_in():
            return "not signed in"
        if time.time() < self.tripped_until:
            return self.trip_why or "out of usage for now"
        return ""

    def handles(self, task: str) -> bool:
        """Only these tasks go to Claude (prefixes; config llm.claude.tasks): the writing. Everything else (drawing animations, picture checks, sound cues) goes to the ChatGPT plan."""
        if self.skip_prefixes and task.startswith(self.skip_prefixes):
            return False
        return not self.only_prefixes or task.startswith(self.only_prefixes)

    def trip(self, minutes: float = 30.0, why: str = "out of usage for now") -> None:
        """The plan said it is out of usage (or not signed in): stop asking for a while (the caller moves on to its fallback) instead of
        failing every call four times."""
        self.tripped_until = time.time() + minutes * 60
        self.trip_why = why

    def save_usage(self, info: dict) -> None:
        """The plan's usage windows as of this answer (5 hours and the week, 0-1, with their reset times), for the app's sidebar."""
        if not self.config_dir:
            return
        win = info.get("unifiedWindows") or {}
        def one(k):
            w = win.get(k) or ({"utilization": info.get("utilization"), "resetsAt": info.get("resetsAt")} if info.get("rateLimitType") == k else None)
            return {"used": float(w["utilization"]), "resets": float(w.get("resetsAt") or 0)} if w and w.get("utilization") is not None else None
        try:
            d = Path(self.config_dir).expanduser()
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / "usage.json.tmp"
            tmp.write_text(json.dumps({"at": time.time(), "status": info.get("status"), "five_hour": one("five_hour"), "week": one("seven_day")}))
            tmp.replace(d / "usage.json")
        except (OSError, TypeError, ValueError):
            pass

    def model_for(self, task: str) -> str:
        if task.startswith(self.script_prefixes):
            return self.models["script"]
        if task in self.cheap_tasks:
            return self.models["cheap"]
        return self.models["default"]

    def effort_for(self, task: str) -> str:
        if task.startswith(self.script_prefixes):
            return self.efforts["script"]
        return self.efforts["cheap"] if task in self.cheap_tasks else self.efforts["default"]

    def complete_json(self, system: str, prompt: str, schema: dict, model: str = "sonnet", effort: str | None = None, images: list[bytes] | None = None) -> tuple[dict, str]:
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}         # the plan, never an API key
        if self.config_dir:
            env["CLAUDE_CONFIG_DIR"] = str(Path(self.config_dir).expanduser())
        with tempfile.TemporaryDirectory(prefix="claude_cli") as td:
            text = prompt.encode()
            cmd = [self.bin or "claude", "-p", "--model", model, "--effort", effort or self.effort, "--output-format", "stream-json", "--verbose",
                   "--json-schema", json.dumps(schema),
                   "--system-prompt", system, "--tools", "", "--no-session-persistence",
                   "--strict-mcp-config", "--disable-slash-commands"]
            # Without those two flags every call also carries the account's claude.ai connectors and skills: measured 63,500 input
            # tokens for one footage description, 2,200 without. Pictures go in the message itself (no Read tool, which took a second
            # turn and doubled it again), shrunk to about half a megapixel.
            if images:
                import base64
                content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(_small(b)).decode()}}
                           for b in images] + [{"type": "text", "text": prompt}]
                text = (json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n").encode()
                cmd += ["--input-format", "stream-json"]
            try:
                r = self.run(cmd, input=text, capture_output=True, cwd=td, env=env, timeout=self.timeout)
            except subprocess.TimeoutExpired as e:
                raise ClaudeError(f"claude did not answer within {self.timeout}s") from e
        out = (r.stdout or b"").decode(errors="replace")
        events = []                                                          # stream-json: one event per line; the last is the result
        for line in out.splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if isinstance(ev, dict):
                events.append(ev)
        if not events:
            raise ClaudeError(f"claude gave no JSON (exit {r.returncode}): {(r.stderr or b'').decode(errors='replace')[:300] or out[:300]}")
        d = next((ev for ev in reversed(events) if ev.get("type") == "result"), events[-1])
        limits = [ev["rate_limit_info"] for ev in events if ev.get("type") == "rate_limit_event" and isinstance(ev.get("rate_limit_info"), dict)]
        if limits:
            self.save_usage(limits[-1])
        if d.get("is_error"):
            raise ClaudeError(f"claude reported an error: {str(d.get('result'))[:300]}")
        data = d.get("structured_output")
        if not isinstance(data, dict):
            try:
                data = json.loads(d.get("result") or "")
            except ValueError as e:
                raise ClaudeError(f"claude's answer was not the requested JSON: {str(d.get('result'))[:200]}") from e
        used = next(iter(d.get("modelUsage") or {model: 1}))
        return data, used
