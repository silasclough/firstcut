"""FirstCut for AI apps: an MCP server (Model Context Protocol, stdio) so Claude Desktop, Claude Code, Cursor, LM Studio and any other
MCP app can use FirstCut: make a video from a folder, cut clips, get a transcript, search the footage, change or finish a video.

North star: "Let your AI use FirstCut." It is a door into the app that is already running on this Mac, not a second copy: every tool is a
thin wrapper over the app engine's own local HTTP API (the port is in <home>/server.json). Off until the person turns it on in
Settings › Connect an AI. Nothing deletes; long jobs return an id at once and the AI checks on them with project_status.

    python -m clipfarm mcp          (the app's launcher, Contents/Resources/bin/firstcut-mcp, runs this with the app's own Python)

JSON-RPC 2.0 over stdin/stdout, one message per line (MCP 2025-06-18): initialize, ping, tools/list, tools/call, resources/list,
resources/read. Written by hand: no dependency beyond the standard library (Pillow, when present, makes search pictures smaller).
Nothing but protocol messages goes to stdout; notes go to stderr.
"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PROTOCOL = "2025-06-18"
SERVER = {"name": "firstcut", "version": "1.0"}
OFF = "FirstCut isn't letting AI apps use it yet: turn on Settings › Connect an AI in FirstCut, then try again."
CLOSED = "FirstCut isn't open on this Mac. Open FirstCut, then try again."
NOTES = ("Chapters.txt", "Description.txt", "Quality check.md", "plan.md", "credits.txt")
BUNDLE_ID = "app.firstcut.mac"


class ToolError(Exception):
    """A plain-language failure, shown to the AI as the tool's result (isError)."""


# ------------------------------------------------------------------------------------------------ the app engine
def home() -> Path:
    from . import studio
    return studio.home()


def enabled(h: Path | None = None) -> bool:
    try:
        s = json.loads(((h or home()) / "settings.json").read_text())
        return bool((s.get("mcp") or {}).get("enabled"))
    except (OSError, ValueError):
        return False


class Engine:
    """The running app's local API (127.0.0.1:<port from server.json>). Opens the app once if it isn't running."""

    def __init__(self, h: Path | None = None, opener=None, wait_s: float = 40.0):
        self.home = h or home()
        self.opener = opener or self._open_app
        self.wait_s = wait_s
        self.base = ""

    def _port(self) -> int | None:
        try:
            return int(json.loads((self.home / "server.json").read_text())["port"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _answers(self, port: int) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/app/ping", timeout=3) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _open_app() -> None:
        subprocess.run(["open", "-g", "-b", BUNDLE_ID], capture_output=True, timeout=20)

    def connect(self) -> str:
        if self.base:
            return self.base
        port = self._port()
        if port and self._answers(port):
            self.base = f"http://127.0.0.1:{port}"
            return self.base
        try:
            self.opener()
        except Exception:  # noqa: BLE001
            pass
        t0 = time.time()
        while time.time() - t0 < self.wait_s:
            time.sleep(1.0)
            port = self._port()
            if port and self._answers(port):
                self.base = f"http://127.0.0.1:{port}"
                return self.base
        raise ToolError(CLOSED)

    def call(self, method: str, path: str, body: dict | None = None, raw: bool = False, timeout: float = 60.0):
        url = self.connect() + path
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = r.read()
                return out if raw else (json.loads(out) if out else {})
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("error") or e.reason
            except Exception:  # noqa: BLE001
                msg = e.reason
            raise ToolError(str(msg)) from None
        except urllib.error.URLError:
            self.base = ""
            raise ToolError(CLOSED) from None

    def get(self, path: str, **kw):
        return self.call("GET", path, **kw)

    def post(self, path: str, body: dict | None = None, **kw):
        return self.call("POST", path, body or {}, **kw)


# ------------------------------------------------------------------------------------------------ tools
def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or [], "additionalProperties": False}


S = {"type": "string"}
PROJECT = {**S, "description": "Which video: its id, or its title or part of it (e.g. 'a workshop shoot'). Leave out for the most recent one."}
TOOLS = [
    {"name": "list_projects", "description": "Every video FirstCut has made or is making on this Mac (long videos, short-clip batches, transcripts), newest first, with status.",
     "inputSchema": _obj({})},
    {"name": "project_status", "description": "How a video, a batch of short clips or a transcript is doing: stage, progress, what it is waiting on; when finished, where the files are, its chapters and quality-report summary, or (a transcript) the text.",
     "inputSchema": _obj({"project": PROJECT})},
    {"name": "find_files", "description": "Find footage on this Mac (read-only): lists the videos, photos or folders in a folder under the person's home or an external drive, with sizes and video lengths. 'where' can be words like 'Desktop/videos/raw', '~/Movies', 'Downloads'; names match without caring about capitals. Use it to turn 'the raw folder in my videos folder on my desktop' into a path.",
     "inputSchema": _obj({"where": {**S, "description": "A folder: a full path, or relative to the home folder, e.g. 'Desktop/videos/raw'."},
                          "query": {**S, "description": "Only names containing these words (optional)."},
                          "kind": {"type": "string", "enum": ["videos", "photos", "folders", "all"], "description": "What to list (default videos)."}},
                         ["where"])},
    {"name": "make_long_video", "description": "Start editing a folder of footage into one video (reads the footage where it is; nothing copied). Returns at once; the video takes minutes, check with project_status. Two ways to use it: (1) leave the cut to FirstCut: give notes (and a numbered outline in them if there is one) and it plans the edit; (2) take control: give edit_list, the exact clips in order, and FirstCut plays exactly that with no planning (use footage_details and get_transcript first to learn which clips are which, and check_edit_list to see the result before starting). Always take control when the person has told you which clips to use or in what order. B-roll, captions, music and graphics are then set separately.",
     "inputSchema": _obj({"folder": {**S, "description": "Full path of the footage folder (find_files helps)."},
                          "notes": {**S, "description": "What the video should be, e.g. 'a 2-minute recap, open with the best line'. With an edit_list the notes only guide B-roll and the title."},
                          "edit_list": {"type": "array", "description": "The exact edit: clips in play order. Each entry is a file name (or its number, like '1.01') or an object. FirstCut plays these and nothing else, trimmed as given, in this order.",
                                        "items": {"type": "object", "properties": {
                                            "file": {**S, "description": "The clip's file name, its name without extension, or its leading number ('3.02'). Must name exactly one file."},
                                            "start": {"type": "number", "description": "Seconds into the clip to start (default: its start)."},
                                            "end": {"type": "number", "description": "Seconds into the clip to stop (default: its end)."},
                                            "audio": {**S, "enum": ["own", "silent"], "description": "'own' plays the clip's sound (default); 'silent' plays the picture only."},
                                            "label": {**S, "description": "Your name for this entry (shown in the plan and the notes)."}},
                                            "required": ["file"], "additionalProperties": False}},
                          "broll": {**S, "enum": ["auto", "none"], "description": "'auto' lays B-roll from clips not in the edit list over it (with an edit_list, default); 'none' plays the list alone."},
                          "broll_files": {"type": "array", "items": S, "description": "Only these files (names or numbers) may be used as B-roll. Leave out to allow every clip that is not in the edit list."},
                          "avoid": {"type": "array", "items": S, "description": "Words: no B-roll shot whose description or file name contains one is used (e.g. 'prison yard')."},
                          "min_shot_seconds": {"type": "number", "description": "No B-roll shot shorter than this many seconds (e.g. 1.5)."},
                          "song": {**S, "description": "Full path of a song to play under the whole video (an audio file on this Mac)."},
                          "minutes": {"type": "number", "description": "Length in minutes (leave out to let the editor decide; ignored with an edit_list)."},
                          "pace": {"type": "string", "enum": ["auto", "fast", "slow"], "description": "'fast' for fast-paced cuts, 'slow' to take it slow."},
                          "captions": {"type": "string", "enum": ["none", "clean", "bold", "pop"], "description": "Default none."},
                          "graphics": {"type": "string", "enum": ["", "title", "all"], "description": "'' none, 'title' title and key points, 'all' full."},
                          "music": {**S, "description": "'' none, 'auto' to match the content, or a mood: chill, upbeat, emotional."},
                          "style": {"type": "object", "description": "A partial style object (see capabilities): only the settings you give change, e.g. {\"pace\": {\"name\": \"fast\"}, \"music\": {\"policy\": \"none\"}, \"broll\": {\"share\": 0.2}}. Allowed values and ranges are in capabilities."},
                          "title": {**S, "description": "The video's name."},
                          "plan": {"type": "object", "description": "Render a plan you edited (from plan_long_video) instead of planning: {\"items\": [...]} in the shapes plan_long_video returned. Needs `project`. Every id is checked first; a plan with a hard problem is refused and nothing changes. Use check_plan before this to see the problems."},
                          "project": {**S, "description": "The video the plan belongs to (its id or part of its title). Only with `plan`."}},
                         [])},
    {"name": "footage_details", "description": "What is in a footage folder, so you can choose clips yourself: every video's name, length, frame rate, size and whether it has sound, in the order they were shot, with the folder it is in. When the folder has been looked at by an earlier edit, each video also shows its role (a_roll, broll, voiceover_broll...), how much is spoken, the largest face, camera movement, sharpness, camera type, when it was shot, and its described scenes with shot size. Read-only. Use it before writing an edit_list, then get_transcript on a file to learn what is said in it.",
     "inputSchema": _obj({"folder": {**S, "description": "Full path of the folder (find_files helps)."}}, ["folder"])},
    {"name": "check_edit_list", "description": "Check an edit_list without making anything: which file each entry is, how long each plays, the total length, and what is wrong (a clip that can't be found or is named by more than one file, a trim past the end, a very short entry). Free and instant: use it until the list is right, then pass it to make_long_video.",
     "inputSchema": _obj({"folder": {**S, "description": "Full path of the footage folder."},
                          "edit_list": {"type": "array", "items": {"type": "object"}, "description": "The same list make_long_video takes."},
                          "min_shot_seconds": {"type": "number", "description": "Mention entries shorter than this."}},
                         ["folder", "edit_list"])},
    {"name": "make_short_clips", "description": "Start cutting a video (a file on this Mac, or a link such as YouTube) into vertical short clips with captions. Returns at once; check with project_status.",
     "inputSchema": _obj({"source": {**S, "description": "Full path of a video on this Mac (find_files helps), or an http(s) link."},
                          "notes": {**S, "description": "Anything to look for, e.g. 'only the funny moments'."},
                          "clips": {"type": "integer", "minimum": 1, "maximum": 30, "description": "How many clips (default 8)."}},
                         ["source"])},
    {"name": "get_transcript", "description": "Write down every word of a video or recording on this Mac, or a link: text, text with times, captions (.srt). Returns at once; project_status gives the text.",
     "inputSchema": _obj({"video": {**S, "description": "Full path of a video or audio file, or an http(s) link."}}, ["video"])},
    {"name": "change_video", "description": "Change a finished long video in plain words ('cut the part about parking', 'make it shorter and more fast-paced'). Give length_minutes and pace when the person asks for them: 'shorter' with no number means about 70% of the current length; 'fast-paced' means pace 'fast'. It renders again.",
     "inputSchema": _obj({"project": PROJECT,
                          "instruction": {**S, "description": "What to change, in the person's words."},
                          "length_minutes": {"type": "number", "description": "The new length in minutes (optional)."},
                          "pace": {"type": "string", "enum": ["auto", "fast", "slow"], "description": "The new pace (optional)."},
                          "style": {"type": "object", "description": "A partial style object (see capabilities): only the settings you give change, e.g. {\"pace\": {\"name\": \"fast\"}, \"music\": {\"policy\": \"none\"}, \"broll\": {\"share\": 0.2}}. Allowed values and ranges are in capabilities."}},
                         ["instruction"])},
    {"name": "capabilities", "description": "What FirstCut can do and the limits, so you can plan inside them without trial and error: the style profiles with every default number, the style object's schema, caption styles, graphics themes, the shortest clip and longest video, and the checks a cut is held to. Read-only; call it before setting a style.",
     "inputSchema": _obj({})},
    {"name": "style_profile", "description": "Read or change the numbers that shape one video's cut (pace, B-roll share and shot length, captions, graphics, music, length): the video's style object. action 'get' shows it in words; 'set' changes only the settings in `style` (a partial object, see capabilities) and says what is wrong when a value is out of range. scope 'always' also keeps the settings it states for every later video. A change shows in the next render (change_video renders again).",
     "inputSchema": _obj({"project": PROJECT, "action": {**S, "enum": ["get", "set"], "description": "get (default) or set."},
                          "style": {"type": "object", "description": "A partial style object (see capabilities): only the settings you give change, e.g. {\"pace\": {\"name\": \"fast\"}, \"music\": {\"policy\": \"none\"}, \"broll\": {\"share\": 0.2}}. Allowed values and ranges are in capabilities."},
                          "scope": {**S, "enum": ["video", "always"], "description": "'video' (default) or 'always'."}},
                         [])},
    {"name": "learn_style", "description": "Learn how a video is EDITED (cut rhythm, pauses, captions, music, zooms) from a YouTube video, short, channel or playlist link, an Instagram reel link (it may need a dropped file instead: no sign-in is ever used) or a video file on this Mac, and keep it as a named editing style preset. Give `preset` to add the video to an existing style (the numbers are combined) or `name` for a new one. Takes about a minute; it returns the plain lines and the measurements, or tells you to check list_style_presets if it is still running. Nothing is applied to any video until apply_style_preset.",
     "inputSchema": _obj({"url": {**S, "description": "A YouTube, Instagram or other video link, or the full path of a video file on this Mac."},
                          "preset": {**S, "description": "The id or name of an existing style to add this video to (optional)."},
                          "name": {**S, "description": "The name for a new style (optional)."}}, ["url"])},
    {"name": "list_style_presets", "description": "The editing styles learned or written so far: name, the plain lines, the person's own rules, the videos each was learned from and what was measured. Read-only.",
     "inputSchema": _obj({})},
    {"name": "apply_style_preset", "description": "Use a style preset for short clips or long videos: its rules and lines become the \"How it edits\" points of that mode and the settings those points state, so every later video of that kind follows it. The person can still edit the points in the app. Hard rules of the app (quality checks, real-life content only) are not changed by a style.",
     "inputSchema": _obj({"preset": {**S, "description": "The id or name of the style."}, "mode": {**S, "enum": ["short", "long"], "description": "short = vertical clips, long = long videos."}},
                         ["preset", "mode"])},
    {"name": "search_footage", "description": "Search the footage library on this Mac in plain words ('students walking to class at sunset'). Returns the best shots with a small picture each, the file and the time in it.",
     "inputSchema": _obj({"query": {**S, "description": "What the shot shows."},
                          "n": {"type": "integer", "minimum": 1, "maximum": 12, "description": "How many shots (default 6)."}}, ["query"])},
    {"name": "replace_shot", "description": "Swap one B-roll shot in a finished long video for another: 'which' says the shot to replace (what it shows, e.g. 'person waving', or seconds into the video), 'with' says what the new shot should show (searched in the project and the whole footage library). Everything else stays; it renders again. If 'which' could be several shots, it lists them with their times instead of guessing.",
     "inputSchema": _obj({"project": PROJECT,
                          "which": {**S, "description": "What the shot to replace shows, or a time in seconds."},
                          "with": {**S, "description": "What the new shot should show (optional: else the editor's best suggestion)."}},
                         ["which"])},
    {"name": "add_song", "description": "Put a song under a finished long video, e.g. 'upbeat': the best match in the person's own music first, else free stock music (Openverse, credited). Empty query takes the music out. The cut stays the same; it renders again.",
     "inputSchema": _obj({"project": PROJECT, "query": {**S, "description": "A mood or description ('upbeat', 'calm piano'), or '' for no music."}}, ["query"])},
    {"name": "catalogue_status", "description": "How far FirstCut has got looking through the footage library: videos and photos, how many are described, what it is doing now.",
     "inputSchema": _obj({})},
    {"name": "open_in_final_cut", "description": "Open a finished long video's project in Final Cut Pro on this Mac.",
     "inputSchema": _obj({"project": PROJECT})},
    {"name": "show_in_finder", "description": "Show a finished video, a batch of clips or a transcript in the Finder on this Mac.",
     "inputSchema": _obj({"project": PROJECT})},
    {"name": "transcript_search", "description": "Find a quote or a topic in a video's transcript and get where it is: word numbers, seconds, the words, and how much silence is either side (the room a cut may use). Use it to ground an instruction ('the part about forgiveness', 'where he says we were wrong') before you edit a plan. mode 'quote' finds words as said (no AI, instant); 'topic' reads the transcript for a subject (slower). Two hits far apart means ask the person which one. Not for reading the whole transcript: use get_transcript.",
     "inputSchema": _obj({"project": PROJECT, "query": {**S, "description": "The words to find, or the topic."}, "mode": {**S, "enum": ["quote", "topic"], "description": "Default quote."}}, ["query"])},
    {"name": "shot_details", "description": "What FirstCut knows about ONE footage file: its role (talking head, B-roll, screen...), whether the speaker is on camera, camera movement and shake, sharpness, camera type, shot time, and every scene in it with its id, seconds and description. Use it after footage_details when you must choose between a few clips. Needs a folder that was edited before. Not for searching the library: use search_footage.",
     "inputSchema": _obj({"folder": {**S, "description": "Full path of the footage folder."}, "file": {**S, "description": "The file's name, its name without extension, or its number."}}, ["folder", "file"])},
    {"name": "plan_long_video", "description": "Plan a long video WITHOUT rendering it: FirstCut reads the footage and your notes, decides the cut, checks every instruction in the notes in code, and returns the plan as lines (each passage with an id, its seconds and its words), the instructions with how each was understood, the problems found and what could not be met. Then you can edit the plan and send it back with make_long_video(plan=...). Same arguments as make_long_video (the first call takes minutes: it returns at once with an id if the plan is not ready, then call plan_long_video again with that `project`). Use make_long_video when you just want the video.",
     "inputSchema": _obj({"folder": {**S, "description": "Full path of the footage folder. Leave out when you give `project`."}, "notes": {**S, "description": "What the video should be and any rules (order, parts to keep or cut, footage not to use, length)."},
                          "minutes": {"type": "number", "description": "Length in minutes."}, "pace": {**S, "enum": ["auto", "fast", "slow"]}, "title": {**S},
                          "project": {**S, "description": "Read the plan of a video already planned (its id or part of its title)."}}, [])},
    {"name": "check_plan", "description": "Check a plan you edited, without rendering: unknown ids, passages that end before they start, a trim past the end of a file, a line played twice, a required section missing, a part that should be cut still in, the length against what was asked, a cut inside a word. Returns each problem in plain words with a hint. Hard problems stop make_long_video(plan=...). Use it after every edit of a plan.",
     "inputSchema": _obj({"project": PROJECT, "plan": {"type": "object", "description": "{\"items\": [{\"id\":\"i0\",\"kind\":\"talk\",\"a\":12,\"b\":15}, {\"kind\":\"visual\",\"start\":140.0,\"end\":146.5}]}: sentence numbers (a, b) come from plan_long_video; times are seconds in the footage."}}, ["plan"])},
    {"name": "render_preview", "description": "A cheap look at a finished video without a full render: with frames_only, one still at each cut in the range (up to 12, as pictures); otherwise a 480-pixel MP4 of the range (at most 60 s; returns its path). Use it to check one passage after a change. Not for judging colour or sound quality.",
     "inputSchema": _obj({"project": PROJECT, "start_s": {"type": "number", "description": "Start second in the finished video (default 0)."}, "end_s": {"type": "number", "description": "End second (default start + 20)."},
                          "frames_only": {"type": "boolean", "description": "Stills at the cuts instead of a clip."}})},
    {"name": "rerender", "description": "Render a finished or planned video again from its saved plan. FirstCut re-encodes the whole video (B-roll placement is kept when the cut did not change); it cannot re-encode only the changed passages, and says so. Use change_video for a change in words; use make_long_video(plan=...) to render a plan you edited.",
     "inputSchema": _obj({"project": PROJECT, "changed_segment_ids": {"type": "array", "items": S, "description": "The passages you changed (noted in the answer only)."}})},
]

STOP = {"the", "a", "an", "of", "to", "in", "on", "with", "and", "shot", "clip", "b-roll", "broll", "video", "one", "that", "this", "my", "is", "at"}


def _words(s) -> set:
    import re
    return {w for w in re.findall(r"[a-z0-9']+", str(s or "").lower()) if w not in STOP and len(w) > 1}


def _state(eng: Engine) -> dict:
    return eng.get("/api/app")


def _all(st: dict) -> list[tuple[str, dict]]:
    return [("long", p) for p in st.get("projects") or []] + [("short", j) for j in st.get("shorts") or []] + \
           [("transcript", t) for t in st.get("transcripts") or []]


def _when(p: dict) -> float:
    return float(p.get("updated") or p.get("created") or 0)


def _find(st: dict, ref, kinds=("long", "short", "transcript")) -> tuple[str, dict]:
    """A project by id, by title (or part of it), or (none given) the most recent one."""
    items = [(k, p) for k, p in _all(st) if k in kinds]
    if not items:
        raise ToolError("nothing has been made yet" if kinds != ("long",) else "there's no long video yet")
    ref = str(ref or "").strip()
    if not ref:
        return max(items, key=lambda kp: _when(kp[1]))
    for k, p in items:
        if p.get("id") == ref:
            return k, p
    want = _words(ref) - {"video", "project"}
    scored = sorted(((len(want & _words(p.get("title"))), _when(p), k, p) for k, p in items), key=lambda x: (-x[0], -x[1]))
    if want and scored and scored[0][0]:
        if len(scored) > 1 and scored[1][0] == scored[0][0] and scored[0][0] < len(want):
            raise ToolError("which one? " + "; ".join(f"{p.get('title')} (id {p['id']})" for n, _, k, p in scored[:3] if n == scored[0][0]))
        return scored[0][2], scored[0][3]
    raise ToolError(f"there's no video called {ref!r}. These are here: " + "; ".join(f"{p.get('title')} (id {p['id']})" for _, p in items[:8]))


def _read(path: Path, limit: int = 6000) -> str:
    try:
        t = path.read_text(errors="replace")
    except OSError:
        return ""
    return t if len(t) <= limit else t[:limit] + "\n…"


def _newest(folder: Path, pattern: str) -> str | None:
    try:
        xs = sorted(folder.glob(pattern), key=lambda x: x.stat().st_mtime)
    except OSError:
        return None
    return str(xs[-1]) if xs else None


def _txt(s: str) -> list[dict]:
    return [{"type": "text", "text": s}]


def tool_list_projects(eng: Engine, a: dict) -> list[dict]:
    lines = []
    for k, p in sorted(_all(_state(eng)), key=lambda kp: -_when(kp[1])):
        if k == "long":
            js = p.get("job_state") or {}
            lines.append(f"- long video “{p.get('title') or 'Untitled'}” (id {p['id']}): {js.get('status') or 'draft'}"
                         + (f", {round(100 * (js.get('progress') or 0))}%, {js.get('stage')}" if js.get("status") not in (None, "done", "failed", "draft") else ""))
        elif k == "short":
            lines.append(f"- short clips “{p.get('title') or 'Clips'}” (id {p['id']}): {p.get('status')}, {len(p.get('clips') or [])} clip(s)")
        else:
            lines.append(f"- transcript “{p.get('title') or 'Transcript'}” (id {p['id']}): {p.get('status')}")
    return _txt("\n".join(lines) if lines else "Nothing has been made yet.")


def tool_project_status(eng: Engine, a: dict) -> list[dict]:
    kind, p = _find(_state(eng), a.get("project"))
    if kind == "transcript":
        t = eng.get(f"/api/app/transcripts/{urllib.parse.quote(p['id'])}")
        if t.get("status") != "done":
            return _txt(f"Transcript of “{t.get('title')}”: {t.get('stage') or t.get('status')}, {round(100 * (t.get('progress') or 0))}%"
                        + (f". It failed: {t.get('error')}" if t.get("status") == "failed" else ""))
        text = "\n\n".join(x.get("text", "") for x in t.get("paragraphs") or [])
        out = t.get("out") or {}
        return _txt(f"Transcript of “{t.get('title')}” ({t.get('words')} words). Files: {out.get('txt')}, {out.get('timed')}, {out.get('srt')}\n\n"
                    + (text if len(text) < 60000 else text[:60000] + "\n…"))
    if kind == "short":
        lines = [f"Short clips “{p.get('title')}”: {p.get('status')}" + (f", {round(100 * (p.get('progress') or 0))}%, {p.get('stage')}"
                                                                         if p.get("status") not in ("done", "failed") else "")]
        if p.get("detail"):
            lines.append(f"Waiting on: {p['detail']}")
        lines += [f"- clip {c.get('index')}: {c.get('title')} ({round(c.get('duration') or 0)} s)" for c in p.get("clips") or []]
        if p.get("home"):
            lines.append(f"Saved in: {p['home']}")
        return _txt("\n".join(lines))
    js = p.get("job_state") or {}
    status = js.get("status") or "draft"
    lines = [f"Long video “{p.get('title')}” (id {p['id']}): {status}"]
    if status not in ("done", "failed", "draft"):
        lines[0] += f", {round(100 * (js.get('progress') or 0))}%, {js.get('stage') or ''}"
        if js.get("detail"):
            lines.append(f"Waiting on: {js['detail']}")
    if status == "failed" and js.get("error"):
        lines.append(f"It failed: {js['error']}")
    saved = Path(p.get("saved_to") or p.get("folder") or "")
    if status == "done" and saved.is_dir():
        mp4, xml = _newest(saved / "Output", "*.mp4"), _newest(saved, "*.fcpxml")
        lines += [x for x in (f"Video: {mp4}" if mp4 else "", f"Final Cut project: {xml}" if xml else "", f"Folder: {saved}") if x]
        ch = _read(saved / "Notes" / "Chapters.txt", 2000)
        if ch:
            lines += ["", "Chapters:", ch.strip()]
        qc = _read(saved / "Notes" / "Quality check.md", 1500)
        if qc:
            lines += ["", "Quality check (first lines):", qc.strip()]
    return _txt("\n".join(lines))


# finding files (read-only, never outside the home folder or an external drive)
VIDEO = {".mp4", ".mov", ".m4v", ".mts", ".mxf", ".avi", ".mkv", ".webm", ".insv", ".lrv"}
PHOTO = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".dng", ".arw", ".cr3", ".cr2", ".nef", ".raf"}


def _allowed(p: Path) -> bool:
    try:
        r = p.resolve()
    except OSError:
        return False
    return r == Path.home().resolve() or Path.home().resolve() in r.parents or (len(r.parts) > 2 and r.parts[1] == "Volumes")


def resolve_place(where: str) -> Path:
    """'Desktop/videos/raw', '~/Movies', '/Volumes/Work/Footage' -> a real folder, matching each name without caring about capitals."""
    w = str(where or "").strip().strip("'\"")
    if not w:
        raise ToolError("say which folder")
    p = Path(w).expanduser()
    if not p.is_absolute():
        p = Path.home() / w
    cur = Path(p.anchor)                                                  # walk down to the real spelling, matching each part loosely
    for part in p.parts[1:]:
        try:
            kids = list(cur.iterdir())
        except OSError:
            kids = []
        hit = next((c for c in kids if c.name == part), None) or next((c for c in kids if c.name.lower() == part.lower()), None) \
            or next((c for c in kids if part.lower() in c.name.lower()), None)
        if hit is None:
            raise ToolError(f"there's no folder like {part!r} in {cur}")
        cur = hit
    if not _allowed(cur):
        raise ToolError("FirstCut only looks in your home folder and on external drives")
    return cur


def tool_find_files(eng: Engine, a: dict) -> list[dict]:
    root = resolve_place(a.get("where"))
    kind = str(a.get("kind") or "videos")
    q = _words(a.get("query"))
    if root.is_file():
        return _txt(f"That is a file: {root} ({root.stat().st_size / 1e6:.0f} MB)")
    rows, more = [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and not d.endswith((".fcpbundle", ".app"))) if depth < 3 else []
        items = [Path(dirpath) / d for d in dirnames] if kind == "folders" else \
            [Path(dirpath) / f for f in sorted(filenames) if not f.startswith(".") and (
                kind == "all" or (kind == "videos" and Path(f).suffix.lower() in VIDEO) or (kind == "photos" and Path(f).suffix.lower() in PHOTO))]
        for f in items:
            if q and not q <= _words(f.name) and not any(w in f.name.lower() for w in q):
                continue
            if len(rows) >= 120:
                more += 1
                continue
            rows.append(f)
    if not rows:
        return _txt(f"Nothing {('matching ' + ' '.join(sorted(q)) + ' ') if q else ''}in {root}.")
    lines = [f"In {root}:"]
    for f in rows:
        if f.is_dir():
            lines.append(f"- folder {f}")
            continue
        size = f.stat().st_size / 1e6
        dur = ""
        if f.suffix.lower() in VIDEO and len(lines) <= 40:
            try:
                from ..core import ffmpeg
                d = ffmpeg.duration(f)
                dur = f", {int(d // 60)}:{int(d % 60):02d}" if d else ""
            except Exception:  # noqa: BLE001
                dur = ""
        lines.append(f"- {f} ({size:.0f} MB{dur})")
    if more:
        lines.append(f"…and {more} more (narrow it with 'query').")
    return _txt("\n".join(lines))


def tool_make_long_video(eng: Engine, a: dict) -> list[dict]:
    if isinstance(a.get("plan"), dict):                                        # a plan an AI edited (plan_long_video): checked in code, then rendered
        p = _long(eng, a.get("project"))
        r = eng.post(f"/api/app/long/{urllib.parse.quote(p['id'])}/plan", {"plan": a["plan"]})
        return _txt(f"Rendering “{p.get('title')}” from your plan ({r.get('totals', {}).get('length_s', 0):.0f} s). It takes minutes; project_status says when it is done."
                    + ((" Things to look at: " + "; ".join(x["message"] for x in r.get("problems") or [])[:600]) if r.get("problems") else ""))
    if not a.get("folder"):
        raise ToolError("give the footage `folder` (or a `plan` with its `project`)")
    folder = str(resolve_place(a.get("folder")) if a.get("folder") else "")
    opts = {}
    if a.get("dry_run") is True:
        opts["dry_run"] = True
    if a.get("minutes") is not None:
        opts["long_minutes"] = float(a["minutes"])
    for k_in, k_out in (("pace", "pace"), ("captions", "caption_style"), ("graphics", "graphics"), ("music", "music")):
        if a.get(k_in) is not None:
            opts[k_out] = a[k_in]
    if isinstance(a.get("style"), dict) and a["style"]:
        opts["style"] = a["style"]
    if a.get("edit_list"):
        opts["edit_list"] = a["edit_list"]
        opts.setdefault("caption_style", "none")                               # (an exact edit is not captioned unless asked)
    for k_in, k_out in (("broll", "broll_mode"), ("broll_files", "broll_files"), ("avoid", "avoid"), ("min_shot_seconds", "min_shot_s")):
        if a.get(k_in) not in (None, "", []):
            opts[k_out] = a[k_in]
    body = {"source": folder, "notes": str(a.get("notes") or ""), "options": opts}
    if a.get("title"):
        body["title"] = str(a["title"])[:120]
    if a.get("song"):
        body["song"] = str(resolve_place(a["song"]))
    r = eng.post("/api/app/long", body)
    if a.get("dry_run") is True:
        return r
    how = f"{len(a['edit_list'])} clips in the order you gave" if a.get("edit_list") else "FirstCut plans the cut"
    return _txt(f"Started making a video from {folder} (id {r.get('id')}): {how}. It reads the footage where it is. Check on it with project_status.")


def _plan_text(v: dict) -> str:
    """A plan view (planapi.view) as text an AI can read and edit: passages with ids, the instructions, the problems."""
    lines = [f"Plan for “{v.get('title')}” ({v.get('totals', {}).get('length_s', 0):.0f} s). Status: {v.get('stage') or v.get('status')}.", "", "PASSAGES (edit these):"]
    for it in v.get("items") or []:
        if it.get("kind") == "talk":
            lines.append(f"- {it['id']} talk sentences {it.get('a')}-{it.get('b')} ({it.get('from_s')}-{it.get('to_s')} s)" + (f" satisfies {it['satisfies']}" if it.get("satisfies") else "")
                         + f": {it.get('text', '')}")
        else:
            lines.append(f"- {it['id']} visual {it.get('start')}-{it.get('end')} s")
    cons = v.get("constraints") or []
    if cons:
        lines += ["", "INSTRUCTIONS AS UNDERSTOOD:"] + [f"- {c['id']} {c['type']} ({'hard' if c.get('hard') else 'soft'}) {c['status']}: “{c['source'][:100]}” {c.get('note', '')}" for c in cons]
    probs = v.get("problems") or []
    if probs:
        lines += ["", "PROBLEMS:"] + [f"- {'MUST FIX' if p.get('hard') else 'look'} {p['code']}: {p['message']}" + (f" Hint: {p['fix_hint']}" if p.get("fix_hint") else "") for p in probs[:20]]
    if v.get("unmet"):
        lines += ["", "NOT MET: " + "; ".join(f"{u['constraint']} ({u['why']})" for u in v["unmet"])]
    if (v.get("approval") or {}).get("questions"):
        lines += ["", "QUESTIONS FOR THE PERSON: " + " | ".join(v["approval"]["questions"])]
    lines += ["", "Send an edited plan back with check_plan, then make_long_video(plan=..., project=...)."]
    return "\n".join(lines)


def tool_plan_long_video(eng: Engine, a: dict) -> list[dict]:
    if a.get("project") and not a.get("folder"):
        p = _long(eng, a.get("project"))
        v = eng.get(f"/api/app/long/{urllib.parse.quote(p['id'])}/plan")
        return _txt(_plan_text(v))
    r = tool_make_long_video(eng, {**a, "dry_run": True})
    pid = r.get("id") if isinstance(r, dict) else None
    if not pid:
        return _txt("Planning started.")
    import time as _t
    for _ in range(12):                                                         # a plan from cached transcripts is ready in seconds; a first one takes minutes
        _t.sleep(2.0)
        try:
            v = eng.get(f"/api/app/long/{urllib.parse.quote(pid)}/plan")
        except ToolError:
            continue
        if v.get("awaiting_approval") or v.get("dry_run"):
            return _txt(_plan_text(v))
    return _txt(f"Planning {a.get('folder')} (id {pid}) is still working: reading the footage the first time takes minutes. Call plan_long_video with project '{pid}' to read the plan when project_status says it is waiting for approval.")


def tool_check_plan(eng: Engine, a: dict) -> list[dict]:
    p = _long(eng, a.get("project"))
    r = eng.post(f"/api/app/long/{urllib.parse.quote(p['id'])}/check-plan", {"plan": a.get("plan")})
    probs = r.get("problems") or []
    head = "The plan passes every hard check." if r.get("ok") else "The plan has problems that must be fixed:"
    lines = [head + f" Length {r.get('totals', {}).get('length_s', 0):.0f} s."] + [f"- {'MUST FIX' if x.get('hard') else 'look'} {x['code']}: {x['message']}" + (f" Hint: {x['fix_hint']}" if x.get("fix_hint") else "") for x in probs[:20]]
    return _txt("\n".join(lines))


def tool_transcript_search(eng: Engine, a: dict) -> list[dict]:
    p = _long(eng, a.get("project"))
    r = eng.post(f"/api/app/long/{urllib.parse.quote(p['id'])}/search", {"query": a.get("query"), "mode": a.get("mode") or "quote"})
    hits = r.get("hits") or []
    if not hits:
        return _txt(f"“{r.get('query')}” was not found in the transcript ({r.get('mode')} search).")
    lines = [f"{len(hits)} place(s) for “{r.get('query')}” ({r.get('mode')} search)" + (" — more than one place, far apart: ask which is meant" if r.get("ambiguous") else "") + ":"]
    for h in hits:
        lines.append(f"- words {h['first_word']}-{h['last_word']}, {_mmss(h['start_s'])}-{_mmss(h['end_s'])} (silence before {h['lead_silence_s']} s, after {h['trail_silence_s']} s): {h['text'][:160]}")
    return _txt("\n".join(lines))


def tool_shot_details(eng: Engine, a: dict) -> list[dict]:
    from . import editlist as EL
    files = _scan_folder(str(a.get("folder") or ""))
    f, cands = EL.find_file(str(a.get("file") or ""), files)
    if f is None:
        raise ToolError("which file? " + ("; ".join(cands[:6]) if cands else f"none matches {a.get('file')!r}"))
    notes_ = _facts_notes(str(a.get("folder") or ""), [f]).get(f["name"], "")
    return _txt(f"{f.get('rel') or f['name']}: {float(f['dur']):.1f} s, {int(f.get('w') or 0)}x{int(f.get('h') or 0)} at {float(f.get('fps') or 0):g} fps, "
                + ("with sound" if f.get("audio") else "silent") + (notes_ or " (nothing worked out about it yet: it was never edited from this folder)"))


def tool_render_preview(eng: Engine, a: dict) -> list[dict]:
    p = _long(eng, a.get("project"))
    r = eng.post(f"/api/app/long/{urllib.parse.quote(p['id'])}/preview", {"start_s": a.get("start_s") or 0, "end_s": a.get("end_s"), "frames_only": bool(a.get("frames_only"))})
    if a.get("frames_only"):
        out: list[dict] = [{"type": "text", "text": f"{len(r.get('frames') or [])} still(s), one at each cut:"}]
        for fr in r.get("frames") or []:
            out.append({"type": "text", "text": f"at {fr['at']:.1f} s"})
            try:
                out.append({"type": "image", "data": base64.b64encode(_small_jpeg(Path(fr["path"]).read_bytes(), 480)).decode(), "mimeType": "image/jpeg"})
            except OSError:
                pass
        return out
    return _txt(f"A 480 px preview of {r.get('from_s')}-{r.get('to_s')} s is at {r.get('path')}")


def tool_rerender(eng: Engine, a: dict) -> list[dict]:
    p = _long(eng, a.get("project"))
    r = eng.post(f"/api/app/long/{urllib.parse.quote(p['id'])}/rerender-changed", {"changed": a.get("changed_segment_ids") or []})
    return _txt(f"Rendering “{p.get('title')}” again. {r.get('note', '')} project_status says when it is done.")


def _scan_folder(folder: str) -> list[dict]:
    from .localproj import scan
    root = resolve_place(folder)
    if not root.is_dir():
        raise ValueError(f"{root} is not a folder")
    return scan(root)


def _facts_notes(folder: str, files: list[dict]) -> dict[str, str]:
    """{file name: ", role a_roll (why), speech 0.82, ... ; scenes: ..."} from what an earlier edit of this folder already worked out (the facts cache and
    the scene descriptions in the folder's .aivideo). Nothing is analysed here and nothing is written: a folder never edited has none."""
    from . import footage as FT
    try:
        root = resolve_place(folder)
    except Exception:  # noqa: BLE001 - the facts are a bonus on top of the list
        return {}
    cache = root / ".aivideo" / "facts"
    try:
        broll = json.loads((root / ".aivideo" / "broll.json").read_text())
    except (OSError, ValueError):
        broll = {}
    try:
        pool = (json.loads((root / ".aivideo" / "plan.json").read_text()) or {}).get("pool") or []
    except (OSError, ValueError):
        pool = []
    def _r1(v) -> float:
        try:
            return round(float(v), 1)
        except (TypeError, ValueError):
            return -1.0
    sid_of = {(s.get("file"), _r1(s.get("fstart", -1))): s.get("sid") for s in pool if isinstance(s, dict) and s.get("sid")}
    by = {}
    for f in files:
        rec = FT.load_cached(cache, Path(f["path"]), "fast") if (cache.is_dir() and f.get("path")) else None
        bits = []
        if rec:
            sp, fa, mo, q, cam = (rec.get("speech") or {}), (rec.get("faces") or {}), (rec.get("motion") or {}), (rec.get("quality") or {}), (rec.get("camera") or {})
            bits.append(f"role {rec.get('role')} ({rec.get('role_why')})")
            if sp.get("ratio") is not None:
                bits.append(f"speech {sp['ratio']:.2f}")
            if fa.get("largest_ratio_median"):
                bits.append(f"largest face {fa['largest_ratio_median']:.2f} of the frame height")
            if mo.get("type") and mo["type"] != "unknown":
                bits.append(f"camera {mo['type']}")
            if "blur_floor_ok" in q:
                bits.append("sharp" if q["blur_floor_ok"] else "soft or flat")
            if rec.get("slowmo_eligible"):
                bits.append("fine in slow motion")
            if cam.get("family") not in (None, "", "unknown"):
                bits.append(f"{cam['family']} camera")
            if cam.get("created"):
                bits.append(f"shot {str(cam['created'])[:16].replace('T', ' ')}")
            if rec.get("duplicates_of"):
                bits.append(f"same shot as {rec['duplicates_of']}")
        scs = (broll.get(f["name"]) or {}).get("scenes") or []
        if scs:
            parts = []
            for s in scs[:8]:
                try:
                    sid = sid_of.get((f["name"], _r1(s["start"])))
                    parts.append(f"{(sid + ' ') if sid else ''}{float(s['start']):.1f}-{float(s['end']):.1f} s {str(s.get('what') or '')[:70]}"
                                 + (f" [{s['shot_size']}{', talking head' if s.get('is_talking_head') else ''}]" if s.get("shot_size") or s.get("is_talking_head") else ""))
                except (TypeError, ValueError, KeyError, AttributeError):
                    continue                                              # (one odd scene in a cache never hides the rest of the folder)
            bits.append("scenes: " + "; ".join(parts))
        by[f["name"]] = (", " + ", ".join(bits)) if bits else ""
    return by


def tool_footage_details(eng: Engine, a: dict) -> list[dict]:
    files = _scan_folder(str(a.get("folder") or ""))
    if not files:
        return _txt("No videos in that folder.")
    lines = [f"{len(files)} videos, in the order they were shot ({sum(float(f['dur']) for f in files) / 60:.1f} minutes in all):"]
    notes_ = _facts_notes(str(a.get("folder") or ""), files)
    for f in files[:400]:
        lines.append(f"- {f.get('rel') or f['name']}: {float(f['dur']):.1f} s, {int(f.get('w') or 0)}x{int(f.get('h') or 0)} at {float(f.get('fps') or 0):g} fps, "
                     + ("with sound" if f.get("audio") else "silent") + notes_.get(f["name"], ""))
    if len(files) > 400:
        lines.append(f"... and {len(files) - 400} more")
    return _txt("\n".join(lines))


def tool_capabilities(eng: Engine, a: dict) -> list[dict]:
    from . import styles_profile as SPF
    return _txt(json.dumps(SPF.capabilities(), indent=1))


def tool_style_profile(eng: Engine, a: dict) -> list[dict]:
    p = _long(eng, a.get("project"))
    pid = urllib.parse.quote(p["id"])
    if str(a.get("action") or "get") == "set":
        if not isinstance(a.get("style"), dict) or not a["style"]:
            raise ToolError("give the settings to change in `style` (see capabilities)")
        r = eng.post(f"/api/app/long/{pid}/style", {"style": a["style"], "scope": "always" if a.get("scope") == "always" else ""})
        return _txt(f"Changed the style of “{p.get('title')}”:\n{r.get('describe', '')}\nIt takes effect when the video renders again (change_video does that)."
                    + (" These settings are kept for every later video too." if a.get("scope") == "always" else ""))
    r = eng.get(f"/api/app/long/{pid}/style")
    return _txt(f"Style of “{p.get('title')}”:\n{r.get('describe', '')}\n\n" + json.dumps(r.get("style") or {}, indent=1)
                + ("\n\nProblems: " + "; ".join(r["problems"]) if r.get("problems") else ""))


def _preset(eng: Engine, ref) -> dict:
    items = eng.get("/api/app/styles/presets").get("presets") or []
    r = str(ref or "").strip().lower()
    hit = next((p for p in items if p["id"].lower() == r), None) or next((p for p in items if (p.get("name") or "").lower() == r), None) \
        or next((p for p in items if r and r in (p.get("name") or "").lower()), None)
    if hit is None:
        raise ToolError("no style called that: " + (", ".join(f"“{p.get('name')}”" for p in items) or "none have been made yet (learn_style makes one)"))
    return hit


def _style_text(p: dict) -> str:
    out = [f"Style “{p.get('name')}” (id {p['id']})"]
    if p.get("rules"):
        out.append("Rules:\n" + "\n".join(f"- {x}" for x in p["rules"]))
    out.append("Lines:\n" + ("\n".join(f"- {x}" for x in p.get("points") or []) or "- (none yet)"))
    if p.get("evidence"):
        out.append("What was measured:\n" + "\n".join(f"- {x}" for x in p["evidence"]))
    if p.get("sources"):
        out.append("Learned from: " + "; ".join(s.get("title") or s.get("url") or "a video" for s in p["sources"]))
    return "\n\n".join(out)


def tool_learn_style(eng: Engine, a: dict) -> list[dict]:
    url = str(a.get("url") or "").strip()
    if not url:
        raise ToolError("give a video link or the path of a video file in `url`")
    body: dict = {"file" if url.startswith(("/", "~", "file://")) else "url": url}
    if a.get("preset"):
        body["preset_id"] = _preset(eng, a["preset"])["id"]
    if a.get("name"):
        body["name"] = str(a["name"])
    job = eng.post("/api/app/styles/learn", body)["job"]
    t0 = time.time()
    while time.time() - t0 < 150:
        time.sleep(3.0)
        r = eng.get(f"/api/app/styles/learn/{job}")
        if r.get("state") == "error":
            raise ToolError(r.get("error") or "that video could not be learned from")
        if r.get("state") == "done" and r.get("preset"):
            return _txt(_style_text(r["preset"]) + "\n\nNothing is applied yet: apply_style_preset uses it for short clips or long videos.")
    return _txt("Still learning (" + str((eng.get(f"/api/app/styles/learn/{job}") or {}).get("text") or "working") + "). Check list_style_presets in a minute.")


def tool_list_style_presets(eng: Engine, a: dict) -> list[dict]:
    items = eng.get("/api/app/styles/presets").get("presets") or []
    return _txt("\n\n---\n\n".join(_style_text(p) for p in items) if items else "No styles yet: learn_style makes one from a video link.")


def tool_apply_style_preset(eng: Engine, a: dict) -> list[dict]:
    p = _preset(eng, a.get("preset"))
    mode = str(a.get("mode") or "")
    if mode not in ("short", "long"):
        raise ToolError("mode is short (vertical clips) or long (long videos)")
    r = eng.post(f"/api/app/styles/presets/{urllib.parse.quote(p['id'])}/apply", {"mode": mode})
    return _txt(f"“{p.get('name')}” is now how {'short clips' if mode == 'short' else 'long videos'} are edited:\n" + "\n".join(f"- {x}" for x in r.get("points") or []))


def tool_check_edit_list(eng: Engine, a: dict) -> list[dict]:
    from . import editlist as EL
    files = _scan_folder(str(a.get("folder") or ""))
    try:
        entries = EL.parse(a.get("edit_list"))
        items, problems, how = EL.items_for(entries, files, float(a.get("min_shot_seconds") or 0))
    except ValueError as e:
        return _txt(f"The edit list isn't ready: {e}")
    by = {f["name"]: f for f in files}
    lines, total = [], 0.0
    for k, it in enumerate(items, 1):
        d = float(it["end"]) - float(it["start"])
        total += d
        f = by.get(it.get("file")) or {}
        lines.append(f"{k}. {it.get('file')} {float(it['start']) - float(f.get('start') or 0):.1f}-{float(it['end']) - float(f.get('start') or 0):.1f} s ({d:.1f} s), "
                     + (how.get(it.get("file")) or "own") + " sound")
    spare = [f for f in files if f["name"] not in how]
    out = [f"{len(items)} clips, {total:.1f} s ({total / 60:.1f} min) in all:", *lines,
           f"{len(spare)} other clips in the folder could be B-roll." if spare else "No other clips in the folder: there is nothing for B-roll."]
    if problems:
        out += ["", "Notes:", *[f"- {p}" for p in problems]]
    return _txt("\n".join(out))


def tool_make_short_clips(eng: Engine, a: dict) -> list[dict]:
    src = str(a.get("source") or "").strip()
    if src and not src.startswith(("http://", "https://")):
        src = str(resolve_place(src))
    opts = {"clips": int(a["clips"])} if a.get("clips") else {}
    r = eng.post("/api/app/short", {"source": src, "notes": str(a.get("notes") or ""), "options": opts})
    return _txt(f"Started cutting short clips from {src} (id {r.get('id')}). Check on it with project_status.")


def tool_get_transcript(eng: Engine, a: dict) -> list[dict]:
    src = str(a.get("video") or "").strip()
    if src and not src.startswith(("http://", "https://")):
        src = str(resolve_place(src))
    r = eng.post("/api/app/transcripts", {"src": src})
    return _txt(f"Writing down the words (id {r.get('id')}). project_status gives the text when it's done, about a minute for a 30-minute talk.")


def _long(eng: Engine, ref) -> dict:
    return _find(_state(eng), ref, kinds=("long",))[1]


def tool_change_video(eng: Engine, a: dict) -> list[dict]:
    p = _long(eng, a.get("project"))
    text = str(a.get("instruction") or "").strip()
    extra = []
    if a.get("length_minutes"):
        m = float(a["length_minutes"])
        extra.append(f"Make it {round(m * 60)} seconds long." if m < 3 else f"Make it about {m:g} minutes long.")
    pace = str(a.get("pace") or "")
    if pace == "fast":
        extra.append("Make it more fast-paced: tighter pauses and quicker cuts.")
    elif pace == "slow":
        extra.append("Take it slower: leave room to breathe between lines.")
    if text and extra and text[-1] not in ".!?":
        text += "."
    text = " ".join([text] + extra).strip()
    if isinstance(a.get("style"), dict) and a["style"]:                       # settings first (a value out of range is told here, nothing is changed)
        eng.post(f"/api/app/long/{urllib.parse.quote(p['id'])}/style", {"style": a["style"]})
        text = text or "Apply the new editing settings."
    if not text:
        raise ToolError("say what to change")
    body = {"instruction": text[:1500]}
    if pace in ("fast", "slow"):                                              # (a pace change is a style change: the pause between lines changes, not only the words)
        body["pace"] = "fast" if pace == "fast" else "slow"
    eng.post(f"/api/ai/projects/{urllib.parse.quote(p['id'])}/revise", body)
    return _txt(f"Changing “{p.get('title')}”: {text} It renders again, usually a few minutes; project_status says when it's done.")


def _small_jpeg(data: bytes, width: int = 320) -> bytes:
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(data)).convert("RGB")
        if im.width > width:
            im = im.resize((width, max(1, round(im.height * width / im.width))))
        out = io.BytesIO()
        im.save(out, "JPEG", quality=78)
        return out.getvalue()
    except Exception:  # noqa: BLE001 - the picture as it came
        return data


def tool_search_footage(eng: Engine, a: dict) -> list[dict]:
    q = str(a.get("query") or "").strip()
    if not q:
        raise ToolError("say what the shot shows")
    n = max(1, min(12, int(a.get("n") or 6)))
    try:
        shots = eng.get(f"/api/library/search?q={urllib.parse.quote(q)}&n={n}").get("shots") or []
    except ToolError as e:
        raise ToolError(f"{e} (add a footage folder in FirstCut's Find footage first)") from None
    if not shots:
        return _txt(f"No shots found for “{q}”. The footage may still be being catalogued: catalogue_status says how far it is.")
    out: list[dict] = [{"type": "text", "text": f"{len(shots)} shot(s) for “{q}”, best first:"}]
    for i, s in enumerate(shots[:n], 1):
        out.append({"type": "text", "text": f"{i}. {s.get('what') or '(not described yet)'} · {s.get('rel')} at {s.get('start')}–{s.get('end')} s"})
        try:
            img = eng.get(f"/api/library/thumb/{urllib.parse.quote(str(s['id']))}/{int(s.get('k') or 0)}", raw=True, timeout=20)
            out.append({"type": "image", "data": base64.b64encode(_small_jpeg(img)).decode(), "mimeType": "image/jpeg"})
        except (ToolError, KeyError, ValueError):
            pass
    return out


def _mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def tool_replace_shot(eng: Engine, a: dict) -> list[dict]:
    p = _long(eng, a.get("project"))
    pid = urllib.parse.quote(p["id"])
    shots = eng.get(f"/api/app/long/{pid}/shots").get("shots") or []
    if not shots:
        raise ToolError(f"“{p.get('title')}” has no B-roll shots to swap")
    which = str(a.get("which") or "").strip()
    try:
        at = float(which.rstrip("s").strip())
    except ValueError:
        at = None
    if at is not None:
        shot = min(shots, key=lambda s: 0 if s["at"] <= at <= s["at"] + s["length"] else min(abs(at - s["at"]), abs(at - s["at"] - s["length"])))
    else:
        want = _words(which)
        scored = sorted(((len(want & _words(s.get("what"))) / max(1, len(want)), s) for s in shots), key=lambda x: -x[0])
        if not want or scored[0][0] == 0:
            raise ToolError("no B-roll shot in it looks like that. Its shots: " + "; ".join(f"{_mmss(s['at'])} {s.get('what')}" for s in shots[:12]))
        close = [s for sc, s in scored if sc >= scored[0][0] - 1e-9]
        if len(close) > 1:
            return _txt("Which one? " + "; ".join(f"at {_mmss(s['at'])}: {s.get('what')}" for s in close[:3])
                        + ". Say the time (e.g. which='72') and I'll swap that one.")
        shot = scored[0][1]
    k = shot["k"]
    words = str(a.get("with") or "").strip()
    found = (eng.get(f"/api/app/long/{pid}/shots/{k}/search?q={urllib.parse.quote(words)}").get("results") or []) if words else []
    if not found:
        if words:
            raise ToolError(f"nothing in the project or the footage library looks like “{words}”")
        found = eng.get(f"/api/app/long/{pid}/shots/{k}/choices").get("choices") or []
    if not found:
        raise ToolError("nothing else fits that moment")
    pick = found[0]
    eng.post(f"/api/app/long/{pid}/shots/{k}", {"choice": pick["choice"]})
    return _txt(f"Swapping the shot at {_mmss(shot['at'])} ({shot.get('what')}) for: {pick.get('what') or pick.get('choice')}"
                + (f", from your footage library ({pick.get('folder')})" if pick.get("where") == "library" else ", from this video's own B-roll" if pick.get("where") == "project" else "")
                + ". Everything else stays; it renders again in a few minutes.")


def tool_add_song(eng: Engine, a: dict) -> list[dict]:
    p = _long(eng, a.get("project"))
    item = f"/api/app/item/long/{urllib.parse.quote(p['id'])}/song"
    q = str(a.get("query") or "").strip()
    if not q:
        eng.post(item, {})
        return _txt(f"Taking the music out of “{p.get('title')}”; it renders again.")
    want = _words(q)
    try:
        mine = eng.get(f"/api/app/music/search?q={urllib.parse.quote(q)}&n=5").get("tracks") or []
    except ToolError:
        mine = []
    fits = [t for t in mine if want & _words(" ".join(str(t.get(k) or "") for k in ("title", "artist", "feel", "mood", "tags", "path")))
            or float(t.get("score") or 0) >= 0.25]
    if fits:
        t = fits[0]
        eng.post(item, {"track": t["id"]})
        return _txt(f"Re-rendering “{p.get('title')}” with “{t.get('title') or Path(str(t.get('path') or '')).stem}” from your music. The cut stays the same; about a few minutes.")
    tracks = eng.get(f"/api/app/music/online?source=openverse&q={urllib.parse.quote(q)}").get("tracks") or []
    if not tracks:
        raise ToolError(f"no song fits “{q}”, in your music or in free stock music")
    t = tracks[0]
    got = eng.post("/api/app/music/fetch", {"track": t}, timeout=180)
    eng.post(item, {"song": got["path"]})
    lic = t.get("license") or ""
    return _txt(f"Re-rendering “{p.get('title')}” with “{t.get('title')}” by {t.get('artist') or 'unknown'}"
                + (f" ({lic})" if lic else "") + " from free stock music. The cut stays the same."
                + (f" Credit (also in the project's Notes): {got.get('credit') or t.get('credit')}" if (got.get("credit") or t.get("credit")) else ""))


def tool_catalogue_status(eng: Engine, a: dict) -> list[dict]:
    lib = _state(eng).get("library")
    if not lib:
        return _txt("No footage folders yet: add one in FirstCut's Find footage.")
    stt = lib.get("status") or {}
    now = (f"Now: {stt.get('stage')}" + (f" ({stt.get('done', 0)} of {stt.get('total')})" if stt.get("total") else "")) if stt.get("running") else "Not cataloguing right now."
    roots = ", ".join(r.get("path", "") for r in lib.get("roots") or [])
    return _txt(f"{lib.get('videos') or 0} videos and {lib.get('photos') or 0} photos catalogued, {lib.get('described') or 0} described. {now}\nFolders: {roots}")


def tool_open_in_final_cut(eng: Engine, a: dict) -> list[dict]:
    p = _long(eng, a.get("project"))
    r = eng.post(f"/api/ai/projects/{urllib.parse.quote(p['id'])}/open-fcp")
    return _txt(f"Opening “{p.get('title')}” in Final Cut Pro" + (f": {r.get('path')}" if r.get("path") else "."))


def tool_show_in_finder(eng: Engine, a: dict) -> list[dict]:
    kind, p = _find(_state(eng), a.get("project"))
    if kind == "long":
        eng.post(f"/api/ai/projects/{urllib.parse.quote(p['id'])}/reveal")
    elif kind == "short":
        folder = (eng.post(f"/api/app/item/short/{urllib.parse.quote(p['id'])}/folder") or {}).get("folder") or p.get("home")
        if not folder:
            raise ToolError("those clips aren't saved in a folder yet")
        eng.post("/api/app/open", {"path": folder, "reveal": True, "nearest": True})
    else:
        out = (p.get("out") or {}).get("txt")
        if not out:
            raise ToolError("that transcript isn't finished yet")
        eng.post("/api/app/open", {"path": out, "reveal": True})
    return _txt(f"Showing “{p.get('title')}” in the Finder.")


HANDLERS = {t["name"]: globals()[f"tool_{t['name']}"] for t in TOOLS}


# ------------------------------------------------------------------------------------------------ resources
def resources(eng: Engine) -> list[dict]:
    st = _state(eng)
    out = []
    for p in st.get("projects") or []:
        saved = Path(p.get("saved_to") or p.get("folder") or "")
        if (p.get("job_state") or {}).get("status") != "done" or not (saved / "Notes").is_dir():
            continue
        for name in NOTES:
            if (saved / "Notes" / name).is_file():
                out.append({"uri": f"firstcut://project/{p['id']}/notes/{urllib.parse.quote(name)}", "name": f"{p.get('title')}: {name}",
                            "mimeType": "text/markdown" if name.endswith(".md") else "text/plain"})
    for t in st.get("transcripts") or []:
        if t.get("status") == "done":
            out.append({"uri": f"firstcut://transcript/{t['id']}", "name": f"Transcript: {t.get('title')}", "mimeType": "text/plain"})
    return out


def read_resource(eng: Engine, uri: str) -> dict:
    u = urllib.parse.urlparse(uri)
    parts = [urllib.parse.unquote(x) for x in u.path.strip("/").split("/") if x]
    st = _state(eng)
    if u.scheme == "firstcut" and u.netloc == "project" and len(parts) == 3 and parts[1] == "notes" and parts[2] in NOTES:
        _, p = _find(st, parts[0])
        f = Path(p.get("saved_to") or p.get("folder") or "") / "Notes" / parts[2]
        if f.is_file():
            return {"uri": uri, "mimeType": "text/plain", "text": _read(f, 200_000)}
    if u.scheme == "firstcut" and u.netloc == "transcript" and len(parts) == 1:
        t = eng.get(f"/api/app/transcripts/{urllib.parse.quote(parts[0])}")
        return {"uri": uri, "mimeType": "text/plain", "text": "\n\n".join(x.get("text", "") for x in t.get("paragraphs") or [])}
    raise ToolError("there's no such FirstCut resource")


# ------------------------------------------------------------------------------------------------ the protocol
class Server:
    def __init__(self, eng: Engine | None = None, h: Path | None = None):
        self.home = h or (eng.home if eng else home())
        self.eng = eng or Engine(self.home)

    def handle(self, msg: dict) -> dict | None:
        mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
        if mid is None:                                                   # a notification (notifications/initialized...): no answer
            return None
        try:
            return {"jsonrpc": "2.0", "id": mid, "result": self._result(method, params)}
        except _RpcError as e:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": e.code, "message": str(e)}}
        except Exception as e:  # noqa: BLE001 - never crash the conversation
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": f"FirstCut: {e}"}}

    def _result(self, method: str, params: dict):
        if method == "initialize":
            return {"protocolVersion": params.get("protocolVersion") or PROTOCOL if params.get("protocolVersion") in (PROTOCOL, "2025-03-26", "2024-11-05") else PROTOCOL,
                    "capabilities": {"tools": {"listChanged": False}, "resources": {"listChanged": False}},
                    "serverInfo": SERVER,
                    "instructions": "FirstCut edits video on this Mac. Start long jobs (make_long_video, make_short_clips, get_transcript, change_video) "
                                    "and check on them with project_status: they take minutes. Paths are full paths on this Mac. Nothing here deletes anything. "
                                    "You are the editor and FirstCut is your tool: when the person says which clips to use, in what order, or what to avoid, do not leave "
                                    "that to FirstCut's own planning. Look at the footage (footage_details, get_transcript on the clips), decide the cut yourself, "
                                    "write it as an edit_list, check it with check_edit_list, then start make_long_video with it, the B-roll rules (broll, avoid, "
                                    "min_shot_seconds) and the song. Use free-text notes only for what you cannot express as a list. When the person gives rules in words "
                                    "(keep this part, cut that, open with this, no B-roll while he prays, avoid that camera, a length), plan first: plan_long_video returns "
                                    "the plan and checks every rule in code; ground a quote or a topic with transcript_search, look at a file with shot_details, fix the "
                                    "problems, check_plan, then make_long_video with the plan. Say what you could not do; never claim a rule was met unless check_plan says so."}
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": TOOLS}
        if method == "tools/call":
            name, args = params.get("name"), params.get("arguments") or {}
            if name not in HANDLERS:
                raise _RpcError(-32602, f"unknown tool {name!r}")
            if not enabled(self.home):
                return {"content": _txt(OFF), "isError": True}
            try:
                return {"content": HANDLERS[name](self.eng, args), "isError": False}
            except ToolError as e:
                return {"content": _txt(str(e)), "isError": True}
        if method == "resources/list":
            if not enabled(self.home):
                return {"resources": []}
            try:
                return {"resources": resources(self.eng)}
            except ToolError:
                return {"resources": []}
        if method == "resources/read":
            if not enabled(self.home):
                raise _RpcError(-32002, OFF)
            try:
                return {"contents": [read_resource(self.eng, str(params.get("uri") or ""))]}
            except ToolError as e:
                raise _RpcError(-32002, str(e)) from None
        if method in ("prompts/list",):
            return {"prompts": []}
        if method in ("resources/templates/list",):
            return {"resourceTemplates": []}
        raise _RpcError(-32601, f"FirstCut doesn't know {method!r}")


class _RpcError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def serve(stdin=None, stdout=None, server: Server | None = None) -> int:
    """Read JSON-RPC messages, one per line, until stdin closes."""
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    srv = server or Server()
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            out = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "that wasn't JSON"}}
        else:
            if isinstance(msg, list):                                     # a batch (older clients)
                outs = [o for o in (srv.handle(m) for m in msg if isinstance(m, dict)) if o]
                out = outs or None
            else:
                out = srv.handle(msg) if isinstance(msg, dict) else None
        if out is not None:
            stdout.write(json.dumps(out) + "\n")
            stdout.flush()
    return 0


# ------------------------------------------------------------------------------------------------ the app side (Settings › Connect an AI)
def launcher_path() -> str:
    """Where the app's launcher is: Contents/Resources/bin/firstcut-mcp in the app; in a source checkout, app/VideoBuilder/firstcut-mcp."""
    b = os.environ.get("VIDEOBUILDER_BIN")
    if b and (Path(b) / "firstcut-mcp").exists():
        return str(Path(b) / "firstcut-mcp")
    here = Path(__file__).resolve().parents[2] / "app" / "VideoBuilder" / "firstcut-mcp"
    return str(here)


CLAUDE_DESKTOP = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"


def claude_desktop_state(cfg_path: Path | None = None) -> dict:
    p = cfg_path or CLAUDE_DESKTOP
    installed = p.parent.is_dir() or Path("/Applications/Claude.app").exists()
    try:
        cfg = json.loads(p.read_text()) if p.exists() else {}
        entry = (cfg.get("mcpServers") or {}).get("firstcut")
    except (OSError, ValueError):
        entry = None
    return {"installed": installed, "added": bool(entry), "path": str(p)}


def add_to_claude_desktop(launcher: str, cfg_path: Path | None = None) -> dict:
    """Merge FirstCut into Claude Desktop's config: other servers and settings stay; the old file is backed up first."""
    p = cfg_path or CLAUDE_DESKTOP
    p.parent.mkdir(parents=True, exist_ok=True)
    cfg: dict = {}
    backup = None
    if p.exists():
        raw = p.read_text()
        try:
            cfg = json.loads(raw) if raw.strip() else {}
        except ValueError:
            raise ValueError("Claude Desktop's settings file isn't valid JSON, so FirstCut won't change it: fix it, or add FirstCut by hand "
                             "(the snippet is on this page)") from None
        if not isinstance(cfg, dict):
            raise ValueError("Claude Desktop's settings file isn't what FirstCut expects: add FirstCut by hand (the snippet is on this page)")
        backup = p.with_name(p.name + time.strftime(".before-firstcut-%Y%m%d-%H%M%S"))
        backup.write_text(raw)
    servers = cfg.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    servers["firstcut"] = {"command": launcher, "args": []}
    cfg["mcpServers"] = servers
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=2) + "\n")
    tmp.replace(p)
    return {"added": True, "path": str(p), "backup": str(backup) if backup else None, "servers": sorted(servers)}


CODEX = Path.home() / ".codex" / "config.toml"                          # Codex CLI and the Codex IDE extension share it


def _codex_header(line: str) -> bool:
    """[mcp_servers.firstcut], [mcp_servers."firstcut"] or one of its sub-tables ([mcp_servers.firstcut.env])."""
    import re
    return bool(re.match(r'^\s*\[\s*mcp_servers\s*\.\s*("firstcut"|firstcut)\s*(\]|\.)', line))


def codex_state(cfg_path: Path | None = None) -> dict:
    p = cfg_path or CODEX
    try:
        added = any(_codex_header(x) for x in p.read_text().splitlines()) if p.exists() else False
    except OSError:
        added = False
    return {"installed": p.parent.is_dir(), "added": added, "path": str(p)}


def add_to_codex(launcher: str, cfg_path: Path | None = None) -> dict:
    """Put FirstCut into Codex's config.toml: only the [mcp_servers.firstcut] table is written (replaced if it is there), every other line
    and comment stays as it was, and the old file is backed up first. A careful text edit, not a TOML round trip that could drop comments.
    startup_timeout_sec 60: the launcher may have to open FirstCut first (Codex's default is 10)."""
    import re
    p = cfg_path or CODEX
    p.parent.mkdir(parents=True, exist_ok=True)
    old = p.read_text() if p.exists() else ""
    backup = None
    if p.exists():
        backup = p.with_name(p.name + time.strftime(".before-firstcut-%Y%m%d-%H%M%S"))
        backup.write_text(old)
    keep, skipping = [], False
    for line in old.splitlines():
        if re.match(r"^\s*\[", line):                                    # a table header: ours starts skipping, any other ends it
            skipping = _codex_header(line)
            if skipping and keep and keep[-1].startswith("# FirstCut for AI apps"):
                keep.pop()                                                # (our comment goes with our table)
        if not skipping:
            keep.append(line)
    while keep and not keep[-1].strip():
        keep.pop()
    q = launcher.replace("\\", "\\\\").replace('"', '\\"')
    block = ["# FirstCut for AI apps (FirstCut › Settings › Connect an AI)", "[mcp_servers.firstcut]", f'command = "{q}"', "args = []",
             "startup_timeout_sec = 60", "tool_timeout_sec = 180"]
    while keep and keep[-1].strip() == block[0]:                            # (our own comment from a previous run)
        keep.pop()
        while keep and not keep[-1].strip():
            keep.pop()
    text = "\n".join(keep + ([""] if keep else []) + block) + "\n"
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(p)
    return {"added": True, "path": str(p), "backup": str(backup) if backup else None}


def test_launcher(launcher: str, timeout: float = 60.0) -> dict:
    """Start the launcher the way an AI app would and ask for its tools: {"ok", "tools", "error"}."""
    msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": PROTOCOL, "capabilities": {},
                                                                          "clientInfo": {"name": "firstcut-test", "version": "1"}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]
    if not Path(launcher).exists():
        return {"ok": False, "tools": 0, "error": f"the launcher isn't there: {launcher}"}
    try:
        r = subprocess.run([launcher], input="".join(json.dumps(m) + "\n" for m in msgs), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "tools": 0, "error": "it didn't answer within a minute"}
    except OSError as e:
        return {"ok": False, "tools": 0, "error": str(e)}
    answers = {}
    for line in r.stdout.splitlines():
        try:
            m = json.loads(line)
            answers[m.get("id")] = m
        except ValueError:
            continue
    tools = ((answers.get(2) or {}).get("result") or {}).get("tools") or []
    if not tools:
        return {"ok": False, "tools": 0, "error": (r.stderr.strip().splitlines() or ["it didn't answer"])[-1][:300]}
    return {"ok": True, "tools": len(tools), "error": None}


def main() -> int:
    try:
        return serve()
    except KeyboardInterrupt:
        return 0
