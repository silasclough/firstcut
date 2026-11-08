"""Instructions you can check (WP5 of docs/plans/editor-improvement-plan.md): notes -> typed constraints -> anchors in the footage -> a validator
that code runs on the plan -> a small repair loop -> a report -> a second check on the finished video.

Why: only numbered outlines were ever verified; every other wish ("open with the drone shot", "no B-roll during the prayer", "keep every mention of
grace") was pasted into prompts as prose and never checked. Research (see the plan) says a model follows about three constraints reliably, does not
catch its own mistakes, and is helped by an outside checker that says in words what is wrong. So:

  extract    ONE cheap call turns the notes into constraints, each carrying the exact words of the notes it came from (an item whose quote is not in the
             notes is thrown away: nothing is invented).
  ground     points each constraint at the footage: word numbers (transcript_search.py), file names from the footage facts, or a number. Two possible
             places = `ambiguous`, never a guess. Nothing here lets an AI name a file, a scene id or a word number that the code did not list.
  filters    negatives ("don't use the shaky clips", "no B-roll during the prayer", "cut the part about X") become filters the planner never sees as
             a "don't": files and windows are taken out of its choices, sentences out of what it may keep.
  check_plan deterministic: unknown ids, trims past a file, repeated lines, order, missing sections, length, excluded files and windows, cuts inside a
             word. Every problem is plain words with a `fix_hint`. Hard problems block rendering; soft ones go in the report.
  repair     at most three calls with ONLY the plan, the problems and the checklist (never the whole brief again); what cannot be fixed goes to
             `plan["unmet"]` with one sentence, and the job stops for the person unless auto-approve is on.
  post       after rendering, the finished video's words are checked again (a section that must be there, words that must not be, the length, a title
             card in the first seconds, a line said twice).
"""

from __future__ import annotations

import re
from typing import Any, Callable

from ..core.log import get_logger
from . import transcript_search as TS

log = get_logger("clipper.constraints")

TYPES = ("order", "include", "exclude_words", "exclude_broll_window", "exclude_files", "duration", "graphic", "camera", "style", "take", "caption_text", "chapter")
MAX_CONSTRAINTS = 12
REPAIR_TRIES = 3                                   # (proposal; LLM-Modulo used 10: we keep the cost low)
SHAKY = 0.4                                        # footage facts: motion.shake above this is "shaky"
MIN_TALK_S = 1.0
COVER_MIN = 0.95                                   # an include span counts as kept when this share of its words is inside the played stretches
HARD_RE = re.compile(r"\b(must|always|never|don'?t|do not|doesn'?t|no\b|only|exactly|every|all of|whole|entire)\b|\d", re.I)
SOFT_RE = re.compile(r"\b(prefer|try|maybe|ideally|if possible|would be nice|if you can|consider)\b", re.I)
OUTLINE_LINE = re.compile(r"^\s*(?:\d{1,2}(?:\.\d{1,3})?[.)]|[-*•])\s+\S")
DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?|s|m)\b", re.I)

HARD_CODES = {"UNKNOWN_ID", "END_BEFORE_START", "TRIM_BEYOND_DURATION", "REPEATED_SOURCE", "DURATION_OUT_OF_BAND", "ORDER_FIRST", "ORDER_LAST",
              "INCLUDE_NOT_COVERED", "BROLL_IN_EXCLUDED_WINDOW", "EXCLUDED_FILE_USED", "GRAPHIC_MISSING_AT_START", "CUT_INSIDE_WORD", "POOL_OUTSIDE_FILES"}

EXTRACT_SYSTEM = (
    "You turn a video editor's notes into a list of checkable instructions. Return JSON {\"constraints\": [...]}. Each item: \"type\" (one of order, include, "
    "exclude_words, exclude_broll_window, exclude_files, duration, graphic, camera, style, take, caption_text, chapter), \"hard\" (true when the notes say must, "
    "always, never, don't, no, only, exactly, or give a number; false for prefer, try, maybe, ideally), \"source\" (the exact words from the notes, quoted, not "
    "paraphrased), and \"want\" (what must be true, in one plain sentence). Rules: \"open with X\" and \"end on X\" are order; \"keep every mention of X\" and "
    "\"keep the whole section about X\" are include; \"cut the part about X\" is exclude_words; \"no B-roll during X\" and \"stay on him while he X\" are "
    "exclude_broll_window; \"don't use the shaky clips\", \"no footage from the phone\" are exclude_files; any length or time target is duration; \"put the "
    "title on screen at the start\", \"show his name\" are graphic; \"use the second camera when he walks\" is camera; words about pace, music, captions, tone "
    "are style (soft unless numeric); \"use the second take\", \"the last take is best\" are take; \"a chapter card before each section\" is chapter. One "
    "instruction per item. Do not invent instructions the notes do not contain. If the notes are a numbered outline, do not list its steps here (they are "
    "handled separately); list only the sentences around it.")
SCHEMA = {"type": "object", "properties": {"constraints": {"type": "array", "items": {"type": "object", "properties": {
    "type": {"type": "string"}, "hard": {"type": "boolean"}, "source": {"type": "string"}, "want": {"type": "string"}},
    "required": ["type", "source"]}}}, "required": ["constraints"]}
MATCH_SCHEMA = {"type": "object", "properties": {"sids": {"type": "array", "items": {"type": "string"}}}, "required": ["sids"]}
MATCH_SYSTEM = ("You match a video editor's instruction to the B-roll scenes that fit it. Each scene is one line: its id, its file and what it shows. Return the ids "
                "of the scenes the instruction is about, best match first. Use only ids from the list. Return an empty list when none fits.")
REPAIR_SYSTEM = ("You fix an edit plan so it passes the listed checks. Change as little as possible. Return the whole plan in the same JSON shape. Where a check "
                 "cannot be passed with the footage available, leave the plan as it is and add the check to \"unmet\" with one sentence why. Never invent ids.")
REPAIR_SCHEMA = {"type": "object", "properties": {
    "items": {"type": "array", "items": {"type": "object", "properties": {
        "kind": {"type": "string"}, "a": {"type": "integer"}, "b": {"type": "integer"}, "start": {"type": "number"}, "end": {"type": "number"},
        "file": {"type": "string"}, "satisfies": {"type": "array", "items": {"type": "string"}}}, "required": ["kind"]}},
    "unmet": {"type": "array", "items": {"type": "object", "properties": {"constraint": {"type": "string"}, "why": {"type": "string"}}}}},
    "required": ["items"]}


# ------------------------------------------------------------------------------------------------------------------ small helpers
def _flat(text: str) -> str:
    return re.sub(r"[\s\"'“”‘’]+", " ", str(text or "")).strip().lower()


def _data(res) -> dict:
    d = getattr(res, "data", res)
    return d if isinstance(d, dict) else {}


def _num(v, default: float | None = None) -> float | None:
    try:
        f = float(v)
        return f if f == f and abs(f) != float("inf") else default
    except (TypeError, ValueError):
        return default


def parse_duration(text: str) -> float | None:
    """'a 2-minute recap', 'about 90 seconds', '1.5 minutes' -> seconds. The first time-like number in the text."""
    m = DURATION_RE.search(str(text or "").replace("-", " "))
    if not m:
        return None
    n, unit = float(m.group(1)), m.group(2).lower()
    return n * (3600.0 if unit.startswith("h") else 60.0 if unit.startswith("m") else 1.0)


def _quoted(text: str) -> list[str]:
    return [q.strip() for q in re.findall(r"[\"“]([^\"”]{3,200})[\"”]|(?<!\w)'([^']{6,200})'(?!\w)", str(text or ""))
            for q in q if q.strip()]


def strip_outline(notes: str) -> str:
    """The notes without the lines of a numbered outline (the outline is handled by flow.py)."""
    return "\n".join(ln for ln in str(notes or "").splitlines() if not OUTLINE_LINE.match(ln))


# ------------------------------------------------------------------------------------------------------------------ extract
def extract(ai, notes: str, style: dict | None = None) -> list[dict]:
    """The notes as typed constraints (ONE call, task `constraints_extract`). Items whose `source` is not in the notes, or whose type is unknown, are dropped.
    No notes (or only an outline) = no call. A failed call = no constraints (the plan is made as before)."""
    text = strip_outline(notes).strip()
    if not text or ai is None:
        return []
    try:
        res = ai.complete_json("constraints_extract", EXTRACT_SYSTEM, "NOTES:\n" + text[:6000] + "\n\nList the checkable instructions in these notes.", SCHEMA)
    except Exception as e:  # noqa: BLE001 - extraction is a help; a plan without it is still a plan
        log.warning(f"reading the notes as instructions failed: {e}")
        return []
    flat_notes = _flat(text)
    out: list[dict] = []
    rows = _data(res).get("constraints")
    for row in (rows if isinstance(rows, list) else [])[:40]:
        if not isinstance(row, dict):
            continue
        kind = str(row.get("type") or "").strip().lower()
        src = str(row.get("source") or "").strip()
        if kind not in TYPES or not src or _flat(src) not in flat_notes:            # (what the notes do not say is not an instruction)
            continue
        want = str(row.get("want") or src).strip()
        hard = bool(row.get("hard")) if isinstance(row.get("hard"), bool) else bool(HARD_RE.search(src))
        if SOFT_RE.search(src):
            hard = False
        if kind == "style" and not re.search(r"\d", src):
            hard = False                                                             # style words are soft unless numeric
        out.append({"id": f"c{len(out) + 1}", "type": kind, "hard": hard, "source": src, "status": "ungrounded",
                    "anchor": {"words": None, "spans": [], "file_ids": [], "scene_ids": [], "seconds": None, "text": ""},
                    "rule": {"text": want}, "candidates": [], "note": ""})
        if len(out) >= MAX_CONSTRAINTS:
            break
    return out


# ------------------------------------------------------------------------------------------------------------------ ground
def _query_of(c: dict) -> tuple[str, str]:
    q = _quoted(c["source"])
    if q:
        return q[0], "quote"
    return (c["rule"].get("text") or c["source"]), "topic"


def _is_visual(c: dict) -> bool:
    return not _quoted(c["source"]) and bool(re.search(r"\b(shots?|footage|clips?|drone|aerial|camera|cam|b-?roll|phone|shaky|handheld)\b", c["source"], re.I))


def _facts(f: dict) -> dict:
    return f.get("facts") or {}


def match_files(c: dict, files: list[dict], pool: list[dict], ai=None) -> tuple[list[str], list[str]]:
    """(file names, scene ids) an instruction about footage points at, from the footage facts first and the AI only for what facts cannot say."""
    text = f"{c['source']} {c['rule'].get('text', '')}".lower()
    names: list[str] = []
    if re.search(r"\b(shaky|shake|shaking|unstable|handheld wobble|jittery)\b", text):
        names = [f["name"] for f in files if float((_facts(f).get("motion") or {}).get("shake") or 0) > SHAKY]
    elif re.search(r"\b(drone|aerial|dji)\b", text):
        names = [f["name"] for f in files if (_facts(f).get("camera") or {}).get("family") == "dji"]
    elif re.search(r"\b(phone|iphone|android)\b", text):
        names = [f["name"] for f in files if (_facts(f).get("camera") or {}).get("family") == "phone"]
    elif re.search(r"\bgopro\b", text):
        names = [f["name"] for f in files if (_facts(f).get("camera") or {}).get("family") == "gopro"]
    elif re.search(r"\b(second|2nd|other) (camera|cam)\b|\bcam(era)?[ _-]?(b|2)\b", text):
        names = [f["name"] for f in files if re.search(r"cam(era)?[ _-]?(b|2)\b|\bb[ _-]?cam|\bcamb", f["name"], re.I)]
    sids: list[str] = []
    if names:
        keep = set(names)
        sids = [s["sid"] for s in pool if s.get("sid") and s.get("file") in keep]
        return names, sids
    if ai is None or not pool:
        return [], []
    listing = "\n".join(f"{s['sid']}: {s.get('file', '')}, {str(s.get('what') or '')[:110]}" for s in pool[:300] if s.get("sid"))
    if not listing:
        return [], []
    try:
        res = ai.complete_json("scene_match", MATCH_SYSTEM, f"INSTRUCTION: {c['source']}\n\nSCENES:\n{listing}", MATCH_SCHEMA)
    except Exception as e:  # noqa: BLE001
        log.warning(f"matching an instruction to scenes failed: {e}")
        return [], []
    valid = {s["sid"]: s for s in pool if s.get("sid")}
    got = [str(x) for x in _data(res).get("sids") or [] if str(x) in valid]            # (an id the AI made up is dropped)
    names = sorted({valid[x].get("file") for x in got if valid[x].get("file")})
    return names, got


def ground(constraints: list[dict], words: list[dict], sents, files: list[dict], pool: list[dict], ai=None, style: dict | None = None) -> list[dict]:
    """Point every constraint at the footage. Sets `status` (grounded | ambiguous | ungrounded), `anchor`, `candidates` and a plain `note`.
    Two places far apart for something that should be ONE place = ambiguous: the person is asked, nothing is guessed."""
    out = []
    band = float(((style or {}).get("length") or {}).get("band") or 0.10)
    for c in constraints:
        c = {**c, "anchor": dict(c.get("anchor") or {}), "rule": dict(c.get("rule") or {}), "candidates": []}
        t = c["type"]
        if t in ("duration",):
            sec = parse_duration(c["source"]) or parse_duration(c["rule"].get("text", ""))
            if sec:
                c["anchor"]["seconds"] = sec
                c["rule"]["band"] = [round(sec * (1 - band), 1), round(sec * (1 + band), 1)]
                c["rule"]["flag"] = "stated"
                c["status"] = "grounded"
            else:
                c["status"], c["note"] = "ungrounded", "I could not read a length from this."
        elif t in ("include", "exclude_words", "exclude_broll_window") or (t == "order" and not _is_visual(c)):
            q, mode = _query_of(c)
            hits = TS.search(words, sents, q, mode, ai)
            every = bool(re.search(r"\b(every|each|all|whole|entire)\b", c["source"], re.I))
            if t == "order":
                c["rule"]["position"] = "last" if re.search(r"\b(end|finish|close|last|final|wrap)\b", c["source"], re.I) else "first"
            if not hits:
                c["status"], c["note"] = "ungrounded", f"I could not find “{q}” in the transcript."
            elif len(hits) > 1 and TS.far_apart(hits) and not every and t != "exclude_broll_window":
                c["status"] = "ambiguous"
                c["candidates"] = [{"words": [h["first_word"], h["last_word"]], "start_s": h["start_s"], "end_s": h["end_s"], "text": h["text"]} for h in hits[:5]]
                c["note"] = "“" + q + "” could be " + " or ".join(f"{_mmss(h['start_s'])}-{_mmss(h['end_s'])}" for h in hits[:3]) + ": pick one."
            else:
                spans = [[h["first_word"], h["last_word"]] for h in hits]
                c["status"] = "grounded"
                c["anchor"].update({"words": spans[0], "spans": spans, "text": hits[0]["text"], "texts": [h["text"] for h in hits],
                                    "start_s": hits[0]["start_s"], "end_s": hits[-1]["end_s"] if every or t == "exclude_broll_window" else hits[0]["end_s"],
                                    "lead_silence_s": hits[0]["lead_silence_s"], "trail_silence_s": hits[-1]["trail_silence_s"]})
                c["anchor"]["times"] = [[h["start_s"], h["end_s"]] for h in hits]
                c["note"] = f"Taken to be {_mmss(hits[0]['start_s'])}: “{hits[0]['text'][:80]}”."
        elif t in ("exclude_files", "camera") or (t == "order" and _is_visual(c)):
            names, sids = match_files(c, files, pool, ai)
            if t == "order":
                c["rule"]["position"] = "last" if re.search(r"\b(end|finish|close|last|final)\b", c["source"], re.I) else "first"
            if not names:
                c["status"], c["note"] = "ungrounded", "I could not tell which footage this means."
            elif t == "order" and len(names) > 1:
                c["status"] = "ambiguous"
                c["candidates"] = [{"file": n} for n in names[:5]]
                c["note"] = "This could be " + ", ".join(names[:4]) + ": pick one."
            else:
                c["status"] = "grounded"
                c["anchor"].update({"file_ids": names, "scene_ids": sids})
                c["note"] = "Taken to be " + ", ".join(names[:4]) + ("…" if len(names) > 4 else "") + "."
        elif t == "graphic":
            c["rule"]["before_s"] = 3.0 if re.search(r"\b(start|begin|open|first|top)\b", c["source"], re.I) else None
            q = _quoted(c["source"])
            c["rule"]["text"] = q[0] if q else c["rule"].get("text", "")
            c["status"] = "grounded"
        elif t in ("take", "caption_text", "chapter", "style"):
            c["status"] = "grounded"
        out.append(c)
    return out


def _mmss(t: float) -> str:
    t = int(max(0, round(float(t or 0))))
    return f"{t // 60}:{t % 60:02d}"


# ------------------------------------------------------------------------------------------------------------------ filters
def filters(constraints: list[dict], sents=None, words: list[dict] | None = None) -> dict:
    """The negatives as filters the planner never sees as a "don't": {"exclude_files": [...], "exclude_windows": [{words, start_s, end_s, constraint}],
    "exclude_sentences": {idx...}}. Only grounded constraints produce a filter."""
    ex_files: list[str] = []
    wins: list[dict] = []
    sent_idx: set[int] = set()
    for c in constraints:
        if c.get("status") != "grounded":
            continue
        a = c.get("anchor") or {}
        if c["type"] == "exclude_files":
            ex_files += [n for n in a.get("file_ids") or [] if n not in ex_files]
        elif c["type"] == "exclude_broll_window":
            for sp, (t0, t1) in zip(a.get("spans") or [], a.get("times") or []):
                wins.append({"words": list(sp), "start_s": float(t0), "end_s": float(t1), "constraint": c["id"]})
        elif c["type"] == "exclude_words" and sents:
            for t0, t1 in a.get("times") or []:
                for s in sents:
                    if float(t0) - 0.05 <= (float(s.start) + float(s.end)) / 2 <= float(t1) + 0.05:
                        sent_idx.add(s.idx)
    return {"exclude_files": ex_files, "exclude_windows": wins, "exclude_sentences": sent_idx}


def _win_se(w):
    from .poolrules import window_se
    return window_se(w)


def _win_pairs(wins):
    return [se for se in (_win_se(w) for w in (wins or [])) if se]


def apply_pool_filters(pool: list[dict], flt: dict) -> list[dict]:
    """The B-roll pool without excluded files, and without scenes lying inside an excluded window (virtual seconds)."""
    ex = set(flt.get("exclude_files") or [])
    wins = flt.get("exclude_windows") or []
    out = []
    for s in pool:
        if s.get("file") in ex:
            continue
        a, b = _num(s.get("start"), None), _num(s.get("end"), None)
        if a is not None and b is not None and any(a < e and b > st_ for st_, e in _win_pairs(wins)):
            continue
        out.append(s)
    return out


def enforce(cutaways: list[dict], plan: dict, pool: list[dict], intervals, out_time: Callable) -> list[dict]:
    """B-roll the planner placed, minus what a hard rule forbids: a shot from an excluded file, a shot over an excluded window (`out_time` maps
    virtual seconds to the finished video's seconds). Called from render(); a dropped shot is a plain gap, never a failure."""
    ex = set(plan.get("exclude_files") or [])
    wins = plan.get("exclude_windows") or []
    if not (ex or wins):
        return cutaways
    spans = []
    for w in wins:
        se = _win_se(w)
        if not se:
            continue
        if not isinstance(w, dict):                       # (a plain pair is already in the finished video's seconds)
            spans.append(se)
            continue
        try:
            spans.append((float(out_time(intervals, se[0])), float(out_time(intervals, se[1]))))
        except Exception:  # noqa: BLE001 - a window that maps to nothing blocks nothing
            continue
    keep = []
    for c in cutaways:
        sc = pool[c["scene"]] if isinstance(c.get("scene"), int) and 0 <= c["scene"] < len(pool) else {}
        if sc.get("file") in ex:
            continue
        a, b = float(c.get("at", 0)), float(c.get("at", 0)) + float(c.get("length", 0))
        if any(a < y and b > x for x, y in spans):
            continue
        keep.append(c)
    return keep


# ------------------------------------------------------------------------------------------------------------------ the plan as segments
def _by_idx(sents) -> dict:
    return {s.idx: s for s in sents or []}


def _idx(v: Any) -> int | None:
    """A sentence number from whatever a saved plan holds (None for anything that is not a whole number)."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v == int(v):
        return int(v)
    return None


def segments(plan: dict, sents, files: list[dict] | None = None) -> list[dict]:
    """The plan's items as [{id, i, kind, start, end, a, b, file, satisfies, score}] in play order (seconds in the footage's shared timeline).
    A damaged saved plan (an item that is not an object, numbers that are text or lists) is read as far as it makes sense: bad items are skipped."""
    by = _by_idx(sents)
    out = []
    items = plan.get("items") if isinstance(plan, dict) else None
    for i, it in enumerate(items if isinstance(items, list) else []):
        if not isinstance(it, dict):
            continue
        sat = it.get("satisfies")
        seg: dict[str, Any] = {"id": str(it.get("id") or f"i{i}"), "i": i, "kind": it.get("kind"), "satisfies": list(sat) if isinstance(sat, (list, tuple)) else [],
                               "score": _num(it.get("score")), "protect": bool(it.get("protect")), "start": None, "end": None, "a": None, "b": None, "file": it.get("file")}
        if it.get("kind") == "talk":
            a, b = _idx(it.get("a")), _idx(it.get("b"))
            seg["a"], seg["b"] = a, b
            if a in by and b in by:
                t0, t1 = _num(it.get("t0")), _num(it.get("t1"))
                seg["start"] = t0 if t0 is not None else float(by[a].start)
                seg["end"] = t1 if t1 is not None else float(by[b].end)
        elif it.get("kind") == "visual":
            seg["start"], seg["end"] = _num(it.get("start")), _num(it.get("end"))
        out.append(seg)
    return out


def _file_at(files: list[dict], t: float) -> dict | None:
    for f in files or []:
        if float(f.get("start", 0)) - 1e-3 <= t < float(f.get("end", 0)) - 1e-3:
            return f
    return None


def _problem(code: str, message: str, constraint_id: str | None = None, segment_id: str | None = None, fix_hint: str = "", hard: bool | None = None) -> dict:
    return {"code": code, "message": message, "constraint_id": constraint_id, "segment_id": segment_id, "fix_hint": fix_hint,
            "hard": (code in HARD_CODES) if hard is None else bool(hard)}


PER_ITEM_S = 0.3      # what the render adds around each talking passage (a lead before the first word, a tail after the last, the pauses kept inside): measured 2026-11


def render_extra_s(plan: dict, n_talk: int | None = None) -> float:
    """Seconds the render adds to the sum of the plan's passages, so the length the validator checks is the length the person gets: an end card
    (`plan["render_extra"]["end_card_s"]`, set when the folder has a logo), a fixed extra, and a little around each talking passage. The per-passage
    figure is the one MEASURED on this project's last render (`plan["length_model"]["per_item_s"]`) when there is one."""
    rx = plan.get("render_extra") if isinstance(plan.get("render_extra"), dict) else {}
    lm = plan.get("length_model") if isinstance(plan.get("length_model"), dict) else {}
    if n_talk is None:
        n_talk = sum(1 for it in plan.get("items") or [] if isinstance(it, dict) and it.get("kind") == "talk")
    try:
        per = float(lm.get("per_item_s")) if lm.get("per_item_s") is not None else float(rx.get("per_item_s", PER_ITEM_S))
        return max(0.0, float(rx.get("end_card_s") or 0.0) + float(rx.get("fixed_s") or 0.0) + max(0.0, per) * n_talk)
    except (TypeError, ValueError):
        return PER_ITEM_S * n_talk


def measured_per_item_s(plan_talk_s: float, rendered_s: float, n_talk: int, end_card_s: float = 0.0) -> float | None:
    """What this render really added around each passage: (rendered - end card - the plan's own sum) / passages, kept between 0.05 and 1.0 s."""
    if n_talk <= 0 or rendered_s <= 0:
        return None
    return round(min(1.0, max(0.05, (rendered_s - end_card_s - plan_talk_s) / n_talk)), 3)


def check_plan(plan: dict, constraints: list[dict] | None, files: list[dict], words: list[dict], pool: list[dict] | None, style: dict | None = None,
               sents=None, cutaways: list[dict] | None = None, to_out: Callable | None = None) -> dict:
    """Run every check code can run on a plan. -> {"ok", "problems": [{code, message, constraint_id, segment_id, fix_hint, hard}], "totals": {...}}.
    `ok` is True when no HARD problem is left. Nothing here calls an AI. Checks that need a render (placed B-roll, graphics) run only when the caller
    passes what the render knows (`cutaways` with `to_out`, or `plan["gfx_cards"]`)."""
    constraints = [c for c in (constraints if constraints is not None else plan.get("constraints") or []) if isinstance(c, dict)]
    style = style or plan.get("style_obj") or {}
    segs = segments(plan, sents, files)
    problems: list[dict] = []
    by = _by_idx(sents)
    names = {f.get("name") for f in files or []}
    sids = {s.get("sid") for s in pool or [] if s.get("sid")}
    hard_of = {c["id"]: bool(c.get("hard")) for c in constraints}

    # -- ids and shapes
    for sg in segs:
        if sg["kind"] == "talk":
            if sg["a"] is None or sg["b"] is None or sg["a"] not in by or sg["b"] not in by:
                problems.append(_problem("UNKNOWN_ID", f"a talking passage names sentences {sg['a']}-{sg['b']} that are not in the transcript", None, sg["id"],
                                         "Use only sentence numbers from the transcript."))
                continue
            if sg["b"] < sg["a"]:
                problems.append(_problem("END_BEFORE_START", f"a talking passage ends (sentence {sg['b']}) before it starts (sentence {sg['a']})", None, sg["id"],
                                         "Swap the two numbers or drop the passage."))
        elif sg["kind"] == "visual":
            if sg["start"] is None or sg["end"] is None:
                problems.append(_problem("UNKNOWN_ID", "a picture-only stretch has no start or end", None, sg["id"], "Give it start and end seconds."))
                continue
            if sg["end"] <= sg["start"]:
                problems.append(_problem("END_BEFORE_START", f"a picture-only stretch ends ({sg['end']:.1f} s) before it starts ({sg['start']:.1f} s)", None, sg["id"],
                                         "Swap the two times or drop the stretch."))
                continue
            f = _file_at(files, sg["start"])
            if sg.get("file") and sg["file"] not in names:
                problems.append(_problem("UNKNOWN_ID", f"a picture-only stretch uses the file {sg['file']}, which is not in the folder", None, sg["id"],
                                         "Use a file from the footage list."))
            elif f is None:
                problems.append(_problem("UNKNOWN_ID", f"a picture-only stretch starts at {sg['start']:.1f} s, which is in none of the files", None, sg["id"],
                                         "Use a stretch inside one file."))
            elif sg["end"] > float(f["end"]) + 0.05:
                d = sg["end"] - sg["start"]
                problems.append(_problem("TRIM_BEYOND_DURATION", f"{f['name']} at {sg['start'] - float(f['start']):.2f} s for {d:.2f} s (the file is "
                                         f"{float(f['end']) - float(f['start']):.2f} s)", None, sg["id"], "Shorten it so it ends inside the file."))
    for sg in segs:                                                                  # sentences too short to be heard
        if sg["kind"] == "talk" and sg["start"] is not None and sg["end"] - sg["start"] < MIN_TALK_S and sg["end"] > sg["start"]:
            problems.append(_problem("SEGMENT_TOO_SHORT", f"a talking passage of {sg['end'] - sg['start']:.1f} s is too short to be a clip", None, sg["id"],
                                     "Drop it or join it to its neighbour.", hard=False))
    shot_lo = float(((style.get("broll") or {}).get("shot_s") or [2.0])[0])
    for k, c in enumerate(cutaways or []):                                             # placed B-roll (after a render)
        sc = pool[c["scene"]] if isinstance(c.get("scene"), int) and pool and 0 <= c["scene"] < len(pool) else None
        if sc is None or (sc.get("sid") and sc["sid"] not in sids):
            problems.append(_problem("UNKNOWN_ID", f"B-roll shot {k + 1} names a scene that is not in the pool", None, f"b{k}", "Use a scene id from the pool."))
        elif float(c.get("length", 0)) < min(shot_lo, 1.0) - 1e-6:
            problems.append(_problem("SEGMENT_TOO_SHORT", f"B-roll shot {k + 1} plays {float(c.get('length', 0)):.1f} s", None, f"b{k}",
                                     f"Make it at least {shot_lo:g} s or drop it.", hard=False))

    # -- repeated lines: the same sentences (or the same stretch) played twice
    seen: list[tuple[int, int, str]] = []
    for sg in segs:
        if sg["kind"] == "talk" and sg["a"] in by and sg["b"] in by and sg["b"] >= sg["a"]:
            for a, b, other in seen:
                lo, hi = max(a, sg["a"]), min(b, sg["b"])
                if lo <= hi and hi - lo + 1 >= 1:
                    problems.append(_problem("REPEATED_SOURCE", f"sentences {lo}-{hi} are played twice: “{by[lo].text[:60]}”", None, sg["id"],
                                             "Keep them only where they are heard first."))
                    break
            seen.append((sg["a"], sg["b"], sg["id"]))

    # -- the constraints
    played = [(sg["start"], sg["end"]) for sg in segs if sg["start"] is not None and sg["end"] is not None and sg["end"] > sg["start"]]
    total_s = sum(b - a for a, b in played)
    plan_sum_s = total_s
    total_s += render_extra_s(plan) if total_s > 0 else 0.0           # (the length the person gets: the render adds an end card and a little around each passage)
    for c in constraints:
        cid, a = c["id"], c.get("anchor") or {}
        if c.get("status") not in ("grounded", "met", "unmet"):
            continue
        t = c["type"]
        if t == "duration" and (c.get("rule") or {}).get("band"):
            lo, hi = c["rule"]["band"]
            if not lo <= total_s <= hi:
                weakest = sorted((sg for sg in segs if not sg["protect"] and not sg["satisfies"] and sg["start"] is not None and sg["end"] is not None),
                                 key=lambda sg: (sg["score"] if sg["score"] is not None else 1e9, sg["end"] - sg["start"]))
                if total_s > hi:
                    hint = (f"Cut {total_s - hi:.0f} s: drop the weakest optional passages first: "
                            + ", ".join(f"{sg['id']} ({sg['end'] - sg['start']:.0f} s)" for sg in weakest[:6])) if weakest else f"Cut {total_s - hi:.0f} s."
                else:
                    hint = f"Add {lo - total_s:.0f} s of the best remaining material."
                problems.append(_problem("DURATION_OUT_OF_BAND", f"the cut is {total_s:.0f} s; {a.get('seconds', 0):.0f} s was asked for (between {lo:.0f} and {hi:.0f})",
                                         cid, None, hint, hard=hard_of.get(cid, True)))
        elif t == "order" and segs:
            pos = (c.get("rule") or {}).get("position", "first")
            seg_ = segs[0] if pos == "first" else segs[-1]
            ok = True
            if a.get("file_ids"):
                f = _file_at(files, seg_["start"]) if seg_["start"] is not None else None
                ok = bool(f and f.get("name") in a["file_ids"]) or seg_.get("file") in a["file_ids"]
            elif a.get("start_s") is not None and seg_["start"] is not None:
                t_anchor = float(a["start_s"] if pos == "first" else a.get("end_s", a["start_s"]))
                ok = seg_["start"] - 0.3 <= t_anchor <= seg_["end"] + 0.3
            if not ok:
                problems.append(_problem("ORDER_FIRST" if pos == "first" else "ORDER_LAST",
                                         f"the video should {'open' if pos == 'first' else 'end'} with “{(a.get('text') or c['source'])[:70]}” but it does not",
                                         cid, seg_["id"], f"Move the passage that holds it to the {'start' if pos == 'first' else 'end'}.", hard=hard_of.get(cid, True)))
        elif t == "include":
            miss = _uncovered(words, a.get("spans") or [], played)
            if miss:
                problems.append(_problem("INCLUDE_NOT_COVERED", "required words are not in the cut: “" + " ".join(miss[:12]) + ("…" if len(miss) > 12 else "") + "”",
                                         cid, None, "Add the sentences with these words back.", hard=hard_of.get(cid, True)))
        elif t == "exclude_words":
            inside = _covered(words, a.get("spans") or [], played)
            if inside:
                problems.append(_problem("INCLUDE_NOT_COVERED", "words that should be cut are still in: “" + " ".join(inside[:12]) + "”", cid, None,
                                         "Take these sentences out.", hard=hard_of.get(cid, True)))
        elif t == "exclude_files":
            bad = [sg for sg in segs if sg["kind"] == "visual" and (sg.get("file") in (a.get("file_ids") or [])
                                                                    or (_file_at(files, sg["start"] or 0) or {}).get("name") in (a.get("file_ids") or []))]
            for sg in bad:
                problems.append(_problem("EXCLUDED_FILE_USED", f"a picture-only stretch uses {(_file_at(files, sg['start']) or {}).get('name', sg.get('file'))}, which you said not to use",
                                         cid, sg["id"], "Drop the stretch or take it from another file.", hard=hard_of.get(cid, True)))
            for k, cu in enumerate(cutaways or []):
                sc = pool[cu["scene"]] if isinstance(cu.get("scene"), int) and pool and 0 <= cu["scene"] < len(pool) else {}
                if sc.get("file") in (a.get("file_ids") or []):
                    problems.append(_problem("EXCLUDED_FILE_USED", f"B-roll shot {k + 1} is from {sc.get('file')}, which you said not to use", cid, f"b{k}",
                                             "Drop the shot.", hard=hard_of.get(cid, True)))
        elif t == "exclude_broll_window" and cutaways and to_out:
            for w in (a.get("times") or []):
                try:
                    x, y = float(to_out(w[0])), float(to_out(w[1]))
                except Exception:  # noqa: BLE001
                    continue
                for k, cu in enumerate(cutaways):
                    if float(cu.get("at", 0)) < y and float(cu.get("at", 0)) + float(cu.get("length", 0)) > x:
                        problems.append(_problem("BROLL_IN_EXCLUDED_WINDOW", f"B-roll shot {k + 1} plays over a stretch you said to stay on the speaker ({_mmss(x)}-{_mmss(y)})",
                                                 cid, f"b{k}", "Drop the shot.", hard=hard_of.get(cid, True)))
        elif t == "graphic" and (c.get("rule") or {}).get("before_s") and plan.get("gfx_cards") is not None:
            lim = float(c["rule"]["before_s"])
            if not any(float(g.get("start", g.get("at", 99))) <= lim for g in plan.get("gfx_cards") or []):
                problems.append(_problem("GRAPHIC_MISSING_AT_START", f"no graphic appears in the first {lim:g} s (“{c['source'][:60]}”)", cid, None,
                                         "Put the title or name card at the start.", hard=hard_of.get(cid, True)))
        elif t == "camera" and a.get("file_ids"):
            uses = any((_file_at(files, sg["start"] or 0) or {}).get("name") in a["file_ids"] or sg.get("file") in a["file_ids"] for sg in segs) or any(
                (pool[cu["scene"]].get("file") in a["file_ids"]) for cu in cutaways or [] if isinstance(cu.get("scene"), int) and pool and 0 <= cu["scene"] < len(pool))
            if not uses and (cutaways or plan.get("rendered")):
                problems.append(_problem("CAMERA_PREFERENCE_IGNORED", f"you asked for {', '.join(a['file_ids'][:3])} but it is not used", cid, None,
                                         "Prefer that camera where it fits.", hard=False))

    # -- a cut inside a word
    margin = float((style.get("pace") or {}).get("cut_margin_s") or 0.08)
    ws = sorted(words or [], key=lambda w: float(w.get("start", 0)))
    for sg in segs:
        if sg["kind"] != "talk" or sg["start"] is None:
            continue
        for edge, t_ in (("start", sg["start"]), ("end", sg["end"])):
            w = next((w for w in ws if float(w["start"]) + margin < t_ < float(w["end"]) - margin), None)
            if w is not None:
                problems.append(_problem("CUT_INSIDE_WORD", f"a cut at {t_:.2f} s falls inside the word “{w.get('text', '')}”", None, sg["id"],
                                         f"Move the {edge} of the passage to the word's edge."))
                break

    # -- a pool whose scenes are outside their files (WP1)
    try:
        from . import localproj as LP
        for msg in (LP.check_pool_inside_files(pool or [], files) if pool and files else [])[:6]:
            problems.append(_problem("POOL_OUTSIDE_FILES", str(msg), None, None, "Drop the scene or re-read the folder."))
    except Exception as e:  # noqa: BLE001 - the pool check is a bonus here
        log.debug(f"pool check skipped: {e}")

    # -- soft: the B-roll share and the longest unchanged stretch, only when a render placed the B-roll
    broll_share = None
    if cutaways and total_s > 0:
        broll_share = round(sum(float(c.get("length", 0)) for c in cutaways) / total_s, 2)
        want = (style.get("broll") or {}).get("share")
        if want is not None and abs(broll_share - float(want)) > 0.15:
            problems.append(_problem("BROLL_SHARE_OFF", f"B-roll is {broll_share * 100:.0f}% of the video; the style asks for about {float(want) * 100:.0f}%", None, None,
                                     "Add or drop B-roll shots.", hard=False))
    totals = {"length_s": round(total_s, 1), "plan_s": round(plan_sum_s, 1), "talk_s": round(sum(b - a for sg in segs if sg["kind"] == "talk" for a, b in [(sg["start"], sg["end"])] if a is not None), 1),
              "broll_share": broll_share, "graphics_per_min": None}
    return {"ok": not any(p["hard"] for p in problems), "problems": problems, "totals": totals}


def _inside(t0: float, t1: float, played: list[tuple[float, float]]) -> bool:
    mid = (t0 + t1) / 2
    return any(a - 0.05 <= mid <= b + 0.05 for a, b in played)


def _uncovered(words: list[dict], spans: list, played) -> list[str]:
    """The words of the required spans that are NOT inside the played stretches (empty when at least COVER_MIN of them are)."""
    miss, total = [], 0
    for sp in spans:
        try:
            i, j = int(sp[0]), int(sp[1])
        except (TypeError, ValueError, IndexError):
            continue
        for w in (words or [])[max(0, i):j + 1]:
            total += 1
            if not _inside(float(w["start"]), float(w["end"]), played):
                miss.append(str(w.get("text", "")))
    return miss if total and (total - len(miss)) / total < COVER_MIN else []


def _covered(words: list[dict], spans: list, played) -> list[str]:
    """The words of an excluded span that ARE still played (empty when under 20 % of them are)."""
    inside, total = [], 0
    for sp in spans:
        try:
            i, j = int(sp[0]), int(sp[1])
        except (TypeError, ValueError, IndexError):
            continue
        for w in (words or [])[max(0, i):j + 1]:
            total += 1
            if _inside(float(w["start"]), float(w["end"]), played):
                inside.append(str(w.get("text", "")))
    return inside if total and len(inside) / total >= 0.2 else []


# ------------------------------------------------------------------------------------------------------------------ the checklist and the repair loop
def checklist(constraints: list[dict]) -> str:
    """The constraints as a checklist for the END of a planner's prompt (what is said last is remembered best), one line each. '' when there are none."""
    rows = []
    for c in constraints or []:
        if c.get("status") not in ("grounded",):
            continue
        a = c.get("anchor") or {}
        where = ""
        if a.get("words"):
            where = f" words {a['words'][0]}-{a['words'][1]}"
        elif a.get("file_ids"):
            where = " files " + ", ".join(a["file_ids"][:3])
        elif a.get("seconds"):
            where = f" {a['seconds']:.0f} s"
        rows.append(f"{c['id']} ({'hard' if c.get('hard') else 'soft'}) {c['type'].replace('_', ' ')}{where}: {c['source'][:140]}")
    return ("INSTRUCTIONS TO SATISFY (check each before answering):\n" + "\n".join(rows)) if rows else ""


def repair_prompt(plan: dict, problems: list[dict], constraints: list[dict] | None = None) -> str:
    """ONLY the plan (ids and seconds), the problems with their hints, and the checklist: never the whole brief (it would be re-planned from scratch)."""
    lines = []
    for i, it in enumerate(plan.get("items") or []):
        if it.get("kind") == "talk":
            lines.append(f"{it.get('id') or f'i{i}'}: talk sentences {it.get('a')}-{it.get('b')}" + (f" satisfies {it['satisfies']}" if it.get("satisfies") else ""))
        else:
            lines.append(f"{it.get('id') or f'i{i}'}: visual {it.get('start')}-{it.get('end')} s")
    probs = "\n".join(f"- {p['code']}: {p['message']}" + (f" [{p['segment_id']}]" if p.get("segment_id") else "") + (f" Hint: {p['fix_hint']}" if p.get("fix_hint") else "")
                      for p in problems)
    return "CURRENT PLAN (in play order):\n" + "\n".join(lines) + "\n\nCHECKS THAT FAILED:\n" + probs + ("\n\n" + checklist(constraints or []) if constraints else "")


def _items_ok(rows) -> list[dict] | None:
    """The repair call's items, cleaned to the shapes the renderer reads (None when the answer is not a usable list)."""
    if not isinstance(rows, list) or not rows:
        return None
    out = []
    for r in rows:
        if not isinstance(r, dict):
            return None
        if r.get("kind") == "talk" and isinstance(r.get("a"), int) and isinstance(r.get("b"), int):
            it = {"kind": "talk", "a": r["a"], "b": r["b"]}
        elif r.get("kind") == "visual" and _num(r.get("start")) is not None and _num(r.get("end")) is not None:
            it = {"kind": "visual", "start": float(r["start"]), "end": float(r["end"])}
            if r.get("file"):
                it["file"] = str(r["file"])
        else:
            return None
        if isinstance(r.get("satisfies"), list):
            it["satisfies"] = [str(x) for x in r["satisfies"]][:8]
        out.append(it)
    return out


def _merge_back(old_items: list[dict], new_items: list[dict]) -> list[dict]:
    """Keep what the old items carried (t, t0, t1, score, id ...) when the repaired item is the same stretch."""
    key = lambda it: (it.get("kind"), it.get("a"), it.get("b"), it.get("start"), it.get("end"))      # noqa: E731
    old = {key(it): it for it in old_items}
    return [{**old[key(it)], **it} if key(it) in old else it for it in new_items]


def trim_to_band(plan: dict, sents, constraints: list[dict], band: list[float]) -> dict:
    """Drop the lowest-scored OPTIONAL passages (none that satisfies a hard constraint or is protected) until the cut is inside the band. Never drops the
    first or last passage. Returns a new plan."""
    items = list(plan.get("items") or [])
    by = _by_idx(sents)
    hard_ids = {c["id"] for c in constraints if c.get("hard")}
    req = []                                                  # (the spans a hard include/order instruction needs: any item that touches one is kept, tagged or not)
    for c in constraints:
        if c.get("hard") and c.get("type") in ("include", "order") and c.get("status") in ("grounded", "met"):
            for tt in (c.get("anchor") or {}).get("times") or []:
                try:
                    req.append((float(tt[0]), float(tt[1])))
                except (TypeError, ValueError, IndexError):
                    continue

    def needed(it):
        if it.get("kind") != "talk" or it.get("a") not in by or it.get("b") not in by:
            return False
        t0, t1 = by[it["a"]].start, by[it["b"]].end
        return any(t0 - 0.05 < b and t1 + 0.05 > a for a, b in req)

    def length(its):
        total = render_extra_s(plan, sum(1 for it in its if it.get("kind") == "talk"))
        for it in its:
            if it.get("kind") == "talk" and it.get("a") in by and it.get("b") in by:
                total += float(it.get("t1", by[it["b"]].end)) - float(it.get("t0", by[it["a"]].start))
            elif it.get("kind") == "visual":
                total += float(it.get("end", 0)) - float(it.get("start", 0))
        return total
    while len(items) > 2 and length(items) > band[1]:
        cand = [(k, it) for k, it in enumerate(items[1:-1], 1) if not it.get("protect") and not (set(it.get("satisfies") or []) & hard_ids) and not needed(it)]
        if not cand:
            break
        k, _ = min(cand, key=lambda kv: (_num(kv[1].get("score"), 1e9), length([kv[1]])))
        items.pop(k)
    return {**plan, "items": items}


def run_repair(ai, plan: dict, constraints: list[dict], run_check: Callable[[dict], dict], tries: int = REPAIR_TRIES) -> tuple[dict, dict, int]:
    """The generate-test-critique loop: while hard problems remain and tries are left, ONE `plan_repair` call fixes the plan (it sees only the plan, the
    problems and the checklist). A repair that adds a problem code the plan did not have (a made-up id) is thrown away. -> (plan, last check result, calls made)."""
    res = run_check(plan)
    calls = 0
    unmet: list[dict] = list(plan.get("unmet") or [])
    while not res["ok"] and calls < tries and ai is not None:
        hard = [p for p in res["problems"] if p["hard"]]
        calls += 1
        try:
            out = ai.complete_json("plan_repair", REPAIR_SYSTEM, repair_prompt(plan, hard, constraints), REPAIR_SCHEMA)
        except Exception as e:  # noqa: BLE001 - a failed repair is an unmet constraint, not a failed video
            log.warning(f"plan repair failed: {e}")
            break
        data = _data(out)
        items = _items_ok(data.get("items"))
        for u in data.get("unmet") or []:
            if isinstance(u, dict) and u.get("constraint") and not any(x.get("constraint") == u["constraint"] for x in unmet):
                unmet.append({"constraint": str(u["constraint"]), "why": str(u.get("why") or "")[:200]})
        if items is None:
            continue
        cand = {**plan, "items": _merge_back(plan.get("items") or [], items)}
        new = run_check(cand)
        before = {(p["code"], p.get("constraint_id")) for p in res["problems"] if p["hard"]}
        added = [p for p in new["problems"] if p["hard"] and (p["code"], p.get("constraint_id")) not in before and p["code"] in ("UNKNOWN_ID", "END_BEFORE_START")]
        if added:
            continue
        plan, res = cand, new
    plan = {**plan, "unmet": unmet}
    return plan, res, calls


# ------------------------------------------------------------------------------------------------------------------ the whole pass for plan()
def prepare(ai, notes: str, style: dict | None, words: list[dict], sents, files: list[dict], pool: list[dict]) -> dict:
    """Before planning: extract and ground the notes' instructions, and turn the negatives into filters. -> {"constraints", "filters", "checklist"}."""
    cons = extract(ai, notes, style)
    if not cons:
        return {"constraints": [], "filters": {"exclude_files": [], "exclude_windows": [], "exclude_sentences": set()}, "checklist": ""}
    cons = ground(cons, words, sents, files, pool, ai, style)
    return {"constraints": cons, "filters": filters(cons, sents, words), "checklist": checklist(cons)}


def finish(ai, plan: dict, prep: dict, words: list[dict], sents, files: list[dict], pool: list[dict], style: dict | None = None, auto_approve: bool = False,
           drop_sentences: Callable | None = None) -> dict:
    """After planning: apply the filters to the plan, check it, repair it (at most three calls), record what is unmet, and decide whether to stop for the
    person. Writes plan["constraints"], ["unmet"], ["problems"], ["exclude_files"], ["exclude_windows"], ["totals"] and, when it must stop, ["approval"]."""
    cons = [dict(c) for c in prep.get("constraints") or []]
    flt = prep.get("filters") or {}
    if flt.get("exclude_sentences") and drop_sentences:
        plan = {**plan, "items": drop_sentences(plan.get("items") or [], set(flt["exclude_sentences"]))}
    plan = {**plan, "exclude_files": list(flt.get("exclude_files") or []), "exclude_windows": list(flt.get("exclude_windows") or [])}
    plan = tag_satisfies(plan, cons, sents, words)

    def run_check(p: dict) -> dict:
        return check_plan(p, cons, files, words, pool, style, sents=sents)
    dur = next((c for c in cons if c["type"] == "duration" and c.get("status") == "grounded" and c["rule"].get("band")), None)
    if dur:
        res0 = run_check(plan)
        if any(p["code"] == "DURATION_OUT_OF_BAND" and "Cut" in p["fix_hint"] for p in res0["problems"]):
            plan = trim_to_band(plan, sents, cons, dur["rule"]["band"])               # (the length is a number: code trims, the model is not asked)
    plan, res, calls = run_repair(ai, plan, cons, run_check)
    failing = {p.get("constraint_id") for p in res["problems"] if p["hard"] and p.get("constraint_id")}
    failing |= {c["id"] for c in cons if c["status"] == "ungrounded" and c.get("hard")}
    unmet = [u for u in (plan.get("unmet") or []) if isinstance(u, dict) and u.get("constraint") in failing]      # (a model's "I could not" is only believed when the check agrees)
    for p in res["problems"]:
        if p["hard"] and p.get("constraint_id") and not any(u["constraint"] == p["constraint_id"] for u in unmet):
            unmet.append({"constraint": p["constraint_id"], "why": p["message"]})
    for c in cons:                                                                       # ungrounded/ambiguous constraints
        if c["status"] == "ungrounded" and c.get("hard") and not any(u["constraint"] == c["id"] for u in unmet):
            unmet.append({"constraint": c["id"], "why": c.get("note") or "I could not tell what this refers to."})
    unmet_ids = {u["constraint"] for u in unmet}
    for c in cons:
        if c["status"] == "grounded":
            c["status"] = "unmet" if c["id"] in unmet_ids else "met"
    questions = [c for c in cons if c["status"] == "ambiguous" and c.get("hard")]
    for c in cons:                                                                       # a soft ambiguous one is decided: the first candidate
        if c["status"] == "ambiguous" and not c.get("hard") and c["candidates"]:
            first = c["candidates"][0]
            if first.get("words"):
                c["anchor"].update({"words": first["words"], "spans": [first["words"]], "text": first.get("text", ""), "start_s": first.get("start_s"), "end_s": first.get("end_s"),
                                    "times": [[first.get("start_s"), first.get("end_s")]]})
            elif first.get("file"):
                c["anchor"]["file_ids"] = [first["file"]]
            c["status"], c["note"] = "grounded", "I took it to be " + (first.get("file") or f"“{(first.get('text') or '')[:60]}”") + "."
    plan = {**plan, "constraints": cons, "unmet": unmet, "problems": list(res["problems"]), "totals": res["totals"], "repair_calls": calls}
    stop = [c for c in cons if c["status"] == "ambiguous" and c.get("hard")]
    hard_left = [p for p in res["problems"] if p["hard"]]
    reasons = [c["note"] for c in stop]
    if hard_left and not auto_approve:
        reasons += [f"{p.get('constraint_id') or 'plan'}: {p['message']}" for p in hard_left[:8]]
    if stop or (hard_left and not auto_approve):
        plan["approval"] = {"reason": "Some of your instructions need a look before I render.", "questions": reasons, "report": report(cons, plan["problems"], unmet)}
        plan["awaiting_approval"] = True
    else:
        if plan.pop("approval", None) is not None:                # (a second look that finds nothing left to ask: the stop from the first look is lifted)
            plan.pop("awaiting_approval", None)
    return plan


def tag_satisfies(plan: dict, constraints: list[dict], sents, words: list[dict]) -> dict:
    """Give every talk item the ids of the constraints whose words it covers (computed from word ranges in code, so a planner that did not return
    `satisfies` costs nothing). Existing values are kept."""
    by = _by_idx(sents)
    items = []
    for it in plan.get("items") or []:
        it = dict(it)
        if it.get("kind") == "talk" and it.get("a") in by and it.get("b") in by:
            t0, t1 = by[it["a"]].start, by[it["b"]].end
            got = set(it.get("satisfies") or [])
            for c in constraints:
                if c.get("type") in ("include", "order") and c.get("status") in ("grounded", "met"):
                    for tt in (c.get("anchor") or {}).get("times") or []:
                        try:
                            a0, a1 = float(tt[0]), float(tt[1])
                        except (TypeError, ValueError, IndexError):
                            continue
                        if a1 >= a0 and t0 - 0.05 < a1 and t1 + 0.05 > a0:      # (an item that covers ANY part of the span is part of it: a span over two sentences tags both)
                            got.add(c["id"])
            if got:
                it["satisfies"] = sorted(got)
        items.append(it)
    return {**plan, "items": items}


# ------------------------------------------------------------------------------------------------------------------ the report
def report(constraints: list[dict], problems: list[dict] | None = None, unmet: list[dict] | None = None, checked: list[dict] | None = None) -> str:
    """Plain words (Notes/plan.md and the page): what was asked, how each instruction was understood, what is not met and why."""
    unmet_by = {u["constraint"]: u["why"] for u in unmet or []}
    lines = ["## Your instructions", ""]
    if not constraints:
        lines.append("No separate instructions were found in the notes.")
    for c in constraints:
        state = {"met": "done", "unmet": "NOT done", "grounded": "understood", "ambiguous": "needs your answer", "ungrounded": "not understood"}.get(c["status"], c["status"])
        extra = (" — " + unmet_by[c["id"]]) if c["id"] in unmet_by else (" — " + c["note"] if c.get("note") else "")
        lines.append(f"- {c['id']} ({'must' if c.get('hard') else 'would like'}) “{c['source'][:120]}”: {state}{extra}")
    soft = [p for p in problems or [] if not p.get("hard")]
    if soft:
        lines += ["", "## Things to look at", ""] + [f"- {p['message']}" for p in soft[:12]]
    if checked:
        lines += ["", "## What I checked after rendering", ""] + [f"- {'ok' if x.get('ok') else 'PROBLEM'}: {x.get('what')}" for x in checked]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------------------------ after the render
def _tok_ratio(needle: list[str], hay: list[str]) -> float:
    import difflib
    if not needle:
        return 1.0
    sm = difflib.SequenceMatcher(a=needle, b=hay, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / len(needle)


def post_render_check(constraints: list[dict], rendered_words: list[dict], duration_s: float | None = None, first_frames_text: str | None = None,
                      sents=None) -> list[dict]:
    """Check the FINISHED video against the constraints that can be checked from its words, its length and its first frames.
    -> [{"constraint": id|None, "ok": bool, "what": plain words}]. Every `include` span must be there (85 % of its words, in order); every `exclude_words`
    span must not be (under 60 %); the length in its band; a graphic's text found by OCR in the first seconds; no line said twice within 20 s."""
    out: list[dict] = []
    hay = TS.tokens(" ".join(str(w.get("text", "")) for w in rendered_words or []))
    for c in constraints or []:
        a = c.get("anchor") or {}
        if c.get("status") not in ("grounded", "met", "unmet"):
            continue
        t = c["type"]
        if t == "include":
            for tx in (a.get("texts") or [a.get("text")] or []):
                toks = TS.tokens(tx or "")
                if len(toks) >= 2:
                    r = _tok_ratio(toks, hay)
                    out.append({"constraint": c["id"], "ok": r >= 0.85, "what": f"“{(tx or '')[:60]}” is {'in' if r >= 0.85 else 'MISSING from'} the finished video"})
        elif t == "exclude_words":
            for tx in (a.get("texts") or [a.get("text")] or []):
                toks = TS.tokens(tx or "")
                if len(toks) >= 3:
                    r = _tok_ratio(toks, hay)
                    out.append({"constraint": c["id"], "ok": r < 0.6, "what": f"“{(tx or '')[:60]}” is {'gone' if r < 0.6 else 'STILL in'} the finished video"})
        elif t == "duration" and (c.get("rule") or {}).get("band") and duration_s:
            lo, hi = c["rule"]["band"]
            out.append({"constraint": c["id"], "ok": lo <= duration_s <= hi, "what": f"length {duration_s:.0f} s (asked {a.get('seconds', 0):.0f} s, band {lo:.0f}-{hi:.0f})"})
        elif t == "graphic" and (c.get("rule") or {}).get("text") and first_frames_text is not None:
            want = TS.tokens(c["rule"]["text"])
            ok = bool(want) and _tok_ratio(want, TS.tokens(first_frames_text)) >= 0.6
            out.append({"constraint": c["id"], "ok": ok, "what": f"“{c['rule']['text'][:50]}” is {'on' if ok else 'NOT on'} screen in the first seconds"})
    if sents:                                                                           # a retake: two sentences within 20 s that say the same thing
        import difflib
        ss = list(sents)
        for i, s in enumerate(ss):
            for o in ss[i + 1:]:
                if o.start - s.end > 20.0:
                    break
                if len(TS.tokens(s.text)) >= 5 and difflib.SequenceMatcher(a=TS.tokens(s.text), b=TS.tokens(o.text), autojunk=False).ratio() >= 0.85:
                    out.append({"constraint": None, "ok": False, "what": f"the same line is said twice, at {_mmss(s.start)} and {_mmss(o.start)}: “{s.text[:60]}”"})
                    break
    return out


# ------------------------------------------------------------------------------------------------------------------ revisions
def revise_constraints(ai, plan: dict, text: str, sents, words: list[dict], files: list[dict], drop_sentences: Callable | None = None) -> tuple[dict, str]:
    """A lasting rule given in a change request ("never use the phone footage", "keep the part about the flood"): read like the notes, merged into the
    plan's constraints (a rule already there, with the same words, is not added twice), grounded, applied to the items and checked. -> (plan, what was done)."""
    new = extract(ai, text, plan.get("style_obj"))
    have = [dict(c) for c in plan.get("constraints") or []]
    seen = {_flat(c.get("source")) for c in have}
    new = [c for c in new if _flat(c["source"]) not in seen]
    if not new:
        return plan, ""
    n = max([int(str(c["id"])[1:]) for c in have if str(c.get("id", "")).startswith("c") and str(c["id"])[1:].isdigit()] or [0])
    for k, c in enumerate(new, 1):
        c["id"] = f"c{n + k}"
    new = ground(new, words, sents, files, plan.get("pool") or [], ai, plan.get("style_obj"))
    allc = have + new
    prep = {"constraints": allc, "filters": filters(allc, sents, words), "checklist": checklist(allc)}
    plan = finish(ai, plan, prep, words, sents, files, plan.get("pool") or [], plan.get("style_obj"), True, drop_sentences)
    return plan, f"{len(new)} new rule(s) from your note"


def validate_edited_plan(plan: dict, files: list[dict], words: list[dict], pool: list[dict], sents, style: dict | None = None) -> dict:
    """check_plan for a plan an outside AI edited (the MCP `check_plan` and `make_long_video(plan=...)`): the plan's own constraints are checked too."""
    return check_plan(plan, plan.get("constraints") or [], files, words, pool, style or plan.get("style_obj"), sents=sents)
