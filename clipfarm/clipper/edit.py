"""Change a finished clip by telling the AI what you want, in plain words.

    "start with the line about the roof, and cut the part where he laughs"
    "trim the sponsor sentence"        "make the first sentence the hook"        "call it 'He fell off a roof'"

The AI sees the clip's transcript (numbered sentences, plus a little of what comes before and after in the source video in
case you point at it) and the edit as it stands now, and returns the new edit as an ordered list of sentence ranges - so
a sentence can be moved to the front as a cold-open hook, others dropped - and optionally a new hook title. The clip is then
re-rendered from the original footage (nothing is lost; "Revert" restores the first version).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.log import get_logger
from . import moments as M

log = get_logger("clipper.edit")
MAX_S = 180.0
SYSTEM = (
    "You are a video editor applying one instruction from the user to a short clip. You get the clip's numbered sentences "
    "(CLIP) and some sentences from just before / after it in the source video (CONTEXT, usable only if the user asks for "
    "something from there), plus the edit as it stands now. Return the NEW edit as an ordered list of sentence ranges "
    "(inclusive): the clip plays the ranges in the order you list them. Rules: apply ONLY what the user asked and keep everything "
    "else as it is now. 'Put X at the front / as the hook' = X becomes the first range and is removed from its old place. "
    "'Cut / trim / remove Y' = leave Y out. Match the user's wording to the closest sentence. Every range starts and ends on a "
    "whole sentence. If the user wants a different on-screen hook title, return it in 'title' (short, max 70 characters), "
    "otherwise leave 'title' empty. If you cannot do what was asked, return the current edit unchanged and explain in 'problem'. "
    "'summary' = one short sentence saying what you changed.")
SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {"type": "array", "items": {"type": "object", "properties": {
            "first": {"type": "integer"}, "last": {"type": "integer"}}, "required": ["first", "last"]}},
        "title": {"type": "string"}, "summary": {"type": "string"}, "problem": {"type": "string"}},
    "required": ["segments", "title", "summary", "problem"],
}


@dataclass
class Plan:
    ranges: list[tuple[int, int]]          # sentence index ranges, in play order
    segments: list[tuple[float, float]]    # the same as source seconds
    title: str
    summary: str
    problem: str


def window(sents: list[M.Sentence], start: float, end: float, pad: float = 45.0) -> list[M.Sentence]:
    return [s for s in sents if s.end > start - pad and s.start < end + pad]


def plan(llm, win: list[M.Sentence], clip_start: float, clip_end: float, current: list[tuple[int, int]], instruction: str,
         title: str, max_s: float = MAX_S) -> Plan:
    idx = {s.idx: s for s in win}
    lines = []
    for s in win:
        tag = "CLIP" if clip_start - 0.05 <= s.start and s.end <= clip_end + 0.05 else "CONTEXT"
        lines.append(f"  [{s.idx}] ({tag}) {s.text}")
    prompt = (f"Clip title now: {title}\nThe edit now plays these ranges in order: "
              f"{', '.join(f'[{a}]..[{b}]' for a, b in current)}\n\nSentences:\n" + "\n".join(lines) +
              f"\n\nTHE USER'S INSTRUCTION: {instruction}")
    res = llm.complete_json("clip_edit", SYSTEM, prompt, SCHEMA)
    ranges, used = [], set()
    for r in res.data.get("segments") or []:
        try:
            a, b = int(r["first"]), int(r["last"])
        except (KeyError, TypeError, ValueError):
            continue
        if a not in idx or b not in idx or a > b:
            continue
        keep = [i for i in range(a, b + 1) if i in idx and i not in used]       # no sentence plays twice
        if not keep:
            continue
        used.update(keep)
        # split where earlier sentences were already used
        run = [keep[0]]
        for i in keep[1:]:
            if i == run[-1] + 1:
                run.append(i)
            else:
                ranges.append((run[0], run[-1]))
                run = [i]
        ranges.append((run[0], run[-1]))
    segs = [(idx[a].start, idx[b].end) for a, b in ranges]
    if not segs or sum(b - a for a, b in segs) < 3.0 or sum(b - a for a, b in segs) > max_s:
        return Plan(current, [(idx[a].start, idx[b].end) for a, b in current if a in idx and b in idx], "",
                    "", str(res.data.get("problem") or "That edit would leave the clip too short or too long - nothing changed."))
    return Plan(ranges, segs, str(res.data.get("title") or "").strip()[:70], str(res.data.get("summary") or "").strip(),
                str(res.data.get("problem") or "").strip())
