"""Long-video mode: several clips -> ONE coherent video, in the order they were shot.

    1. combine.py already puts the files on one timeline in recording order (creation time, else name).
    2. Files that are mostly pictures (little speech) are kept whole (silent lead-in / tail trimmed).
    3. For the talking parts an AI editor reads the transcript in order and chooses which sentences to keep so the video
       follows one thread: no repeats, false starts, retakes, tangents or filler - each kept stretch a complete thought.
       It also remembers what it already kept ("the story so far") so later chunks continue, not repeat.
    4. Dead air inside what's kept is trimmed (the same pacing as whole-video mode).
Without a working AI it falls back to keeping everything (still in order, dead space trimmed).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.log import get_logger
from . import moments as M

log = get_logger("clipper.story")
MIN_SHARE = 0.25                     # every talking clip keeps at least this share of its length
SPARSE_WPS = 0.35                    # below this many words / second a file counts as "mostly visual"

SYSTEM = (
    "You are a film editor cutting ONE coherent video out of raw clips that were shot one after another. You never "
    "reorder anything: the video must follow the order it happened, like a story with a beginning, middle and end. "
    "From the numbered sentences you are given, choose the stretches to KEEP. Drop: repeated takes and restarts, false "
    "starts, filler and chit-chat, talking to someone off camera, tangents that go nowhere, and dead stretches. Keep what "
    "moves the story forward, the reactions and the payoffs. Every clip is part of the story: keep something from each "
    "one unless it is truly irrelevant, and never cut an action or a payoff that the next clip depends on. Every kept stretch must start at the beginning of a thought "
    "and end when the thought is finished - never mid-sentence. Return ranges of sentence numbers (inclusive, ascending, "
    "not overlapping) and a one-line summary of what the kept part of THIS chunk says (used as 'the story so far').")
SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "keep": {"type": "array", "items": {"type": "object", "properties": {
            "first": {"type": "integer"}, "last": {"type": "integer"}}, "required": ["first", "last"]}},
        "summary": {"type": "string"}},
    "required": ["title", "keep", "summary"],
}


@dataclass
class Plan:
    intervals: list[tuple[float, float]]
    title: str
    summary: str
    kept_files: list[dict]


def _file_of(t: float, files: list[dict]) -> int:
    for k, f in enumerate(files):
        if f["start"] <= t < f["end"] + 1e-6:
            return k
    return max(0, len(files) - 1)


def _chunks(sents: list[M.Sentence], files: list[dict], max_words: int = 700) -> list[list[M.Sentence]]:
    """Consecutive sentences, ~max_words per chunk; a new file always starts a new chunk if the current one is big."""
    out, cur, n = [], [], 0
    for s in sents:
        if cur and (n + len(s.text.split()) > max_words or (n > max_words * 0.5 and _file_of(s.start, files) != _file_of(cur[-1].start, files))):
            out.append(cur)
            cur, n = [], 0
        cur.append(s)
        n += len(s.text.split())
    if cur:
        out.append(cur)
    return out


def _valid(keep: list[dict], lo: int, hi: int) -> list[tuple[int, int]]:
    out, last = [], lo - 1
    for r in sorted(keep, key=lambda r: int(r.get("first", 0))):
        try:
            a, b = int(r["first"]), int(r["last"])
        except (KeyError, TypeError, ValueError):
            continue
        a = max(a, last + 1)
        if lo <= a <= b <= hi:
            out.append((a, b))
            last = b
    return out


def select(llm, sents: list[M.Sentence], files: list[dict], title: str, fraction: float,
           brief: str = "") -> tuple[list[tuple[int, int]], str, str]:
    """-> (kept sentence ranges, a title, the story summary)."""
    kept: list[tuple[int, int]] = []
    so_far: list[str] = []
    best_title = ""
    chunks = _chunks(sents, files)
    for n, chunk in enumerate(chunks, 1):
        lo, hi = chunk[0].idx, chunk[-1].idx
        lines, prev_file = [], None
        for s in chunk:
            f = _file_of(s.start, files)
            if prev_file is not None and f != prev_file:
                lines.append(f"  --- next clip (shot later: {files[f]['name']}) ---")
            prev_file = f
            lines.append(f"  [{s.idx}] {s.text}")
        prompt = (f"Video: {title}\n"
                  + (f"THE USER'S BRIEF for this video (follow it when choosing what to keep): {brief}\n" if brief else "")
                  + (f"Story so far (what you already kept): {' '.join(so_far)[-700:]}\n" if so_far else "This is the start of the video.\n")
                  + f"Keep roughly {int(fraction * 100)}% of this part. Chunk {n} of {len(chunks)}; sentences [{lo}]..[{hi}]:\n"
                  + "\n".join(lines))
        try:
            res = llm.complete_json("story_edit", SYSTEM, prompt, SCHEMA)
            ranges = _valid(res.data.get("keep") or [], lo, hi)
            so_far.append(str(res.data.get("summary") or "").strip())
            best_title = best_title or str(res.data.get("title") or "").strip()
        except Exception as e:  # noqa: BLE001 - keep this chunk whole rather than lose it
            log.warning(f"story edit of chunk {n} failed ({e}); keeping it whole")
            ranges = [(lo, hi)]
        kept.extend(ranges if ranges else [(lo, hi)])               # never drop a whole chunk because of a bad answer
    return kept, best_title, " ".join(s for s in so_far if s)[:800]


def plan(engine, words: list[dict], files: list[dict], total: float, title: str, target_s: float, gap: float,
         brief: str = "") -> Plan:
    """The whole edit as keep-intervals in source seconds."""
    from .engine import repace
    files = files or [{"name": "video", "start": 0.0, "end": total}]
    pool = {k for k, f in enumerate(files) if f.get("broll")}              # B-roll kept for cutaways: not part of the story itself
    sents = M.sentences_from_words(words) if words else []
    speech: list[M.Sentence] = []
    sparse_idx = set()
    for k, f in enumerate(files):
        fw = [w for w in words if f["start"] <= w["start"] < f["end"]]
        dur = max(1.0, f["end"] - f["start"])
        if len(fw) / dur < SPARSE_WPS and k not in pool:
            sparse_idx.add(k)
    for s in sents:
        k_ = _file_of(s.start, files)
        if k_ not in sparse_idx and k_ not in pool:
            speech.append(s)
    kept_s = sum(f["end"] - f["start"] for k, f in enumerate(files) if k in sparse_idx)
    talk_s = sum(f["end"] - f["start"] for k, f in enumerate(files) if k not in sparse_idx and k not in pool)
    fraction = 1.0 if talk_s <= 0 else max(0.3, min(1.0, (target_s - kept_s) / talk_s))
    router = engine.creative_router()                         # the stronger AI (the server's) when reachable, else the local one
    ranges, ai_title, summary = (select(router, speech, files, title, fraction, brief) if speech else ([], "", ""))
    keep_by_file: dict[int, list[tuple[float, float]]] = {}
    by_idx = {s.idx: s for s in sents}
    for a, b in ranges:
        sa, sb = by_idx[a], by_idx[b]
        keep_by_file.setdefault(_file_of(sa.start, files), []).append((sa.start, sb.end))
    for k, f in enumerate(files):                               # safety net: no clip may vanish from the story
        if k in sparse_idx or k in pool:
            continue
        have = sum(b - a for a, b in keep_by_file.get(k, []))
        need = MIN_SHARE * (f["end"] - f["start"])
        if have >= need:
            continue
        mine = [s for s in speech if _file_of(s.start, files) == k and not any(a <= s.start < b for a, b in keep_by_file.get(k, []))]
        for s_ in sorted(mine, key=lambda s: -len(s.text.split())):      # the wordiest sentences first (that's where things happen)
            keep_by_file.setdefault(k, []).append((s_.start, s_.end))
            have += s_.end - s_.start
            if have >= need:
                break
        keep_by_file[k].sort()
    intervals: list[tuple[float, float]] = []
    for k, f in enumerate(files):
        if k in pool:
            continue                                            # cutaway material only
        if k in sparse_idx:                                     # mostly visual: the whole file, minus a silent lead-in
            fw = [w for w in words if f["start"] <= w["start"] < f["end"]]
            a = f["start"] if not fw or fw[0]["start"] - f["start"] < 3 else fw[0]["start"] - 1.0
            intervals.append((a, f["end"]))
            continue
        for a, b in keep_by_file.get(k, []):
            for iv in repace(words, a, b, gap):                 # trim the dead air inside what was kept
                intervals.append((max(f["start"], iv[0]), min(f["end"], iv[1])))
    intervals = [iv for iv in sorted(intervals) if iv[1] - iv[0] > 0.15]
    merged: list[tuple[float, float]] = []
    for a, b in intervals:
        if merged and a <= merged[-1][1] + 0.05:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    if not merged:
        merged = [(0.0, total)]
    kept_files = [{"name": f["name"], "kept_s": round(sum(min(b, f["end"]) - max(a, f["start"]) for a, b in merged
                                                          if b > f["start"] and a < f["end"]), 1)} for f in files]
    return Plan(merged, ai_title or title, summary, kept_files)
