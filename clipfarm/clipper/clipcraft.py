"""Short-clip craft (WP8): the rubric, the hook checks, where a clip ends and which caption look suits it. Pure functions, no AI and no ffmpeg.

    rubric     the AI grades a clip 1-5 on Hook, Flow and Value with a reason each; the 1-10 score is computed here, not by the model
    lead       the first word starts 0.15 s into the clip (no dead air before the hook)
    tail       the clip ends a quarter second after its last word; a very short clip that calls back to its first line keeps a little more (a loop)
    hook       first speech within `style.hook.speech_by_s`, a face within `face_by_s`: reported as warnings, never a rejection
    caption    karaoke (one to three words) for teaching, phrase (three to five) for stories, emotion, comedy and interviews
"""

from __future__ import annotations

import re

WEIGHTS = (0.4, 0.3, 0.3)             # hook, flow, value (proposal: the hook is the first thing a viewer judges)
LEAD_S = 0.15                         # speech starts this far into the clip
TAIL_S = 0.25                         # the clip ends this long after the last word
LOOP_ROOM_S = 0.15                    # extra room tone kept for a clip that loops
LOOP_UNDER_S = 15.0
REHOOK_OVER_S = 45.0

KARAOKE = ("pop", "fire", "bold")     # one to three words at a time (the existing caption looks, by how many words they show)
PHRASE = ("clean", "sky")             # three to five words
PHRASE_SOURCES = ("streamer", "twitch", "twitch_clip", "podcast", "interview")
TEACHING = re.compile(r"\b(how to|how do you|step|steps|tip|tips|tutorial|learn|here'?s why|the reason|number one|first,|second,|third,|\d+ (?:ways|things|reasons|tips|steps|rules)|"
                      r"explain|means|called|formula|recipe|ingredient|trick|hack)\b", re.I)


def _grade(v) -> int | None:
    try:
        g = int(round(float(v)))
    except (TypeError, ValueError, OverflowError):
        return None
    return g if 1 <= g <= 5 else None


def rubric_of(c: dict) -> dict | None:
    """-> {"hook", "flow", "value" (1-5), "why_hook", "why_flow", "why_value"} from one AI clip answer, or None when a grade is missing or out of range."""
    grades = [_grade(c.get(k)) for k in ("hook", "flow", "value")]          # (the caller maps the schema's hook_grade onto "hook": "hook" is already the one-line reason in a clip answer)
    if any(g is None for g in grades):
        return None
    out = dict(zip(("hook", "flow", "value"), grades))
    for k in ("why_hook", "why_flow", "why_value"):
        out[k] = re.sub(r"\s+", " ", str(c.get(k) or "")).strip()[:140]
    return out


def score_of(r: dict | None, fallback: float = 5.0) -> float:
    """1-10 from the three grades (a 3-3-3 clip is a 6.0; 5-5-5 is a 10): code, not the model, so the scale is the same for every clip."""
    if not r:
        return float(fallback)
    s = (r["hook"] * WEIGHTS[0] + r["flow"] * WEIGHTS[1] + r["value"] * WEIGHTS[2]) * 2
    return round(max(1.0, min(10.0, s)), 1)


def fallback_score(c: dict, hooky=None) -> float:
    """A 1-10 score for a clip answer that has no usable rubric (the model left grades out; the schema has no free `score` any more): the grades it did
    give, averaged, else what the clip itself shows (a question or a hook word in the hook line, a concrete title with a number or a name). 5.0-7.5, so
    a batch of such clips still ranks instead of tying at 5.0."""
    got = [g for g in (_grade(c.get(k)) for k in ("hook_grade", "hook", "flow", "value")) if g is not None]
    if got:
        return round(max(1.0, min(10.0, sum(got) / len(got) * 2)), 1)
    hook = str(c.get("hook") or "")
    title = str(c.get("title") or "")
    pts = 5.0
    if hooky is not None:
        pts += min(1.2, 0.4 * len(hooky.findall(hook + " " + title)))
    if "?" in hook or "?" in title:
        pts += 0.4
    words = title.split()
    if any(re.search(r"\d", w) for w in words) or any(w[:1].isupper() and len(w) >= 3 for w in words[1:]):
        pts += 0.6
    if 4 <= len(words) <= 12:
        pts += 0.3
    return round(min(7.5, pts), 1)


def length_band(min_len: float, max_len: float) -> str:
    """The length sentence of the prompt, from the options: no fixed 'sweet spot' that overrides what the person asked for."""
    lo, hi = int(round(min_len)), int(round(max_len))
    return (f"Each clip must be {lo}-{hi} seconds long. Let the thought decide where it lands inside that band, and never pad a clip to reach a length: "
            f"a clip that finishes at {lo + (hi - lo) // 3} seconds is better than one stretched to {hi}.")


# ------------------------------------------------------------------------------------------------------------------ the edges of the clip
def trim_lead(intervals: list[tuple[float, float]], words: list[dict], lead_s: float = LEAD_S) -> list[tuple[float, float]]:
    """The first interval starts `lead_s` before its first word (silence before the hook is dead air), never later than it was."""
    if not intervals:
        return intervals
    a, b = intervals[0]
    first = next((w for w in words if float(w["end"]) > a and float(w["start"]) < b), None)
    if first is None:
        return intervals
    start = max(a, float(first["start"]) - lead_s)
    if start >= b - 0.5:
        return intervals
    return [(round(start, 3), b), *intervals[1:]]


def could_loop(first_text: str, last_text: str) -> bool:
    """A very short clip loops when its last line calls back to its first: they share a content word (five letters or more)."""
    def keys(t: str) -> set[str]:
        return {w for w in re.findall(r"[a-z']{5,}", str(t).lower())}
    return bool(keys(first_text) & keys(last_text))


def end_after_last_word(intervals: list[tuple[float, float]], words: list[dict], tail_s: float = TAIL_S, loop: bool = False) -> list[tuple[float, float]]:
    """The last interval ends `tail_s` after the last word it plays (plus `LOOP_ROOM_S` when the clip loops), never into the next word."""
    if not intervals:
        return intervals
    a, b = intervals[-1]
    inside = [w for w in words if float(w["start"]) < b and float(w["end"]) > a]
    if not inside:
        return intervals
    last = max(inside, key=lambda w: float(w["end"]))
    want = float(last["end"]) + tail_s + (LOOP_ROOM_S if loop else 0.0)
    nxt = next((float(w["start"]) for w in words if float(w["start"]) >= float(last["end"]) - 1e-6 and w is not last), None)
    last_end = float(last["end"])
    end = max(last_end + 0.05, want if nxt is None else min(want, nxt - 0.03))      # (cut back when it ran on into dead air or the next thought, grown a little when it was cut at the last word)
    if nxt is not None and end > nxt - 0.02:                                    # never into the next word: in a tiny gap the cut is half way across it
        end = nxt - 0.02 if nxt - 0.02 > last_end else last_end + max(0.0, nxt - last_end) / 2
    return [*intervals[:-1], (a, round(end, 3))]


# ------------------------------------------------------------------------------------------------------------------ hook checks
def hook_warnings(cap_words: list[dict], out_len: float, first_face_s: float | None, faces_measured: bool, speech_by_s: float = 0.5, face_by_s: float = 1.0) -> list[dict]:
    """Warnings (never a rejection) about the first second: speech starting late, no face when the source has faces. Words are on the clip's own timeline."""
    out: list[dict] = []
    if out_len < 3:
        return out
    first = min((float(w["start"]) for w in cap_words), default=None)
    if first is None:
        out.append({"level": "warn", "code": "hook_late", "what": "nobody speaks in the clip's opening"})
    elif first > speech_by_s + 0.001:
        out.append({"level": "warn", "code": "hook_late", "what": f"the first word comes {first:.1f} s into the clip (the aim is under {speech_by_s:.1f} s)"})
    if faces_measured and (first_face_s is None or first_face_s > face_by_s + 0.001):
        out.append({"level": "warn", "code": "no_face_early",
                    "what": "no face on screen in the first second" if first_face_s is None else f"the first face appears {first_face_s:.1f} s in (the aim is under {face_by_s:.1f} s)"})
    return out


def first_face_time(samples: list, rate: float) -> float | None:
    """Seconds into the sampled stretch of the first sample that has a face (None: none in any sample)."""
    for k, faces in enumerate(samples or []):
        if faces:
            return k / max(rate, 0.1)
    return None


# ------------------------------------------------------------------------------------------------------------------ captions
def caption_kind(source: dict | None, text: str = "", ai_answer: str | None = None) -> str:
    """'karaoke' | 'phrase'. Streamer and podcast-like sources are always phrase. An AI answer of karaoke|phrase decides the rest; without one, teaching
    words decide."""
    s = source if isinstance(source, dict) else {}
    meta = " ".join(str(s.get(k) or "") for k in ("title", "topic", "channel", "uploader")).lower()
    if str(s.get("kind") or "") in PHRASE_SOURCES or re.search(r"\b(podcast|interview|stream|streamer|conversation)\b", meta):
        return "phrase"
    if ai_answer in ("karaoke", "phrase"):
        return ai_answer
    return "karaoke" if TEACHING.search(text or "") else "phrase"


def caption_style_for(kind: str, index: int = 0) -> str:
    """One of the engine's caption looks that shows the right number of words; the clips of a batch rotate through the matching ones."""
    pool = KARAOKE if kind == "karaoke" else PHRASE
    return pool[index % len(pool)]
