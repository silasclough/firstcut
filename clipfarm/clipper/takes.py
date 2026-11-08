"""Repeated takes: the same line said more than once, grouped, with the best one chosen (WP6 of docs/plans/editor-improvement-plan.md).

A retake tool works on consecutive repeated or restarted sentences with a similarity tolerance, and "the best take is not always the recording with
the fewest mistakes". So a group lists every take of a line with what is known about each (is it complete, how many fillers, did it restart, how it
was recorded), `best` picks one by a weighted score (or the last / first take when the style says so), and a repeat the speaker meant ("let me say
that again", a refrain that comes back through the talk) is marked `deliberate` and is never reduced.

A take here is one sentence (a `Sentence` of moments.py); `a` and `b` are sentence numbers, so a longer take is a run `a..b` made by the caller.
"""

from __future__ import annotations

import re

from . import moments as M

SIM = 0.8                  # token-set similarity from which two sentences are takes of one line (the retake tools' tolerance)
WINDOW_S = 20.0            # takes of a line follow each other within this many seconds (a director cue between them gives them twice as long)
RESTART_S = 8.0            # a sentence that starts over the first words of another one within this long is a restart of it
RESTART_WORDS = 4
REFRAIN_N, REFRAIN_GAP_S = 3, 60.0     # a line that comes back 3+ times, each more than a minute after the last, is a refrain, not a retake
MIN_TOKENS = 4             # shorter sentences ("Yes.") are not takes of anything
FILLERS = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "hmm", "mm", "mmm", "hm"}
# the speaker means to say it twice (checked before the director cues: "let me say that again" is not a crew call)
DELIBERATE = re.compile(r"\b(let me (?:say|put) (?:that|it|this) again|i'?ll say (?:that|it) again|(?:i'?ll )?(?:say|repeat) (?:that|it|this) one more time|"
                        r"let me repeat (?:that|this|it)?|i'?ll repeat (?:that|this|it)|say it again)\b", re.I)
# the crew or the speaker calling for another go: a short sentence of these words, or "let me do that again"
CUE = re.compile(r"^\W*(?:again|one more(?: time)?|cut|from the top|take (?:\w+)|let me (?:do|try|start) (?:that|it|this)? ?again|let'?s (?:do|try) (?:that|it)? ?again|"
                 r"sorry|wait|hold on|start over|(?:can|could) (?:we|i) (?:do|go) (?:that|it)? ?again)\W*$|\blet me (?:do|try|start) (?:that|it|this)? ?again\b", re.I)
WEIGHTS = {"complete": 0.40, "clean": 0.25, "sound": 0.15, "picture": 0.10, "recent": 0.10}      # (proposal: tuned in the evaluation, WP15)
SOUND_SCORE = {"good": 1.0, "ok": 0.7, "weak": 0.4, "bad": 0.0}


def tokens(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", str(text).lower().replace("’", "'")).split()


def similarity(a: list[str], b: list[str]) -> float:
    """Token-set similarity (Dice) of two sentences: word order and repeats of a word do not matter, so a restart with a filler still matches."""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return 2.0 * len(sa & sb) / (len(sa) + len(sb))


def _is_cue(text: str) -> bool:
    t = " ".join(str(text).split())
    return bool(t) and not DELIBERATE.search(t) and len(t.split()) <= 9 and bool(CUE.search(t))


def _fillers_in(words: list[dict], a: float, b: float) -> int:
    return sum(1 for w in words if a - 0.01 <= float(w["start"]) <= b + 0.01 and re.sub(r"[^a-z]", "", str(w.get("text", "")).lower()) in FILLERS)


def _take(s: M.Sentence, words: list[dict], order: int, grades: dict | None, facts: dict | None, restart: bool) -> dict:
    blur = True
    if isinstance(facts, dict):
        per = (facts.get("blur_ok_by_sentence") or {}).get(s.idx)
        blur = bool(per) if per is not None else bool((facts.get("quality") or {}).get("blur_floor_ok", facts.get("blur_ok", True)))
    return {"a": s.idx, "b": s.idx, "start": float(s.start), "end": float(s.end), "words": len(tokens(s.text)), "complete": bool(M.END.search(s.text)),
            "fillers": _fillers_in(words, s.start, s.end), "restart": bool(restart), "sound": str((grades or {}).get(s.idx) or "good"),
            "blur_ok": blur, "order": order}


def groups(sents: list, words: list, facts: dict | None = None, grades: dict | None = None) -> list[dict]:
    """-> [{"id": "t3", "line": "the opening about the farm", "takes": [{"a", "b", "complete", "fillers", "restart", "sound", "blur_ok", "order", ...}], "deliberate": False}]
    The groups with two or more takes, in order of their first take. `grades` maps a sentence number to its sound grade (good / weak / bad)."""
    ss = sorted(sents, key=lambda s: s.start)
    cues = {s.idx for s in ss if _is_cue(s.text)}
    toks = {s.idx: tokens(s.text) for s in ss}
    raw: list[dict] = []
    cue_seen = False
    last_cue_at = -1e9
    for s in ss:
        if s.idx in cues:
            cue_seen, last_cue_at = True, s.end
            continue
        t = toks[s.idx]
        if len(t) < MIN_TOKENS:
            continue
        home, restart_of = None, None
        for g in reversed(raw):
            prev = g["_last"]
            window = WINDOW_S * (2.0 if last_cue_at > prev.end else 1.0)
            if s.start - prev.end > window:
                continue
            if similarity(t, g["_tok"]) >= SIM:
                home = g
                break
            if len(toks[prev.idx]) >= RESTART_WORDS and t[:RESTART_WORDS] == toks[prev.idx][:RESTART_WORDS] and s.start - prev.end <= RESTART_S:
                home, restart_of = g, prev
                break
        if home is None:
            raw.append({"_tok": t, "_last": s, "sents": [s], "restarts": set(), "preceded": False})
            continue
        home["sents"].append(s)
        if restart_of is not None:
            home["restarts"].add(restart_of.idx)
        before = next((x for x in reversed(ss) if x.end <= s.start + 1e-6 and x.idx != s.idx), None)
        if DELIBERATE.search(s.text) or (before is not None and DELIBERATE.search(before.text) and s.start - before.end < 6.0):
            home["preceded"] = True
        home["_last"] = s
    out: list[dict] = []
    for g in raw:
        if len(g["sents"]) < 2:
            continue
        takes = [_take(s, words, n, grades, facts, s.idx in g["restarts"]) for n, s in enumerate(g["sents"])]
        out.append({"id": "", "line": " ".join(g["sents"][-1].text.split()[:8]), "takes": takes, "deliberate": bool(g["preceded"])})
    out += _refrains(ss, toks, words, grades, facts, out)
    out.sort(key=lambda g: g["takes"][0]["start"])
    for n, g in enumerate(out):
        g["id"] = f"t{n}"
    return out


def _refrains(ss: list, toks: dict, words: list, grades: dict | None, facts: dict | None, have: list[dict]) -> list[dict]:
    """A line that is said again and again through the talk, each time long after the last: deliberate (a refrain), never reduced."""
    used = {t["a"] for g in have for t in g["takes"]}
    found: list[dict] = []
    seen: set[int] = set()
    for i, s in enumerate(ss):
        if s.idx in seen or len(toks[s.idx]) < MIN_TOKENS or s.idx in used:
            continue
        same = [s] + [x for x in ss[i + 1:] if x.idx not in used and len(toks[x.idx]) >= MIN_TOKENS and similarity(toks[s.idx], toks[x.idx]) >= SIM]
        apart = [same[0]]
        for x in same[1:]:
            if x.start - apart[-1].end > REFRAIN_GAP_S:
                apart.append(x)
        if len(apart) >= REFRAIN_N:
            seen.update(x.idx for x in apart)
            found.append({"id": "", "line": " ".join(apart[-1].text.split()[:8]), "deliberate": True,
                          "takes": [_take(x, words, n, grades, facts, False) for n, x in enumerate(apart)]})
    return found


def score(take: dict, group: dict) -> float:
    """How good one take is: complete 40 %, few fillers and no restart 25 %, recorded well 15 %, picture 10 %, later 10 % (a prior, not a rule)."""
    n = len(group["takes"])
    longest = max(1, max(int(t.get("words") or 0) for t in group["takes"]))
    complete = 0.5 * (1.0 if take.get("complete") else 0.0) + 0.5 * (int(take.get("words") or 0) / longest)
    clean = max(0.0, 1.0 - (int(take.get("fillers") or 0) + 2 * (1 if take.get("restart") else 0)) / 4.0)
    sound = SOUND_SCORE.get(str(take.get("sound") or "good"), 0.7)
    picture = 1.0 if take.get("blur_ok", True) else 0.0
    recent = (int(take.get("order") or 0) / (n - 1)) if n > 1 else 1.0
    return (WEIGHTS["complete"] * complete + WEIGHTS["clean"] * clean + WEIGHTS["sound"] * sound + WEIGHTS["picture"] * picture + WEIGHTS["recent"] * recent)


def best(group: dict, policy: str = "best") -> int:
    """-> the index (into group["takes"]) of the take to keep. policy "last" / "first" override the score. A deliberate group has nothing to choose:
    the caller keeps every take of it (this still answers, with the last)."""
    takes = group["takes"]
    if not takes:
        raise ValueError("a take group with no takes")
    if policy == "first":
        return 0
    if policy == "last":
        return len(takes) - 1
    scores = [score(t, group) for t in takes]
    top = max(scores)
    return max(i for i, sc in enumerate(scores) if sc >= top - 1e-9)           # (a tie goes to the later take)


def alternates(grps: list[dict], chosen: dict[str, int]) -> list[dict]:
    """-> [{"group", "a", "b"}] for every take that was not chosen (kept in plan["alt_takes"] for the Final Cut export and the review page).
    `chosen` maps a group id to the index of its kept take; a group not in it (deliberate, or not used) lists nothing."""
    out = []
    for g in grps:
        if g["id"] not in chosen:
            continue
        for n, t in enumerate(g["takes"]):
            if n != chosen[g["id"]]:
                out.append({"group": g["id"], "a": t["a"], "b": t["b"]})
    return out
