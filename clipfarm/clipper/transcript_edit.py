"""Edit a long video from its transcript (Descript-style).

The transcript panel on a long-form project shows every word the guest said. Words that are in the video are marked; selecting some and
choosing "Cut from the video" removes them, selecting words that are not in the video and choosing "Add to the video" puts them in.

How an edit is kept. The plan (localproj `plan.json`) already says which sentences play (`items`, sentence ranges `a..b`). A transcript edit
adds two small things to it, both in seconds of the recording (so they do not depend on word numbering):

  * `plan["te_cuts"]`  - [[start, end], ...]: stretches of words taken out by hand. `LocalRunner.intervals_of` subtracts them from what an item plays
                         (kept apart from the neighbouring words the same way `word_limits` keeps every cut: never into the word before or after).
  * an item's `t0` / `t1` - a talk item that starts or ends inside a sentence (words added by hand that were not part of any item).

Edits are staged first (`te_staged.json` in the project's state folder): the page shows what would change ("Cut 12 s, 3 sentences") with
an Undo, and one button renders it. If the video is still being made, the same button queues the edit; the runner applies it when the
build is done. Applying keeps the placed B-roll where it still fits (`te_carry` in the render cache) and drops only the shots whose
stretch was cut. The plan before the edit is kept (`te_prev.json`), so a rendered edit can be undone too (a new version, as always).

Everything here is pure (plan + words in, plan out) except the `Staging` helpers at the bottom, which read and write the project's state.
"""
from __future__ import annotations

import bisect
import json
import time

EDGE = 0.02                      # seconds of slack when asking "is this word inside that stretch" (a word counts by its middle)
BEFORE, AFTER = 0.04, 0.03       # how close to a cut word the kept side may reach (the same limits localproj.word_limits keeps)
STAGED, PREV = "te_staged.json", "te_prev.json"


# ------------------------------------------------------------------------------------------------------------- words and sentences
def mid(w: dict) -> float:
    return (float(w["start"]) + float(w["end"])) / 2.0


def sentence_of(sents, starts: list[float], w: dict) -> int:
    """Index (the sentence's own `idx`) of the sentence a word belongs to."""
    k = max(0, bisect.bisect_right(starts, float(w["start"]) + 1e-6) - 1)
    return sents[k].idx


def snap_range(words: list[dict], t0: float, t1: float) -> tuple[int, int] | None:
    """Word indexes (first, last) of every word that overlaps the seconds t0..t1, so a cut never lands inside a word. None if no word does."""
    if not words or t1 < t0:
        return None
    hit = [k for k, w in enumerate(words) if float(w["end"]) > t0 + 1e-6 and float(w["start"]) < t1 - 1e-6]
    return (hit[0], hit[-1]) if hit else None


def runs_of(idx: list[int]) -> list[tuple[int, int]]:
    """[3,4,5,9,10] -> [(3,5),(9,10)]"""
    out: list[list[int]] = []
    for k in sorted(set(idx)):
        if out and k == out[-1][1] + 1:
            out[-1][1] = k
        else:
            out.append([k, k])
    return [(a, b) for a, b in out]


# ------------------------------------------------------------------------------------------------------------- what is in the video
def item_bounds(it: dict, by: dict) -> tuple[float, float] | None:
    """The seconds a talk item covers: its own t0/t1 when it has them, else its sentences."""
    if it.get("kind") != "talk" or it.get("a") not in by or it.get("b") not in by:
        return None
    return float(it.get("t0", by[it["a"]].start)), float(it.get("t1", by[it["b"]].end))


def cut_spans(plan: dict, ref: list[dict]) -> list[tuple[float, float]]:
    """The stretches of sound taken out by hand, as seconds: each held clear of the words either side (`ref` = EVERY word, sorted)."""
    out = []
    starts = [float(w["start"]) for w in ref]
    for s, e in plan.get("te_cuts") or []:
        s, e = float(s), float(e)
        i = bisect.bisect_left(starts, s - 1e-6) - 1                    # the word said just before
        j = bisect.bisect_right(starts, e - 0.001)                      # the word said just after
        lo = s - BEFORE
        hi = e + AFTER
        if i >= 0 and ref:
            lo = max(lo, float(ref[i]["end"]) + 0.01)
        if j < len(ref):
            hi = min(hi, float(ref[j]["start"]) - 0.005)
        lo, hi = min(lo, s), max(hi, e)                                  # (never less than the words themselves)
        out.append((lo, hi))
    return out


def subtract(a: float, b: float, spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """a..b without the spans."""
    parts = [(a, b)]
    for s, e in sorted(spans):
        nxt = []
        for x, y in parts:
            if e <= x or s >= y:
                nxt.append((x, y))
                continue
            if s > x:
                nxt.append((x, s))
            if e < y:
                nxt.append((e, y))
        parts = nxt
    return parts


def kept_flags(plan: dict, sents, words: list[dict]) -> list[bool]:
    """For every guest word: is it in the video the plan describes (an item plays it and no hand cut took it out)?"""
    by = {s.idx: s for s in sents}
    spans = [b for b in (item_bounds(it, by) for it in plan.get("items") or []) if b]
    cuts = [(float(s), float(e)) for s, e in plan.get("te_cuts") or []]
    out = []
    for w in words:
        m = mid(w)
        out.append(any(a - EDGE <= m <= b + EDGE for a, b in spans) and not any(s - 1e-6 <= m <= e + 1e-6 for s, e in cuts))
    return out


def item_extent(it: dict, by: dict, words: list[dict]) -> tuple[int, int] | None:
    """Word indexes (first, last) an item plays, ignoring hand cuts."""
    bnd = item_bounds(it, by)
    if not bnd:
        return None
    hit = [k for k, w in enumerate(words) if bnd[0] - EDGE <= mid(w) <= bnd[1] + EDGE]
    return (hit[0], hit[-1]) if hit else None


# ------------------------------------------------------------------------------------------------------------- the two edits
def _clone(plan: dict) -> dict:
    return json.loads(json.dumps(plan))


def _normalize(cuts: list[list[float]]) -> list[list[float]]:
    cuts = sorted([round(float(a), 3), round(float(b), 3)] for a, b in cuts if float(b) > float(a))
    out: list[list[float]] = []
    for a, b in cuts:
        if out and a <= out[-1][1] + 0.001:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _words_in(cuts: list[list[float]], words: list[dict]) -> set[int]:
    return {k for k, w in enumerate(words) if any(float(s) - 1e-6 <= mid(w) <= float(e) + 1e-6 for s, e in cuts)}


def cut(plan: dict, sents, words: list[dict], i: int, j: int) -> dict:
    """Words i..j (indexes into the guest words) are taken out of the video. Words that are not in it already are left alone. An item with
    nothing left is dropped."""
    plan = _clone(plan)
    flags = kept_flags(plan, sents, words)
    hit = [k for k in range(max(0, i), min(len(words) - 1, j) + 1) if flags[k]]
    if not hit:
        return plan
    cuts = list(plan.get("te_cuts") or [])
    for a, b in runs_of(hit):
        cuts.append([float(words[a]["start"]), float(words[b]["end"])])
    plan["te_cuts"] = _normalize(cuts)
    after = kept_flags(plan, sents, words)
    by = {s.idx: s for s in sents}
    keep = []
    for it in plan.get("items") or []:
        ext = item_extent(it, by, words) if it.get("kind") == "talk" else None
        if it.get("kind") == "talk" and (ext is None or not any(after[k] for k in range(ext[0], ext[1] + 1))):
            continue                                                        # (everything it played was cut)
        keep.append(it)
    plan["items"] = keep
    return plan


def add(plan: dict, sents, words: list[dict], i: int, j: int) -> dict:
    """Words i..j that are not in the video are put in. A word taken out by hand comes back; words no item plays join the item next to
    them (no needless join), or become an item of their own, placed by time among the others."""
    plan = _clone(plan)
    flags = kept_flags(plan, sents, words)
    want = [k for k in range(max(0, i), min(len(words) - 1, j) + 1) if not flags[k]]
    if not want:
        return plan
    # 1. words cut by hand come back
    cut_idx = _words_in(plan.get("te_cuts") or [], words)
    left = cut_idx - set(want)
    plan["te_cuts"] = _normalize([[float(words[a]["start"]), float(words[b]["end"])] for a, b in runs_of(sorted(left))]) if left else []
    by = {s.idx: s for s in sents}
    starts = [s.start for s in sents]
    after = kept_flags(plan, sents, words)
    # 2. words that no item plays
    rest = [k for k in want if not after[k]]
    for a, b in runs_of(rest):
        items = plan["items"]
        wa, wb = words[a], words[b]
        sa, sb = sentence_of(sents, starts, wa), sentence_of(sents, starts, wb)
        exts = [item_extent(it, by, words) if it.get("kind") == "talk" else None for it in items]
        before_k = next((n for n, e in enumerate(exts) if e and e[1] == a - 1), None)
        after_k = next((n for n, e in enumerate(exts) if e and e[0] == b + 1), None)
        if before_k is not None:
            it = items[before_k]
            it["t0"] = it.get("t0", by[it["a"]].start)
            it["t1"] = float(wb["end"])
            it["b"] = max(it["b"], sb)
            if after_k is not None and after_k == before_k + 1:             # the gap closed completely: one item
                nxt = items[after_k]
                it["t1"] = float(nxt.get("t1", by[nxt["b"]].end))
                it["b"] = max(it["b"], nxt["b"])
                del items[after_k]
        elif after_k is not None:
            it = items[after_k]
            it["t1"] = it.get("t1", by[it["b"]].end)
            it["t0"] = float(wa["start"])
            it["a"] = min(it["a"], sa)
        else:
            new = {"kind": "talk", "a": sa, "b": sb, "t0": float(wa["start"]), "t1": float(wb["end"])}
            pos = len(items)
            for n, e in enumerate(exts):
                if e and e[0] > b:
                    pos = n
                    break
            items.insert(pos, new)
    return plan


def apply_ops(plan: dict, sents, words: list[dict], ops: list[dict]) -> dict:
    """The staged edits, in order. Each op names its words by seconds ({"op", "t0", "t1"}) and is snapped to whole words here."""
    for op in ops:
        rng = snap_range(words, float(op["t0"]), float(op["t1"]))
        if not rng:
            continue
        plan = cut(plan, sents, words, *rng) if op["op"] == "cut" else add(plan, sents, words, *rng)
    return plan


# ------------------------------------------------------------------------------------------------------------- what would change
def diff(before: list[bool], after: list[bool], sents, words: list[dict]) -> dict:
    """{cut_s, added_s, cut_sentences, added_sentences, cut_words, added_words} between two kept-flag lists."""
    starts = [s.start for s in sents]
    out = {"cut_s": 0.0, "added_s": 0.0, "cut_words": 0, "added_words": 0}
    seen = {"cut": set(), "add": set()}
    for kind, want in (("cut", (True, False)), ("add", (False, True))):
        idx = [k for k, (x, y) in enumerate(zip(before, after)) if (x, y) == want]
        for a, b in runs_of(idx):
            out["cut_s" if kind == "cut" else "added_s"] += float(words[b]["end"]) - float(words[a]["start"])
        out["cut_words" if kind == "cut" else "added_words"] = len(idx)
        seen[kind] = {sentence_of(sents, starts, words[k]) for k in idx}
    out["cut_s"], out["added_s"] = round(out["cut_s"], 1), round(out["added_s"], 1)
    out["cut_sentences"], out["added_sentences"] = len(seen["cut"]), len(seen["add"])
    return out


def sentence_line(d: dict) -> str:
    """'Cut 12 s (3 sentences)' / 'Add 4 s' / both."""
    bits = []
    if d["cut_words"]:
        bits.append(f"Cut {d['cut_s']:g} s" + (f" ({d['cut_sentences']} sentence{'s' if d['cut_sentences'] != 1 else ''})" if d["cut_sentences"] else ""))
    if d["added_words"]:
        bits.append(f"Add {d['added_s']:g} s" + (f" ({d['added_sentences']} sentence{'s' if d['added_sentences'] != 1 else ''})" if d["added_sentences"] else ""))
    return " · ".join(bits) or "No change"


def paragraph_breaks(words: list[dict], sents, gap: float = 1.5, per: int = 4) -> list[int]:
    """Word indexes where a new paragraph starts: after a long pause, or about every `per` sentences."""
    out, n_s, prev_idx = [], 0, None
    starts = [s.start for s in sents]
    for k, w in enumerate(words):
        sidx = sentence_of(sents, starts, w) if sents else 0
        new_sentence = prev_idx is not None and sidx != prev_idx
        if new_sentence:
            n_s += 1
        if k and ((float(w["start"]) - float(words[k - 1]["end"]) >= gap) or (new_sentence and n_s >= per)):
            out.append(k)
            n_s = 0
        prev_idx = sidx
    return out


# ------------------------------------------------------------------------------------------------------------- B-roll that stays
def _source_time(iv: list, t: float) -> float | None:
    """A time in the finished cut -> where it is in the recording (None past the end)."""
    acc = 0.0
    for a, b in iv:
        if acc - 1e-6 <= t <= acc + (b - a) + 1e-6:
            return a + (t - acc)
        acc += b - a
    return None


def _out_time(iv: list, t: float) -> float | None:
    acc = 0.0
    for a, b in iv:
        if a - 1e-6 <= t <= b + 1e-6:
            return acc + min(max(t, a), b) - a
        acc += b - a
    return None


def carry_cutaways(carry: dict, new_iv: list) -> list[dict]:
    """The B-roll placed in the last render, moved to the new cut: a shot stays when the stretch it covered is still there (the whole of it,
    played in one piece); a shot over a stretch that was cut is dropped. `carry` = {"iv": the last cut's intervals, "cutaways": its shots}."""
    old_iv = [tuple(x) for x in carry.get("iv") or []]
    out = []
    for c in carry.get("cutaways") or []:
        try:
            at, ln = float(c["at"]), float(c["length"])
        except (KeyError, TypeError, ValueError):
            continue
        s0, s1 = _source_time(old_iv, at + 0.02), _source_time(old_iv, at + ln - 0.02)
        if s0 is None or s1 is None:
            continue
        n0, n1 = _out_time(new_iv, s0), _out_time(new_iv, s1)
        if n0 is None or n1 is None or abs((n1 - n0) - (ln - 0.04)) > 0.35:
            continue                                                         # (cut through, or no longer in one piece)
        out.append({**c, "at": round(max(0.0, n0 - 0.02), 2)})
    out.sort(key=lambda c: c["at"])
    return out


# ------------------------------------------------------------------------------------------------------------- the project's state
def load_words(st) -> tuple[list[dict], list[dict]]:
    """(guest words, every word) as the plan was made from them; ([], []) while there is no transcript yet."""
    from . import speakers as SP
    from .parakeet import split_glued_numbers
    pw = st.get("plan_words.json") or {}
    if not pw.get("words"):
        return [], []
    allw = split_glued_numbers(pw["words"])
    return SP.drop_spans(allw, [tuple(x) for x in pw.get("spans") or []]), allw


def view(runner, pid: str) -> dict:
    """What the transcript panel needs:
       state  "none" (no transcript yet), "words" (a transcript but no plan yet), "plan" (words marked, editing on)
       words  [[text, start, end], ...]; flags one character per word: 0 not in the video, 1 in it, 2 staged to be cut, 3 staged to be added
       breaks paragraph starts (word indexes); staged {ops, summary}; busy; undo (an applied edit that can be undone)."""
    from .localproj import State, home_of, seg
    proj = runner.projects.get(pid)
    st = State(home_of(proj))
    words, _ = load_words(st)
    busy = runner.running == pid
    out = {"state": "none", "words": [], "flags": "", "breaks": [], "staged": {"ops": [], "summary": "", "queued": False}, "busy": busy,
           "undo": bool(st.get(PREV)), "version": (st.get("plan.json") or {}).get("version")}
    if not words:
        return out
    sents = seg(words)
    plan = st.get("plan.json") or {}
    staged = st.get(STAGED) or {}
    out["words"] = [[w["text"], round(float(w["start"]), 2), round(float(w["end"]), 2)] for w in words]
    out["breaks"] = paragraph_breaks(words, sents)
    if not plan.get("items") or plan.get("awaiting_approval"):                # (a script map waiting for approval is reviewed there, not here)
        out.update(state="words", flags="0" * len(words))
        out["staged"]["queued"] = bool(staged.get("queued"))
        return out
    before = kept_flags(plan, sents, words)
    ops = list(staged.get("ops") or [])
    after = kept_flags(apply_ops(plan, sents, words, ops), sents, words) if ops else before
    out["flags"] = "".join(("1" if b else "0") if b == a else ("2" if b else "3") for b, a in zip(before, after))
    d = diff(before, after, sents, words)
    out["staged"] = {"ops": ops, "summary": sentence_line(d) if ops else "", "diff": d, "queued": bool(staged.get("queued"))}
    out["state"] = "plan"
    return out


def stage(runner, pid: str, op: str, first: int, last: int) -> dict:
    """Add one edit (word indexes first..last of the page's list) to what is staged. Raises ValueError for a request that changes nothing."""
    from .localproj import State, home_of, seg
    if op not in ("cut", "add"):
        raise ValueError("op is cut or add")
    proj = runner.projects.get(pid)
    st = State(home_of(proj))
    words, _ = load_words(st)
    plan = st.get("plan.json") or {}
    if not words or not plan.get("items") or plan.get("awaiting_approval"):
        raise ValueError("the plan isn't ready yet: the words in the video show once it is")
    if not (0 <= int(first) <= int(last) < len(words)):
        raise ValueError("those words aren't in this transcript")
    sents = seg(words)
    staged = st.get(STAGED) or {}
    ops = list(staged.get("ops") or [])
    new = {"op": op, "t0": round(float(words[int(first)]["start"]), 3), "t1": round(float(words[int(last)]["end"]), 3)}
    now = kept_flags(apply_ops(plan, sents, words, ops), sents, words)
    nxt = kept_flags(apply_ops(plan, sents, words, ops + [new]), sents, words)
    if now == nxt:
        raise ValueError("that is " + ("already out of the video" if op == "cut" else "already in the video"))
    st.put(STAGED, {**staged, "ops": ops + [new], "at": int(time.time())})
    return view(runner, pid)


def undo_staged(runner, pid: str) -> dict:
    """Take back the last staged edit (all of them with all_=True through `clear`)."""
    from .localproj import State, home_of
    st = State(home_of(runner.projects.get(pid)))
    staged = st.get(STAGED) or {}
    ops = list(staged.get("ops") or [])[:-1]
    if ops:
        st.put(STAGED, {**staged, "ops": ops})
    else:
        st.put(STAGED, {})
    return view(runner, pid)


def clear(runner, pid: str) -> dict:
    from .localproj import State, home_of
    State(home_of(runner.projects.get(pid))).put(STAGED, {})
    return view(runner, pid)


def request_apply(runner, pid: str) -> dict:
    """The button: render the staged edits. Idle -> starts at once; busy -> marked to run when the build is done."""
    from .localproj import State, home_of
    st = State(home_of(runner.projects.get(pid)))
    staged = st.get(STAGED) or {}
    if not staged.get("ops"):
        raise ValueError("nothing is staged")
    if (st.get("plan.json") or {}).get("awaiting_approval"):
        raise ValueError("approve the script map first")
    if runner.running:
        st.put(STAGED, {**staged, "queued": True})
        return {**view(runner, pid), "queued": True}
    runner.start(pid, "transcript_edit")
    return {**view(runner, pid), "started": True}


def request_undo(runner, pid: str) -> dict:
    """Go back to the cut before the last applied edit (rendered again: a new version)."""
    from .localproj import State, home_of
    st = State(home_of(runner.projects.get(pid)))
    if not st.get(PREV):
        raise ValueError("there is no edit to undo")
    if runner.running:
        raise ValueError("a video is being made right now - undo it when it's done")
    runner.start(pid, "transcript_undo")
    return {"started": True}


def run_apply(runner, pid: str) -> None:
    """Runner action `transcript_edit`: the staged edits go into the plan, which renders again. Everything else (song, B-roll that still fits,
    graphics) is as it was."""
    from .localproj import seg
    proj, folder, st, ai, engine = runner._setup(pid)
    staged = st.get(STAGED) or {}
    ops = list(staged.get("ops") or [])
    if not ops:
        return
    files, group, all_words, words = runner.saved_inputs(folder, st, ai, runner.store.get(proj["job"])["options"].get("transcriber", "parakeet"))
    plan = st.get("plan.json")
    if not plan or not plan.get("items"):
        raise RuntimeError("there is no saved edit to change yet - make the video first")
    sents = seg(words)
    new = apply_ops(plan, sents, words, ops)
    d = diff(kept_flags(plan, sents, words), kept_flags(new, sents, words), sents, words)
    st.snapshot("plan.json", plan, "before-transcript-edit")
    st.put(PREV, {k: v for k, v in plan.items() if k not in ("pending_render", "pending_note")})
    _carry_broll(runner, proj, st, plan)
    note = "edited in the transcript: " + sentence_line(d).lower()
    new["pending_render"], new["pending_note"] = True, note
    st.put("plan.json", new)
    st.put(STAGED, {})
    scenes = runner.describe(files, words, ai, st, lambda f, m: None)
    runner.render(pid, new, files, words, scenes, ai, engine, st, folder, note=note, all_words=all_words)


def run_undo(runner, pid: str) -> None:
    """Runner action `transcript_undo`: the plan from before the last applied edit, rendered again."""
    proj, folder, st, ai, engine = runner._setup(pid)
    prev = st.get(PREV)
    if not prev:
        raise RuntimeError("there is no edit to undo")
    files, group, all_words, words = runner.saved_inputs(folder, st, ai, runner.store.get(proj["job"])["options"].get("transcriber", "parakeet"))
    cur = st.get("plan.json") or {}
    st.snapshot("plan.json", cur, "before-undoing-transcript-edit")
    _carry_broll(runner, proj, st, cur)
    note = "transcript edit undone"
    st.put("plan.json", {**prev, "version": cur.get("version", prev.get("version", 1)), "history": cur.get("history", prev.get("history", [])),
                         "pending_render": True, "pending_note": note})
    st.put(PREV, {})
    scenes = runner.describe(files, words, ai, st, lambda f, m: None)
    runner.render(pid, st.get("plan.json"), files, words, scenes, ai, engine, st, folder, note=note, all_words=all_words)


def _carry_broll(runner, proj: dict, st, plan: dict) -> None:
    """Remember the last render's B-roll and the cut it was placed over, for `render` to move onto the new cut."""
    try:
        iv = json.loads((runner.store.dir(proj["job"]) / "plan.json").read_text()).get("intervals") or []
    except (OSError, ValueError):
        iv = []
    rc = st.get("render_cache.json") or {}
    placed = (rc.get("final") or {}).get("cutaways")
    if iv and placed:
        st.put("render_cache.json", {**rc, "te_carry": {"iv": iv, "cutaways": placed}})


def after_run(runner, pid: str, kind: str, ok: bool) -> None:
    """Called when a run for pid has finished: an edit that was queued while the video was being made starts now (only after a good run)."""
    if not ok or kind in ("transcript_edit", "transcript_undo"):
        return
    from .localproj import State, home_of
    try:
        st = State(home_of(runner.projects.get(pid)))
        staged = st.get(STAGED) or {}
        if staged.get("queued") and staged.get("ops") and not (st.get("plan.json") or {}).get("awaiting_approval"):
            st.put(STAGED, {**staged, "queued": False})
            runner.start(pid, "transcript_edit")
    except Exception:  # noqa: BLE001 - a queued edit that cannot start stays staged; the page offers the button again
        pass
