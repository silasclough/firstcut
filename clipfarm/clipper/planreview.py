"""The plan of a long video, before it renders and after, in plain words for the page (WP16 of docs/plans/editor-improvement-plan.md).

    build      (pure) a plan + the footage + the sentences -> what the person reads: one sentence about what will be made, what the editor understood from the
               notes (each instruction with the words it came from and `will do` / `done` / `could not` / `which one?`), the questions it needs answered with
               simple choices, the cut as rows (when it plays, which file, the first words, why it is there), what the footage is, the style in five lines, and
               what to look at. No internal names leave this module: the codes stay in the JSON files.
    review     build() for a saved video.
    answer     the person picked one of the candidates for an instruction that could mean two places: it is settled and the plan is checked again.
    leave_out  take one passage out of the plan (or put it back); a change that would break something that was asked for is refused, in words.
    use_take   swap a passage for another take of the same line.
    checked    after the render: what was checked, what passed, what was dropped or skipped and where.

Everything here works on the saved files of the video; no AI is called.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from . import constraints as CN
from . import styles_profile as SPF

PROFILE_WORDS = {"short_talk": "short talk", "reel": "reel", "interview_clip": "interview clip", "longform_talk": "talk", "seminar": "seminar",
                 "event_recap": "event recap", "promo": "promo", "ad": "ad"}
ROLE_WORDS = {"a_roll": "people talking", "broll": "B-roll", "voiceover_broll": "B-roll with talking", "multicam_angle": "other camera angles", "screen": "screen recordings",
              "photo": "photos", "music": "songs", "audio": "sound recordings"}
KIND_WORDS = {"include": "Keep this in", "exclude_words": "Leave this out", "exclude_broll_window": "No B-roll here", "exclude_files": "Don't use this footage",
              "order": "Put this in order", "duration": "Length", "graphic": "Show a graphic", "camera": "Camera", "style": "Look and feel", "take": "Which take",
              "caption_text": "Caption text", "chapter": "Chapter"}
STATE_WORDS = {"met": "done", "grounded": "will do", "unmet": "could not", "ambiguous": "which one?", "ungrounded": "could not"}
PACE_WORDS = {"slow": "Slow", "natural": "Natural", "tight": "Tight", "fast": "Fast"}
CAPTION_WORDS = {"none": "None", "phrase": "A few words at a time", "karaoke": "Word by word"}
GRAPHICS_WORDS = {"none": "None", "title": "Title only", "informational": "When it helps", "all": "Everywhere it fits"}
MUSIC_WORDS = {"none": "None", "under_broll": "Under the B-roll", "throughout": "All the way through"}

# words that belong in the JSON files and must never reach a person (a test reads every text field for them)
INTERNAL = re.compile(r"\b(ambiguous|ungrounded|unmet|grounded|abstain\w*|constraint_id|style_obj|segment_id|HSR|SSR|CSL|WP\d+|stub|json|sid|fstart|fend)\b", re.I)


def plain(text: Any, limit: int = 220, quoted: bool = False) -> str:
    """A line from the code made safe for the page: one line, no internal words, a sensible length.
    quoted=True is the person's own words (their notes, the transcript): nothing is taken out of them."""
    s = " ".join(str(text or "").split())
    if not quoted:
        s = INTERNAL.sub("", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" -:;,")
    return s[:limit]


def sentence(text: Any, limit: int = 220) -> str:
    s = plain(text, limit)
    return s[:1].upper() + s[1:]


def clock(t: Any) -> str:
    try:
        t = int(max(0, round(float(t))))
    except (TypeError, ValueError):
        return "0:00"
    return f"{t // 60}:{t % 60:02d}"


def _words_of(sents, a: Any, b: Any, n: int = 16) -> str:
    by = {s.idx: s for s in sents or []}
    try:
        a, b = int(a), int(b)
    except (TypeError, ValueError):
        return ""
    text = " ".join(by[i].text for i in range(a, min(b, a + 40) + 1) if i in by)
    ws = text.split()
    return " ".join(ws[:n]) + (" …" if len(ws) > n else "")


def _role_of(f: dict) -> str:
    facts = f.get("facts") if isinstance(f.get("facts"), dict) else {}
    role = facts.get("role")
    if role:
        return str(role)
    return "broll" if f.get("broll") else "a_roll"


ONE_WORDS = {"people talking": "person talking", "B-roll": "B-roll shot", "B-roll with talking": "B-roll shot with talking", "other camera angles": "other camera angle",
             "screen recordings": "screen recording", "photos": "photo", "songs": "song", "audio": "sound recording", "sound recordings": "sound recording"}


def _count_label(n: int, what: str) -> str:
    """'1 person talking', '3 people talking', '1 B-roll shot', '4 B-roll shots'."""
    if n == 1:
        return f"1 {ONE_WORDS.get(what, what)}"
    return f"{n} {what + ' shots' if what == 'B-roll' else what}"


def footage_summary(files: list[dict]) -> dict:
    counts: dict[str, int] = {}
    total = 0.0
    for f in files or []:
        if not isinstance(f, dict) or f.get("library") or f.get("stock"):
            continue
        w = ROLE_WORDS.get(_role_of(f), "other")
        counts[w] = counts.get(w, 0) + 1
        try:
            total += float(f.get("dur") or 0)
        except (TypeError, ValueError):
            pass
    rows = [{"what": k, "n": v} for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
    n = sum(counts.values())
    return {"files": n, "minutes": round(total / 60.0, 1), "rows": rows,
            "line": (f"{n} file{'s' if n != 1 else ''}, {round(total / 60.0)} min: " + ", ".join(_count_label(r["n"], r["what"]) for r in rows)) if n else "No footage was found."}


def _choice(c: dict, n: int, cand: dict) -> dict:
    if cand.get("file"):
        return {"id": str(n), "label": str(cand["file"])[:90], "at": ""}
    return {"id": str(n), "label": "“" + plain(cand.get("text"), 110, quoted=True) + "”", "at": clock(cand.get("start_s"))}


def understood(constraints: list[dict], problems: list[dict] | None = None) -> list[dict]:
    """The instructions the editor found in the notes, one row each."""
    hard_by = {}
    for p in problems or []:
        if isinstance(p, dict) and p.get("hard") and p.get("constraint_id"):
            hard_by.setdefault(p["constraint_id"], p.get("message"))
    out = []
    for c in constraints or []:
        if not isinstance(c, dict):
            continue
        st = c.get("status")
        state = STATE_WORDS.get(st, "will do")
        if st in ("grounded", "met") and c.get("id") in hard_by:
            state = "could not"
        row = {"id": c.get("id"), "kind": KIND_WORDS.get(c.get("type"), "Instruction"), "quote": plain(c.get("source"), 160, quoted=True), "must": bool(c.get("hard")), "state": state,
               "note": sentence(hard_by.get(c.get("id")) or c.get("note")), "choices": []}
        if st == "ambiguous":
            row["choices"] = [_choice(c, n, cd) for n, cd in enumerate((c.get("candidates") or [])[:5]) if isinstance(cd, dict)]
            if not row["note"]:
                row["note"] = "I found more than one place this could mean. Which one?"
        out.append(row)
    return out


def cut_rows(plan: dict, files: list[dict], sents) -> list[dict]:
    """The plan as rows in play order: when it plays, which file, the first words, and why it is there."""
    cons = {c.get("id"): c for c in plan.get("constraints") or [] if isinstance(c, dict)}
    segs = CN.segments(plan, sents, files)
    left = {str((e or {}).get("item", {}).get("id")) for e in plan.get("left_out_items") or [] if isinstance(e, dict)}
    alts: dict[str, list[dict]] = {}
    groups = {g.get("id"): g for g in plan.get("take_groups") or [] if isinstance(g, dict)}
    for al in plan.get("alt_takes") or []:
        g = groups.get(al.get("group")) or {}
        for sg in segs:
            if sg["kind"] == "talk" and sg["a"] is not None and any(t.get("a") == sg["a"] and t.get("b") == sg["b"] for t in g.get("takes") or []):
                alts.setdefault(sg["id"], []).append({"a": al.get("a"), "b": al.get("b"), "text": _words_of(sents, al.get("a"), al.get("b"), 12)})
    rows, at = [], 0.0
    for sg in segs:
        dur = max(0.0, float(sg["end"] - sg["start"])) if sg["start"] is not None and sg["end"] is not None else 0.0
        why = ""
        if sg["satisfies"]:
            q = [cons[i].get("source") for i in sg["satisfies"] if i in cons]
            why = "You asked for this: “" + plain(q[0], 90, quoted=True) + "”" if q else ""
        item = (plan.get("items") or [])[sg["i"]] if sg["i"] < len(plan.get("items") or []) else {}
        why = why or plain(item.get("why") or item.get("reason"), 120)
        f = CN._file_at(files, sg["start"]) if sg["start"] is not None else None
        row = {"id": sg["id"], "kind": "talk" if sg["kind"] == "talk" else "picture", "at": clock(at), "length_s": round(dur, 1),
               "file": str((f or {}).get("name") or sg.get("file") or "")[:80], "words": _words_of(sents, sg["a"], sg["b"]) if sg["kind"] == "talk" else "",
               "why": why, "required": bool(sg["protect"] or sg["satisfies"]), "other_takes": alts.get(sg["id"], [])[:3]}
        rows.append(row)
        at += dur
    out_rows = [dict(r) for r in rows]
    for e in sorted([e for e in plan.get("left_out_items") or [] if isinstance(e, dict)], key=lambda e: int(e.get("index") or 0)):   # what the person left out stays where it was, with a way back
        it = (e or {}).get("item") or {}
        if str(it.get("id")) in left and it.get("kind") == "talk":
            out_rows.insert(min(int(e.get("index") or 0), len(out_rows)), {"id": str(it.get("id")), "kind": "talk", "at": "", "length_s": 0.0, "file": "", "words": _words_of(sents, it.get("a"), it.get("b")),
                                                                              "why": "", "required": False, "other_takes": [], "left_out": True})
    return out_rows


def style_lines(plan: dict) -> list[str]:
    sty = plan.get("style_obj") or SPF.build(str(plan.get("style") or "story"))
    try:
        out = []
        for x in SPF.describe(sty).splitlines():
            if not x.strip():
                continue
            x = re.sub(r"^Kind: ([a-z_ ]+);", lambda m: "Kind of video: " + PROFILE_WORDS.get(m.group(1).strip().replace(" ", "_"), m.group(1)) + ";", x)
            x = {"Captions off.": "No captions."}.get(x, x)
            out.append(plain(x, 160))
        return out
    except Exception:  # noqa: BLE001
        return []


def options_of(plan: dict) -> dict:
    sty = plan.get("style_obj") or SPF.build(str(plan.get("style") or "story"))
    cap = (sty.get("captions") or {}).get("style") or "none"
    return {"profile": sty.get("profile"), "profiles": [{"id": k, "label": v} for k, v in PROFILE_WORDS.items()],
            "pace": (sty.get("pace") or {}).get("name") or "natural", "paces": [{"id": k, "label": v} for k, v in PACE_WORDS.items()],
            "captions": cap if cap in CAPTION_WORDS else "none", "captions_all": [{"id": k, "label": v} for k, v in CAPTION_WORDS.items()],
            "graphics": (sty.get("graphics") or {}).get("policy") or "informational", "graphics_all": [{"id": k, "label": v} for k, v in GRAPHICS_WORDS.items()],
            "music": (sty.get("music") or {}).get("policy") or "under_broll", "music_all": [{"id": k, "label": v} for k, v in MUSIC_WORDS.items()],
            "length_s": (sty.get("length") or {}).get("target_s")}


def _problem_lines(problems: list[dict]) -> list[dict]:
    out, seen = [], set()
    for p in problems or []:
        if not isinstance(p, dict):
            continue
        line = sentence(p.get("message"), 240)
        if line and line not in seen:
            seen.add(line)
            out.append({"line": line, "must": bool(p.get("hard"))})
    return out[:14]


def build(plan: dict, files: list[dict], sents, res: dict | None = None, status: str = "") -> dict:
    """What the person reads before the video is made."""
    plan = plan if isinstance(plan, dict) else {}
    totals = (res or {}).get("totals") or plan.get("totals") or {}
    problems = (res or {}).get("problems") if res is not None else plan.get("problems")
    sty = plan.get("style_obj") or SPF.build(str(plan.get("style") or "story"))
    rows = cut_rows(plan, files, sents)
    talk = [r for r in rows if r["kind"] == "talk" and not r.get("left_out")]
    und = understood(plan.get("constraints") or [], problems)
    questions = [u for u in und if u["state"] == "which one?" and u["must"]]
    probs = _problem_lines(problems)
    length = totals.get("length_s") or totals.get("plan_s")
    profile_word = PROFILE_WORDS.get(sty.get("profile"), "video")
    summary = (f"I will make a {clock(length)} {profile_word} video from {len(talk)} part{'s' if len(talk) != 1 else ''} of your footage."
               if length else f"I will make a {profile_word} video from {len(talk)} part{'s' if len(talk) != 1 else ''} of your footage.")
    pool = plan.get("pool") or []
    share = int(round(float((sty.get("broll") or {}).get("share") or 0) * 100))
    return {"status": status, "title": plain(plan.get("title"), 90), "summary": summary, "length": clock(length) if length else "", "ready": not questions,
            "awaiting": bool(plan.get("awaiting_approval")), "reason": plain((plan.get("approval") or {}).get("reason")),
            "understood": und, "questions": questions, "cut": rows,
            "broll": {"available": len(pool), "share": share, "line": (f"About {share}% of the video can be B-roll, picked to match the words when it is made. {len(pool)} shot"
                                                                         f"{'s' if len(pool) != 1 else ''} to choose from.") if pool else "No B-roll to choose from in this folder."},
            "footage": footage_summary(files), "style": style_lines(plan), "options": options_of(plan), "problems": probs}


def texts(obj: Any) -> list[str]:
    """Every text value in a review (the test that no internal word reaches the page reads these)."""
    out: list[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("id", "kind_id"):
                continue
            out += texts(v)
    elif isinstance(obj, list):
        for v in obj:
            out += texts(v)
    return out


# ------------------------------------------------------------------------------------------------------------------ saved videos
def _load(runner, pid: str):
    from . import planapi as PA
    proj, folder, st, ai, job, files, words, sents = PA._inputs(runner, pid)
    plan = st.get("plan.json")
    if not plan:
        raise ValueError("this video has no plan yet")
    return st, job, files, words, sents, PA.with_ids(plan)


def _check(plan: dict, files: list[dict], words: list[dict], sents) -> dict:
    return CN.validate_edited_plan(plan, files, words, plan.get("pool") or [], sents)


def review(runner, pid: str) -> dict:
    st, job, files, words, sents, plan = _load(runner, pid)
    res = _check(plan, files, words, sents)
    return build(plan, files, sents, res, str(job.get("status") or ""))


def _save(st, plan: dict, res: dict, why: str) -> None:
    old = st.get("plan.json")
    if old is not None:
        st.snapshot("plan.json", old, why)
    plan["problems"], plan["totals"] = res["problems"], res["totals"]
    st.put("plan.json", plan)


def _hard(res: dict) -> set:
    return {(p.get("code"), p.get("constraint_id")) for p in res.get("problems") or [] if p.get("hard")}


def _unique_ids(items: list[dict]) -> list[dict]:
    """A range that was split around a cut part keeps its id on the first piece; the others get a letter, so every part can still be told apart."""
    seen, out = set(), []
    for it in items:
        it = dict(it)
        base, n = str(it.get("id") or f"i{len(out)}"), 0
        cur = base
        while cur in seen:
            n += 1
            cur = f"{base}{chr(96 + n) if n < 27 else n}"
        it["id"] = cur
        seen.add(cur)
        out.append(it)
    return out


def _sentences_between(sents, t0: float, t1: float) -> tuple[int, int] | None:
    hit = [s.idx for s in sents or [] if s.start < t1 and s.end > t0]
    return (min(hit), max(hit)) if hit else None


def answer(runner, pid: str, cid: str, choice: Any) -> dict:
    """The person picked one of the places an instruction could mean. It becomes that place and the plan is checked again."""
    st, job, files, words, sents, plan = _load(runner, pid)
    cons = [dict(c) for c in plan.get("constraints") or [] if isinstance(c, dict)]
    c = next((x for x in cons if x.get("id") == cid), None)
    if c is None:
        raise ValueError("that instruction is not in this plan")
    cands = c.get("candidates") or []
    if isinstance(choice, str) and choice.strip().lstrip("+-").isdigit():           # (the page sends the radio's value, a string)
        choice = int(choice.strip())
    if isinstance(choice, bool) or not isinstance(choice, int) or not 0 <= choice < len(cands):
        raise ValueError("pick one of the choices that were shown")
    cand = cands[choice]
    c["anchor"] = dict(c.get("anchor") or {})
    if cand.get("words"):
        c["anchor"].update({"words": cand["words"], "spans": [cand["words"]], "text": cand.get("text", ""), "start_s": cand.get("start_s"), "end_s": cand.get("end_s"),
                            "times": [[cand.get("start_s"), cand.get("end_s")]]})
        if c.get("type") == "include":                                          # the chosen place must be in the cut: add it if no passage covers it
            t0, t1 = float(cand.get("start_s") or 0), float(cand.get("end_s") or 0)
            if not any(s["kind"] == "talk" and s["start"] is not None and s["start"] - 0.05 < t1 and s["end"] + 0.05 > t0 for s in CN.segments(plan, sents, files)):
                span = _sentences_between(sents, t0, t1)
                if span:
                    items = list(plan.get("items") or [])
                    pos = len(items)
                    for i, it in enumerate(items):
                        if it.get("kind") == "talk" and isinstance(it.get("a"), int) and it["a"] > span[0]:
                            pos = i
                            break
                    items.insert(pos, {"kind": "talk", "a": span[0], "b": span[1], "protect": True, "satisfies": [c["id"]], "id": f"i{len(items)}a"})
                    plan["items"] = items
        elif c.get("type") == "exclude_words":                                  # the chosen place must be out of the cut: the plan is edited around it
            from .localproj import drop_sentences
            t0, t1 = float(cand.get("start_s") or 0), float(cand.get("end_s") or 0)
            bad = {s.idx for s in sents if s.start < t1 and s.end > t0}
            if bad:
                plan["items"] = _unique_ids(drop_sentences([it for it in plan.get("items") or [] if isinstance(it, dict)], bad))
    elif cand.get("file"):
        c["anchor"]["file_ids"] = [cand["file"]]
        if c.get("type") == "exclude_files":
            plan["exclude_files"] = sorted(set(plan.get("exclude_files") or []) | {cand["file"]})
    c["status"], c["note"], c["candidates"] = "grounded", "You chose this one.", []
    plan["constraints"] = [c if x.get("id") == cid else x for x in cons]
    plan = CN.tag_satisfies(plan, plan["constraints"], sents, words)
    res = _check(plan, files, words, sents)
    left = [x for x in plan["constraints"] if x.get("status") == "ambiguous" and x.get("hard")]
    ap = dict(plan.get("approval") or {})
    if not left:
        ap.pop("questions", None)
        ap["reason"] = ""
    else:
        ap["questions"] = [x.get("note") or "" for x in left]
    plan["approval"] = ap if (ap.get("questions") or ap.get("reason")) else None
    if plan["approval"] is None:
        plan.pop("approval")
    plan["unmet"] = [u for u in plan.get("unmet") or [] if isinstance(u, dict) and u.get("constraint") != cid]
    _save(st, plan, res, "answered-a-question")
    return build(plan, files, sents, res, str(job.get("status") or ""))


def leave_out(runner, pid: str, item_id: str, out: bool = True) -> dict:
    """Take a passage out of the plan (or put it back). Something that was asked for cannot be left out: the reason is returned in words."""
    st, job, files, words, sents, plan = _load(runner, pid)
    before = _check(plan, files, words, sents)
    items = list(plan.get("items") or [])
    stash = [e for e in plan.get("left_out_items") or [] if isinstance(e, dict)]
    if out:
        idx = next((i for i, it in enumerate(items) if str(it.get("id")) == str(item_id)), None)
        if idx is None:
            raise ValueError("that part is not in the plan")
        cand = {**plan, "items": items[:idx] + items[idx + 1:]}
        res = _check(cand, files, words, sents)
        new = _hard(res) - _hard(before)
        if new:
            msgs = [p["message"] for p in res["problems"] if p.get("hard") and (p.get("code"), p.get("constraint_id")) in new]
            raise ValueError("it cannot be left out: " + plain(" ".join(msgs[:2]), 200))
        stash.append({"item": items[idx], "index": idx})
        cand["left_out_items"] = stash
    else:
        e = next((x for x in stash if str((x.get("item") or {}).get("id")) == str(item_id)), None)
        if e is None:
            raise ValueError("that part was not left out")
        items.insert(min(int(e.get("index") or 0), len(items)), e["item"])
        cand = {**plan, "items": items, "left_out_items": [x for x in stash if x is not e]}
        res = _check(cand, files, words, sents)
    _save(st, cand, res, "left-out-a-part" if out else "put-a-part-back")
    return build(cand, files, sents, res, str(job.get("status") or ""))


def _int(v: Any, d: int = -1) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return d


def _same_line(plan: dict, it: dict, a: int, b: int) -> bool:
    """The other take (words a..b) is a take of the same line as this part: both sit in one take group."""
    ia, ib = _int(it.get("a")), _int(it.get("b"))
    for g in plan.get("take_groups") or []:
        if not isinstance(g, dict):
            continue
        takes = [t for t in g.get("takes") or [] if isinstance(t, dict)]
        mine = any(_int(t.get("a")) <= ib and _int(t.get("b")) >= ia for t in takes)
        theirs = any(_int(t.get("a")) == a and _int(t.get("b")) == b for t in takes)
        if mine and theirs:
            return True
    return False


def use_take(runner, pid: str, item_id: str, a: Any, b: Any) -> dict:
    """Swap a passage for another take of the same line (the one it replaces becomes an other take)."""
    st, job, files, words, sents, plan = _load(runner, pid)
    items = [dict(it) for it in plan.get("items") or []]
    it = next((x for x in items if str(x.get("id")) == str(item_id) and x.get("kind") == "talk"), None)
    if it is None:
        raise ValueError("that part is not in the plan")
    try:
        a, b = int(a), int(b)
    except (TypeError, ValueError):
        raise ValueError("pick one of the other takes that were shown") from None
    ok = any(_int(al.get("a")) == a and _int(al.get("b")) == b for al in plan.get("alt_takes") or [] if isinstance(al, dict))
    if not ok:
        raise ValueError("that is not one of the other takes of this line")
    if not _same_line(plan, it, a, b):
        raise ValueError("that take belongs to a different line, not this part")
    old = (it.get("a"), it.get("b"))
    it["a"], it["b"] = a, b
    for k in ("t0", "t1"):
        it.pop(k, None)
    alts = [al for al in plan.get("alt_takes") or [] if not (int(al.get("a", -1)) == a and int(al.get("b", -1)) == b)]
    for al in plan.get("alt_takes") or []:
        if int(al.get("a", -1)) == a and int(al.get("b", -1)) == b:
            alts.append({"group": al.get("group"), "a": old[0], "b": old[1]})
    cand = {**plan, "items": items, "alt_takes": alts}
    cand = CN.tag_satisfies(cand, cand.get("constraints") or [], sents, words)
    res = _check(cand, files, words, sents)
    _save(st, cand, res, "used-another-take")
    return build(cand, files, sents, res, str(job.get("status") or ""))


def checked(runner, pid: str) -> dict:
    """After the render: what was checked and how it went, in words."""
    from . import notes as NT
    st = runner._setup(pid)[2]
    plan = st.get("plan.json") or {}
    cons = [c for c in plan.get("constraints") or [] if isinstance(c, dict)]
    rows = understood(cons, plan.get("problems"))
    checks = [{"ok": bool(x.get("ok")), "what": plain(x.get("what"), 200)} for x in plan.get("post_check") or [] if isinstance(x, dict)]
    try:
        ni = runner.projects.get(pid).get("notes_info") or {}
        issues = ni.get("issues")
        if issues is None:
            from pathlib import Path
            issues = NT.rows_from_md(Path(ni.get("report") or "").read_text())
    except Exception:  # noqa: BLE001 - an older video without the rows
        issues = []
    skipped, dropped, other = [], [], []
    for r in issues or []:
        what = str(r.get("what") or "")
        line = {"at": str(r.get("where") or ""), "start": r.get("start"), "line": plain(what, 200)}
        if re.search(r"stayed on the speaker|nothing matched", what, re.I):
            skipped.append(line)
        elif re.search(r"dropped|left out|its clip ended|shortened", what, re.I):
            dropped.append(line)
        elif r.get("level") != "Info" and re.search(r"graphic|card|caption|face|lower third|logo|overlap", what, re.I):
            other.append(line)
    met = sum(1 for r in rows if r["state"] == "done")
    return {"instructions": rows, "checks": checks, "dropped": dropped[:20], "skipped": skipped[:20], "placement": other[:20],
            "summary": ("No separate instructions were found in the notes." if not rows else f"{met} of {len(rows)} instruction{'s' if len(rows) != 1 else ''} done.") +
                       (f" {sum(1 for c in checks if not c['ok'])} check{'s' if sum(1 for c in checks if not c['ok']) != 1 else ''} did not pass." if any(not c["ok"] for c in checks) else
                        (" Everything I checked passed." if checks else ""))}


def clip_notes(clip: dict) -> dict:
    """The rubric of a short clip as three small numbers and the reasons, for My Clips and the feed (text only: no colours for grades)."""
    rb = (clip or {}).get("rubric") if isinstance((clip or {}).get("rubric"), dict) else {}
    if not rb:
        return {}

    def grade(k):
        try:
            return max(1, min(5, int(round(float(rb.get(k))))))
        except (TypeError, ValueError, OverflowError):
            return None
    warns = []
    for i in ((clip or {}).get("qa") or {}).get("issues", []) or []:
        if isinstance(i, dict) and i.get("what"):
            warns.append(plain(i["what"], 160))
    return {"hook": grade("hook"), "flow": grade("flow"), "value": grade("value"),
            "why": {k: plain(rb.get("why_" + k), 160) for k in ("hook", "flow", "value")}, "warnings": warns[:4]}


__all__ = ["build", "review", "answer", "leave_out", "use_take", "checked", "clip_notes", "plain", "clock", "texts", "copy"]
