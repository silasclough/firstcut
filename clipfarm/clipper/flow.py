"""Following the creator's outline. When the notes lay out how the video goes (a numbered list of beats), that outline is the script:
every step is looked up in the transcript, the strongest passages for it are chosen, and the steps play IN ORDER. Afterwards each step
is checked against what was actually used, so a step with nothing behind it is reported instead of silently skipped."""

from __future__ import annotations

import difflib
import re

from ..core.log import get_logger
from . import moments as M
from . import styles_profile as SPF

log = get_logger("clipper.flow")
STEP = re.compile(r"^\s*(?:(?:step\s*)?(\d{1,2})\s*[.)\]:\-]|([\u2022\u00b7\u25e6\u25aa\u2023*\-\u2013\u2014]))\s*(.+?)\s*$", re.I)


# A numbered line like "1.01 Welcome" or "2.3 The first seminar": section.item (the creator's own numbering of their clips or sections).
# Not "1." / "1)" (a plain list) and not a decimal in a sentence ("2.5 minutes long": a list of this kind has to start at section 1).
DOTTED = re.compile(r"^\s*(?:step\s*)?(\d{1,2})\.(\d{1,3})\s*[.)\]:\-\u2013\u2014]?\s+(.+?)\s*$", re.I)


def steps(notes: str) -> list[str]:
    """The script in the notes: a numbered or bulleted list of 3 or more beats (numbers must count up from 1), else []. A beat's text may
    continue on the next indented line. Non-breaking spaces (pasted from Word / Pages) count as spaces. A list numbered section.item
    (1.01, 1.02, 2.01 ... 8.10) is a list too: each beat keeps its number, so "1.01 Welcome" reads as that and not as step 1 "01 Welcome"."""
    lines = [ln.replace("\xa0", " ").rstrip() for ln in str(notes or "").splitlines()]
    dotted = _dotted_steps(lines)
    if dotted:
        return dotted
    out: list[str] = []
    kind = None
    for line in lines:
        m = STEP.match(line)
        if m and not DOTTED.match(line):
            this = "num" if m.group(1) else "bullet"
            if kind is None:
                if this == "num" and int(m.group(1)) != 1:
                    continue
                kind = this
            if this != kind:
                continue
            if this == "num" and int(m.group(1)) != len(out) + 1:
                continue
            out.append(m.group(3))
        elif out and line.strip() and line.startswith((" ", "\t")):
            out[-1] += " " + line.strip()
        elif out and not line.strip():
            continue
        elif out:
            break                                                            # a plain line after the list: the list is over
    return out if len(out) >= 3 else []


def _dotted_steps(lines: list[str]) -> list[str]:
    """The beats of a list numbered section.item, each as "1.01 text" (in the order written; the numbers only have to go up, they may skip)."""
    out: list[str] = []
    last: tuple[int, int] | None = None
    for line in lines:
        m = DOTTED.match(line)
        if m:
            key = (int(m.group(1)), int(m.group(2)))
            if last is None:
                if key[0] != 1:
                    continue                                                 # (a decimal in a sentence, not the start of a list)
            elif key <= last:
                continue
            last = key
            out.append(f"{m.group(1)}.{m.group(2)} {m.group(3)}")
        elif out and line.strip() and line.startswith((" ", "\t")):
            out[-1] += " " + line.strip()
        elif out and line.strip() and not DOTTED.match(line):
            break
    return out if len(out) >= 3 else []


SCHEMA = {"type": "object", "properties": {
    "title": {"type": "string"}, "summary": {"type": "string"},
    "steps": {"type": "array", "items": {"type": "object", "properties": {
        "n": {"type": "integer"}, "covered": {"type": "boolean"}, "note": {"type": "string"},
        "ranges": {"type": "array", "items": {"type": "object", "properties": {"first": {"type": "integer"}, "last": {"type": "integer"}},
                                              "required": ["first", "last"]}}},
        "required": ["n", "covered", "ranges"]}}},
    "required": ["title", "summary", "steps"]}

SYSTEM = (
    "You are the editor of an interview video and the creator has written the script as an outline of steps. Read the ENTIRE outline and "
    "ENTIRE transcript together, then assign every kept passage exactly once. FOLLOW THE STEPS IN ORDER: the finished video tells step 1, then "
    "step 2, and so on. For each step choose one to three passages (ranges of consecutive sentence numbers) that say it best. A passage must "
    "be a whole thought: start and end cleanly, never in the middle of an idea. When one passage changes topic, split it at the sentence where it "
    "moves to the next step. Do not repeat or share a sentence between steps, and do not take a sentence that belongs to another step. Use only "
    "the guest's words as they are. If the transcript truly has nothing for a step, say covered=false and leave its ranges empty; never invent or "
    "force a match. Keep to the time budget, spending more time on the steps that matter most. The notes may also contain other instructions "
    "(tone, length, what to show): honour them.")


STEP_SCHEMA = {"type": "object", "properties": {"covered": {"type": "boolean"}, "note": {"type": "string"},
                                                "ranges": {"type": "array", "items": {"type": "object", "properties": {"first": {"type": "integer"}, "last": {"type": "integer"}},
                                                                                       "required": ["first", "last"]}}}, "required": ["covered", "ranges"]}
TITLE_SCHEMA = {"type": "object", "properties": {"title": {"type": "string"}, "summary": {"type": "string"}}, "required": ["title", "summary"]}


def name_it(ai, sents, notes: str = "") -> str:
    """A short title (3-8 words, no quotes) for a video that is kept as it is, from what is said in it: "" when there is nothing to go on."""
    text = " ".join(s.text for s in sents)[:6000].strip()
    if len(text.split()) < 8:
        return ""
    prompt = ((f"The creator's notes: {SPF.fit_brief(notes, 600)}\n\n" if notes.strip() else "") + f"What is said in the video:\n{text}\n\n"
              "Write a short, specific title for this video (3 to 8 words, plain words, no quotes, no clickbait) and a one-sentence summary.")
    try:
        res = ai.complete_json("video_title", "You name finished videos for a college marketing team.", prompt, TITLE_SCHEMA)
        return " ".join(str(res.data.get("title") or "").replace('"', "").split())[:90]
    except Exception:  # noqa: BLE001 - the file's name is a fine fallback
        return ""


def plan(ai, sents: list[M.Sentence], outline: list[str], notes: str, budget_s: float, title: str = "") -> tuple[list[dict], str, str, list[dict]]:
    """-> (items, title, summary, report) where items are talk items in outline order and report lists each step and what covers it.
    The model assigns all passages in ONE pass, so it can split an idea where it changes steps instead of letting an earlier step claim it."""
    lines = [f"{s.idx}|{s.end - s.start:.2f}s|{s.text}" for s in sents]
    must = requirements(outline, " ".join(x.text for x in sents))
    script = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(outline))
    required = "\n".join(f"step {n}: " + ", ".join(x for x, _ in must[n]) for n in sorted(must))
    prompt = (f"The creator's notes:\n{notes}\n\nTHE WHOLE SCRIPT, in required playback order:\n{script}\n\n"
              f"Exact names, numbers, and quoted words to preserve when heard:\n{required}\n\nTarget duration: about {budget_s:.0f} seconds.\n\n"
              "ENTIRE GUEST TRANSCRIPT, one complete sentence per line as number|length|text:\n" + "\n".join(lines))
    try:
        answer = ai.complete_json("flow_plan", SYSTEM, prompt, SCHEMA).data
    except Exception as e:  # noqa: BLE001
        log.warning(f"one-pass script mapping failed: {e}")
        raise
    got_steps = {int(s.get("n", -1)): s for s in answer.get("steps") or []}
    by = {s.idx: s for s in sents}
    items: list[dict] = []
    report: list[dict] = []
    used: set[int] = set()
    for n in range(1, len(outline) + 1):
        st = got_steps.get(n, {})
        got = []
        for r in st.get("ranges") or []:
            a, b = int(r.get("first", -1)), int(r.get("last", -1))
            if a > b:
                a, b = b, a
            runs: list[list[int]] = []                                   # the sentences of this range that exist and no earlier step took, as whole runs
            for i in range(a, b + 1):
                if i in by and i not in used:
                    if runs and runs[-1][-1] == i - 1:
                        runs[-1].append(i)
                    else:
                        runs.append([i])
            if not runs:
                log.warning(f"script map: range {a}..{b} for step {n} is empty or already used by another step")
            for run in runs:
                if len(run) < b - a + 1:
                    log.info(f"script map: step {n} range {a}..{b} overlapped another step; kept {run[0]}..{run[-1]}")
                used.update(run)
                items.append({"kind": "talk", "a": run[0], "b": run[-1], "step": n})
                got.append((run[0], run[-1]))
        report.append({"n": n, "step": outline[n - 1], "covered": bool(got), "ranges": got,
                       "text": " … ".join(" ".join(by[i].text for i in range(a, b + 1)) for a, b in got), "note": str(st.get("note", "") )[:200]})
    total = lambda its: sum(by[it["b"]].end - by[it["a"]].start for it in its)          # noqa: E731
    for _ in range(3):                                                                   # over budget: ask for tighter passages, keeping every step
        if total(items) <= budget_s * 1.1:
            break
        items = trim(ai, by, items, budget_s, notes)
    items = complete(ai, by, items, budget_s)                                            # clean starts and ends (never into another passage)
    while items and total(items) > budget_s * 1.15:                                      # still over: drop a whole passage of a step that has several
        per: dict[int, list[dict]] = {}
        for it in items:
            per.setdefault(it["step"], []).append(it)
        big = max((v for v in per.values() if len(v) > 1), key=lambda v: total(v), default=None)
        if big is None:
            break
        items.remove(min(big, key=lambda it: by[it["b"]].end - by[it["a"]].start))        # the shortest of the biggest step's passages
    return items, str(answer.get("title") or title).strip()[:90], str(answer.get("summary") or "").strip(), report


TRIM_SCHEMA = {"type": "object", "properties": {"passages": {"type": "array", "items": {"type": "object", "properties": {
    "n": {"type": "integer"}, "first": {"type": "integer"}, "last": {"type": "integer"}}, "required": ["n", "first", "last"]}}},
    "required": ["passages"]}


def trim(ai, by: dict, items: list[dict], budget_s: float, notes: str) -> list[dict]:
    """The chosen passages are too long for the time budget: keep the strongest whole thought (consecutive sentences) of each, so every
    step is still told, in order, and the whole is about `budget_s` seconds."""
    lens = [by[it["b"]].end - by[it["a"]].start for it in items]
    scale = min(1.0, budget_s / max(1.0, sum(lens)))
    allow = [max(8.0, x * scale * 0.95) for x in lens]                                   # each passage keeps its share of the budget
    lines = []
    for k, it in enumerate(items):
        lines.append(f"PASSAGE {k} (step {it['step']}, {lens[k]:.0f}s now; keep AT MOST {allow[k]:.0f}s):")
        lines += [f"  {i}|{by[i].end - by[i].start:.0f}s|{by[i].text}" for i in range(it["a"], it["b"] + 1) if i in by]
    prompt = (f"The creator's notes:\n{notes}\n\nThese passages tell the video's steps in order but together are too long. The whole video may use about "
              f"{budget_s:.0f} seconds. For EACH passage return the part to keep: consecutive whole sentences (first and last sentence number), "
              "the strongest complete thought, starting and ending cleanly. Cut each passage to about its share of the budget; passages for steps "
              "that matter more may keep more. Keep every passage (n = its PASSAGE number).\n\n" + "\n".join(lines))
    try:
        res = ai.complete_json("flow_trim", "You tighten an interview edit without losing any step of the story.", prompt, TRIM_SCHEMA)
    except Exception as e:  # noqa: BLE001 - the longer cut stands
        log.warning(f"trimming the passages failed: {e}")
        return items
    out = [dict(it) for it in items]
    for p in res.data.get("passages") or []:
        k, a, b = int(p.get("n", -1)), int(p.get("first", -1)), int(p.get("last", -1))
        if 0 <= k < len(items) and items[k]["a"] <= a <= b <= items[k]["b"]:
            out[k]["a"], out[k]["b"] = a, b
    for k, it in enumerate(out):                                      # whatever the AI left too long is cut back sentence by sentence
        while it["b"] > it["a"] and by[it["b"]].end - by[it["a"]].start > allow[k] * 1.3:
            it["b"] -= 1
    return out


EDGE_SCHEMA = {"type": "object", "properties": {"passages": {"type": "array", "items": {"type": "object", "properties": {
    "n": {"type": "integer"}, "drop_first": {"type": "integer"}, "drop_last": {"type": "integer"}, "extend_last": {"type": "integer"}},
    "required": ["n", "drop_first", "drop_last", "extend_last"]}}},
    "required": ["passages"]}


def complete(ai, by: dict, items: list[dict], budget_s: float = 0.0) -> list[dict]:
    """Every passage must begin and end on a whole thought. The AI looks at the edges of each (with the sentences around it): a start that is the
    tail of an earlier thought is trimmed, an end that stops mid-thought runs on until it is finished. The last passage matters most: the video
    never ends mid-sentence."""
    lines = []
    for k, it in enumerate(items):
        pre = [i for i in range(it["a"] - 3, it["a"]) if i in by]
        body = list(range(it["a"], it["b"] + 1))
        post = [i for i in range(it["b"] + 1, it["b"] + 5) if i in by]
        shown = body                                                        # An edge review still needs the actual sentences, never a clipped placeholder.
        lines.append(f"PASSAGE {k}:")
        lines += [f"  (before, not used) {by[i].text}" for i in pre]
        lines += [f"  >> [{i}] {by[i].text}" for i in shown]
        lines += [f"  (after, not used) +{j + 1}: {by[i].text}" for j, i in enumerate(post)]
    taken = {i for it in items for i in range(it["a"], it["b"] + 1)}
    prompt = ("For each passage (the >> lines are what is used), decide: drop_first = how many of its FIRST sentences to drop because they are the tail "
              "end of an earlier thought, or are the interviewer's acknowledgement or filler such as 'okay', 'yes', 'right', 'that's good' (0 to 2); "
              "drop_last = how many of its LAST sentences to drop because they are filler or the interviewer talking ('okay that's good', 'yes', "
              "a question) after the real answer has ended (0 to 2); extend_last = how many of the '(after)' sentences must be added so the passage ends on a finished "
              "sentence and a finished thought (0 to 4; 0 if it already ends cleanly). Words like 'and', 'but', 'so', 'the' at the end mean it is "
              "unfinished.\n\n" + "\n".join(lines))
    try:
        res = ai.complete_json("flow_edges", "You are checking where an edit's passages begin and end.", prompt, EDGE_SCHEMA)
    except Exception as e:  # noqa: BLE001
        log.warning(f"checking the passage edges failed: {e}")
        return items
    out = [dict(it) for it in items]
    room = 0.2 * budget_s if budget_s else 1e9                                           # all the running-on together may add at most 20% to the length
    plist = sorted(res.data.get("passages") or [], key=lambda p: (int(p.get("n", -1)) != len(out) - 1, int(p.get("n", 0))))      # the closing first
    for p in plist:
        k = int(p.get("n", -1))
        if not 0 <= k < len(out):
            continue
        df, dl, ex = (max(0, min(2, int(p.get("drop_first", 0)))), max(0, min(2, int(p.get("drop_last", 0)))),
                      max(0, min(2, int(p.get("extend_last", 0)))))
        it = out[k]
        if df and it["b"] - it["a"] >= df + 1:
            it["a"] += df
        if dl and it["b"] - it["a"] >= dl + 1:
            it["b"] -= dl
        elif ex:
            while ex and it["b"] + 1 in by and it["b"] + 1 not in taken:                 # never into a sentence another passage already tells
                add = by[it["b"] + 1].end - by[it["b"] + 1].start
                if add > room:
                    break
                room -= add
                it["b"] += 1
                ex -= 1
    return out


# ----------------------------------------------------------------------------------------------- the script supervisor
AUDIT_SCHEMA = {"type": "object", "properties": {"steps": {"type": "array", "items": {"type": "object", "properties": {
    "n": {"type": "integer"}, "missing": {"type": "array", "items": {"type": "string"}}}, "required": ["n", "missing"]}}}, "required": ["steps"]}


NOT_NAMES = {"Mom", "Dad", "I", "The", "A", "An", "They", "Their", "He", "She", "It", "We", "Our", "His", "Her"}


def _name_stats(text: str) -> tuple[dict[str, int], dict[str, int]]:
    """How often each word is written with a capital in the MIDDLE of a sentence, and how often in lower case, in `text`."""
    cap: dict[str, int] = {}
    low: dict[str, int] = {}
    prev = "."
    for tok in text.split():
        w = tok.strip(".,;:!?\"'()\u201c\u201d")
        if w:
            if w[:1].isupper() and prev[-1:] not in ".!?":
                cap[w] = cap.get(w, 0) + 1
            elif w[:1].islower():
                low[w.lower()] = low.get(w.lower(), 0) + 1
        prev = tok
    return cap, low


def requirements(outline: list[str], text: str = "") -> dict[int, list[tuple[str, bool]]]:
    """What each step must tell, taken VERBATIM from the step. Heard word for word (literal): anything in quotes, every proper name (a person,
    place, program) and every number. Only checked by the AI (ideas): the things listed after a colon or 'including', and, if a step has nothing else,
    the step itself. -> {step: [(item, literal?)]}.

    A capital letter in the middle of a sentence marks a name. A capitalised word at the START of a step, or in a topic label before a colon
    ("Unexpected pregnancy:", "Parenting classes ..."), is only a capital because it starts something, so it counts as a name only if the
    context says so: `text` (the interview) writes it with a capital mid-sentence more often than without ("Jolene", "Harbor Clinic"), or,
    with no interview text, the rest of the script writes it that way too."""
    texts = [" ".join(str(t).split()) for t in outline]
    script_cap, _ = _name_stats(" ".join(texts))
    talk_cap, talk_low = _name_stats(text) if text else ({}, {})
    talk_caps_anywhere = sorted({t.strip(".,;:!?\"'()\u201c\u201d") for t in text.split() if t[:1].isupper()}) if text else []

    def is_name(w: str, label: bool = False) -> bool:
        if text:
            c, l = talk_cap.get(w, 0), talk_low.get(w.lower(), 0)
            if c == 0 and l == 0 and not label:                      # never heard: a name only if something CLOSE is heard with a capital ("Jolane" for Jolene)
                import difflib
                return bool(difflib.get_close_matches(w, talk_caps_anywhere, n=1, cutoff=0.7))
            return c >= (max(2, l) if label else max(1, l))             # a topic label needs its capital seen twice (once could be stray chatter)             # never heard at all: maybe a name the recogniser misspelled ("Eolene")
        return script_cap.get(w, 0) >= 1

    proper: set[str] = set()
    for t in texts:
        toks = t.split()
        colon = next((k for k, w in enumerate(toks[:6]) if w.endswith(":")), None)
        start = (colon + 1) if colon is not None else 1                  # the words of a label (or just the first word) start something
        for j, w in enumerate(toks):
            c = re.sub(r"['\u2019]s$", "", w.strip(".,;:!?\"'()"))
            if not c[:1].isupper() or c in NOT_NAMES:
                continue
            begins = j == 0 or toks[j - 1][-1:] in ".!?" or j < start         # starts a step, a sentence, or sits inside the label
            if not begins or is_name(c, label=j < start and colon is not None):
                proper.add(c)
    out: dict[int, list[tuple[str, bool]]] = {}
    for n, text in enumerate(texts, 1):
        items: list[tuple[str, bool]] = [(m.strip(" .;"), True) for m in re.findall(r'["\u201c]([^"\u201d]{6,160})["\u201d]', text)]
        bare = re.sub(r'["\u201c][^"\u201d]*["\u201d]', " ", text)
        run: list[str] = []
        for w in bare.split() + [""]:
            c = re.sub(r"['\u2019]s$", "", w.strip(".,;:!?\"'()"))
            if c and (c in proper or (c.isupper() and len(c) >= 2)) and c not in NOT_NAMES:
                run.append(c)
                if w[-1:] in ",;:.!?)":                                      # punctuation ends a name ("Hannah Halden, Little Steps" are two)
                    name = " ".join(run)
                    if all(name != x for x, _ in items):
                        items.append((name, True))
                    run = []
            else:
                if run:
                    name = " ".join(run)
                    if all(name != x for x, _ in items):
                        items.append((name, True))
                run = []
        for d in re.findall(r"\b\d[\d,.]*\b", bare):
            if all(d != x for x, _ in items):
                items.append((d, True))
        tail = ""
        if ":" in bare:
            tail = bare.split(":", 1)[1]
        elif re.search(r"\bincluding\b", bare):
            tail = re.split(r"\bincluding\b", bare, 1)[1]
        for part in re.split(r",|;|\.\s| and | or |\band even\b", tail):
            part = re.sub(r"^(even|also|plus)\s+", "", part.strip(" .;\"'\u201c\u201d"))
            if part and 1 <= len(part.split()) <= 7 and part.split()[-1].lower() not in ("were", "was", "who", "that", "the", "a", "of", "to", "is") \
                    and all(part.lower() != x.lower() and (not lit or x.lower() not in part.lower()) for x, lit in items):
                items.append((part, False))
        if not items:
            items.append((text.rstrip("."), False))                      # the step as a whole must be told
        out[n] = items[:10]
    return out


def _toks(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def _number_words(n: int) -> str:
    """English forms used by speech recognition for an integer in a script (22 -> "twenty two")."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + (" " + _ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return _ONES[n // 100] + " hundred" + (" " + _number_words(n % 100) if n % 100 else "")
    if n < 10000:
        return _ONES[n // 1000] + " thousand" + (" " + _number_words(n % 1000) if n % 1000 else "")
    return ""


def _number_forms(token: str) -> set[str]:
    try:
        n = int(token.replace(",", ""))
    except ValueError:
        return set()
    forms = {_number_words(n)} if 0 <= n < 10000 else set()
    if 1000 <= n <= 2099:                                                # "2019" is commonly transcribed as "twenty nineteen"
        forms.add(_number_words(n // 100) + " " + _number_words(n % 100))
    return {x for x in forms if x}


def heard(item: str, text: str) -> bool:
    """Is `item` in `text`? Numbers and names tolerate the speech recogniser's spelling; longer phrases must match closely."""
    want, have = _toks(item), _toks(text)
    if not want:
        return True
    if len(want) == 1 and want[0].isdigit():
        spoken = _number_forms(want[0])
        padded = " " + " ".join(have) + " "                              # whole words only: "two" is not in "network", "ten" not in "often"
        return want[0] in have or any(f" {form} " in padded for form in spoken)
    n = len(want)
    if n <= 3:                                                           # a name: at least half of its words, each allowing a different spelling
        need = max(1, (n + 1) // 2)
        thr = 0.8 if n == 1 else 0.66
        for k in range(0, max(1, len(have) - n + 2)):
            win = have[k:k + n + 1]
            ok = sum(1 for w in want if win and max(difflib.SequenceMatcher(None, w, x).ratio() for x in win) >= thr)
            if ok >= need:
                return True
        return False
    for k in range(0, max(1, len(have) - n + 1)):
        for w in (n - 1, n, n + 1):
            if w >= 1 and difflib.SequenceMatcher(None, " ".join(want), " ".join(have[k:k + w])).ratio() >= 0.84:
                return True
    return False


_THOUGHT_LEAD = re.compile(r"^(and|but|so|because|then|also|which|that|who|while|when)\b", re.I)
_THOUGHT_TAIL = re.compile(r"\b(and|but|so|or|because|the|a|an|to|of|for|with|that|which|who|if|when|then|like|is|was|are|were)$", re.I)


def whole_thought(sents: list[M.Sentence], idx: int, max_extra: int = 4) -> tuple[int, int]:
    """The smallest clean passage around a found sentence. Literal repairs never drop a name or number in as an isolated fragment."""
    by = {s.idx: s for s in sents}
    a = b = idx
    for _ in range(max_extra):
        cur, prev = by.get(a), by.get(a - 1)
        if not cur or not prev or cur.start - prev.end > 1.5 or not _THOUGHT_LEAD.search(cur.text.strip()):
            break
        a -= 1
    for _ in range(max_extra):
        cur, nxt = by.get(b), by.get(b + 1)
        last = (cur.text if cur else "").strip().rstrip(".,;:!?\"')]")
        if not cur or not nxt or nxt.start - cur.end > 1.5 or (M.END.search(cur.text) and not _THOUGHT_TAIL.search(last)):
            break
        b += 1
    return a, b


def correct_names(words: list[dict], outline: list[str]) -> tuple[list[dict], list[dict]]:
    """Correct high-confidence speech-recogniser spellings using names in the creator's own script.

    Only the visible transcript/captions change; timing and the recorded voice remain untouched. A correction requires the same number of
    words and a close fuzzy match, so a general script word cannot casually rewrite the transcript.
    """
    names = []
    for reqs in requirements(outline, " ".join(w["text"] for w in words)).values():
        for item, literal in reqs:
            toks = re.findall(r"[A-Za-z][A-Za-z'’-]*", item)
            if literal and toks and not any(x.isdigit() for x in toks) and any(x[:1].isupper() for x in toks):
                names.append(toks)
    out, fixes = [dict(w) for w in words], []
    for wanted in names:
        n = len(wanted)
        for i in range(0, len(out) - n + 1):
            have = [re.sub(r"[^A-Za-z'’-]", "", str(w["text"])) for w in out[i:i + n]]
            if not all(have):
                continue
            scores = [difflib.SequenceMatcher(None, x.lower(), y.lower()).ratio() for x, y in zip(wanted, have)]
            if min(scores) < 0.72 or sum(scores) / n < 0.80 or all(x.lower() == y.lower() for x, y in zip(wanted, have)):
                continue
            before = " ".join(str(w["text"]) for w in out[i:i + n])
            for k, want in enumerate(wanted):
                suffix = re.search(r"[^A-Za-z'’-]+$", str(out[i + k]["text"]))
                out[i + k]["text"] = want + (suffix.group(0) if suffix else "")
            fixes.append({"from": before, "to": " ".join(wanted), "at": out[i]["start"]})
    return out, fixes


FIND_SCHEMA = {"type": "object", "properties": {"found": {"type": "array", "items": {"type": "object", "properties": {
    "item": {"type": "string"}, "first": {"type": "integer"}, "last": {"type": "integer"}, "in_footage": {"type": "boolean"}}, "required": ["item", "in_footage"]}}},
    "required": ["found"]}


def find(ai, sents: list, step: int, step_text: str, missing: list[str], used: set[int]) -> list[dict]:
    """Search the WHOLE transcript for what a step is missing: the shortest whole thought that tells each missing thing, or in_footage=false."""
    lines = [f"{s.idx}|{s.end - s.start:.0f}s|{s.text}" for s in sents if s.idx not in used]
    prompt = (f"Script step {step}: {step_text}\n\nThis must still be told: " + "; ".join(missing)
              + "\n\nTranscript of the guest (number|length|text), only sentences not yet used:\n" + "\n".join(lines)
              + "\n\nFor each thing that must be told, give the first and last sentence number of the shortest whole thought in which the guest tells it "
                "(start and end cleanly). If the guest never says it anywhere, in_footage=false. A passage that merely touches the topic is not a match.")
    try:
        return ai.complete_json("flow_find", "You search an interview transcript for specific content.", prompt, FIND_SCHEMA).data.get("found") or []
    except Exception as e:  # noqa: BLE001
        log.warning(f"searching the transcript for step {step} failed: {e}")
        return []


def _idea_gaps(ai, by: dict, items: list[dict], must: dict) -> dict[int, list[str]]:
    """The ideas the script lists (not names / numbers / quotes) that the passages under each step do NOT tell, by the AI's honest reading."""
    ideas = {n: [m for m, lit in ms if not lit] for n, ms in must.items()}
    if not any(ideas.values()):
        return {}
    cut = []
    for k, it in enumerate(items):
        cut.append(f"PASSAGE {k} (step {it.get('step')}):")
        cut += [f"  {by[i].text}" for i in range(it["a"], it["b"] + 1) if i in by]
    try:
        res = ai.complete_json("flow_audit", "You check an edit against a script. Be fair: an idea counts as told if a passage clearly says it, in any words. "
                               "Something told only under a different step does not count for this step.",
                               "THE SCRIPT'S IDEAS PER STEP:\n" + "\n".join(f"step {n}: " + "; ".join(v) for n, v in ideas.items() if v)
                               + "\n\nTHE CUT:\n" + "\n".join(cut) + "\n\nFor each step list in 'missing' the ideas NOT told in the passages placed under that step.", AUDIT_SCHEMA)
    except Exception as e:  # noqa: BLE001
        log.warning(f"the AI check of the script's ideas failed: {e}")
        return {}
    return {int(x["n"]): [str(m) for m in x.get("missing") or []] for x in res.data.get("steps") or [] if x.get("missing")}


def recheck(ai, sents: list, items: list[dict], outline: list[str], old: list[dict] | None = None) -> list[dict]:
    """After a change to a finished video: is every step of the script still told somewhere in the changed cut? Names, numbers and quotes
    are looked for word for word, the ideas by the AI over the whole cut. Nothing is added or moved - the cut stays as the creator asked;
    this only keeps the report honest. -> the same per-step report as supervise() (with "rechecked": True)."""
    by = {s.idx: s for s in sents}
    must = requirements(outline, " ".join(x.text for x in sents))
    talk = [it for it in items if it.get("kind", "talk") == "talk"]

    def said(its: list[dict]) -> str:
        return " ".join(by[i].text for it in its for i in range(it["a"], it["b"] + 1) if i in by)
    never = {int(r["n"]): list(r.get("not_in_footage") or []) for r in old or [] if r.get("n") is not None}
    whole_text = " ".join(x.text for x in sents)
    ideas = {n: [m for m, lit in ms if not lit] for n, ms in must.items()}
    gaps: dict[int, list[str]] = {}
    if any(ideas.values()) and talk:
        cut = "\n".join(f"  {by[i].text}" for it in talk for i in range(it["a"], it["b"] + 1) if i in by)
        try:
            res = ai.complete_json("flow_recheck", "You check an edit against a script. Be fair: an idea counts as told if the cut clearly says it, "
                                   "in any words, anywhere in the cut.",
                                   "THE SCRIPT'S IDEAS PER STEP:\n" + "\n".join(f"step {n}: " + "; ".join(v) for n, v in ideas.items() if v)
                                   + "\n\nTHE CUT, in play order:\n" + cut + "\n\nFor each step list in 'missing' the ideas the cut does NOT tell anywhere.",
                                   AUDIT_SCHEMA)
            gaps = {int(x["n"]): [str(m) for m in x.get("missing") or []] for x in res.data.get("steps") or [] if x.get("missing")}
        except Exception as e:  # noqa: BLE001 - the word-for-word checks below still run
            log.warning(f"re-checking the script's ideas failed: {e}")
    report = []
    for n, step in enumerate(outline, 1):
        mine = [it for it in talk if it.get("step") == n]
        mine_text = said(mine)
        lit_gone = [m for m, lit in must.get(n, []) if lit and not heard(m, mine_text)]
        nowhere = [m for m in lit_gone if m in never.get(n, []) or not heard(m, whole_text)]      # the guest never says it anywhere (e.g. her own name)
        report.append({"n": n, "step": step, "ranges": [(it["a"], it["b"]) for it in mine],
                       "missing": [m for m in lit_gone if m not in nowhere] + gaps.get(n, []),
                       "not_in_footage": nowhere,
                       "covered": bool(mine) and not lit_gone and not gaps.get(n), "text": mine_text, "rechecked": True})
    return report


def supervise(ai, sents: list, items: list[dict], outline: list[str], notes: str, budget_s: float, rounds: int = 1, clean=None,
              banned: set | None = None) -> tuple[list[dict], list[dict]]:
    """Make the cut follow the script, and PROVE it. Every name, number and quoted phrase the script lists is looked for, word for word, in what
    the viewer will hear; one that is missing is searched for in the whole transcript and its sentence is added to its step (the cut is then
    tightened to the time again). What the guest never says is reported, not faked. Ideas are checked by the AI and reported honestly.
    -> (items, per-step report)."""
    by = {s.idx: s for s in sents}
    must = requirements(outline, " ".join(x.text for x in sents))
    banned = banned if banned is not None else set()                                      # sentences that must never be used (the interviewer's)
    total = lambda its: sum(by[it["b"]].end - by[it["a"]].start for it in its)               # noqa: E731

    def said(step_items: list[dict]) -> str:
        return " ".join(by[i].text for it in step_items for i in range(it["a"], it["b"] + 1) if i in by)
    not_in_footage: dict[int, list[str]] = {}
    for _ in range(rounds):
        used = {i for it in items for i in range(it["a"], it["b"] + 1)}
        added = 0
        for n, ms in must.items():
            for item, lit in ms:
                if not lit or heard(item, said([it for it in items if it.get("step") == n])):
                    continue
                hit = next((s for s in sents if s.idx not in used and s.idx not in banned and heard(item, s.text)), None)
                if hit is None:
                    if not any(heard(item, s.text) for s in sents):
                        not_in_footage.setdefault(n, []).append(item)
                    continue
                a, b = whole_thought(sents, hit.idx)
                if any(i in used or i in banned for i in range(a, b + 1)):
                    continue
                at = max([k for k, it in enumerate(items) if it.get("step") == n] or [max([k for k, it in enumerate(items) if it.get("step", 0) < n] or [-1])])
                items.insert(at + 1, {"kind": "talk", "a": a, "b": b, "step": n, "added": True})
                used.update(range(a, b + 1))
                added += 1
        miss = _idea_gaps(ai, by, items, must)                                              # ideas the script lists that the cut does not tell
        for n, ms in sorted(miss.items()):
            if added >= 6:
                break
            used = {i for it in items for i in range(it["a"], it["b"] + 1)}
            for f in find(ai, sents, n, outline[n - 1], ms, used | banned):
                if not f.get("in_footage", True) or "first" not in f or "last" not in f or added >= 6:
                    continue
                a, b = int(f["first"]), int(f["last"])
                if a > b:
                    a, b = b, a
                a, b = whole_thought(sents, a)
                idx = [i for i in range(a, b + 1) if i in by and i not in used and i not in banned]
                if len(idx) != b - a + 1 or b - a > 5:
                    continue
                at = max([k for k, it in enumerate(items) if it.get("step") == n] or [max([k for k, it in enumerate(items) if it.get("step", 0) < n] or [-1])])
                items.insert(at + 1, {"kind": "talk", "a": idx[0], "b": idx[-1], "step": n, "added": True})
                used.update(range(idx[0], idx[-1] + 1))
                added += 1
        if not added:
            break
        for _t in range(3):                                                                   # the additions must fit the time
            if total(items) <= budget_s * 1.1:
                break
            items = trim(ai, by, items, budget_s, notes + "\n\nWhatever is cut, keep telling: " + "; ".join(f"step {n}: " + ", ".join(x for x, _ in v) for n, v in must.items()))
        items = complete(ai, by, items, budget_s)
        while len(items) > len(outline) and total(items) > budget_s * 1.2:                    # still over: drop a whole extra passage (never what a step needs)
            per: dict[int, list[dict]] = {}
            for it in items:
                per.setdefault(it.get("step", 0), []).append(it)
            pool = [it for v in per.values() if len(v) > 1 for it in v if not it.get("added")]
            if not pool:
                break
            items.remove(max(pool, key=lambda it: by[it["b"]].end - by[it["a"]].start))
    if clean:                                                                             # everything added or extended is checked for the interviewer too...
        items = clean(items)
        used = {i for it in items for i in range(it["a"], it["b"] + 1)}
        for n, ms in must.items():                                                        # ...and then the words the script lists are GUARANTEED again
            for item, lit in ms:
                if not lit or heard(item, said([it for it in items if it.get("step") == n])):
                    continue
                hit = next((s for s in sents if s.idx not in used and s.idx not in banned and heard(item, s.text)), None)
                if hit is None:
                    continue
                a, b = whole_thought(sents, hit.idx)
                if any(i in used or i in banned for i in range(a, b + 1)):
                    continue
                at = max([k for k, it in enumerate(items) if it.get("step") == n] or [max([k for k, it in enumerate(items) if it.get("step", 0) < n] or [-1])])
                items.insert(at + 1, {"kind": "talk", "a": a, "b": b, "step": n, "added": True})
                used.update(range(a, b + 1))
    gaps = _idea_gaps(ai, by, items, must)                                               # the final, honest verdict on the ideas
    report = []
    for n, step in enumerate(outline, 1):
        rs = [(it["a"], it["b"]) for it in items if it.get("step") == n]
        heard_text = said([it for it in items if it.get("step") == n])
        gone = [m for m, lit in must.get(n, []) if lit and m not in not_in_footage.get(n, []) and not heard(m, heard_text)] + gaps.get(n, [])
        report.append({"n": n, "step": step, "ranges": rs, "covered": bool(rs) and not gone, "missing": gone,
                       "not_in_footage": not_in_footage.get(n, []),
                       "text": " … ".join(" ".join(by[i].text for i in range(a, b + 1) if i in by)[:200] for a, b in rs)[:600]})
    return items, report


FINAL_SCHEMA = {"type": "object", "properties": {"steps": {"type": "array", "items": {"type": "object", "properties": {
    "n": {"type": "integer"}, "missing": {"type": "array", "items": {"type": "string"}},
    "quotes": {"type": "array", "items": {"type": "string"}}}, "required": ["n", "missing", "quotes"]}}}, "required": ["steps"]}


def script_map(sents: list[M.Sentence], items: list[dict], outline: list[str], report: list[dict] | None = None) -> list[dict]:
    """The review screen's map: for every script step the exact lines that will be said (with their times), the unused line next to each end
    of a passage (so it can be added with one click), and a word-for-word check of the names, numbers and quotes the step lists. `report` (the
    planner's own per-step report) adds what the AI noted about ideas; pass None after the creator has edited, since it would be out of date."""
    by = {s.idx: s for s in sents}
    talk = [it for it in items if it.get("kind", "talk") == "talk"]
    used = {i for it in talk for i in range(it["a"], it["b"] + 1)}
    whole = " ".join(x.text for x in sents)
    must = requirements(outline, whole)
    old = {int(r["n"]): r for r in report or [] if r.get("n") is not None}
    out = []
    for n, step in enumerate(outline, 1):
        passages, ids = [], []
        for it in talk:
            if it.get("step") != n:
                continue
            lines = [{"sentence": i, "text": by[i].text, "start": round(by[i].start, 1), "end": round(by[i].end, 1)} for i in range(it["a"], it["b"] + 1) if i in by]
            before, after = by.get(it["a"] - 1), by.get(it["b"] + 1)
            passages.append({"first": it["a"], "last": it["b"], "lines": lines,
                             "before": {"sentence": before.idx, "text": before.text} if before and before.idx not in used else None,
                             "after": {"sentence": after.idx, "text": after.text} if after and after.idx not in used else None})
            ids += range(it["a"], it["b"] + 1)
        said = " ".join(by[i].text for i in ids if i in by)
        gone = [m for m, lit in must.get(n, []) if lit and not heard(m, said)]
        never = [m for m in gone if not heard(m, whole) or m in (old.get(n, {}).get("not_in_footage") or [])]
        literal = {m for m, lit in must.get(n, []) if lit}
        ideas = [m for m in (old.get(n, {}).get("missing") or []) if m not in literal]
        out.append({"n": n, "step": step, "passages": passages, "missing": [m for m in gone if m not in never], "not_in_footage": never,
                    "ideas": ideas})
    return out


def edit_items(sents: list[M.Sentence], items: list[dict], op: dict, steps: int | None = None) -> list[dict]:
    """One change from the review screen, applied to the plan's items (nothing else is touched):
    drop   {first, last, sentence}  remove that line from the passage (an end line trims it, a middle line splits it, the last line removes it)
    extend {first, last, edge}      add the unused line before ("start") or after ("end") the passage
    add    {step, first, last}      use these lines as a new passage of the step (after the step's other passages)
    Raises ValueError with a plain-words reason when the change cannot be made."""
    by = {s.idx: s for s in sents}
    items = [dict(it) for it in items]
    kind = op.get("op")
    for key in ("first", "last", "sentence", "step"):                     # whole numbers only (a click sends numbers; anything else is a mistake)
        if key in op and (isinstance(op[key], bool) or not isinstance(op[key], int)):
            raise ValueError(f"'{key}' must be a whole number")

    def idx_of(first, last):
        return next((k for k, it in enumerate(items) if it.get("kind", "talk") == "talk" and it["a"] == first and it["b"] == last), None)

    def used_by_others(skip: int | None) -> set[int]:
        return {i for k, it in enumerate(items) if k != skip and it.get("kind", "talk") == "talk" for i in range(it["a"], it["b"] + 1)}
    if kind in ("drop", "extend"):
        k = idx_of(op.get("first"), op.get("last"))
        if k is None:
            raise ValueError("that passage has changed: reload the plan")
        it = items[k]
        if kind == "drop":
            sid = op.get("sentence")
            if not it["a"] <= sid <= it["b"]:
                raise ValueError("that line is not in the passage")
            left = [{**it, "a": it["a"], "b": sid - 1}] if sid > it["a"] else []
            right = [{**it, "a": sid + 1, "b": it["b"]}] if sid < it["b"] else []
            items[k:k + 1] = left + right
        else:
            edge = op.get("edge")
            if edge not in ("start", "end"):
                raise ValueError("say whether to add the line before or after")
            nid = it["a"] - 1 if edge == "start" else it["b"] + 1
            if nid not in by:
                raise ValueError("there is no line there")
            if nid in used_by_others(k):
                raise ValueError("that line is already used in another passage")
            it["a" if edge == "start" else "b"] = nid
        return items
    if kind == "add":
        a, b, n = op.get("first"), op.get("last"), op.get("step")
        if n is None or n < 1 or (steps is not None and n > steps):
            raise ValueError("there is no such step")
        if a is None or b is None or a > b or any(i not in by for i in range(a, b + 1)):
            raise ValueError("those lines do not exist")
        if any(i in used_by_others(None) for i in range(a, b + 1)):
            raise ValueError("some of those lines are already used")
        at = max([k for k, it in enumerate(items) if it.get("step") == n] or [max([k for k, it in enumerate(items) if (it.get("step") or 0) < n] or [-1])])
        items.insert(at + 1, {"kind": "talk", "a": a, "b": b, "step": n, "added": True, "by_you": True})
        return items
    raise ValueError("unknown change")


CAND_SCHEMA = {"type": "object", "properties": {"passages": {"type": "array", "items": {"type": "object", "properties": {
    "first": {"type": "integer"}, "last": {"type": "integer"}}, "required": ["first", "last"]}}}, "required": ["passages"]}


def find_candidates(ai, sents: list[M.Sentence], items: list[dict], step_text: str, query: str, limit: int = 3) -> list[dict]:
    """The creator describes a line they want ("the part about the recliner"): up to `limit` whole-thought passages of the guest's words that tell
    it, from the sentences not used yet. -> [{first, last, lines}] (best first)."""
    by = {s.idx: s for s in sents}
    used = {i for it in items if it.get("kind", "talk") == "talk" for i in range(it["a"], it["b"] + 1)}
    lines = [f"{s.idx}|{s.end - s.start:.0f}s|{s.text[:170]}" for s in sents if s.idx not in used]
    prompt = (f"The video step: {step_text}\nThe creator is looking for: {query[:300]}\n\nTranscript of the guest, sentences not used yet "
              f"(number|length|text):\n" + "\n".join(lines[:2500])
              + f"\n\nGive up to {limit} different passages that tell what the creator is looking for, best first. A passage is the first and last sentence "
                "number of one whole thought (1 to 5 sentences) that starts and ends cleanly. Only passages that really tell it; none if nothing does.")
    res = ai.complete_json("flow_find", "You search an interview transcript for what the creator describes.", prompt, CAND_SCHEMA)
    out = []
    for p in res.data.get("passages") or []:
        a, b = int(p.get("first", -1)), int(p.get("last", -1))
        if a > b:
            a, b = b, a
        if b - a > 6 or any(i not in by or i in used for i in range(a, b + 1)):
            continue
        out.append({"first": a, "last": b, "lines": [{"sentence": i, "text": by[i].text, "start": round(by[i].start, 1), "end": round(by[i].end, 1)}
                                                   for i in range(a, b + 1)]})
        if len(out) >= limit:
            break
    return out


def final_check(ai, rendered_sents: list[M.Sentence], outline: list[str]) -> list[dict]:
    """Strict post-render audit. Each step is judged on what the script WROTE for it, not on details the reviewer imagines: names, numbers and quotes
    are looked for word for word in the finished render; the step's other points go to the reviewer, who must quote the render as proof and may only
    call something missing if it is on that step's checklist. Saying the same thing in other words counts as telling it."""
    transcript = "\n".join(f"[{s.idx}] {s.text}" for s in rendered_sents)
    heard_all = " ".join(s.text for s in rendered_sents)
    must = requirements(outline, heard_all)
    script = "\n".join(f"{n}. {step}" for n, step in enumerate(outline, 1))
    lists = []
    for n in range(1, len(outline) + 1):
        words = [m for m, lit in must.get(n, []) if lit]
        ideas = [m for m, lit in must.get(n, []) if not lit]
        lists.append(f"step {n}: the step's own point" + (f"; also: {'; '.join(ideas)}" if ideas else "") + (f" (names/numbers already checked by machine: {', '.join(words)})" if words else ""))
    prompt = ("SCRIPT:\n" + script + "\n\nCHECKLIST PER STEP (judge ONLY these):\n" + "\n".join(lists)
              + "\n\nTRANSCRIPT OF THE FINISHED RENDER, IN PLAYBACK ORDER:\n" + transcript +
              "\n\nFor every script step, say whether the render tells that step's point, in ANY words (a paraphrase counts; the speaker need not use the script's "
              "wording). Quote the exact words from the render that prove it, verbatim, in 'quotes'. Put an item in 'missing' ONLY when it is on that step's "
              "checklist and nothing in the render tells it - copy the checklist wording. Never invent extra details, ages, names or phrases the script did not "
              "write, and do not list names or numbers (checked separately). Never use words not present in the finished-render transcript.")
    res = ai.complete_json("flow_final", "You are a fair but evidence-based final video verifier. The script's own words set the standard, nothing more.", prompt, FINAL_SCHEMA)
    by_n = {int(x.get("n", -1)): x for x in res.data.get("steps") or []}
    out = []
    for n, step in enumerate(outline, 1):
        row = by_n.get(n, {})
        quotes = [str(x).strip(" \"'\u201c\u201d") for x in row.get("quotes") or []]
        proved = [q for q in quotes if len(_toks(q)) >= 3 and heard(q, heard_all)]        # a quote counts only if those words really are in the render
        if len(proved) < len(quotes):
            log.warning(f"final check: step {n} cited {len(quotes) - len(proved)} quote(s) that are not in the finished video")
        listed = [m for m, lit in must.get(n, []) if not lit]
        said = " ".join(proved).lower()

        def on_list(m: str) -> bool:
            m = m.lower()
            return any(m in x.lower() or x.lower() in m for x in listed) or (not listed and bool(proved) is False)
        missing = [str(x) for x in row.get("missing") or [] if on_list(str(x)) and str(x).lower() not in said]     # reviewer's inventions are dropped
        missing += [m for m, lit in must.get(n, []) if lit and not heard(m, heard_all)]                       # names / numbers / quotes: word for word
        if not proved and not missing:
            missing = ["no proof for this step could be quoted from the finished video"]
        out.append({"n": n, "step": step, "missing": missing, "quotes": proved, "rendered": True})
    return out
