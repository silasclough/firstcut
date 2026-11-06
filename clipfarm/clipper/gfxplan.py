"""Which graphics a video gets, and when - decided from what is SAID.

The editing AI reads the transcript of the finished cut (with its timeline seconds) and proposes slides, charts, stat call-outs, pull quotes
and lower thirds. Nothing is invented: every number, name and quote must really be in the transcript, and what cannot be proved is dropped
here in code (not left to the AI's good will). The density, the kinds and the restraint come from the kind of video (profiles).

    plan(ai, sents, brief, profile, out_len, hook_end) -> [{"kind", "at", "len", ...data}]   sorted, spaced, checked
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from ..core.log import get_logger
from . import flow as FL
from . import moments as M

log = get_logger("clipper.gfxplan")

SCHEMA = {"type": "object", "properties": {"graphics": {"type": "array", "items": {"type": "object", "properties": {
    "kind": {"type": "string", "enum": ["slide", "chart", "stat", "quote", "lower", "entity", "chips", "compare", "timeline", "hub", "statement", "fullslide", "bigstat", "fullquote", "chapter", "search", "twocol", "ledger", "recap", "animation", "drawchart", "count", "kinetic"]},
    "concept": {"type": "string"},
    "emphasis": {"type": "string"}, "number": {"type": "string"}, "answer": {"type": "string"},
    "titles": {"type": "array", "items": {"type": "string"}}, "steps": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
    "tag": {"type": "object", "properties": {"side": {"type": "integer"}, "text": {"type": "string"}}},
    "rows": {"type": "array", "items": {"type": "object", "properties": {"label": {"type": "string"}, "value": {"type": "string"}}, "required": ["label", "value"]}},
    "total": {"type": "object", "properties": {"label": {"type": "string"}, "value": {"type": "string"}}},
    "events": {"type": "array", "items": {"type": "object", "properties": {"date": {"type": "string"}, "label": {"type": "string"}}, "required": ["date", "label"]}},
    "joiner": {"type": "string", "enum": ["vs", "none"]},
    "items": {"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string"}, "type": {"type": "string", "enum": ["country", "company", "product", "person", "place", "thing"]},
        "domain": {"type": "string"}, "at": {"type": "number"}}, "required": ["name", "type", "at"]}},
    "at": {"type": "number"},
    "title": {"type": "string"}, "bullets": {"type": "array", "items": {"type": "string"}},
    "chart_type": {"type": "string", "enum": ["bar", "column", "line", "donut"]},
    "labels": {"type": "array", "items": {"type": "string"}}, "values": {"type": "array", "items": {"type": "string"}}, "unit": {"type": "string"},
    "value": {"type": "string"}, "label": {"type": "string"},
    "text": {"type": "string"}, "who": {"type": "string"},
    "name": {"type": "string"}, "role": {"type": "string"}},
    "required": ["kind", "at"]}}}, "required": ["graphics"]}

SYSTEM = (
    "You are the motion-graphics editor of a video. You get the transcript of the FINISHED video with timeline seconds. Choose a few on-screen "
    "graphics that make what is said clearer or stronger, placed where it is said. Kinds: "
    "'slide' (a title and 3-5 short bullets, when the speaker lists steps, reasons, features, benefits or a framework - the bullets are 2-6 words "
    "each, taken from what is said); "
    "'chart' (ONLY when the speaker states two or more comparable numbers - chart_type bar/column/line/donut, labels and values exactly as said, a "
    "title, a unit like % or $ if said); "
    "'stat' (one striking number the speaker says - 'value' like '22%' or '$1.2M' or '3x', and 'label' saying what it is); "
    "'statement' (an EMPHASIS CARD: the picture is replaced by a quiet page with one sentence appearing word by word as it is spoken - use it for the one or two sentences that "
    "carry the whole point or a surprising claim, 6-16 words copied word for word in 'text', and 'emphasis' = the 2-5 key words in it to underline; never for a list or a number); "
    "FULL-SCREEN cards take over the whole picture with a transition (the voice carries on): 'fullslide' (like slide: title + 3-5 bullets, for a list that matters), 'bigstat' (like stat, the one number that matters most), 'fullquote' (like quote, a line worth a page of its own), 'chapter' (a heading for a new part of the video: 'text' = 2-6 word heading, 'number' = '01', '02'... only for videos longer than a minute). Use a full-screen card for the strongest moments, roughly one every 20-30 seconds, and the small cards in between. "
    "MORE FULL-SCREEN cards: 'search' (a question the speaker poses, typed into a search box: 'text' = the question in the speaker's words, e.g. 'What is MIPAP?', and 'answer' = the 1-5 word answer if it is said right after; use it to open a part of the video); "
    "'twocol' (two paths or people compared step by step, e.g. Buyer A vs Buyer B: 'titles' = two short names, 'steps' = two lists of 2-4 short steps taken from what is said, in the order said, 'tag' = {side 0 or 1, text} for a label like 'Ruled out' or 'Wins' on one side if the speaker says so, 'title' = a one-line heading); "
    "'ledger' (a sum: 'rows' = 2-5 {label, value} items whose amounts are SAID, 'total' = {label, value} if the total is said, 'title' optional); "
    "'recap' (the bottom line: 'bullets' = 2-4 short points the speaker states near the END of the video, 'title' optional). "
    "'quote' (a short memorable line the speaker says, copied word for word, at most 16 words; who = the speaker's name if it is said); "
    "'lower' (a name and role, ONLY when someone says who they are, or is introduced - name and role as said). "
    "'chips' (2-6 short facts as small pills when the speaker rattles off specs or attributes, e.g. 'M5 PRO', '15-CORE CPU': put them in 'bullets', 1-4 words each, "
    "as said); "
    "'compare' (two numbers set against each other, like a price range or before and after: 'labels' = the two names, 'values' = the two numbers exactly as said, "
    "e.g. ['$1,699', '$2,699'], 'unit' if said; 'title' optional); "
    "'timeline' (2-4 dated events the speaker walks through in order: 'events' = [{date as said, label of 2-6 words}]); "
    "'hub' (one main company/product/place with 2-5 others connected to it, e.g. a company and its partners: 'items' = the main one FIRST, then the others, "
    "same fields as entity); "
    "'entity' (show the REAL picture of something NAMED, the moment it is named: a country -> its flag, a company / product / app -> its logo, a person, "
    "a place or a famous object -> a photo. items = [{name exactly as said, type country|company|product|person|place|thing, domain (a company's website like "
    "openai.com, only if you are sure), at = the timeline second it is said}]; set joiner 'vs' when exactly two are being compared against each other. Group names said within a few seconds of each other in ONE entity "
    "graphic (up to 5), e.g. 'OpenAI, Anthropic and Google' = three logos that pop up one by one. Use it for the specific named things, not for vague "
    "words like 'a company' or 'the country'; do not use the speaker's own name here (that is a lower third)."
    "'animation' (a CUSTOM ANIMATED DIAGRAM drawn just for this moment, taking over the picture for 7-12 s while the voice carries on: use it ONLY when the speaker EXPLAINS how "
    "something works or happens - a mechanism, a process, a structure, how two things relate or differ - that a moving diagram would make clear. 'concept' = ONE sentence saying exactly "
    "what is drawn and how it moves, using only things the speaker says (e.g. 'a selector fork slides a clutch sleeve sideways until its teeth lock into a gear'). Never for a list "
    "(that is a slide), a number or a person. 'at' = the second the explanation begins).\n"
    "ANIMATED full-screen cards (drawn on this Mac, no extra software): 'drawchart' (like chart, but a chart that DRAWS ITSELF on a page of its "
    "own - use it instead of chart when the speaker walks through 3 or more numbers that change over time or rank something: chart_type line for over time, bar "
    "for a ranking; labels and values exactly as said); 'count' (like stat, but the number counts up on a page of its own; a percentage fills a ring - use it for "
    "the ONE number that is the point of a part of the video); 'kinetic' (like statement, but punchier: a short line of 5-12 words copied word for word that "
    "lands word by word with a spring, the key words in the accent colour - use it for a punchline or a bold claim; 'emphasis' = the 1-3 key words).\n"
    "Rules: use ONLY facts that are in the transcript; never invent or round a number, a name or a quote; if nothing in the transcript fits a kind, "
    "do not use it. 'at' is the timeline second where the graphic should appear, shortly after the statement it illustrates begins. Spread them out "
    "over the video; never in the first few seconds (the title card is there). Prefer FEWER, better graphics to many.")


FULL_OF = {"fullslide": "slide", "bigstat": "stat", "fullquote": "quote", "drawchart": "chart", "count": "stat", "kinetic": "statement"}                  # (a full-screen card is checked exactly like its small cousin)
FULL = ("fullslide", "bigstat", "fullquote", "chapter", "statement", "search", "twocol", "ledger", "recap", "animation", "drawchart", "count", "kinetic")        # cards that take over the whole picture


def _norm(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(s).lower().replace("’", "'"))


_WORD_NUMS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
              "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
              "seventy": 70, "eighty": 80, "ninety": 90}


def _digits_of(num: str) -> str:
    """The digits in a figure; for one written in words ('six hours', 'forty-seven percent') the number those words mean."""
    raw = re.sub(r"[^\d.]", "", str(num))
    if raw:
        return raw
    parts = [p for p in re.findall(r"[a-z]+", str(num).lower()) if p in _WORD_NUMS]
    return str(sum(_WORD_NUMS[p] for p in parts[:2])) if parts else ""


def _said_number(num: str, spoken: str) -> bool:
    raw = _digits_of(num)
    if not raw:
        return False
    spoken = re.sub(r"(?<=\d),(?=\d)", "", spoken)                                        # "$1,699" is heard as 1699
    if "." not in raw:
        return FL.heard(raw, spoken) or bool(re.search(rf"(?<![\d.]){re.escape(raw)}(?!\d)", spoken))     # ("3x more": the 3 is glued to its x)
    whole, frac = raw.split(".", 1)
    return (raw in spoken.replace(",", "")) or (FL.heard(whole, spoken) and "point" in spoken.lower())


def _content(words: list[str]) -> list[str]:
    stop = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "was", "it", "that", "this", "we", "you", "i", "our", "your", "my"}
    return [w for w in words if w not in stop and len(w) > 2]


def _row_times(items: list[str], words: list[dict] | None, start: float, horizon: float = 45.0) -> list[float] | None:
    """When each item is SAID: the first distinctive word of it, moving strictly forward through the speech (two items that share a word do not get the same moment).
    Seconds from `start`; None if the words are not there."""
    if not words:
        return None
    flat = []
    for w in words:
        try:
            ts = float(w["start"])
        except (KeyError, TypeError, ValueError):
            continue
        if start - 1.0 <= ts <= start + horizon:
            flat.extend((t, ts) for t in _norm(w.get("text", "")))
    out, pos = [], 0
    for it in items:
        toks = sorted(set(_content(_norm(it))), key=len, reverse=True)[:4]
        j = next((i for i in range(pos, len(flat)) if flat[i][0] in toks or flat[i][0].rstrip("s") in toks), None)
        if j is None:
            return None
        out.append(round(max(0.5, flat[j][1] - start + 0.1), 2))
        pos = j + 1                                                                      # the next item is looked for AFTER this word
    return out


def validate(item: dict, spoken: str, sents: list[M.Sentence], out_len: float, hook_end: float, words: list[dict] | None = None) -> dict | None:
    """Wrapper: a wrongly-shaped item (the AI returns a string where a list belongs) is simply not valid, never an error."""
    try:
        return _validate(item, spoken, sents, out_len, hook_end, words)
    except (TypeError, KeyError, AttributeError, ValueError, IndexError):
        return None


def _validate(item: dict, spoken: str, sents: list[M.Sentence], out_len: float, hook_end: float, words: list[dict] | None = None) -> dict | None:
    """The checked, completed item (with 'len' and its data in the shape gfxcards wants) - or None if it cannot be proved from the transcript."""
    kind = item.get("kind")
    if kind in FULL_OF:                                                                  # a full-screen card is checked exactly like its small cousin
        v = _validate({**item, "kind": FULL_OF[kind]}, spoken, sents, out_len, hook_end, words)
        if v is None:
            return None
        v["kind"] = kind
        if kind == "kinetic" and len(str(v.get("text") or "").split()) > 12:
            v["kind"] = kind = "statement"                                                # (too long to land word by word: the quieter card)
        if kind == "drawchart":
            v["len"] = 7.5                                                                # (it draws itself, then rests)
        room = out_len - v["at"] - 0.6
        v["len"] = round(min(max(v["len"], 3.6 if kind == "bigstat" else 4.2 if kind == "count" else v["len"]), room), 2)          # (a big number gets at least 3.6 s - but never runs past the end of the video)
        if v["len"] < 2.4:
            return None
        return v
    try:
        at = float(item["at"])
    except (KeyError, TypeError, ValueError):
        return None
    at = max(at, hook_end + 0.2)
    if at > out_len - 3.0:
        return None
    if not any(s.start - 0.3 <= at <= s.end + 0.3 for s in sents):                       # a graphic starts while someone is speaking: snap to the nearest sentence
        near = min(sents, key=lambda s: abs(s.start - at), default=None)
        if near is None or abs(near.start - at) > 3.0:
            return None
        at = near.start + 0.2
    window = " ".join(s.text for s in sents if s.end >= at - 4 and s.start <= at + 30)       # what is said around then
    out: dict = {"kind": kind, "at": round(at, 2)}
    if kind == "stat":
        value, label = str(item.get("value") or "").strip(), " ".join(str(item.get("label") or "").split())
        mv = re.match(r"^\s*([$€£]?\s*\d[\d,.]*\s*(?:%|[kKmMbB]\b|x\b|\+)?)\s*(.*)$", value)                       # the big figure is JUST the number ("10"), the words go in the label
        if mv:
            value, label = mv.group(1).strip(), " ".join(f"{mv.group(2)} {label}".split())
        if not value or len(value) > 9 or not label or not _said_number(value, spoken):
            return None
        out.update({"value": value, "label": label, "len": 4.2})
        if words:                                                                        # the panel pops up on the very word that says the number
            hit = _find_said(re.sub(r"[^\d]", "", value), words, at)
            if hit is not None and abs(hit - at) <= 8.0:
                out["at"] = round(max(hook_end + 0.2, hit - 0.25), 2)
    elif kind == "quote":
        text = " ".join(str(item.get("text") or "").split())
        w = _norm(text)
        if not (4 <= len(w) <= 18):
            return None
        if " ".join(w) not in " ".join(_norm(spoken)) and max((SequenceMatcher(None, w, _norm(s.text)).find_longest_match(0, len(w), 0, len(_norm(s.text))).size
                                                              for s in sents), default=0) < 0.85 * len(w):
            return None                                                                   # not said (word for word): no quote card
        out.update({"text": text, "who": " ".join(str(item.get("who") or "").split()), "len": min(7.5, 2.6 + 0.34 * len(w))})
        if words:                                                                          # the card comes up as the speaker BEGINS the quoted words, not when the sentence around it starts
            for k in (0, 1, 2):
                t = _find_said(" ".join(w[k:k + 4]), words, at) if len(w) >= k + 4 else None
                if t is not None:
                    out["at"] = round(max(hook_end + 0.2, t - 0.1 - 0.3 * k), 2)
                    break
    elif kind == "animation":
        concept = " ".join(str(item.get("concept") or "").split())[:240]
        if len(concept.split()) < 6:
            return None
        said = set(_content(_norm(window)))
        mine = [w for w in _content(_norm(concept)) if len(w) > 3]
        if len(mine) < 2 or sum(1 for w in mine if w in said or w.rstrip("s") in said) < max(2, int(0.4 * len(mine))):
            return None                                                                  # the diagram is about something that is not being said here
        span = [x for x in sents if x.end > at and x.start < at + 14.0]
        if not span:
            return None
        ln = round(min(12.0, max(7.0, min(span[-1].end, at + 12.0) - at + 0.4), out_len - at - 0.8), 2)
        if ln < 6.0:
            return None
        narr = " ".join(x.text for x in sents if x.end > at and x.start < at + ln)
        timed = [{"text": str(w.get("text", "")), "start": round(float(w["start"]) - at, 3)} for w in (words or []) if at - 0.01 <= float(w.get("start", -1)) <= at + ln]
        out.update({"concept": concept, "len": ln, "narration": narr[:700], "timed": timed[:200],
                    "context": " ".join(x.text for x in sents if abs(x.start - at) < 45)[:4000]})     # (what the diagram may draw from: only the speaker's own words)
    elif kind == "statement":
        text = " ".join(str(item.get("text") or "").split())
        w = _norm(text)
        flat = [(t, float(x["start"]), float(x["end"])) for x in (words or []) for t in _norm(x.get("text", ""))]
        tl = [f[0] for f in flat]
        pos = next((i for i in range(len(tl) - len(w) + 1) if tl[i:i + len(w)] == w and abs(flat[i][1] - at) <= 12.0), None) if 5 <= len(w) <= 22 else None
        if pos is None:
            return None                                                                  # not said word for word, or no word timing: no emphasis card
        st, en = flat[pos][1], flat[pos + len(w) - 1][2]
        start = max(hook_end + 0.2, st - 0.3)
        times, k = [], pos
        for dw in text.split():                                                          # each displayed word appears when it is spoken
            times.append(round(max(0.0, flat[min(k, len(flat) - 1)][1] - start), 3))
            k += max(1, len(_norm(dw)))
        emph = " ".join(str(item.get("emphasis") or "").split())
        if not emph or " ".join(_norm(emph)) not in " ".join(w):
            emph = " ".join(text.split()[-3:])                                           # (nothing sensible chosen: underline the last few words)
        out.update({"text": text, "times": times, "emphasis": emph, "at": round(start, 2), "len": round(en - start + 1.4, 2)})
    elif kind == "recap":
        if at < out_len * 0.55:
            return None                                                                  # the bottom line belongs at the end
        v = validate({**item, "kind": "slide", "title": item.get("title") or "The bottom line", "bullets": (item.get("bullets") or []) + [""] * 0}, spoken, sents, out_len, hook_end, words) \
            if len(item.get("bullets") or []) >= 3 else None
        if v is None:
            bl = [" ".join(str(b).split()) for b in (item.get("bullets") or [])]
            ww = set(_norm(window))
            good = [b for b in bl if (cw := _content(_norm(b))) and sum(1 for x in cw if x in ww or x.rstrip("s") in ww) >= 0.6 * len(cw)]
            if len(good) < 2:
                return None
            tm = _row_times(good[:4], words, at)
            out.update({"title": " ".join(str(item.get("title") or "The bottom line").split())[:40], "bullets": good[:4], "len": round(min(10.0, 2.6 + 1.2 * len(good[:4])), 2)})
            if tm:
                out["times"], out["len"] = tm, round(min(14.0, max(out["len"], tm[-1] + 2.4)), 2)
        else:
            out.update({k: v[k] for k in ("title", "bullets", "len", "times") if k in v})
    elif kind == "search":
        q = " ".join(str(item.get("text") or "").split())
        cw = _content(_norm(q))
        sw = set(_norm(spoken))
        if not 2 <= len(q.split()) <= 12 or not cw or sum(1 for x in cw if x in sw or x.rstrip("s") in sw) < 0.6 * len(cw):
            return None
        ans = " ".join(str(item.get("answer") or "").split())
        aw = _content(_norm(ans))
        ww = set(_norm(window))
        if ans and (len(ans.split()) > 6 or not aw or sum(1 for x in aw if x in ww or x.rstrip("s") in ww) < 0.7 * len(aw)):
            ans = ""
        out.update({"text": q if q.endswith("?") else q + "?", "answer": ans, "number": re.sub(r"[^\d]", "", str(item.get("number") or ""))[:2].zfill(2) if re.search(r"\d", str(item.get("number") or "")) else "", "len": 4.2 + (1.4 if ans else 0.0)})
    elif kind == "twocol":
        ww = set(_norm(window))
        cols = []
        for col in (item.get("steps") or [])[:2]:
            good = []
            for b in col or []:
                cw = _content(_norm(b))
                if cw and sum(1 for x in cw if x in ww or x.rstrip("s") in ww) >= 0.6 * len(cw):
                    good.append(" ".join(str(b).split()))
            cols.append(good[:4])
        titles = [" ".join(str(x).split())[:26] for x in (item.get("titles") or [])][:2]
        if len(cols) < 2 or min(len(c) for c in cols) < 2 or len(titles) < 2 or not all(titles):
            return None
        tag = item.get("tag") if isinstance(item.get("tag"), dict) else {}
        flat = cols[0] + cols[1]
        tm = _row_times(flat, words, at)
        tg = {"side": int(tag["side"]) if str(tag.get("side", "")).lstrip("-").isdigit() and int(tag["side"]) in (0, 1) else -1, "text": " ".join(str(tag.get("text") or "").split())[:22]} if tag else {}
        if tg and (tg["side"] < 0 or not tg["text"]):
            tg = {}
        out.update({"title": " ".join(str(item.get("title") or "").split())[:60], "titles": titles, "steps": cols, "tag": tg, "len": round(min(16.0, 3.4 + 0.9 * len(flat)), 2)})
        if tm and tm[-1] + 3.4 <= 18.0:                                                  # (a speaker so slow that the last step is >15 s in: spread them evenly instead of losing them)
            out["times"] = tm + ([round(tm[-1] + 0.9, 2)] if tg else [])
            out["len"] = round(min(18.0, max(out["len"], out["times"][-1] + 2.4)), 2)
    elif kind == "ledger":
        ww = set(_norm(window))
        rows = []
        for r in item.get("rows") or []:
            if not isinstance(r, dict):
                continue
            lb, val = " ".join(str(r.get("label") or "").split()), " ".join(str(r.get("value") or "").split())
            cw = _content(_norm(lb))
            if not (lb and val and cw and _said_number(val, spoken) and sum(1 for x in cw if x in ww or x.rstrip("s") in ww) >= 0.5 * len(cw)):
                continue
            clauses = re.split(r",(?!\d)|;|\band\b|\bso\b|\bbut\b|\.(?!\d)", window)                           # the item and its amount must be said together (not "price $8,000, fees $400,000" swapped)
            if any(sum(1 for x in cw if x in set(_norm(c_)) or x.rstrip("s") in set(_norm(c_))) >= 0.5 * len(cw) and _said_number(val, c_) for c_ in clauses):
                rows.append({"label": lb[:34], "value": val[:14]})
        if len(rows) < 2:
            return None
        tot = item.get("total") if isinstance(item.get("total"), dict) else {}
        total = {"label": " ".join(str(tot.get("label") or "Total").split())[:24], "value": " ".join(str(tot.get("value") or "").split())[:14]} if tot.get("value") and _said_number(str(tot["value"]), spoken) else {}
        rows = rows[:5]
        tm = _row_times([r["label"] for r in rows], words, at)
        out.update({"title": " ".join(str(item.get("title") or "").split())[:60], "rows": rows, "total": total, "len": round(min(14.0, 3.2 + 1.0 * (len(rows) + (1 if total else 0))), 2)})
        if tm and tm[-1] + 3.4 <= 16.0:
            out["times"] = tm + ([round(tm[-1] + 1.0, 2)] if total else [])
            out["len"] = round(min(16.0, max(out["len"], out["times"][-1] + 2.4)), 2)
    elif kind == "chapter":
        if out_len < 60:
            return None                                                                  # a heading between parts only makes sense in a longer video
        text = " ".join(str(item.get("text") or item.get("title") or "").split())
        cw = _content(_norm(text))
        spoken_words = set(_norm(spoken))
        if not 2 <= len(text.split()) <= 7 or not cw or sum(1 for x in cw if x in spoken_words or x.rstrip("s") in spoken_words) < 0.5 * len(cw):
            return None                                                                  # a heading must be about what is said
        out.update({"text": text, "number": re.sub(r"[^\d]", "", str(item.get("number") or ""))[:2].zfill(2) if re.search(r"\d", str(item.get("number") or "")) else "", "len": 3.4})
    elif kind == "lower":
        name = " ".join(str(item.get("name") or "").split())
        parts = _norm(name)
        if not parts or not all(p in _norm(spoken) for p in parts[:1] + parts[-1:]):
            return None
        out.update({"name": name, "role": " ".join(str(item.get("role") or "").split()), "len": 4.5})
    elif kind == "slide":
        bullets = []
        window_words = set(_norm(window))
        for b in item.get("bullets") or []:
            cw = _content(_norm(b))
            if cw and sum(1 for x in cw if x in window_words or x.rstrip("s") in window_words) >= 0.6 * len(cw):
                bullets.append(" ".join(str(b).split()))
        if len(bullets) < 3 or not str(item.get("title") or "").strip():
            return None
        out.update({"title": " ".join(str(item["title"]).split()), "bullets": bullets[:5], "len": min(9.0, 2.8 + 1.25 * len(bullets[:5]))})
        tm = _row_times(bullets[:5], words, at)
        if tm and tm[-1] + 2.4 <= 14.0:
            out["times"] = tm                                                            # each point lights up as it is said
            out["len"] = round(min(14.0, max(out["len"], tm[-1] + 2.4)), 2)
    elif kind == "chart":
        labels = [" ".join(str(x).split()) for x in item.get("labels") or []]
        values = [str(x) for x in item.get("values") or []]
        pairs = [(lb, v) for lb, v in zip(labels, values) if lb and _said_number(v, spoken)]
        if len(pairs) < 2:
            return None
        ctype = item.get("chart_type") if item.get("chart_type") in ("bar", "column", "line", "donut") else "bar"
        out.update({"chart": {"type": ctype, "title": " ".join(str(item.get("title") or "").split()), "labels": [p[0] for p in pairs],
                              "values": [p[1] for p in pairs], "unit": str(item.get("unit") or "")[:4]}, "len": 7.0})
    elif kind == "chips":
        window_words = set(_norm(window))
        chips = []
        for b in item.get("bullets") or []:
            cw = _content(_norm(b))
            if cw and sum(1 for x in cw if x in window_words or x.rstrip("s") in window_words) >= 0.6 * len(cw):
                chips.append(" ".join(str(b).split()))
        if len(chips) < 2:
            return None
        out.update({"chips": chips[:6], "len": min(9.0, 2.6 + 0.8 * len(chips[:6]))})
    elif kind == "compare":
        labels = [" ".join(str(x).split()) for x in (item.get("labels") or [])][:2]
        values = [str(x).strip() for x in (item.get("values") or [])][:2]
        if len(labels) < 2 or len(values) < 2 or not all(l for l in labels) or not all(_said_number(v, spoken) for v in values):
            return None
        out.update({"labels": labels, "values": values, "unit": str(item.get("unit") or "")[:4], "title": " ".join(str(item.get("title") or "").split()), "len": 6.5})
    elif kind == "timeline":
        window_words = set(_norm(window)) | set(_norm(spoken))
        events = []
        for e in item.get("events") or []:
            date, label = " ".join(str(e.get("date", "")).split()), " ".join(str(e.get("label", "")).split())
            dt = _norm(date)
            nums = [t for t in dt if t.isdigit()]
            if not dt or not label or not all(_said_number(n_, spoken) for n_ in nums) or not all(t in window_words for t in dt if not t.isdigit()):
                continue
            cw = _content(_norm(label))
            if cw and sum(1 for x in cw if x in window_words or x.rstrip("s") in window_words) >= 0.6 * len(cw):
                events.append({"date": date, "label": label})
        if len(events) < 2:
            return None
        out.update({"events": events[:4], "title": " ".join(str(item.get("title") or "").split()), "len": min(11.0, 3.2 + 1.2 * len(events[:4]))})
    elif kind in ("entity", "hub"):
        return None                                                                       # (built by entity_card: it needs the picture lookup)
    else:
        return None
    # a line meant to be read stays up about 0.3 s a word once it has settled (the /brag rule); never shorter than what we already planned
    nwords = sum(len(str(v).split()) for k, v in out.items() if k in ("text", "quote", "title", "label", "who", "role", "sub") and isinstance(v, str))
    nwords += sum(len(str(b).split()) for k in ("bullets", "chips") for b in (out.get(k) or []))
    out["len"] = max(out["len"], 1.0 + 0.3 * nwords) if kind not in ("stat",) else out["len"]
    out["len"] = round(min(out["len"], 18.0 if out.get("times") else 12.0, out_len - out["at"] - 0.6), 2)
    return out if out["len"] >= 2.4 else None


def _find_said(name: str, words: list[dict], near: float) -> float | None:
    """The time (s) the speaker says `name` (its words in a row), the occurrence closest to `near` and within 8 s of it."""
    toks = _norm(name)
    if not toks:
        return None
    flat = [(t, float(w["start"])) for w in words for t in _norm(w.get("text", ""))]
    best = None
    for i in range(len(flat) - len(toks) + 1):
        if [flat[i + k][0] for k in range(len(toks))] == toks:
            d = abs(flat[i][1] - near)
            if d <= 8.0 and (best is None or d < best[0]):
                best = (d, flat[i][1])
    return best[1] if best else None


def entity_card(cfg, item: dict, spoken: str, words: list[dict], sents: list[M.Sentence], out_len: float, hook_end: float) -> dict | None:
    """One entity graphic from the AI's list of named things: every name must really be said, every picture must be found and free to use."""
    from . import entities as EN
    said = " ".join(_norm(spoken))
    found = []
    for it in item.get("items") or []:
        name = " ".join(str(it.get("name") or "").split())
        toks = _norm(name)
        if not toks or " ".join(toks) not in said:
            continue                                                                       # never said: no picture
        try:
            hint = float(it.get("at", 0))
        except (TypeError, ValueError):
            continue
        t = _find_said(name, words, hint)
        if t is None:
            t = hint if any(s.start - 0.3 <= hint <= s.end + 0.3 for s in sents) else None
        if t is None:
            continue
        pic = EN.resolve(cfg, {"name": name, "type": it.get("type"), "domain": it.get("domain")})
        if not pic:
            continue
        found.append({**pic, "name": name, "abs": t})
    if not found:
        return None
    hub = item.get("kind") == "hub"
    if not hub:
        found.sort(key=lambda f: f["abs"])
    group = [f for f in found if f["abs"] - found[0]["abs"] <= (14.0 if hub else 9.0)][:6 if hub else 5]                # names said close together share one graphic
    if hub and (len(group) < 3 or group[0]["name"] != " ".join(str(item["items"][0].get("name", "")).split())):
        return None                                                                       # a hub needs its centre and at least two others
    at = max(hook_end + 0.1, min(f["abs"] for f in group) - 0.35)
    if at > out_len - 3.5:
        return None
    for f in group:
        f["at_rel"] = round(max(0.0, f["abs"] - at), 2)
    ln = min(12.0, max(f["at_rel"] for f in group) + (3.8 if hub else 3.0), out_len - at - 0.6)
    if ln < 2.4:
        return None
    for f in group:
        f["at_rel"] = round(max(0.0, f["abs"] - at), 2)
    if hub:
        group[0]["at_rel"] = 0.0
    joiner = "vs" if (item.get("joiner") == "vs" and len(group) == 2 and not hub) else None
    return {"kind": "hub" if hub else "entity", "at": round(at, 2), "len": round(ln, 2), **({"joiner": joiner} if joiner else {}), "items": [
        {k: f.get(k) for k in ("name", "type", "file", "color", "mark", "credit", "at_rel")} for f in group]}


TRANSITION_NAMES = ("wipe", "iris", "push", "zoom")
def _at_of(x: dict) -> float:
    try:
        return float(x.get("at", 0) or 0)
    except (TypeError, ValueError):
        return 1e9                                                                     # a garbage time sorts last and is dropped by validate


GAP = 0.6                                              # seconds between two graphics


def _min_len(v: dict) -> float:
    return v["len"] if v["kind"] in ("statement", "quote") else max(2.4, 0.6 * v["len"])        # (words that are timed to speech keep their length)


def schedule(cards: list[dict], out_len: float) -> list[dict]:
    """Time order, never overlapping. When two collide the earlier one is shown a little shorter, or the later one starts a little later (not the speech-timed
    cards): a graphic is only dropped when there is truly no room, instead of whenever the speaker is quick."""
    out: list[dict] = []
    for v in sorted((dict(c) for c in cards), key=lambda x: x["at"]):
        if out:
            prev = out[-1]
            need = prev["at"] + prev["len"] + GAP - v["at"]
            if need > 0:
                take = min(max(0.0, prev["len"] - _min_len(prev)), need)
                prev["len"] = round(prev["len"] - take, 2)
                need -= take
            if need > 0 and v["kind"] not in ("statement", "quote") and need <= 4.5 and v["at"] + need + v["len"] <= out_len - 0.6:
                v["at"] = round(v["at"] + need, 2)
                if v.get("times"):                                                       # the card starts later, so what it shows on the words comes earlier inside it
                    v["times"] = [round(max(0.3, t - need), 2) for t in v["times"]]
                need = 0
            if need > 0.01:
                log.info(f"graphics: no room for a {v['kind']} at {v['at']:.1f}s")
                continue
        out.append(v)
    return out


def plan(ai, sents: list[M.Sentence], brief: str, profile: dict, out_len: float, hook_end: float = 3.2, words: list[dict] | None = None, cfg=None) -> list[dict]:
    """-> the graphics for this video, in time order, never overlapping, at most one per `graphics_per_s` seconds on average."""
    kinds = [k for k in profile.get("kinds", []) if k in ("slide", "chart", "stat", "quote", "lower", "entity", "chips", "compare", "timeline", "hub", "statement", "fullslide", "bigstat", "fullquote", "chapter", "search", "twocol", "ledger", "recap", "animation", "drawchart", "count", "kinetic")]
    for anim, cousin in (("drawchart", "chart"), ("count", "bigstat"), ("kinetic", "statement")):     # the animated full-screen kinds (skiagfx; Pillow
        if cousin in kinds and anim not in kinds:                                               #  without it) go wherever their cousin may
            kinds.append(anim)
    if cfg is None:
        kinds = [k for k in kinds if k not in ("entity", "hub")]
    if "animation" in kinds:                                             # animated diagrams need Hyperframes (node + Chrome): never designed
        try:                                                             # by the AI on a Mac that can't render them (the app on a friend's Mac)
            from .animate_hf import cli as _hf
            if cfg is not None and _hf(cfg) is None:                      # (no config: a caller that renders elsewhere decides)
                kinds = [k for k in kinds if k != "animation"]
        except Exception:  # noqa: BLE001
            kinds = [k for k in kinds if k != "animation"]
    if not sents or not kinds or out_len < 12:
        return []
    system = SYSTEM
    cap = max(1, int(out_len / float(profile.get("graphics_per_s", 40))))
    ecap = max(1, int(out_len / float(profile.get("entities_per_s", 25)))) if ("entity" in kinds or "hub" in kinds) else 0           # pictures of named things have a budget of their own
    acap = min(3, max(1, int(out_len / 75))) if "animation" in kinds else 0                                                          # custom animated diagrams: slow to make, so few
    if re.search(r"graphics? test|test (?:of )?the graphics|test every (?:kind|type)", brief or "", re.I):                  # a test of the engine: one of each kind is wanted
        cap, ecap = max(cap, len(kinds) + 2), max(ecap, 4)
        brief = (brief or "") + " (THIS IS A TEST: use one graphic of EVERY allowed kind that the transcript can support, including a statement card, a stat, chips and a compare.)"
        system = SYSTEM.replace("Prefer FEWER, better graphics to many.", "This is a test of the graphics engine: use as many different kinds as the transcript supports, one of each.")
    lines = "\n".join(f"[{s.start:.1f}-{s.end:.1f}] {s.text}" for s in sents)
    from . import gfxdetect
    found = gfxdetect.find(sents, words or [])                                           # moments with a recognisable shape (ranges, figures, dates, lists, claims)
    hint = ("\n\nMOMENTS THAT PROBABLY WANT A GRAPHIC (found by a scan of the transcript; use the kind named, or a better one, unless it truly does not fit):\n"
            + gfxdetect.prompt_lines(found)) if found else ""
    prompt = ((f"The creator's notes: {brief}\n" if brief else "") + f"Kind of video: {profile.get('label', 'video')}. Allowed graphics: {', '.join(kinds)}. "
              f"Use at most {cap} graphics plus at most {ecap} 'entity' graphics" + (f" plus at most {acap} 'animation' diagram(s)" if acap else "") + f" ({profile.get('tone', '')}). The title card covers 0-{hook_end:.1f} s.{hint}\n\nTRANSCRIPT:\n{lines[:12000]}")
    try:
        res = ai.complete_json("graphics_plan", system, prompt, SCHEMA).data
    except Exception as e:  # noqa: BLE001 - no graphics is a fine result
        log.warning(f"planning the graphics failed: {e}")
        return []
    spoken = " ".join(s.text for s in sents)
    out: list[dict] = []
    proposed = list(res.get("graphics") or [])
    for o in found:                                                                       # a recognised range the AI left out is built by code (it needs no judgement)
        if o.get("item") and "compare" in kinds and not any(g.get("kind") in ("compare", "chart", "stat") and abs(float(g.get("at", 0) or 0) - o["at"]) < 8.0 for g in proposed):
            log.info(f"graphics: adding a compare the AI missed at {o['at']:.1f}s ({o['why']})")
            proposed.append(o["item"])
    for it in sorted((i for i in proposed if isinstance(i, dict)), key=_at_of):
        if it.get("kind") not in kinds:
            continue
        try:
            v = entity_card(cfg, it, spoken, words or [], sents, out_len, hook_end) if it.get("kind") in ("entity", "hub") else validate(it, spoken, sents, out_len, hook_end, words)
        except Exception as e:  # noqa: BLE001 - one badly-shaped item from the AI is dropped, the rest of the plan stands
            log.info(f"graphics: dropped a malformed {it.get('kind')}: {e}")
            v = None
        if v is None:
            log.info(f"graphics: dropped a {it.get('kind')} (not supported by the transcript): {str(it)[:240]}")
            continue
        out.append(v)
    final, ne, ng, na = [], 0, 0, 0
    for v in sorted(out, key=lambda x: x["at"]):                                        # keep within each budget
        if v["kind"] in ("entity", "hub"):
            if ne >= ecap:
                continue
        elif v["kind"] == "animation":
            if na >= acap:
                continue
        elif ng >= cap:
            continue
        final.append(v)
        ne += v["kind"] in ("entity", "hub")
        na += v["kind"] == "animation"
        ng += v["kind"] not in ("entity", "hub", "animation")
    cards = schedule(final, out_len)
    last_full, k = -99.0, 0
    keep = []
    for v in cards:                                                                      # a full-screen card at most every 14 s (the picture is the speaker; it is a cutaway, not the whole video), and neighbours get different transitions
        if v["kind"] in FULL:
            if v["at"] - last_full < 14.0:
                log.info(f"graphics: dropped a full-screen {v['kind']} at {v['at']:.1f}s (another is too close)")
                continue
            last_full = v["at"]
            v["style"] = TRANSITION_NAMES[k % len(TRANSITION_NAMES)]
            k += 1
        keep.append(v)
    return keep



def _shows(c: dict) -> str:
    parts = []
    for k in ("title", "text", "name", "role", "value", "label", "answer", "emphasis", "who", "concept"):
        if c.get(k):
            parts.append(str(c[k]))
    for k in ("bullets", "chips", "labels", "values", "titles"):
        if c.get(k):
            parts.append(" / ".join(str(x) for x in c[k]))
    if c.get("steps"):
        parts.append(" | ".join(" > ".join(col) for col in c["steps"]))
    if c.get("rows"):
        parts.append("; ".join(f"{r.get('label', '')} {r.get('value', '')}" for r in c["rows"] if isinstance(r, dict)) + (f" = {c['total'].get('value')}" if isinstance(c.get("total"), dict) and c["total"].get("value") else ""))
    if c.get("chart"):
        ch = c["chart"]
        parts.append(f"{ch.get('title', '')}: " + ", ".join(f"{a} {b}" for a, b in zip(ch.get("labels") or [], ch.get("values") or [])))
    if c.get("events"):
        parts.append("; ".join(f"{e['date']} {e['label']}" for e in c["events"]))
    if c.get("items"):
        parts.append(", ".join(str(i.get("name")) for i in c["items"]))
    return " - ".join(p for p in parts if p)[:220]


def check_notes(cards: list[dict], sents: list[M.Sentence]) -> str:
    """A list for a human to check: every graphic, when it appears, exactly what it shows, and the sentence that supports it (the numbers and names are verified against the transcript
    by code, but speech-to-text can mishear a number or a name, and only you know what you meant)."""
    lines = ["# Graphics to check", "", "Each graphic below was built only from words in the transcript. Check the figures and names against what you meant (speech-to-text can mishear).", ""]
    shown = 0
    for c in cards:
        try:
            at = float(c["at"])
            near = min(sents, key=lambda s: abs(s.start - at), default=None)
            t = int(at)
            body = f"- **{t // 60}:{t % 60:02d}  {c.get('kind', '?')}**: {_shows(c)}"
            said = f"  - you said: \"{' '.join(near.text.split())[:200].replace(chr(34), chr(39))}\"" if near else ""
        except Exception:  # noqa: BLE001 - a malformed card is skipped in the list, never an error
            continue
        lines.append(body.replace("\n", " "))
        if said:
            lines.append(said)
        shown += 1
    if not shown:
        lines.append("(no graphics in this video)")
    return "\n".join(lines) + "\n"
