"""Which graphic, what text, when: the part of the graphics plan that code decides from what is SAID (WP10).

The AI that plans graphics is good at choosing and bad at being exact, so the transcript is scanned in plain code first for five shapes, and each one becomes a
card that gfxplan.validate then proves word by word:

    number     "twenty-three percent", "$1.2 million", "40%"          -> stat    (digits for number words; years are never numbers; a unit is required)
    name       "I'm Sarah Miller, the director of ..."                -> lower   (first mention only)
    list       "three reasons ... first ... second ... third"         -> slide   (the bullets are the words said after each ordinal)
    quote      "Churchill said, ..."                                  -> quote   (the exact words; never paraphrased)
    reference  "John 3:16 says ..."                                   -> reference (scripture.py)

Also here, because they are about the words and the clock and not about drawing: how long a card stays (from its word count), when it enters (three frames before
the word, never after it), the budget per minute, the one-at-a-time spacing, the rule that a card never repeats the caption under it, chapter cards that sit on the
outline's steps, and the genre gate (which kinds a style allows).
"""

from __future__ import annotations

import re

from . import gfxdetect
from . import scripture

LEAD_FRAMES = 3                                  # a beat of stillness before the key word: the card is already there when it is said
FILLERS = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "hmm", "mm", "mmm", "hm"}
QUOTES_VERBATIM = True                           # style.graphics.quotes_verbatim: always on for now
QUOTE_MAX_WORDS = 21                             # 7 s at 3 words a second: a longer quote shows only the clause being spoken

GRAPHICS_KINDS_BY_POLICY = {
    "none": {"lower"},
    "title": {"chapter", "reference", "lower"},
    "informational": {"chapter", "reference", "lower", "stat", "bigstat", "count", "quote", "fullquote", "slide", "fullslide", "recap", "chips", "compare", "chart",
                      "drawchart", "timeline", "ledger"},
}                                                # "all" (and no style at all): everything the profile lists


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def lead_s(fps: float = 24.0) -> float:
    """Three frames at the timeline's rate (0.125 s at 24 fps, 0.1 s at 30)."""
    return LEAD_FRAMES / max(1.0, float(fps or 24.0))


def _words_of(text: str) -> list[str]:
    return re.findall(r"[a-z0-9$%']+", str(text or "").lower())


class _Spans(list):
    """The (start, end, word index) of every word in the flat text, with the sorted starts kept for a binary search (a 2-hour transcript is 20,000+ words)."""
    starts: list[int]


def _flat(words: list[dict]) -> tuple[str, list[tuple[int, int, int]]]:
    out, spans, pos = [], _Spans(), 0
    for i, w in enumerate(words):
        t = str(w.get("text", "") or "").strip()
        if not t:
            continue
        spans.append((pos, pos + len(t), i))
        out.append(t)
        pos += len(t) + 1
    spans.starts = [a for a, _b, _i in spans]
    return " ".join(out), spans


def _wi(spans: list[tuple[int, int, int]], char: int) -> int:
    st = getattr(spans, "starts", None)
    if st is not None:
        import bisect
        k = bisect.bisect_right(st, char) - 1
        if 0 <= k < len(spans) and char < spans[k][1] + 1:
            return spans[k][2]
        return spans[-1][2] if spans else 0
    for a, b, i in spans:
        if a <= char < b + 1:
            return i
    return spans[-1][2] if spans else 0


def _start_of(words: list[dict], i: int) -> float:
    try:
        return float(words[max(0, min(i, len(words) - 1))]["start"])
    except (KeyError, TypeError, ValueError, IndexError):
        return 0.0


def entrance(word_start: float, sents: list, hook_end: float = 0.0, fps: float = 24.0) -> float:
    """When a card whose key word starts at `word_start` comes in: three frames before it, never earlier than the end of the sentence before (a card does not
    arrive during somebody else's line) and never after the word. (The title card's time is kept clear by gfxplan.validate.)"""
    ws = float(word_start)
    at = ws - lead_s(fps)
    inside = next((s for s in sents or [] if float(s.start) - 0.01 <= ws <= float(s.end) + 0.01), None)
    if inside is not None and at < float(inside.start):
        prev_end = max((float(s.end) for s in sents if s is not inside and float(s.end) <= float(inside.start) + 0.01), default=None)
        if prev_end is not None:
            at = max(at, prev_end)
    return round(min(ws, at), 3)


# ---------------------------------------------------------------------------------------------------------------- numbers
_ONES_SMALL = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_NUMWORDS = "|".join(sorted(gfxdetect.ONES, key=len, reverse=True))
NUMBER_RX = re.compile(r"(?<![\w.])(?P<n>\$?\d[\d,]*(?:\.\d+)?|(?:" + _NUMWORDS + r")(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?(?: (?:hundred|thousand))?)"
                       r"\s*(?P<u>%|percent\b|dollars?\b|bucks\b|million\b|billion\b|thousand\b)(?:\s+(?P<d>dollars?|bucks)\b)?", re.I)


def spell_small(n: int) -> str:
    """One to nine are spelled out in running text (the exceptions, percentages, money and years, are digits)."""
    return _ONES_SMALL[n] if 0 <= int(n) <= 9 else str(int(n))


def _compact(v: float) -> str:
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if v >= div:
            x = v / div
            return (f"{x:.1f}".rstrip("0").rstrip(".")) + suffix
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:g}"


def display_number(spoken: str, unit: str = "", money: bool = False) -> str | None:
    """The figure the way it is written on a card, from what was said: "twenty-three percent" -> "23%", "five million dollars" -> "$5M", "$1.2 million" -> "$1.2M",
    "40%" kept, "two thousand twenty" (a year) -> None. A number is never turned into a year ("2,026")."""
    text = " ".join(str(spoken or "").split())
    unit = str(unit or "").lower()
    if re.fullmatch(r"(?:nineteen|twenty)[ -](?:oh[ -])?(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
                    r"(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?)", text.lower()) and unit not in ("%", "percent"):
        return None                                                                  # "twenty twenty-six" is a year, never 46 and never 2,026
    n = gfxdetect.number_of(text.replace("$", "").replace(",", ""))
    if n is None:
        return None
    if unit in ("%", "percent"):
        return f"{n:g}%"
    mult = {"million": 1e6, "billion": 1e9, "thousand": 1e3}.get(unit, 1.0)
    value = n * mult
    if 1900 <= value <= 2099 and mult == 1.0 and not money:
        return None                                                              # a year: left alone (and never given a thousands comma)
    if money or text.startswith("$") or unit in ("dollar", "dollars", "bucks"):
        return "$" + _compact(value)
    if mult > 1.0:
        x = n
        return f"{x:g} {unit}"
    return f"{int(n):,}" if float(n).is_integer() else f"{n:g}"


def number_candidates(words: list[dict], sents: list, fps: float = 24.0, hook_end: float = 0.0) -> list[dict]:
    out: list[dict] = []
    flat, spans = _flat(words)
    for m in NUMBER_RX.finditer(flat):
        unit = (m.group("u") or "").lower()
        money = bool(m.group("d")) or m.group("n").startswith("$") or unit in ("dollar", "dollars", "bucks")
        shown = display_number(m.group("n"), unit, money)
        if not shown or len(shown) > 9:
            continue
        i0 = _wi(spans, m.start())
        i1 = _wi(spans, max(m.start(), m.end() - 1))
        after = [str(w.get("text", "")) for w in words[i1 + 1:i1 + 9]]
        label_words: list[str] = []
        for t in after:
            label_words.append(t.strip())
            if re.search(r"[.!?,;:]$", t):
                break
            if len(label_words) >= 6:
                break
        label = " ".join(label_words).strip(" .,;:!?")
        if len(label.split()) < 2:
            before = [str(w.get("text", "")) for w in words[max(0, i0 - 5):i0]]
            label = " ".join(before).strip(" .,;:!?") or label
        if not label:
            continue
        said = _start_of(words, i0)
        out.append({"type": "number", "kind": "stat", "said_at": said, "at": entrance(said, sents, hook_end, fps),
                    "why": f"a striking figure ({shown})", "rule": f"number {shown} said at {said:.1f} s",
                    "item": {"kind": "stat", "at": entrance(said, sents, hook_end, fps), "value": shown, "label": label}})
    return out


# ---------------------------------------------------------------------------------------------------------------- names
NAME_STOP = {"god", "lord", "jesus", "christ", "holy", "spirit", "english", "january", "february", "march", "april", "may", "june", "july", "august", "september",
             "october", "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "american", "america", "church", "thank",
             "good", "morning", "evening", "happy", "glad", "excited", "here", "going", "just", "also", "very", "so", "not", "now", "the"}
INTRO_RX = re.compile(r"(?:\bmy name is|\bI'?m|\bI am|\bthis is|\bmeet|\bjoined by|\bwith me (?:today )?(?:is|are)|\bour guest(?: today)? is|\bplease welcome|\bsay hello to|\bwelcome)"
                      r"\s+(?P<title>(?:Dr|Doctor|Pastor|Reverend|Rev|Professor|Prof|Mr|Mrs|Ms|Bishop|Elder|Deacon)\.?\s+)?(?P<name>[A-Z][a-z'’-]+(?:\s+[A-Z][a-z'’-]+){0,2})")
ROLE_AFTER = re.compile(r"^[,]?\s*(?:and\s+)?(?:I'm|I am|who is|who's|the|a|an|our|your)\s+(?P<role>[A-Za-z][^.,;!?]{3,55})")


def name_candidates(words: list[dict], sents: list, fps: float = 24.0, hook_end: float = 0.0) -> list[dict]:
    """A person named in an introduction, first mention only. The capital letter is checked against how the transcript writes the word elsewhere
    (flow's name test): a word that is capitalised everywhere else it appears and only here is not a name."""
    from . import flow as FL
    out: list[dict] = []
    flat, spans = _flat(words)
    try:
        caps, lows = FL._name_stats(flat)
    except Exception:  # noqa: BLE001
        caps, lows = {}, {}
    seen: set[str] = set()
    for m in INTRO_RX.finditer(flat):
        name = m.group("name").strip()
        toks = name.split()
        if toks[0].lower() in NAME_STOP or any(t.lower() in NAME_STOP for t in toks[1:]):
            continue
        if len(toks) == 1 and not re.search(r"my name is|this is|meet|please welcome|say hello to", m.group(0), re.I):
            continue                                                             # "I'm Happy": one capital word is a name only after "my name is"
        if lows.get(toks[0].lower(), 0) > max(1, caps.get(toks[0], 0)):
            continue                                                             # the transcript mostly writes it in lower case: not a name
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        tail = flat[m.end():m.end() + 90]
        rm = ROLE_AFTER.match(tail)
        role = " ".join(rm.group("role").split()) if rm else ""
        title = (m.group("title") or "").strip().rstrip(".")
        if not role and title:
            role = {"dr": "Doctor", "doctor": "Doctor", "rev": "Reverend", "prof": "Professor"}.get(title.lower(), title.capitalize())
        i0 = _wi(spans, m.start("name"))
        said = _start_of(words, i0)
        at = entrance(said, sents, hook_end, fps)
        out.append({"type": "name", "kind": "lower", "said_at": said, "at": at, "why": f"{name} introduced", "rule": f"name {name} first said at {said:.1f} s",
                    "item": {"kind": "lower", "at": at, "name": name, "role": role[:60]}})
    return out


# ---------------------------------------------------------------------------------------------------------------- lists
COUNT_NOUNS = r"(?:things|reasons|ways|steps|points|keys|principles|lessons|rules|tips|ideas|mistakes|habits|questions|truths|promises|areas|parts|stages)"
COUNT_RX = re.compile(r"\b(?P<n>two|three|four|five|six|2|3|4|5|6)\s+(?:(?:big|key|main|important|simple|powerful)\s+)?(?P<noun>" + COUNT_NOUNS + r")\b", re.I)
ORD_RX = re.compile(r"\b(?:first(?:ly)?|second(?:ly)?|third(?:ly)?|fourth|fifth|number (?:one|two|three|four|five))\b[,:]?(?:\s+of all)?[,:]?", re.I)
_COUNT_WORD = {"2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "two": "Two", "three": "Three", "four": "Four", "five": "Five", "six": "Six"}


def list_candidates(words: list[dict], sents: list, fps: float = 24.0, hook_end: float = 0.0) -> list[dict]:
    out: list[dict] = []
    flat, spans = _flat(words)
    ords = list(ORD_RX.finditer(flat))
    counts = [(c, _start_of(words, _wi(spans, c.start()))) for c in COUNT_RX.finditer(flat)] if ords else []       # (once, not once per list)
    k = 0
    while k < len(ords):
        group = [ords[k]]
        j = k + 1
        while j < len(ords) and _start_of(words, _wi(spans, ords[j].start())) - _start_of(words, _wi(spans, group[-1].start())) <= 45.0 and len(group) < 5:
            group.append(ords[j])
            j += 1
        k = j
        if len(group) < 3:
            continue
        bullets: list[str] = []
        for o in group:
            i1 = _wi(spans, max(o.start(), o.end() - 1))
            run: list[str] = []
            for w in words[i1 + 1:i1 + 12]:
                t = str(w.get("text", "")).strip()
                run.append(t)
                if re.search(r"[.!?;,:]$", t) and len(run) >= 2:
                    break
                if len(run) >= 6:
                    break
            b = " ".join(run).strip(" .,;:!?")
            if len(b.split()) >= 2:
                bullets.append(b)
        if len(bullets) < 3:
            continue
        first_at = _start_of(words, _wi(spans, group[0].start()))
        cm = None
        for c, c_at in counts:
            if 0 <= first_at - c_at <= 25.0:
                cm = c
        title = (f"{_COUNT_WORD[cm.group('n').lower()]} {cm.group('noun').lower()}" if cm else "Key points")
        said = _start_of(words, _wi(spans, cm.start())) if cm else first_at
        at = entrance(said, sents, hook_end, fps)
        out.append({"type": "list", "kind": "slide", "said_at": said, "at": at, "why": f"a list of {len(bullets)} ({title.lower()})", "rule": f"list of {len(bullets)}, first said at {first_at:.1f} s",
                    "item": {"kind": "slide", "at": at, "title": title, "bullets": bullets[:5]}})
    return out


# ---------------------------------------------------------------------------------------------------------------- quotes
QUOTE_RX = re.compile(r"(?P<who>\b[A-Z][\w'’.-]*(?: [A-Z][\w'’.-]*){0,2})\s+(?:once |famously |also )?(?:said|says|wrote|writes|put it)\s*[,:]?\s+(?:that\s+)?(?P<q>[^.!?]{18,})", re.U)


def quote_candidates(words: list[dict], sents: list, fps: float = 24.0, hook_end: float = 0.0) -> list[dict]:
    out: list[dict] = []
    flat, spans = _flat(words)
    for m in QUOTE_RX.finditer(flat):
        who = m.group("who").strip()
        if who.lower() in ("i", "he", "she", "they", "we", "you", "it", "and", "but", "so", "then", "god", "the"):
            continue
        q = m.group("q").strip()
        i0 = _wi(spans, m.start("q"))
        cl = clause(q.split())
        if cl is None:
            continue
        said = _start_of(words, i0)
        at = entrance(said, sents, hook_end, fps)
        out.append({"type": "quote", "kind": "quote", "said_at": said, "at": at, "why": f"a quote by {who}", "rule": f"quote said at {said:.1f} s",
                    "item": {"kind": "quote", "at": at, "text": " ".join(cl), "who": who}})
    return out


def clause(tokens: list[str]) -> list[str] | None:
    """The words a quote card shows: all of them up to QUOTE_MAX_WORDS, else the first comma clause that is long enough (4 to 18 words) - never a shrunken or
    reworded line. None when no whole clause fits."""
    toks = [t for t in tokens if str(t).strip()]
    if 4 <= len(toks) <= QUOTE_MAX_WORDS:
        return toks
    if len(toks) < 4:
        return None
    run: list[str] = []
    for t in toks:
        run.append(t)
        if re.search(r"[,;:]$", t) and len(run) >= 4:
            return [x.rstrip(",;:") if k == len(run) - 1 else x for k, x in enumerate(run)]
        if len(run) > 18:
            break
    return None


def verbatim_text(text: str, words: list[dict]) -> str | None:
    """The words of a quote exactly as the transcript has them (case and punctuation as transcribed): the AI's wording is only used to FIND the stretch. Fillers
    may be dropped, nothing else may be added, removed or re-ordered. None when the stretch is not there (word for word, fillers aside)."""
    want = [w for w in _words_of(text) if w not in FILLERS]
    if not want:
        return None
    toks: list[tuple[str, str]] = []
    for w in words or []:
        for piece in str(w.get("text", "")).split():
            t = _words_of(piece)
            if t and t[0] not in FILLERS:
                toks.append((t[0], piece))
    flat = [t for t, _p in toks]
    n = len(want)
    for i in range(0, len(flat) - n + 1):
        if flat[i:i + n] == want:
            return " ".join(p for _t, p in toks[i:i + n]).strip().rstrip(",;:")
    return None


# ---------------------------------------------------------------------------------------------------------------- references
def reference_candidates(words: list[dict], sents: list, fps: float = 24.0, hook_end: float = 0.0) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for r in scripture.find(words, sents):
        if r["ref"] in seen:
            continue
        seen.add(r["ref"])
        at = entrance(r["at"], sents, hook_end, fps)
        item = {"kind": "reference", "at": at, "ref": r["ref"], "passage": r["passage"]}
        out.append({"type": "reference", "kind": "reference", "said_at": r["at"], "at": at, "why": f"scripture reference {r['ref']}", "rule": f"reference {r['ref']} said at {r['at']:.1f} s",
                    "item": item})
    return out


def candidates(sents: list, words: list[dict], fps: float = 24.0, hook_end: float = 0.0) -> list[dict]:
    """All the typed candidates, in time order. Each: {type, kind (the card), said_at, at, why, rule, item (a card for gfxplan.validate)}."""
    if not words:
        return []
    found: list[dict] = []
    for fn in (number_candidates, name_candidates, list_candidates, quote_candidates, reference_candidates):
        try:
            found.extend(fn(words, sents, fps, hook_end))
        except Exception:  # noqa: BLE001 - a scan that fails is a scan that finds nothing
            continue
    return sorted(found, key=lambda c: c["at"])


# ---------------------------------------------------------------------------------------------------------------- durations
def _nwords(c: dict) -> int:
    n = 0
    for k in ("text", "quote", "title", "label", "value", "who", "role", "name", "passage", "ref"):
        if isinstance(c.get(k), str):
            n += len(c[k].split())
    for k in ("bullets", "chips"):
        n += sum(len(str(b).split()) for b in (c.get(k) or []))
    return n


def retime(c: dict, style: dict | None = None, out_len: float | None = None) -> dict:
    """The card's length from its word count (the readability rules: about one second per two to three words, a floor of 1.5 to 3 s), never short of the moment its
    last timed step is said plus a second and a half. -> the card with "len" set."""
    c = dict(c)
    kind = c.get("kind")
    cur = float(c.get("len") or 0.0)
    lt = float(((style or {}).get("graphics") or {}).get("lower_third_s", 4.5) or 4.5)
    times = [float(t) for t in (c.get("times") or []) if isinstance(t, (int, float))]
    if kind in ("stat", "bigstat", "count"):
        n = len(str(c.get("value", "")).split()) + len(str(c.get("label", "")).split())
        ln = clamp(1.0 + 0.4 * n, 2.0, 4.0)
        ln = max(ln, 3.6 if kind == "bigstat" else 4.2 if kind == "count" else ln)           # (the full-screen number keeps its own floor)
    elif kind in ("quote", "fullquote"):
        n = len(str(c.get("text", "")).split())
        ln = clamp(1.0 + 0.33 * n, 3.0, 7.0)
    elif kind == "lower":
        ln = lt if c.get("role") else min(lt, 3.0)
    elif kind == "reference":
        ln = 3.0 + 0.3 * len(str(c.get("passage", "")).split())
    elif kind == "chapter":
        ln = 3.4
    else:
        ln = cur
    if times:
        ln = max(ln, times[-1] + 1.5)                                                           # the card holds after the last item is said, never cut before it
    if out_len is not None and c.get("at") is not None:
        ln = min(ln, max(0.0, float(out_len) - float(c["at"]) - 0.6))
    c["len"] = round(ln, 2)
    return c


def snap_exit(c: dict, sents: list, cuts: list[float] | None = None, out_len: float | None = None) -> dict:
    """A card leaves at the end of a sentence if that is within a second after its minimum, or at a shot change within half a second of its end (never before
    the last timed item has had its moment)."""
    c = dict(c)
    try:
        end = float(c["at"]) + float(c["len"])
    except (KeyError, TypeError, ValueError):
        return c
    snapped = None
    if cuts:
        near = min(cuts, key=lambda t: abs(t - end), default=None)
        if near is not None and abs(near - end) <= 0.5 and near - float(c["at"]) >= 1.5:
            snapped = near
    if snapped is None:
        s = next((s for s in sents if float(s.start) < end < float(s.end)), None)
        if s is not None and 0.0 < float(s.end) - end <= 1.0:
            snapped = float(s.end)
    if snapped is not None:
        if out_len is not None:
            snapped = min(snapped, float(out_len) - 0.6)
        times = [float(t) for t in (c.get("times") or []) if isinstance(t, (int, float))]
        floor = (times[-1] + 1.0) if times else 1.5
        c["len"] = round(max(snapped - float(c["at"]), floor), 2)
    return c


# ---------------------------------------------------------------------------------------------------------------- budget and gaps
def cap_per_video(style: dict | None, out_len: float, fallback: int) -> int:
    """How many cards a video may have: style.graphics.max_per_minute for its length (at least one), else the profile's old rule."""
    if not isinstance(style, dict):
        return fallback
    g = style.get("graphics") or {}
    try:
        per_min = float(g.get("max_per_minute"))
    except (TypeError, ValueError):
        return fallback
    return max(1, int(round(per_min * float(out_len) / 60.0 + 0.01)))


def min_gap(style: dict | None, default: float) -> float:
    try:
        return max(0.0, float(((style or {}).get("graphics") or {}).get("min_gap_s", default)))
    except (TypeError, ValueError):
        return default


def allowed_kinds(kinds: list[str], style: dict | None) -> list[str]:
    """The genre gate: the policy of the style decides which of the profile's kinds may be used (no style = no gate, as before)."""
    if not isinstance(style, dict):
        return list(kinds)
    pol = str(((style.get("graphics") or {}).get("policy")) or "").lower()
    allowed = GRAPHICS_KINDS_BY_POLICY.get(pol)
    if allowed is None:
        return list(kinds)
    return [k for k in kinds if k in allowed]


def repeats_caption(card: dict, words: list[dict], captions_on: bool, group: int = 3) -> bool:
    """A card whose few words are the words the caption is showing at that moment is redundant (the viewer reads the same thing twice): at least 80% of the card's
    words are in one caption group (the words said in the first two seconds of the card, three at a time)."""
    if not captions_on or card.get("kind") in ("chapter", "lower", "reference"):
        return False
    shown = [w for k in ("text", "title", "value", "label") for w in _words_of(str(card.get(k, "")))]
    if not shown or len(shown) > 6:
        return False                                                                           # only a few-word card can be a caption's twin
    try:
        at = float(card["at"])
    except (KeyError, TypeError, ValueError):
        return False
    spoken = [w for x in words or [] if at - 0.3 <= float(x.get("start", -1)) <= at + 2.0 for w in _words_of(x.get("text", ""))]
    best = 0.0
    for i in range(0, max(1, len(spoken) - group + 1)):
        g = set(spoken[i:i + group])
        best = max(best, sum(1 for w in shown if w in g) / len(shown))
    return best >= 0.8


# ---------------------------------------------------------------------------------------------------------------- chapters
def shorten_title(text: str, max_words: int = 6) -> str:
    t = " ".join(str(text or "").split())
    t = re.split(r"\s*[:–—]\s+|\s+-\s+", t)[0] or t
    w = t.split()
    if len(w) > max_words:
        w = w[:max_words]
        while w and w[-1].lower() in ("a", "an", "the", "of", "and", "to", "in", "for", "with", "on", "at"):
            w.pop()
    return " ".join(w).strip(" ,;:")


def chapter_cards(marks: list[dict], out_len: float, min_gap_s: float = 10.0, hook_end: float = 0.0) -> list[dict]:
    """Chapter cards on the outline's steps: marks = [{"at": seconds in the finished video, "title": the step's words, "n": step number}]. At least `min_gap_s`
    apart, never in the title card's time, never in the last 6 s."""
    out: list[dict] = []
    for mk in sorted((m for m in marks or [] if isinstance(m, dict)), key=lambda m: float(m.get("at") or 0)):
        try:
            at = float(mk["at"])
        except (KeyError, TypeError, ValueError):
            continue
        title = shorten_title(mk.get("title", ""))
        if not title or at < float(hook_end) + 0.2 or at > out_len - 6.0:
            continue
        if out and at - out[-1]["at"] < min_gap_s:
            continue
        n = mk.get("n")
        out.append({"kind": "chapter", "at": round(at, 2), "text": title, "number": str(int(n)).zfill(2) if isinstance(n, int) or str(n or "").isdigit() else "",
                    "len": 3.4, "rule": f"chapter: outline step {n} at {at:.1f} s"})
    return out


def chapter_marks(flow: list[dict] | None, items: list[dict], start_of, remap) -> list[dict]:
    """One mark per outline step: the first talk item of the step, mapped to the finished video's clock. start_of(sentence index) -> source seconds;
    remap(source seconds) -> video seconds or None."""
    out: list[dict] = []
    titles = {int(r.get("n")): r.get("step") for r in (flow or []) if isinstance(r, dict) and isinstance(r.get("n"), int)}
    seen: set[int] = set()
    for it in items or []:
        n = it.get("step")
        if it.get("kind") != "talk" or not isinstance(n, int) or n in seen or n not in titles:
            continue
        seen.add(n)
        try:
            t = remap(start_of(it["a"]))
        except Exception:  # noqa: BLE001
            t = None
        if t is not None:
            out.append({"n": n, "title": titles[n], "at": float(t)})
    return out
