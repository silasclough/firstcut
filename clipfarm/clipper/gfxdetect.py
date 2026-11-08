"""Where graphics belong: a rule-based scan of the transcript for the moments that ask for one.

The AI that plans graphics is good at choosing but forgetful: in a test it skipped "the numbers could be as high as 2,000 and as low as 1". So the
transcript is scanned first, in plain code, for moments with a recognisable shape:

    range     "as high as X ... as low as Y", "from X to Y", "between X and Y", "X compared to / versus Y"  -> a compare card (built by code if the AI misses it)
    figure    a percentage, a sum of money, "N million"                                                    -> a stat (a hint to the AI)
    dates     three or more years / month-years close together                                              -> a timeline (a hint)
    list      first / second / third, "three things"                                                        -> a slide (a hint)
    claim     a short, strong sentence ("the most", "never", "nobody", "only")                              -> an emphasis card (a hint)

The hints go into the AI's prompt as "moments that probably want a graphic"; the ranges, which need no judgement, are also turned into cards directly when the
AI has put nothing there. Everything still goes through gfxplan.validate, so a card only exists if its numbers and words were really said.
"""

from __future__ import annotations

import re

from . import moments as M

ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
        "seventy": 70, "eighty": 80, "ninety": 90}
MULT = {"hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000}
NUM = r"(?<![\w.])(?:\$?\d[\d,]*(?:\.\d+)?%?|(?:(?:" + "|".join(ONES) + r")(?:[- ](?:" + "|".join(k for k, v in ONES.items() if v < 10) + r"))?(?: (?:hundred|thousand|million|billion))?))(?: (?:hundred|thousand|million|billion|percent|k))?(?![\w])"
HIGH = r"(?:high|large|big|much|many|tall|fast|great|far)"
LOW = r"(?:low|small|little|few|short|slow)"
MULT_OR_UNIT = {"hundred", "thousand", "million", "billion", "percent", "k"}
RANGES = [
    (re.compile(rf"as {HIGH} as ({NUM}).{{0,80}}?as {LOW} as ({NUM})", re.I), ("As high as", "As low as")),
    (re.compile(rf"as {LOW} as ({NUM}).{{0,80}}?as {HIGH} as ({NUM})", re.I), ("As low as", "As high as")),
    (re.compile(rf"\bfrom ({NUM}) (?:all the way )?(?:up |down )?to ({NUM})", re.I), ("From", "To")),
    (re.compile(rf"\bbetween ({NUM}) and ({NUM})", re.I), ("Low", "High")),
    (re.compile(rf"({NUM}) (?:compared to|versus|vs\.?|as opposed to|instead of) ({NUM})", re.I), ("", "")),
]
FIGURE = re.compile(r"\d[\d,.]*\s?(?:%|percent)|\$\s?\d[\d,.]*|\b\d[\d,.]*\s(?:million|billion|thousand)\b", re.I)
YEAR = re.compile(r"\b(?:19|20)\d\d\b")
ORDINAL = re.compile(r"\b(first|second|third|fourth|fifth)\b", re.I)
CLAIM = re.compile(r"\b(most|never|always|nobody|no one|everyone|only|biggest|worst|best|secret|truth|mistake)\b", re.I)


def number_of(text: str) -> float | None:
    """'2,000' -> 2000, '$9' -> 9, 'one' -> 1, 'twenty five' -> 25, 'two thousand' -> 2000. None if it is not a number."""
    t = text.lower().strip().replace("$", "").replace(",", "").rstrip("%")
    try:
        return float(t)
    except ValueError:
        pass
    total, cur, seen = 0.0, 0.0, False
    for tok in re.split(r"[ -]+", t):
        if tok in ONES:
            cur += ONES[tok]
            seen = True
        elif tok in MULT:
            cur = (cur or 1) * MULT[tok]
            if MULT[tok] >= 1000:
                total, cur = total + cur, 0.0
            seen = True
        else:
            return None
    return (total + cur) if seen else None


def shown(text: str) -> str:
    """A figure the way it is written on a card: digits for number words, the speaker's own digits kept."""
    text = text.strip().rstrip(",.;:")
    n = number_of(text)
    if n is None or re.search(r"\d", text):
        return text
    return f"{int(n):,}" if float(n).is_integer() else f"{n:g}"


def find(sents: list[M.Sentence], words: list[dict]) -> list[dict]:
    """-> [{"kind", "at", "why", "item" (a card for gfxplan.validate, only for the ones code can build alone)}], in time order."""
    out: list[dict] = []
    for k, s in enumerate(sents):
        text = " ".join(s.text.split())
        nxt = " ".join(sents[k + 1].text.split()) if k + 1 < len(sents) else ""
        joined = (text + " " + nxt).strip() if nxt and sents[k + 1].start - s.end < 3.0 else text
        for rx, labels in RANGES:
            m = rx.search(joined)
            if not m:
                continue
            a, b = m.group(1).rstrip(",."), m.group(2).rstrip(",.")
            if number_of(a) is None or number_of(b) is None or number_of(a) == number_of(b):
                continue
            na, nb = number_of(a.split(" ")[0] if " " in a.strip() and a.split(" ")[-1] in MULT_OR_UNIT else a), number_of(b.split(" ")[0] if " " in b.strip() and b.split(" ")[-1] in MULT_OR_UNIT else b)
            tail = joined[m.end():m.end() + 14].lower()
            clock = bool(re.match(r"\s*(hours?|minutes?|mins?|days?|weeks?|months?|years?|a\.?m|p\.?m|o'clock)\b", tail)) or (labels in (("From", "To"), ("Low", "High")) and max(na or 0, nb or 0) <= 12 and (na or 0) > (nb or 0))
            weak = labels in (("From", "To"), ("Low", "High")) and clock                  # "from 9 to 5", "between 3 and 4 hours": a time span, not a comparison
            years = all(re.fullmatch(r"(?:19|20)\d\d", x.strip()) for x in (a, b))
            if weak or years:
                continue                                                                 # "from 9 to 5", "between 3 and 4 hours", "from 1999 to 2004": not a comparison to chart
            if not labels[0]:                                                            # "X versus Y": which is which needs judgement: a hint for the AI, not a card
                out.append({"kind": "compare", "at": s.start + 0.2, "why": f"two things set against each other ({shown(a)} versus {shown(b)})"})
                break
            item = {"kind": "compare", "at": s.start + 0.2, "labels": [labels[0], labels[1]], "values": [shown(a), shown(b)]}
            if any(o["kind"] == "compare" and o["item"]["values"] == [shown(a), shown(b)] and abs(o["at"] - s.start) < 20 for o in out):
                break                                                                    # (the same two numbers, seen again through the next sentence)
            out.append({"kind": "compare", "at": s.start + 0.2, "why": f"two numbers set against each other ({shown(a)} and {shown(b)})", "item": item, "first": a})
            break
        else:
            fig = FIGURE.search(text)
            if fig:
                out.append({"kind": "stat", "at": s.start + 0.2, "why": f"a striking figure ({fig.group(0)})"})
            elif CLAIM.search(text) and 6 <= len(text.split()) <= 18 and text.rstrip()[-1:] in ".!":
                out.append({"kind": "statement", "at": s.start + 0.2, "why": "a short, strong claim"})
        if len(set(YEAR.findall(joined))) >= 3 and not any(o["kind"] == "timeline" and abs(o["at"] - s.start) < 20 for o in out):
            out.append({"kind": "timeline", "at": s.start + 0.2, "why": "dates in order"})
        if len(ORDINAL.findall(joined)) >= 2 and not any(o["kind"] == "slide" and abs(o["at"] - s.start) < 20 for o in out):
            out.append({"kind": "slide", "at": s.start + 0.2, "why": "a list (first, second, third...)"})
    try:                                                                                 # typed candidates (numbers, names, lists, quotes, scripture) with the rule that found them
        from . import gfxcontent
        typed = gfxcontent.candidates(sents, words)
    except Exception:  # noqa: BLE001 - a scan that fails finds nothing
        typed = []
    for c in typed:
        out[:] = [o for o in out if not (o.get("type") is None and o["kind"] == c["kind"] and c["kind"] in ("stat", "slide") and abs(o["at"] - c["at"]) < 4.0)]      # (the typed one replaces the plain hint)
        out.append(c)
    out.sort(key=lambda o: o["at"])
    return out


def prompt_lines(found: list[dict]) -> str:
    return "\n".join(f"  [{o['at']:.1f}s] {o['kind']}: {o['why']}" for o in found)
