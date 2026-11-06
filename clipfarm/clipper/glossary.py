"""Names and words the person types once (Settings › How it edits › Names and words): transcripts, captions, titles and graphics spell
them this way.

Speech to text has never heard most names. A recorded interview came back with a school's name as "Nortfeld", "Norfield", "Northfeeld" and
"North Feeld" for one school, Northfield. Parakeet can't be told the words beforehand, so its words are corrected afterwards (conservatively:
a near miss that sounds the same and is not an ordinary word), and the AI that writes titles and captions is told the spellings.

    terms(cfg) -> list[str]                    the app's list, read fresh (a change in Settings takes effect at once), else clipper.glossary
    correct(words, terms) -> list[dict]        the same words with near misses replaced; merged words keep the first start and the last end
    prompt_note(terms) -> str                  "Spell these names exactly as written: ..." ("" when there are none)
"""

from __future__ import annotations

import json
import os
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

MAX_TERMS = 200
RATIO = 0.75                                     # how close (0-1) a word must be to a name to be read as it
_cache = {"at": 0.0, "val": None}

# Ordinary English words are never "corrected" into a name: "a must" stays "a must", whatever the list says.
COMMON = set("""a about above after again against all almost also always am among an and another any are around as ask at away back
be because been before being below best better between big both but by call came can case come could day did different do does done
down during each early end even ever every fact family far feel few find first for found from get give go going good got great group
had hand has have he head help her here high him his home house how i if important in into is it its just keep kind know large last
late later lead learn least leave left less let life like line little live long look lot made make man many may me mean might more
most mother much must my name need never new next night no not now number of off often old on once one only open or order other our
out over own part people place play point program public put question quite rather read real really right room said same say school
see seem set she should show side since small so some something sound start state still story student study such sure system take
talk team tell than that the their them then there these they thing think this those though three through time to today together too
took toward try turn two under until up us use very want was water way we week well went were what when where which while who why
will with without word work world would write year yes yet you young your amazing campus faith chapel class friends god life love
maybe music okay ready right yeah wait actually""".split())


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9']", "", str(t).lower().replace("’", "'"))


def _key(t: str) -> str:
    """A rough sound key: the first letter, then the consonant sounds with look-alikes merged and vowels dropped."""
    t = _norm(t).replace("'", "")
    if not t:
        return ""
    sub = str.maketrans("bfpvcgjkqsxzdtlmnr", "111122222222334556")              # (Soundex groups)
    head, rest = t[0], t[1:]
    head = {"e": "a", "i": "a", "o": "a", "u": "a", "y": "a"}.get(head, head)           # (vowels at the start sound alike)
    out = head
    for ch in rest.translate(sub):
        if ch.isdigit() and (not out or out[-1] != ch):
            out += ch
    return out


def terms(cfg=None) -> list[str]:
    """The names list: the app's settings (read fresh, at most every 5 s), else the config's clipper.glossary."""
    h = os.environ.get("VIDEOBUILDER_HOME")
    if h:
        if time.time() - _cache["at"] > 5:
            try:
                v = json.loads((Path(h) / "settings.json").read_text()).get("glossary")
            except (OSError, ValueError, AttributeError):
                v = None
            _cache.update(at=time.time(), val=v)
        if _cache["val"]:
            return clean(_cache["val"])
    try:
        return clean((cfg.get("clipper.glossary") if cfg is not None else None) or [])
    except Exception:  # noqa: BLE001
        return []


def clean(items) -> list[str]:
    """One name per entry, trimmed, each once, at most MAX_TERMS (a pasted block of lines works too)."""
    if isinstance(items, str):
        items = items.splitlines()
    out, seen = [], set()
    for x in items if isinstance(items, list) else []:
        for line in str(x or "").splitlines():
            t = " ".join(line.split())[:60]
            if t and t.lower() not in seen and any(c.isalpha() for c in t):
                seen.add(t.lower())
                out.append(t)
    return out[:MAX_TERMS]


def prompt_note(names: list[str]) -> str:
    return ("\nSpell these names exactly as written (people, places, brands): " + "; ".join(names) + ".") if names else ""


def _similar(heard: str, name: str) -> float:
    return SequenceMatcher(None, heard, name).ratio()


def _ok(window: list[str], name: str) -> float:
    """How well these heard words stand for the name (0 = not at all)."""
    heard = "".join(_norm(w).replace("'", "") for w in window)
    target = _norm(name).replace(" ", "").replace("'", "")
    if not heard or not target:
        return 0.0
    if heard == target:
        return 1.0
    words = [_norm(w) for w in window]
    if all(w in COMMON for w in words):                                         # ordinary words: never a name ("a must" is not Northfield)
        return 0.0
    if len(window) == 1 and len(heard) < 4:                                     # too short to tell
        return 0.0
    if any(w in COMMON and w not in [_norm(p) for p in name.split()] for w in words) and len(window) > 1:
        return 0.0                                                              # (a merge never swallows an ordinary word)
    if abs(len(heard) - len(target)) > max(1, round(0.2 * len(target))):      # about the same length ("Emma" is someone else, not Northfield)
        return 0.0
    r = _similar(heard, target)
    if r < RATIO:
        return 0.0
    if heard[0] != target[0] and _key(heard)[:2] != _key(target)[:2]:            # the same start, or the same opening sound
        return 0.0
    return r


def _tail(heard: str, name: str) -> str:
    """What follows the name in the heard word: punctuation, and a possessive ("Eli's," keeps its 's)."""
    m = re.search(r"(['\u2019]s)?\W*$", heard, re.I)
    return m.group(0)


def correct(words: list[dict], names: list[str]) -> list[dict]:
    """Near misses of the names fixed in a word list ({"text","start","end",...}). A name of n words is looked for in windows of n and
    n+1 heard words (speech to text splits names: "North Feeld"); a wider window wins only when it fits clearly better. Punctuation
    around the heard words is kept ("Nortfeld," -> "Northfield,"). Names that are themselves ordinary words only get their capitals fixed."""
    if not words or not names:
        return words
    plans = []
    for name in names:
        n = len(name.split())
        common = all(_norm(p) in COMMON for p in name.split())
        plans.append((name, n, common))
    out, i = [], 0
    while i < len(words):
        best = None                                                             # (score, size, name)
        for name, n, common in plans:
            for size in (n, n + 1):
                if i + size > len(words):
                    continue
                window = [str(w.get("text", "")) for w in words[i:i + size]]
                if common:                                                      # an ordinary-word name: capitals only, exact words
                    score = 1.0 if size == n and [_norm(w) for w in window] == [_norm(p) for p in name.split()] else 0.0
                else:
                    score = _ok(window, name)
                if score and size > n:
                    single = max(_ok([window[k]], name) for k in range(size)) if n == 1 else 0.0
                    if score < single + 0.05:                                    # merging must clearly beat fixing one word
                        score = 0.0
                if score and (best is None or score > best[0] + 1e-9 or (abs(score - best[0]) < 1e-9 and size < best[1])):
                    best = (score, size, name)
        if best is None:
            out.append(words[i])
            i += 1
            continue
        _, size, name = best
        span = words[i:i + size]
        first, last = str(span[0].get("text", "")), str(span[-1].get("text", ""))
        lead = re.match(r"^\W*", first).group(0)
        tail = _tail(last, name)
        text = lead + name + tail
        parts = name.split()
        if size == len(parts):                                                  # word for word: each keeps its own timing (captions)
            for k, (w, part) in enumerate(zip(span, parts)):
                t = str(w.get("text", ""))
                lead_k = re.match(r"^\W*", t).group(0) if k == 0 else ""
                tail_k = _tail(t, part)
                out.append(w if t == lead_k + part + tail_k else {**w, "text": lead_k + part + tail_k})
        else:                                                                   # heard as more words than it has: one word, both ends
            out.append({**span[0], "text": text, "end": span[-1]["end"]})
        i += size
    return out
