"""Scripture references in what is said: "John 3:16", "Romans eight twenty-eight", "First Corinthians 13", "Psalm 23".

Found with a table of the 66 books (and the aliases and abbreviations people say and transcribers write) plus a regular expression. A hit is only accepted
when the book is followed by a chapter AND the next twenty words either READ a passage or say so ("says", "reads", "tells us", "verse", "chapter"). That kills
the false positives: "I read 2 John last week" has no chapter, "John 5 times" has no cue.

    find(words, sents) -> [{"ref": "John 3:16", "book", "chapter", "v1", "v2", "at" (when the book is said), "end" (when the number is said),
                           "sent": index of the sentence, "passage": the words read aloud after it, or ""}]
"""

from __future__ import annotations

import re

from . import gfxdetect

# canonical name -> aliases (lower case, no trailing full stop). A leading 1/2/3 is handled by the pattern, the aliases of numbered books are written without it.
_BOOKS: dict[str, list[str]] = {
    "Genesis": ["gen", "ge", "gn"], "Exodus": ["ex", "exod", "exo"], "Leviticus": ["lev", "lv"], "Numbers": ["num", "nu", "nm", "nb"], "Deuteronomy": ["deut", "dt"],
    "Joshua": ["josh", "jos"], "Judges": ["judg", "jdg"], "Ruth": ["rth"], "1 Samuel": ["samuel", "sam", "sa"], "2 Samuel": ["samuel", "sam", "sa"],
    "1 Kings": ["kings", "kgs", "ki"], "2 Kings": ["kings", "kgs", "ki"], "1 Chronicles": ["chronicles", "chron", "chr"], "2 Chronicles": ["chronicles", "chron", "chr"],
    "Ezra": [], "Nehemiah": ["neh"], "Esther": ["est"], "Job": [], "Psalms": ["psalm", "ps", "psa", "pslm"], "Proverbs": ["prov", "pro", "prv"],
    "Ecclesiastes": ["eccl", "ecc", "eccles"], "Song of Solomon": ["song of songs", "song", "sos", "canticles"], "Isaiah": ["isa"], "Jeremiah": ["jer"],
    "Lamentations": ["lam"], "Ezekiel": ["ezek", "eze"], "Daniel": ["dan"], "Hosea": ["hos"], "Joel": [], "Amos": [], "Obadiah": ["obad", "oba"], "Jonah": ["jon"],
    "Micah": ["mic"], "Nahum": ["nah"], "Habakkuk": ["hab"], "Zephaniah": ["zeph", "zep"], "Haggai": ["hag"], "Zechariah": ["zech", "zec"], "Malachi": ["mal"],
    "Matthew": ["matt", "mt"], "Mark": ["mrk", "mk"], "Luke": ["lk"], "John": ["jn", "jhn"], "Acts": ["act"], "Romans": ["rom", "ro"],
    "1 Corinthians": ["corinthians", "cor"], "2 Corinthians": ["corinthians", "cor"], "Galatians": ["gal"], "Ephesians": ["eph"], "Philippians": ["phil", "php"],
    "Colossians": ["col"], "1 Thessalonians": ["thessalonians", "thess", "thes"], "2 Thessalonians": ["thessalonians", "thess", "thes"],
    "1 Timothy": ["timothy", "tim"], "2 Timothy": ["timothy", "tim"], "Titus": [], "Philemon": ["philem", "phm"], "Hebrews": ["heb"], "James": ["jas", "jm"],
    "1 Peter": ["peter", "pet", "pe"], "2 Peter": ["peter", "pet", "pe"], "1 John": ["john", "jn"], "2 John": ["john", "jn"], "3 John": ["john", "jn"],
    "Jude": [], "Revelation": ["rev", "revelations", "revelation of john"],
}
NUMBERED = {"1": 1, "2": 2, "3": 3, "i": 1, "ii": 2, "iii": 3, "first": 1, "second": 2, "third": 3, "1st": 1, "2nd": 2, "3rd": 3}


def _table() -> tuple[dict[tuple[int, str], str], list[str]]:
    """{(book number or 0, spoken name): canonical}, and the names longest first (so "song of solomon" beats "song")."""
    out: dict[tuple[int, str], str] = {}
    for canon, aliases in _BOOKS.items():
        m = re.match(r"^([123]) (.+)$", canon)
        n, base = (int(m.group(1)), m.group(2)) if m else (0, canon)
        for name in {base.lower(), *aliases}:
            out[(n, name)] = canon
    names = sorted({name for _n, name in out}, key=len, reverse=True)
    return out, names


TABLE, NAMES = _table()
_BOOK_RX = r"(?:(?P<num>[123]|i{1,3}|first|second|third|1st|2nd|3rd)\.?\s+)?(?P<book>" + "|".join(re.escape(n) for n in NAMES) + r")\.?"
_NUMWORD = r"(?:\d{1,3}|(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?)"
REF = re.compile(r"(?<![\w])" + _BOOK_RX + r"\s+(?:chapter\s+)?(?P<ch>\d{1,3}|" + _NUMWORD + r")(?:\s*(?::|\.|,|\s+verses?|\s+verse)\s*(?P<v1>\d{1,3}|" + _NUMWORD + r")(?:\s*(?:-|–|to|through)\s*(?P<v2>\d{1,3}))?)?(?![\w])", re.I)
CUE = re.compile(r"\b(says?|reads?|tells us|told us|verse|verses|chapter|writes?|wrote|states?|declares?|reminds us|according to)\b", re.I)
BEFORE_CUE = re.compile(r"\b(?:writes?|wrote|says?|said|reads?|read|found|written|tells us|told us|reminds us|teaches?|declares?) (?:in|at)\b|\baccording to\b|\bin the (?:book|letter|gospel|epistle) of\b|\b(?:turn|turned|open|opened|go|look|looking) (?:with me |up )?(?:to|at)\b", re.I)    # said BEFORE the reference: "Paul writes in 2 Corinthians 5:17 that ..."
FIRST_PERSON = re.compile(r"\b(i|me|my|mine|we|us|our|ours|i'm|i've|i'll|i'd|we're|we've|let's)\b", re.I)


def _n(text: str) -> int | None:
    text = str(text or "").strip()
    if text.isdigit():
        return int(text)
    v = gfxdetect.number_of(text.lower())
    return int(v) if v is not None and float(v).is_integer() else None


def _flat(words: list[dict]) -> tuple[str, list[tuple[int, int, int]]]:
    """The words joined with spaces, and (start char, end char, word index) of each."""
    out, spans, pos = [], [], 0
    for i, w in enumerate(words):
        t = str(w.get("text", "") or "").strip()
        if not t:
            continue
        spans.append((pos, pos + len(t), i))
        out.append(t)
        pos += len(t) + 1
    return " ".join(out), spans


def _word_at(spans: list[tuple[int, int, int]], char: int) -> int:
    for a, b, i in spans:
        if a <= char < b + 1:
            return i
    return spans[-1][2] if spans else 0


def _reads_a_passage(text: str) -> bool:
    """A long line with nobody saying "I" or "we" in it: the Bible being read, not a sentence about the speaker."""
    w = text.split()
    return len(w) >= 10 and not FIRST_PERSON.search(text)


def find(words: list[dict], sents: list | None = None) -> list[dict]:
    """-> the scripture references said in `words` (dicts with text/start/end), in time order. `sents` (objects with .text/.start/.end/.idx) gives the passage."""
    if not words:
        return []
    flat, spans = _flat(words)
    out: list[dict] = []
    for m in REF.finditer(flat):
        book = m.group("book").lower().rstrip(".")
        num = NUMBERED.get((m.group("num") or "").lower().rstrip(".")) if m.group("num") else 0
        canon = TABLE.get((num or 0, book)) or (TABLE.get((0, book)) if not num else None)
        if canon is None:
            continue
        if re.match(r"^[123] ", canon) and not num:                       # "John 3:16" is the Gospel; "1 John 3:16" needs its number
            continue
        if not re.match(r"^[123] ", canon) and num:                       # "2 Romans" is not a book
            continue
        ch, v1 = _n(m.group("ch")), (_n(m.group("v1")) if m.group("v1") else None)
        v2 = int(m.group("v2")) if m.group("v2") else None
        if not ch or ch > 150 or (v1 and v1 > 176) or (v2 and v1 and v2 <= v1):
            continue
        i0, i1 = _word_at(spans, m.start()), _word_at(spans, max(m.start(), m.end() - 1))
        after_words = [str(w.get("text", "")) for w in words[i1 + 1:i1 + 21]]
        after = " ".join(after_words)
        before = " ".join(str(w.get("text", "")) for w in words[max(0, i0 - 7):i0])
        if not (CUE.search(after) or CUE.search(m.group(0)) or BEFORE_CUE.search(before) or _reads_a_passage(after)):
            continue                                                       # "John 5 times", "Mark 3 of them": no cue and nothing read: not a reference
        if v1 is None and not str(m.group("ch")).isdigit() and after_words and (nv := _n(after_words[0].strip(".,:;"))) and nv <= 176:
            v1, after_words = nv, after_words[1:]                          # "Romans eight twenty-eight": chapter and verse spoken as two numbers
        k = 0                                                              # the passage read aloud: after "says" / "reads" (and "that"), to the end of the sentence
        while k < len(after_words) and k < 4 and (CUE.fullmatch(after_words[k].strip(".,:;\"'").lower()) or after_words[k].lower().strip(".,:;") in ("that", "us", "to", "which", "it", "and", "tells", "told", "reminds")):
            k += 1
        read: list[str] = []
        for t in after_words[k:]:
            read.append(t)
            if re.search(r"[.!?][\"')\]]*$", t):
                break
        passage = " ".join(read) if len(read) >= 6 else ""
        near = next((s for s in (sents or []) if s.start - 0.05 <= float(words[i0]["start"]) <= s.end + 0.05), None)
        ref = f"{canon} {ch}" + (f":{v1}" + (f"-{v2}" if v2 else "") if v1 else "")
        out.append({"ref": ref, "book": canon, "chapter": ch, "v1": v1, "v2": v2, "at": float(words[i0]["start"]), "end": float(words[i1]["end"]),
                    "sent": getattr(near, "idx", None), "passage": " ".join(passage.split()[:26])})
    return out
