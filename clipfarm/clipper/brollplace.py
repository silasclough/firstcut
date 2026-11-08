"""B-roll that is motivated by the words (WP9): a matcher that may say "nothing fits", and a placement pass that starts a cutaway a few frames
before the noun it shows, ends it at the clause, never covers a protected sentence, and keeps the speaker on screen for most of the talk.

The AI (broll.plan_cutaways) stays a proposal generator; this is the code that decides. Everything here is pure: pool scenes, nouns
(visualnouns.extract), protected spans and the style go in, cutaway dicts come out, so it is tested without a render.

    match(noun, pool, ...)  -> (scene index | None, score, why)     abstains below THRESHOLD or when the best is not clearly above the pool
    place(nouns, pool, words, style, ...) -> (cutaways, unmatched)    cutaway: {at, length, src, scene, what, file, photo, noun, phrase, score, device}
    snap_beat(at, beats, words, noun_start)                          a beat snap that never lands on a word
    photo_ok(...)                                                    a photo is placed only where it matches (or when photos are "all")
    stock_queries(unmatched)                                         what to ask the stock library for when footage had nothing
"""

from __future__ import annotations

import re
import statistics

THRESHOLD = 0.35                                 # best score below this: nothing fits, the speaker stays (proposal; log score so it can be tuned per folder)
CLEAR_OVER_MEDIAN = 0.10
STRONG = 0.6
CONCRETE = 0.6                                   # a noun is worth a cutaway at this concreteness
INTRO_FREE_S = 3.0
SPEAKER_BETWEEN_S = 1.5
CLAUSE_END = re.compile(r"[,;:.!?]['\")\]]*$")
STOP = frozenset("a an the and or but of in on at to for with from by is are was were be been it its this that these those i you we they he she as so "
                 "very just really some any all".split())


def _tok(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(w) >= 3 and w not in STOP}


def _scene_tokens(sc: dict) -> set[str]:
    return _tok(sc.get("what", "")) | _tok(" ".join(str(x) for x in (sc.get("subject_words") or [])))


def text_score(phrase: str, sc: dict) -> float:
    """How much of what the noun names is in the scene's description: 0.7 * the share of the noun's words found + 0.3 * Jaccard."""
    a, b = _tok(phrase), _scene_tokens(sc)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return 0.7 * (inter / len(a)) + 0.3 * (inter / len(a | b))


def _z(values: list[float]) -> list[float]:
    if len(values) < 2:
        return [0.0] * len(values)
    m, s = statistics.fmean(values), statistics.pstdev(values)
    return [(v - m) / s if s > 1e-9 else 0.0 for v in values]


def match(noun: dict, pool: list[dict], style: dict | None = None, recent: list | None = None, now: float = 0.0, embed=None,
          passage: str | None = None) -> tuple[int | None, float, str]:
    """The pool scene that shows `noun` (visualnouns row), or None when nothing fits.
    recent: [(file, end_timeline_s)] of cutaways already placed; embed(phrase, i) -> cosine (SigLIP, when the catalogue model is loaded) or None."""
    b = ((style or {}).get("broll") or {})
    gap = float(b.get("reuse_gap_s", 150.0) or 0)
    phrase = str(noun.get("phrase") or "")
    live = [i for i, sc in enumerate(pool) if not sc.get("is_talking_head") and float(sc.get("quality", 6) or 0) >= 4 and sc.get("end", 0) > sc.get("start", 0)]
    if not live or not phrase:
        return None, 0.0, "nothing to choose from"
    base = [text_score(phrase, pool[i]) for i in live]
    if embed is not None:
        emb = []
        for i in live:
            try:
                emb.append(embed(phrase, i))
            except Exception:  # noqa: BLE001 - a missing model is no reason to lose the text match
                emb.append(None)
        if all(e is not None for e in emb):
            z = _z([float(e) for e in emb])
            squash = [1.0 / (1.0 + 2.718281828 ** (-x)) for x in z]            # z-scored within the pool (raw SigLIP scores do not compare across prompts)
            base = [0.5 * t + 0.5 * s for t, s in zip(base, squash)]
    scored = []
    for i, s0 in zip(live, base):
        sc, s = pool[i], s0
        if recent and gap > 0 and any(f == sc.get("file") and abs(now - end) < gap for f, end in recent):
            s -= 0.3                                                               # the same file again so soon "feels padded"
        if sc.get("mismatch"):
            s -= 0.2                                                               # a portrait shot in a landscape video (or the reverse): cropped hard
        mood = str(sc.get("mood") or "neutral")
        if mood in ("calm", "busy") and noun.get("mood") in ("calm", "busy") and mood != noun.get("mood"):
            s -= 0.1
        if passage and sc.get("found_for") and str(sc["found_for"]) == str(passage):
            s += 0.2
        scored.append((s, i))
    scored.sort(reverse=True)
    best, bi = scored[0]
    med = statistics.median(s for s, _ in scored)
    if best < THRESHOLD:
        return None, round(best, 3), "nothing in the footage is close to it"
    if len(scored) > 1 and best < STRONG and best - med < CLEAR_OVER_MEDIAN:        # (a strong match needs no margin: two shots of the same thing are both good)
        return None, round(best, 3), "nothing stands out from the rest"
    return bi, round(best, 3), "matched"


def _clause_end(words: list[dict], j0: int, lo_t: float, hi_t: float) -> float | None:
    """The end of the first comma / sentence end after word j0 whose time lies in [lo_t, hi_t]."""
    for j in range(j0, min(len(words), j0 + 40)):
        w = words[j]
        if CLAUSE_END.search(str(w.get("text", "")).strip()) and lo_t <= float(w.get("end", 0)) <= hi_t:
            return float(w["end"])
        if float(w.get("start", 0)) > hi_t:
            break
    return None


def _inside(t0: float, t1: float, spans) -> bool:
    return any(t0 < b and t1 > a for a, b in spans)


def _pairs(wins) -> list[tuple[float, float]]:
    from . import poolrules as _PR                        # (windows arrive as pairs or as dicts: one reader)
    return _PR.window_pairs(wins)


def place(nouns: list[dict], pool: list[dict], words: list[dict], style: dict | None, out_len: float, protected=(), exclude_windows=(), existing=(),
          embed=None, intro_free: float = INTRO_FREE_S) -> tuple[list[dict], list[dict]]:
    """-> (cutaways, unmatched). One cutaway per concrete noun, in time order. `existing`: cutaways already fixed (never overlapped; the speaker is
    kept visible between them). unmatched: nouns that deserved a picture and got none ({phrase, kind, at, why}): graphics / stock read them."""
    b = ((style or {}).get("broll") or {})
    lo, hi = (float(x) for x in (b.get("shot_s") or [2.0, 5.0]))
    lead = float(b.get("lead_s", 0.15))
    face_back = float(b.get("face_back_within_s", 15.0))
    share = float(b.get("share", 0.3))
    budget = share * max(0.0, out_len) - sum(float(c.get("length", 0)) for c in existing)
    blocked = [(float(a), float(z)) for a, z in [*protected, *_pairs(exclude_windows)]]
    placed: list[dict] = [dict(c) for c in existing]
    mine: list[dict] = []
    unmatched: list[dict] = []
    recent = [(c.get("file"), float(c["at"]) + float(c.get("length", 0))) for c in placed]
    for n in sorted(nouns, key=lambda n: (int(n.get("word_first", 0)))):
        try:
            w0 = words[int(n["word_first"])]
            at = float(w0["start"]) - lead
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if float(n.get("concrete", 0)) < CONCRETE or n.get("kind") == "abstract":
            continue
        if n.get("kind") == "number":
            unmatched.append({"phrase": n["phrase"], "kind": "number", "at": round(at, 2), "why": "a number: a graphic, not a picture"})
            continue
        at = max(0.0, at)
        if at < intro_free:
            continue
        i, score, why = match(n, pool, style, recent, at, embed)
        if i is None:
            unmatched.append({"phrase": n["phrase"], "kind": n.get("kind", "object"), "at": round(at, 2), "why": why, "score": score})
            continue
        sc = pool[i]
        room = float(sc["end"]) - float(sc["start"])
        if room < min(lo, 1.0):
            continue
        length = (lo + hi) / 2.0
        end_c = _clause_end(words, int(n["word_first"]), at + lo, at + hi)
        if end_c is not None:
            length = end_c - at
        length = max(min(lo, room), min(length, hi, room, face_back))
        if _inside(at, at + length, blocked):
            continue
        if any(at < float(c["at"]) + float(c.get("length", 0)) + SPEAKER_BETWEEN_S and at + length + SPEAKER_BETWEEN_S > float(c["at"]) for c in placed):
            continue
        if length > budget + 1e-6:
            continue
        c = {"at": round(at, 2), "length": round(length, 2), "src": round(float(sc["start"]), 2), "scene": i, "what": sc.get("what", ""),
             "why": f"shows '{n['phrase']}'", "file": sc.get("file"), "photo": sc.get("photo"), "noun": n["phrase"], "phrase": n["phrase"],
             "score": score, "device": "noun", "word_first": int(n["word_first"]), "noun_at": round(float(w0["start"]), 2)}
        if sc.get("sid"):
            c["sid"] = sc["sid"]
        placed.append(c)
        mine.append(c)
        recent.append((c["file"], c["at"] + c["length"]))
        budget -= length
    mine.sort(key=lambda c: c["at"])
    return mine, unmatched


def snap_beat(at: float, beats: list[float], words: list[dict], noun_start: float | None = None) -> float:
    """The nearest beat to `at` that does not land inside a word and is no later than the noun's first word; else `at` itself."""
    def in_word(t: float) -> bool:
        return any(float(w.get("start", 0)) + 0.01 < t < float(w.get("end", 0)) - 0.01 for w in words)
    latest = noun_start if noun_start is not None else float("inf")
    for t in sorted(beats, key=lambda t: abs(t - at)):
        if abs(t - at) > 1.0:
            break
        if t <= latest + 1e-6 and not in_word(t):
            return round(float(t), 2)
    return at


def photo_ok(c: dict, nouns: list[dict], words: list[dict], pool: list[dict], style: dict | None = None) -> bool:
    """A still photo is placed only where it matches a noun spoken then (score >= THRESHOLD), unless the style says photos: all."""
    if str(((style or {}).get("broll") or {}).get("photos", "match")) == "all":
        return True
    sc = pool[c["scene"]] if isinstance(c.get("scene"), int) and 0 <= c["scene"] < len(pool) else None
    if sc is None:
        return False
    for n in nouns:
        try:
            t = float(words[int(n["word_first"])]["start"])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if float(c["at"]) - 1.0 <= t <= float(c["at"]) + float(c.get("length", 3)) and text_score(n["phrase"], sc) >= THRESHOLD:
            return True
    return False


def stock_queries(unmatched: list[dict], limit: int = 6) -> list[str]:
    """Search phrases for the stock library from the nouns that had no footage (concrete things only; numbers and abstractions are not stock)."""
    out: list[str] = []
    for u in unmatched:
        if u.get("kind") in ("object", "place", "action") and u["phrase"] not in out:
            out.append(u["phrase"])
    return out[:limit]


def report_line(c: dict, pool: list[dict] | None = None) -> str:
    """'41.2 s: 'the oven' -> C0012 (kitchen counter, 0.71)'."""
    return f"{float(c['at']):.1f} s: '{c.get('noun') or c.get('phrase') or c.get('device', 'cutaway')}' -> {c.get('file') or '?'} " \
           f"({str(c.get('what', ''))[:40]}{', ' + format(float(c['score']), '.2f') if c.get('score') is not None else ''})"


def run(ai, cache: dict | None, sents, words: list[dict], proposals: list[dict], pool: list[dict], style: dict | None, out_len: float,
        exclude_windows=(), embed=None) -> tuple[list[dict], list[dict], dict]:
    """The whole pass: name what the words show (visualnouns), place a cutaway on each concrete noun, then let the AI's own proposals fill in where
    they do not clash, are not over a protected sentence and (photos) match something said. -> (cutaways, unmatched nouns, info).
    When the nouns could not be had (no AI, an empty answer) the proposals come back untouched: placement by words is an improvement, never a risk."""
    from . import visualnouns as VN
    info = VN.extract(ai, sents, words, cache)
    if not info["ok"] or not (info["nouns"] or info["tags"]):
        return proposals, [], info
    b = ((style or {}).get("broll") or {})
    prot = VN.protected_spans(info["tags"], sents, b.get("protect"))
    blocked = [*prot, *_pairs(exclude_windows)]
    mine, unmatched = place(info["nouns"], pool, words, style, out_len, prot, exclude_windows, embed=embed)
    budget = float(b.get("share", 0.3)) * max(0.0, out_len) - sum(c["length"] for c in mine)
    kept = list(mine)
    for c in sorted(proposals, key=lambda c: c["at"]):
        c0, c1 = float(c["at"]), float(c["at"]) + float(c["length"])
        if _inside(c0, c1, blocked):
            continue
        if c.get("photo") and not c.get("must") and not photo_ok(c, info["nouns"], words, pool, style):
            continue
        if any(c0 < float(o["at"]) + float(o["length"]) + SPEAKER_BETWEEN_S and c1 + SPEAKER_BETWEEN_S > float(o["at"]) for o in kept):
            continue
        if float(c["length"]) > budget + 1e-6 and not c.get("must"):
            continue
        budget -= float(c["length"])
        kept.append(dict(c, device=c.get("device", "ai")))
    kept.sort(key=lambda c: c["at"])
    return kept, unmatched, info
