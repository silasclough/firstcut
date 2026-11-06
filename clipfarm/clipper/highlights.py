"""A short video out of a LOT of footage ("1 minute from 20 minutes", or from 10 hours).

story.py keeps a share of every part, which is right for a vlog but wrong when only the best 5% is wanted. Here the AI works in two
passes so no single call has to read everything:

    1. candidates()  for every ~5 minutes of talk, the best few moments (a complete thought each), with a score and one line on what happens
    2. assemble()    one call sees ALL candidates (just the one-liners) and the brief, and orders the ones that make the best video

The result is sentence ranges (in play order), the same shape the plain-words editor works with.
"""

from __future__ import annotations

from ..core.log import get_logger
from . import moments as M
from .story import _chunks, _file_of

log = get_logger("clipper.highlights")
MAX_CAND_S = 32.0
PROMO_CAND_S = 11.0

CAND_SYSTEM = (
    "You are a video editor looking for the best moments in raw footage. From the numbered sentences you are shown, pick up to "
    "{n} moments that would make a viewer stop scrolling: a surprise, a strong opinion, a funny line, a payoff, a clear useful "
    "point, an emotional beat. Each moment is a run of consecutive sentences (first..last, inclusive) that makes complete sense "
    "on its own and starts and ends on a whole sentence, at most about 30 seconds. Score each 1-10 for how strong it is as a "
    "highlight (be harsh: 8+ is rare). Skip chit-chat, setup, repeats and false starts. 'what' = one plain sentence saying what "
    "happens in it. If nothing here is worth keeping, return an empty list.")
CAND_SCHEMA = {"type": "object", "properties": {"moments": {"type": "array", "items": {"type": "object", "properties": {
    "first": {"type": "integer"}, "last": {"type": "integer"}, "score": {"type": "number"}, "what": {"type": "string"}},
    "required": ["first", "last", "score", "what"]}}}, "required": ["moments"]}

ASM_SYSTEM = (
    "You are the editor cutting ONE video of about {target:.0f} seconds from a pile of candidate moments. Choose the moments that "
    "make the best video and put them in the order that works: an opening that hooks in the first seconds, a clear flow, a strong "
    "ending. Chronological order is fine unless the creator's notes ask for something else or another order clearly plays better. "
    "Follow the creator's notes. Stay within about 10% of the target length (each moment lists its length). Never use a moment "
    "twice. Return the chosen ids in play order, a title, and one sentence on what the video says.")
ASM_SCHEMA = {"type": "object", "properties": {"order": {"type": "array", "items": {"type": "integer"}},
                                               "title": {"type": "string"}, "summary": {"type": "string"}},
               "required": ["order", "title", "summary"]}


def candidates(ai, sents: list[M.Sentence], files: list[dict], brief: str = "", per_chunk: int = 3, progress=None,
               style: str = "story") -> list[dict]:
    promo = style in ("promo", "ad")
    per_chunk = max(per_chunk, 6) if promo else per_chunk
    max_len = PROMO_CAND_S if promo else MAX_CAND_S
    chunks = _chunks(sents, files, max_words=700)
    out: list[dict] = []
    for n, chunk in enumerate(chunks, 1):
        if progress:
            progress(n - 1, len(chunks))
        lo, hi = chunk[0].idx, chunk[-1].idx
        by = {s.idx: s for s in chunk}
        lines = [f"  [{s.idx}] {s.text}" for s in chunk]
        prompt = ((f"The creator's notes about the video they want: {brief}\n" if brief else "")
                  + f"Footage part {n} of {len(chunks)} (file: {files[_file_of(chunk[0].start, files)]['name']}). Sentences:\n" + "\n".join(lines))
        try:
            res = ai.complete_json("highlight_candidates", CAND_SYSTEM.format(n=per_chunk) + (PROMO_CAND if promo else "") + (AD_CAND if style == "ad" else ""),
                                   prompt, CAND_SCHEMA)
        except Exception as e:  # noqa: BLE001 - one bad part just contributes nothing
            log.warning(f"candidates for part {n} failed: {e}")
            continue
        for m in res.data.get("moments") or []:
            try:
                a, b, sc = int(m["first"]), int(m["last"]), float(m["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (lo <= a <= b <= hi) or a not in by or b not in by:
                continue
            while b > a and by[b].end - by[a].start > max_len:                # too long: end it earlier on a whole sentence
                b -= 1
            out.append({"first": a, "last": b, "start": by[a].start, "end": by[b].end, "score": max(1.0, min(10.0, sc)),
                        "what": " ".join(str(m.get("what", "")).split())[:220]})
    if progress:
        progress(len(chunks), len(chunks))
    out.sort(key=lambda c: c["start"])
    for i, c in enumerate(out):
        c["id"] = i
    return out


PROMO_CAND = (" These will be cut into a fast, upbeat promo, so look for SHORT punchy soundbites: one clear benefit, claim, number, "
              "promise or feeling per moment, 2 to 8 seconds each, said with energy. Prefer lines a viewer could repeat; skip anything that "
              "needs the question before it to make sense.")
AD_CAND = (" Never pick a line where the speaker is unsure, guessing, correcting themselves or talking about the interview itself (for example 'I\'m not sure how long we\'ve been open'): ads only use confident, finished claims. This is an ADVERTISEMENT for a business. Pick lines that SELL it: what makes it different, a benefit to the customer, expertise, "
           "service, trust, the results people get, why people come back. Prefer short, direct, confident statements over anecdotes. "
           "Skip stories about a particular customer or product, jokes, hedging ('kind of', 'I guess'), and anything that is a question "
           "or needs setup. A line is only worth taking if it makes someone want to visit.")
AD_ASM = (" It is an ad, so every soundbite must be a selling point or proof (no anecdotes, no stories about individual shoes or people "
          "unless the point is the result for the customer). End with a line that works as the closing statement or invitation, and make "
          "sure that last line is a COMPLETE sentence that sounds finished.")
PROMO_ASM = (" Build it like a promo with a story shape: (1) open with the single strongest line as a hook (under 6 seconds), "
             "(2) a few moments that name the problem or the need, (3) the key points / benefits, one soundbite each, (4) proof or "
             "emotion (a customer, a result, community), (5) finish on a closing or call-to-action line. Use MANY short soundbites "
             "(roughly one every 3 to 6 seconds: for a minute that is 10 to 16 of them), never one long stretch. Every soundbite must stand "
             "on its own.")


def assemble(ai, cands: list[dict], target_s: float, brief: str = "", title: str = "", style: str = "story") -> tuple[list[tuple[int, int]], str, str]:
    """-> (sentence ranges in play order, title, summary)."""
    if not cands:
        return [], title, ""
    pool = sorted(cands, key=lambda c: -c["score"])[:400]                     # the AI never needs to see more than the best few hundred
    pool.sort(key=lambda c: c["start"])
    lines = [f"  id {c['id']}: [{int(c['start'] // 60)}:{int(c['start'] % 60):02d}] {c['end'] - c['start']:.0f}s score {c['score']:.0f}  {c['what']}"
             for c in pool]
    prompt = ((f"The creator's notes: {brief}\n" if brief else "") + (f"Working title: {title}\n" if title else "")
              + f"Target length: about {target_s:.0f} seconds.\nCandidate moments (in the order they were shot):\n" + "\n".join(lines))
    by = {c["id"]: c for c in pool}
    try:
        res = ai.complete_json("highlight_assemble", ASM_SYSTEM.format(target=target_s) + (PROMO_ASM if style in ("promo", "ad") else "") + (AD_ASM if style == "ad" else ""),
                               prompt, ASM_SCHEMA)
        order = [int(i) for i in res.data.get("order") or [] if int(i) in by]
        ttl, summ = str(res.data.get("title") or title).strip()[:90], str(res.data.get("summary") or "").strip()
    except Exception as e:  # noqa: BLE001 - fall back to the best moments in the order shot
        log.warning(f"assembling highlights failed ({e}); taking the highest-scored moments")
        order, ttl, summ = [], title, ""
    seen, chosen = set(), []
    for i in order:
        if i not in seen:
            seen.add(i)
            chosen.append(i)
    total = lambda ids: sum(by[i]["end"] - by[i]["start"] for i in ids)         # noqa: E731
    while chosen and total(chosen) > target_s * 1.12:                           # too long: drop the weakest
        chosen.remove(min(chosen, key=lambda i: by[i]["score"]))
    spare = sorted((c for c in pool if c["id"] not in seen), key=lambda c: -c["score"])
    while total(chosen) < target_s * 0.85 and spare:                            # too short: add the best that are left, in time order
        nxt = spare.pop(0)
        if total(chosen) + (nxt["end"] - nxt["start"]) > target_s * 1.12:
            continue
        chosen.append(nxt["id"])
        seen.add(nxt["id"])
        if not order:
            chosen.sort(key=lambda i: by[i]["start"])
    if not order:
        chosen.sort(key=lambda i: by[i]["start"])
    return [(by[i]["first"], by[i]["last"]) for i in chosen], ttl, summ
