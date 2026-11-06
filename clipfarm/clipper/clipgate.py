"""A last look at a finished clip before it can be posted: the mistakes a person sees in two seconds and a score of 9 out of 10 does not.

Learned from clips that went up and should not have: a 21-second clip that was the END of a car review ("so that concludes my review of the
Macan", cut off mid-word, plus an aside about buying one - an advert), and a few seconds of a game's face cam with nothing happening.
"""

from __future__ import annotations

import re

MIN_SECONDS = 7.0
MAX_SECONDS = 40.0               # October 2026: posts of 35 s and longer kept ~21% of viewers and got about 600 views against ~1,000 for 12-25 s (your own uploads are exempt, see relay.ingest)
OUTRO = re.compile(r"\b(conclud\w+ my (?:review|video|thoughts)|that wraps (?:it )?up|(?:let me )?wrap(?:ping)? (?:it|this|things|the video) up|thanks? (?:so much )?for (?:watching|listening)|"
                   r"see you (?:in the )?next (?:one|video|time)|(?:like and )?subscribe to (?:the|my|our) channel|like,? comment,? and subscribe|don'?t forget to (?:like|subscribe)|"
                   r"link in the (?:description|bio)|use (?:the )?(?:promo )?code \w+|(?:this|today'?s) (?:video|episode|reel) is sponsored by|today'?s sponsor|check (?:it )?out (?:in|at|below)|in the comments below|"
                   r"final thoughts|that'?s (?:it|all) for (?:today|this|now))\b", re.I)
BUYING = re.compile(r"\b(bargain|per month|a month|lease deal|0% (?:apr|finance)|trade[- ]in|dealership|on sale now|limited time|discount|promo code)\b", re.I)


BARE_QUESTION = re.compile(r"^(?:how|what|why|where|when|who)\s+(?:do|did|does|am|are|is|was|were|can|could|should)\s+(?:i|you|we)\b|"                 # "How do I do that?"
                           r"^(?:how|what)\s+(?:you|u)\s+(?:doing|doin|been)\b|^what\s+just\s+happened\b", re.I)
FILLER = re.compile(r"^(?:wait|wow|omg|whoa|bruh|yo|hey|hi|hello|look|watch|listen)(?![-\w])|\byou won'?t believe\b|\bwill (?:blow|shock|surprise) you\b|"
                    r"\bgoes? (?:viral|wrong)\b|\b(?:crazy|wild) (?:story|moment|thing)\b|\bunexpected (?:moment|thing|twist)\b|"
                    r"\b(?:funny|best|epic|insane|crazy|amazing) moment\b|\bthis (?:video|clip|moment)\b|\breacts? to (?:an? )?(?:\w+ )?(?:moment|thing|clip|video)\b", re.I)


def _concrete(words: list[str]) -> bool:
    """A number, or a name (a capitalised word that is not the first): the title is about something."""
    return any(re.search(r"\d", w) for w in words) or any(w[:1].isupper() and len(w) >= 3 for w in words[1:])


def weak_title(title: str | None) -> str | None:
    """-> why a title sells nothing (None = fine, or no title to judge). The 250-300 view posts were titled "How do I do that?", "How you doing?" and
    "Streamer reacts to unexpected moment"; the 2,800+ ones state a claim or a stake with a concrete name or number in it. A title with a number or a name in it
    is never called filler; a bare "how do I / how you doing" question is weak whatever else it says."""
    t = " ".join(str(title or "").split())
    if not t:
        return None
    words = t.split()
    if len(words) < 2:
        return "a single word"
    if BARE_QUESTION.search(t) and len(words) <= 7:
        return "a bare question"
    hit = FILLER.search(t)
    if hit and not _concrete(words) and (len(words) <= 5 or hit.start() > 0):                  # (a long sentence that merely starts with "Watch" says something)
        return "a greeting or label"
    return None


def problems(clip: dict, source: dict | None = None) -> list[str]:
    """-> the reasons this clip should not be posted (empty = fine)."""
    out: list[str] = []
    dur = float(clip.get("duration") or 0)
    text = " ".join(str(clip.get("transcript") or "").split())
    if dur and dur < MIN_SECONDS:
        out.append(f"only {dur:.0f} seconds long")
    if dur > MAX_SECONDS:
        out.append(f"{dur:.0f} seconds long (over {MAX_SECONDS:.0f}: most viewers have left by then)")
    if clip.get("title") is not None and (why := weak_title(clip.get("title"))):
        out.append(f"weak title \"{clip.get('title')}\" ({why})")
    fan_clip = (source or {}).get("kind") == "twitch_clip"                         # a fan-made Twitch clip: a reaction, often mid-sentence, never a sign-off or a pitch
    if text and not fan_clip and not re.search(r"[.!?][\"')\]]*$", text) and len(text) < 1700:      # (long transcripts are stored cut off)
        out.append("ends in the middle of a sentence")
    if OUTRO.search(text) and not fan_clip:
        out.append("it is the wrap-up / sign-off of the video, not a moment")
    if not fan_clip and len(BUYING.findall(text)) >= 1 and re.search(r"review|deal|buy|price", " ".join(str((source or {}).get(k) or "") for k in ("title", "topic")), re.I):
        out.append("sounds like a sales pitch")
    if (source or {}).get("topic") in ("gaming", "game", "gameplay") and clip.get("layout") in ("track",) and dur < 25:
        out.append("short, tightly cropped gameplay face cam")
    for i in (clip.get("qa") or {}).get("issues", []):                              # the finished-file check (qa.py), run on every clip when it is rendered
        if i.get("level") == "bad":
            out.append(i["what"])
    return out
