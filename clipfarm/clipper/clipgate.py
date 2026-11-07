"""A last look at a finished clip before it can be posted: the mistakes a person sees in two seconds and a score of 9 out of 10 does not.

Learned from clips that went up and should not have: a 21-second clip that was the END of a car review ("so that concludes my review of the
Macan", cut off mid-word, plus an aside about buying one - an advert), and a few seconds of a game's face cam with nothing happening.
"""

from __future__ import annotations

import re

GATE_VERSION = 3                 # bump when a check is added: the scheduler looks again (on the finished file) at every clip that was last looked at under an older version
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


GENERIC = re.compile(r"^(?:taste|watch|listen to|look at|try|feel|see) this\b|\bbefore you (?:even )?(?:touch|know|realize|realise|think|try|taste|see|notice)\b|"
                     r"^your (?:brain|body|mind|heart|gut|eyes|ears|skin) (?:decides|knows|wants|controls|hides|lies|tricks)\b|"
                     r"^how to (?:make|earn|get|build|become|be|win|achieve|save|find)\b.{0,30}\b(?:million|money|rich|wealth|millionaire|successful|success|famous|happy|confident)\b", re.I)
GAME_WORDS = re.compile(r"(?<![a-z0-9])(?:gta ?(?:v|rp|5|6)|nopixel|paleto|valorant|fortnite|minecraft|roblox|call of duty|warzone|apex legends|counter-?strike|cs2|csgo|rainbow six|"
                        r"nba 2k\d*|elden ring|let'?s play|speed ?run|playthrough|gameplay|rp server|league of legends|dota ?2?|rocket league|"
                        r"fall guys|pubg|destiny 2|genshin|twitch plays)(?![a-z0-9])", re.I)     # (only names and game talk that mean a game: not "gaming the system", "roleplay", "overwatch", "game of life", a bare "GTA")
GAME_TAGS = {"gaming", "gameplay", "video games", "gta", "gta rp", "roleplay", "let's play"}     # (a source's own tag list is exact: here the broad words are safe)


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
    if GENERIC.search(t):
        return "a generic hook with no subject"
    hit = FILLER.search(t)
    if hit and not _concrete(words) and (len(words) <= 5 or hit.start() > 0):                  # (a long sentence that merely starts with "Watch" says something)
        return "a greeting or label"
    return None


def gaming_why(clip: dict, source: dict | None = None, cfg=None) -> str | None:
    """-> why this is game footage (None = fine). Real-life content only: the words of the clip's own title and hashtags, and of the source video's title, channel
    and topic, are matched against a list of games and game talk (plus clipper.auto.avoid_words / avoid_channels from the config when it is given). Streamers
    talking to their camera are fine; this only goes by what the clip and its source say, so a gameplay clip with a neutral title and an unlisted channel
    still gets through (the engine's source check looks at the pictures)."""
    src = source or {}
    parts = [clip.get("title"), clip.get("caption"), src.get("title"), src.get("channel"), src.get("topic"), " ".join(map(str, src.get("tags") or []))]
    text = " ".join(str(x) for x in parts if x).lower()
    if not text:
        return None
    m = GAME_WORDS.search(text)
    if m:
        return f"game footage ({m.group(0)}): real-life content only"
    if str(src.get("topic") or "").strip().lower() in ("gaming", "game", "games", "gameplay", "video games"):
        return f"game footage (topic {src.get('topic')}): real-life content only"
    if cfg is not None:
        try:
            ch = {str(c).lower() for c in (cfg.get("clipper.auto.avoid_channels") or [])}
            if str(src.get("channel") or "").lower() in ch:
                return f"game channel ({src.get('channel')}): real-life content only"
            own = " ".join(str(x) for x in (src.get("title"), src.get("channel"), src.get("topic")) if x).lower()      # (the source's words, as purge_avoided always matched them)
            padded = f" {own} "
            for w in (str(w).lower() for w in (cfg.get("clipper.auto.avoid_words") or [])):
                if f" {w} " in padded or (len(w) > 4 and w in own):
                    return f"game footage ({w}): real-life content only"
        except Exception:  # noqa: BLE001 - a broken config is never a reason to lose a clip
            pass
    return None


def check_file(video, clip: dict | None = None) -> dict:
    """The last look at a FINISHED file (the same checks the render makes: moving picture, a real opening, burned-in promo, double captions). -> a clip's `qa`
    entry: {"issues": [...], "gate": GATE_VERSION}. Run again by the scheduler on clips that were rendered before a check existed."""
    from pathlib import Path
    from . import opening
    clip = clip or {}
    return {"issues": opening.check(Path(video), " ".join(str(clip.get("transcript") or "").split()), strict=True), "gate": GATE_VERSION}


def fresh(qa: dict | None) -> bool:
    """True when a clip's `qa` was made under the current set of checks."""
    return bool(qa) and int(qa.get("gate") or 0) >= GATE_VERSION


def problems(clip: dict, source: dict | None = None, cfg=None) -> list[str]:
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
    if why := gaming_why(clip, source, cfg):
        out.append(why)
    for i in (clip.get("qa") or {}).get("issues", []):                              # the finished-file check (qa.py), run on every clip when it is rendered
        if i.get("level") == "bad":
            out.append(i["what"])
    return out
