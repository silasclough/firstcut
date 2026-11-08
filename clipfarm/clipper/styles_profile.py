"""One `style` record per video: every number that shapes a cut (WP4 of docs/plans/editor-improvement-plan.md).

The planner, the validator, the render passes and the MCP read the same object. It is built in layers, lowest first:

    DEFAULTS[profile]   <   the person's "How it edits" points   <   the notes of this video   <   the options (MCP / the page)

A layer only sets what it states ("fast-paced" sets the pace and nothing else). `style["_set"]` lists the dotted keys a person, the notes or the
options set, so the render can tell "stated" from "default" (a promo's default pace is tight, but only a stated pace changes the render).

Profiles that today's code already renders (`longform_talk`, `promo`, `ad`) carry TODAY'S numbers first, so the suite stays green and nothing
changes for a person who writes no notes; `TARGETS` holds the numbers the plan's section 2.3 wants for them, for the follow-up that moves them.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from ..core.log import get_logger

log = get_logger("clipper.styles_profile")

PROFILES = ("short_talk", "reel", "interview_clip", "longform_talk", "seminar", "event_recap", "promo", "ad")
PACE_NAMES = ("slow", "natural", "tight", "fast")
CAPTION_STYLES = ("none", "phrase", "karaoke")
GRAPHICS_POLICIES = ("none", "title", "informational", "all")
MUSIC_POLICIES = ("none", "under_broll", "throughout")
GRAPHIC_KINDS = ("lower", "stat", "quote", "chapter", "reference", "list")
LIMITS = {"min_clip_s": 7, "max_minutes": 120}

# ------------------------------------------------------------------------------------------------------------------ the base object (section 2.2)
_BASE: dict[str, Any] = {
    "profile": "longform_talk",
    "aspect": "source",
    "hook": {"speech_by_s": 0.5, "face_by_s": 1.0, "key_claim_by_s": 3.0, "title_from_frame0": True},
    "pace": {"name": "natural", "gap_shrink_to_s": 0.25, "gap_min_sentence_s": 0.35, "gap_idea_s": 0.8, "gap_shrink_over_s": 0.5, "cut_margin_s": 0.08,
             "max_hold_s": 12.0, "visual_change_every_s": [3.0, 5.0], "fillers": "cut", "atempo_if_wpm_below": 130, "atempo": 1.08},
    "takes": {"policy": "best", "keep_deliberate_repeats": True},
    "multicam": {"min_hold_s": 5.0, "baseline_hold_s": [15.0, 30.0], "switch_on": ["speaker_change", "thought_end"], "wide_share": 0.3,
                 "no_switch_inside_word": True},
    "broll": {"share": 0.30, "per_s": None, "show_s": None, "shot_s": [2.0, 5.0], "shot_moving_max_s": 8.0, "lead_s": 0.15, "face_back_within_s": 15.0,
              "protect": ["hook", "punchline", "key_statement", "direct_address", "emotional"], "reuse_gap_s": 150.0, "never_speaker_as_broll": True,
              "photos": "if_fit", "stock": "never", "jl_cut_s": 0.3, "allow_pip": True, "allow_split": True},
    "captions": {"style": "none", "words_per_group": [3, 5], "font_px_at_1080w": 64, "min_font_px": 48, "highlight": "one_colour", "case": "upper",
                 "band_y": [0.56, 0.65], "avoid_face": True, "keep_under_fullscreen": True},
    "graphics": {"policy": "informational", "max_per_minute": 6, "min_gap_s": 1.5, "one_at_a_time": True, "kinds": list(GRAPHIC_KINDS),
                 "quotes_verbatim": True, "lower_third_s": 4.5, "name_once": True, "chapter_cards_from_outline": True, "theme": "auto"},
    "music": {"policy": "under_broll", "bed_under_speech_lu": -20, "bed_in_gaps_lu": -10, "duck_db": 10, "attack_ms": 50, "release_ms": 400,
              "target_lufs": -14, "voice_lufs": -16, "beat_sync": False},
    "sfx": {"max_per_30s": 2, "same_sound_twice_in_row": False, "on": ["fullscreen_card"]},
    "transitions": {"default": "cut", "dissolve_max_s": 2.0, "dissolve_when": ["time_passes", "prayer"], "fade_out_s": 1.0},
    "length": {"target_s": None, "band": 0.10, "min_s": 7, "loop_if_under_s": 15, "end_on_last_word": True, "rehook_every_s": [45, 90]},
    "chronology": "keep",
    "speakers": {"remove_interviewer": "auto"},
    "voice_over": False,
}

# What each profile changes (section 2.1 / 2.3). longform_talk, promo and ad hold TODAY'S numbers (see the module note); TARGETS below are section 2.3's.
_OVER: dict[str, dict[str, Any]] = {
    "short_talk": {
        "aspect": "9:16", "pace": {"max_hold_s": 5.0, "gap_shrink_to_s": 0.25, "gap_min_sentence_s": 0.35, "gap_idea_s": 0.6, "visual_change_every_s": [3.0, 5.0]},
        "takes": {"policy": "last"}, "broll": {"share": 0.35, "shot_s": [1.5, 3.0]},
        "captions": {"style": "karaoke", "words_per_group": [1, 3]}, "graphics": {"policy": "informational"}, "music": {"policy": "under_broll"},
        "length": {"min_s": 7, "loop_if_under_s": 15}, "sfx": {"max_per_30s": 3, "on": ["whoosh_rare"]}},
    "reel": {
        "aspect": "9:16", "pace": {"max_hold_s": 4.0}, "broll": {"share": 0.80, "shot_s": [2.0, 4.0]}, "captions": {"style": "phrase"},
        "graphics": {"policy": "title"}, "music": {"policy": "throughout", "beat_sync": True}, "length": {"target_s": None}},
    "interview_clip": {
        "aspect": "9:16", "pace": {"max_hold_s": 6.0, "gap_shrink_over_s": 0.7, "gap_shrink_to_s": 0.35, "gap_min_sentence_s": 0.4, "gap_idea_s": 0.8, "fillers": "keep_soft"},
        "broll": {"share": 0.05, "shot_s": [1.5, 3.0]}, "captions": {"style": "phrase", "case": "mixed", "highlight": "none"},
        "graphics": {"policy": "none", "kinds": ["lower"]}, "music": {"policy": "none"}, "multicam": {"min_hold_s": 5.0},
        "sfx": {"max_per_30s": 0, "on": []}, "length": {"min_s": 30}},
    "longform_talk": {
        "pace": {"max_hold_s": 12.0, "gap_shrink_to_s": 0.3, "gap_min_sentence_s": 0.4, "gap_idea_s": 0.8}, "broll": {"share": 0.30, "per_s": 18.0, "show_s": 3.2, "shot_s": [1.5, 4.5]},
        "multicam": {"min_hold_s": 6.0, "baseline_hold_s": [9.0, 12.0]}, "captions": {"style": "none"}, "graphics": {"policy": "informational"},
        "music": {"policy": "under_broll"}},
    "seminar": {
        "pace": {"max_hold_s": 30.0, "gap_shrink_over_s": 0.6, "gap_shrink_to_s": 0.4, "gap_min_sentence_s": 0.5, "gap_idea_s": 1.0, "fillers": "keep_soft"}, "broll": {"share": 0.15, "shot_s": [2.5, 5.0], "protect": ["hook", "key_statement",
                                                                                                                                       "direct_address", "emotional", "prayer"]},
        "captions": {"style": "none"}, "graphics": {"policy": "title", "kinds": ["chapter", "reference", "lower", "quote"], "chapter_cards_from_outline": True},
        "music": {"policy": "none"}, "multicam": {"min_hold_s": 5.0, "wide_share": 0.3}, "length": {"target_s": None}},
    "event_recap": {
        "pace": {"max_hold_s": 4.0}, "broll": {"share": 0.80, "shot_s": [1.5, 5.0]}, "captions": {"style": "none"}, "graphics": {"policy": "title"},
        "music": {"policy": "throughout", "beat_sync": True}, "chronology": "keep", "sfx": {"max_per_30s": 1, "on": ["drop"]}},
    "promo": {
        "pace": {"name": "tight", "max_hold_s": 12.0}, "broll": {"share": 0.65, "per_s": 6.0, "show_s": 3.2, "shot_s": [1.2, 3.2]}, "captions": {"style": "phrase"},
        "graphics": {"policy": "informational"}, "music": {"policy": "throughout"}},
    "ad": {
        "pace": {"name": "tight", "max_hold_s": 12.0}, "broll": {"share": 0.65, "per_s": 3.0, "show_s": 3.2, "shot_s": [1.2, 3.2]}, "captions": {"style": "phrase"},
        "graphics": {"policy": "informational", "kinds": ["stat", "quote", "lower"]}, "music": {"policy": "throughout"}},
}

# Section 2.3's numbers for the three profiles above that keep today's values for now (the follow-up moves DEFAULTS to these).
TARGETS = {"longform_talk": {"pace.max_hold_s": 15.0, "broll.share": 0.30, "broll.shot_s": [2.0, 5.0], "multicam.min_hold_s": 5.0},
           "promo": {"pace.max_hold_s": 6.0, "broll.share": 0.65, "broll.shot_s": [1.2, 3.2]},
           "ad": {"pace.max_hold_s": 3.0, "broll.share": 0.65, "broll.shot_s": [1.2, 3.2]}}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _build() -> dict[str, dict]:
    out = {}
    for p in PROFILES:
        o = _merge(_BASE, _OVER.get(p, {}))
        o["profile"] = p
        out[p] = o
    return out


DEFAULTS: dict[str, dict] = _build()

# ------------------------------------------------------------------------------------------------------------------ resolve / validate


def _leaves(d: dict, prefix: str = "") -> list[str]:
    out = []
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out += _leaves(v, key + ".")
        else:
            out.append(key)
    return out


def layer(*parts: dict | None) -> dict:
    """Overrides stacked lowest first: a later layer wins key by key."""
    out: dict = {}
    for p in parts:
        if p:
            out = _merge(out, p)
    return out


def resolve(profile: str, overrides: dict | None = None, base: dict | None = None) -> dict:
    """The full style object for `profile` with `overrides` (a nested partial dict) laid over it. `_set` lists the dotted keys the overrides set
    (`base` changes the defaults without counting as stated: what a legacy kind adds to its profile)."""
    profile = profile if profile in DEFAULTS else "longform_talk"
    over = {k: v for k, v in (overrides or {}).items() if not str(k).startswith("_") and k != "profile"}
    obj = _merge(_merge(DEFAULTS[profile], base or {}), over)
    obj["profile"] = profile
    obj["_set"] = sorted(_leaves(over))
    return obj


def stated(style: dict | None, key: str) -> bool:
    """Did the person, the notes or an option set this dotted key (rather than the profile's default)?"""
    return bool(style) and key in (style.get("_set") or [])


def get(style: dict | None, path: str, default: Any = None) -> Any:
    """style["a"]["b"] by 'a.b' (the profile's default when the object has none, then `default`)."""
    gone = object()

    def walk(root: Any) -> Any:
        cur = root
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return gone
            cur = cur[part]
        return cur
    cur = walk(style or {})
    if cur is gone:                                                          # (a key the object lacks: the base profile's; an explicit null stays null)
        cur = walk(DEFAULTS["longform_talk"])
    return default if cur is gone or cur is None else cur


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _range(v: Any) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 2 and all(_num(x) for x in v) and v[0] <= v[1]


def validate(style: dict) -> list[str]:
    """Plain-words problems with a style object (empty = fine)."""
    out: list[str] = []
    if not isinstance(style, dict):
        return ["the style is not an object"]
    if style.get("profile") not in PROFILES:
        out.append(f"profile must be one of {', '.join(PROFILES)}")
    ref = DEFAULTS["longform_talk"]

    def missing(d: dict, r: dict, path: str) -> None:
        for k, v in r.items():
            if k not in d:
                out.append(f"{path}{k} is missing")
            elif isinstance(v, dict) and isinstance(d.get(k), dict):
                missing(d[k], v, path + k + ".")
    missing(style, ref, "")
    p, b, c, g, m, ln = (style.get(k) or {} for k in ("pace", "broll", "captions", "graphics", "music", "length"))
    if p.get("name") is not None and p.get("name") not in PACE_NAMES:
        out.append(f"pace.name must be one of {', '.join(PACE_NAMES)}")
    for k in ("max_hold_s", "gap_shrink_to_s", "gap_min_sentence_s", "gap_idea_s", "gap_shrink_over_s", "cut_margin_s"):
        if k in p and (not _num(p[k]) or p[k] <= 0):
            out.append(f"pace.{k} must be a number above 0")
    if p.get("fillers") is not None and p.get("fillers") not in ("cut", "keep_soft", "keep"):
        out.append("pace.fillers must be cut, keep_soft or keep")
    if "share" in b and (not _num(b["share"]) or not 0 <= b["share"] <= 1):
        out.append("broll.share must be between 0 and 1")
    if "shot_s" in b and not _range(b["shot_s"]):
        out.append("broll.shot_s must be [shortest, longest] in seconds")
    elif "shot_s" in b and b["shot_s"][0] < 0.5:
        out.append("broll.shot_s: a shot shorter than half a second is a flash")
    if b.get("per_s") is not None and (not _num(b["per_s"]) or b["per_s"] <= 0):
        out.append("broll.per_s must be a number above 0 (or null)")
    if c.get("style") not in CAPTION_STYLES:
        out.append(f"captions.style must be one of {', '.join(CAPTION_STYLES)}")
    if "words_per_group" in c and not _range(c["words_per_group"]):
        out.append("captions.words_per_group must be [fewest, most] words")
    if g.get("policy") not in GRAPHICS_POLICIES:
        out.append(f"graphics.policy must be one of {', '.join(GRAPHICS_POLICIES)}")
    if m.get("policy") not in MUSIC_POLICIES:
        out.append(f"music.policy must be one of {', '.join(MUSIC_POLICIES)}")
    if ln.get("target_s") is not None and (not _num(ln["target_s"]) or ln["target_s"] < LIMITS["min_clip_s"] or ln["target_s"] > LIMITS["max_minutes"] * 60):
        out.append(f"length.target_s must be between {LIMITS['min_clip_s']} seconds and {LIMITS['max_minutes']} minutes")
    if style.get("chronology") not in ("keep", "free"):
        out.append("chronology must be keep or free")
    if (style.get("speakers") or {}).get("remove_interviewer") not in ("auto", "always", "never"):
        out.append("speakers.remove_interviewer must be auto, always or never")
    return out


def from_kind(kind: str, is_clip: bool = False) -> str:
    """video_kind's words (and the new ones) -> a profile name (the mapping in section 2.3)."""
    kind = str(kind or "").lower()
    if kind in PROFILES:
        return kind
    if kind == "youtube":
        return "short_talk" if is_clip else "longform_talk"
    if kind == "seminar":
        return "seminar"
    if kind == "interview":
        return "interview_clip" if is_clip else "longform_talk"
    if kind == "recap":
        return "event_recap"
    if kind in ("promo", "ad"):
        return kind
    return "short_talk" if is_clip else "longform_talk"                      # story, business, and anything unknown


def kind_overrides(kind: str, is_clip: bool = False) -> dict:
    """The part of a legacy kind that is not the profile itself (business shows every graphic; a long YouTube video is tighter and holds 10 s)."""
    kind = str(kind or "").lower()
    if is_clip:
        return {}
    if kind == "business":
        return {"graphics": {"policy": "all"}}
    if kind == "youtube":
        return {"pace": {"name": "tight", "max_hold_s": 10.0}, "broll": {"per_s": 10.0}}
    return {}


# ------------------------------------------------------------------------------------------------------------------ plain words for prompts


def mean_shot_s(style: dict | None) -> float:
    lo, hi = get(style, "broll.shot_s", [2.0, 5.0])
    return (float(lo) + float(hi)) / 2.0


def show_s(style: dict | None) -> float:
    """How long a cutaway that breaks up a long hold shows: the profile's own figure, else the average shot length."""
    explicit = get(style, "broll.show_s")
    return float(explicit) if explicit else mean_shot_s(style)


def per_s(style: dict | None) -> float:
    """Seconds between the starts of B-roll shots: the profile's own figure, else the average shot length over the share of the video it fills."""
    explicit = get(style, "broll.per_s")
    if explicit:
        return float(explicit)
    share = max(0.02, float(get(style, "broll.share", 0.3)))
    return round(mean_shot_s(style) / share, 3)


def describe(style: dict | None) -> str:
    """The style as five short plain lines, the same wording in every prompt."""
    s = style or DEFAULTS["longform_talk"]
    lo, hi = get(s, "broll.shot_s", [2.0, 5.0])
    cap = get(s, "captions.style", "none")
    gp = get(s, "graphics.policy", "informational")
    mp = get(s, "music.policy", "under_broll")
    gw = {"none": "No graphics.", "title": "A title card only.", "informational": f"One graphic at a time, at most {get(s, 'graphics.max_per_minute', 6)} a minute.",
          "all": f"Every kind of graphic, one at a time, at most {get(s, 'graphics.max_per_minute', 6)} a minute."}.get(gp, "")
    return "\n".join([
        f"Kind: {str(s.get('profile') or 'longform_talk').replace('_', ' ')}; pace {get(s, 'pace.name', 'natural')}; the picture changes at least every {get(s, 'pace.max_hold_s', 12.0):g} s.",
        f"B-roll about {round(float(get(s, 'broll.share', 0.3)) * 100)} % of the time, shots {float(lo):g} to {float(hi):g} s, never over the hook or a punchline.",
        {"none": "Captions off.", "phrase": "Captions in short phrases.", "karaoke": "Captions word by word."}.get(cap, "Captions off."),
        gw,
        {"none": "No music.", "under_broll": "Music only under B-roll.", "throughout": "Music throughout."}.get(mp, "")])


# ------------------------------------------------------------------------------------------------------------------ notes -> style
SERMON_RE = re.compile(r"\b(sermon|seminar|lecture|keynote|talk at|preach\w*)\b", re.I)
INTERVIEW_RE = re.compile(r"\b(podcast|interview|conversation with)\b", re.I)
RECAP_RE = re.compile(r"\b(recap|highlights?|best of|memories|aftermovie|after-movie|montage)\b", re.I)
PACE_FAST_RE = re.compile(r"\b(fast[- ]?paced|fast|snappy|punchy|upbeat|quick cuts)\b", re.I)
PACE_SLOW_RE = re.compile(r"\b(slow(?:er)?|calm|relaxed|take it slow|leisurely)\b", re.I)
NO_MUSIC_RE = re.compile(r"\bno music\b|\bwithout music\b", re.I)
NO_CAPTIONS_RE = re.compile(r"\bno captions\b|\bwithout captions\b|\bno subtitles\b", re.I)
NO_GRAPHICS_RE = re.compile(r"\bno graphics\b|\bwithout graphics\b|\bno on-?screen text\b", re.I)
IN_ORDER_RE = re.compile(r"\b(keep (it )?in order|in order|chronological(ly)?)\b", re.I)
BOTH_SPEAKERS_RE = re.compile(r"\bkeep both speakers\b|\bboth (speakers|voices)\b", re.I)

LENGTH_PHRASE_RE = re.compile(r"(?:\b(?:half a|a|one|two|three|four|five|six|ten|fifteen|twenty|thirty)\b|\d+(?:\.\d+)?)\s*[- ]?(?:minutes?|mins?|seconds?|secs?)\b", re.I)
SETTING_KEYS = ("profile", "pace", "captions", "graphics", "music", "broll_share", "chronology", "remove_interviewer", "voice_over", "target_seconds")
STYLE_SYSTEM = (
    "You read a video editor's notes and return only the editing settings they state or clearly imply, as JSON. Settings: profile (short_talk, reel, "
    "interview_clip, longform_talk, seminar, event_recap, promo, ad), pace (slow, natural, tight, fast), captions (none, phrase, karaoke), graphics "
    "(none, title, informational, all), music (none, under_broll, throughout), broll_share (a number 0 to 1), chronology (keep, free), "
    "remove_interviewer (auto, always, never), voice_over (true or false), target_seconds (a number, or null). Return null for anything the notes do "
    "not say. Do not guess from the subject of the video; \"a sermon\" implies profile seminar, \"podcast clip\" implies interview_clip, "
    "\"fast-paced\" implies pace fast, \"no music\" implies music none, \"keep it in order\" implies chronology keep, \"keep both speakers\" implies "
    "remove_interviewer never. Lengths: \"about eight minutes\" is 480; \"under two minutes\" is 110; ignore times that are positions in the footage.")
STYLE_SCHEMA = {"type": "object", "properties": {
    "profile": {"type": ["string", "null"], "enum": [*PROFILES, None]}, "pace": {"type": ["string", "null"], "enum": [*PACE_NAMES, None]},
    "captions": {"type": ["string", "null"], "enum": [*CAPTION_STYLES, None]}, "graphics": {"type": ["string", "null"], "enum": [*GRAPHICS_POLICIES, None]},
    "music": {"type": ["string", "null"], "enum": [*MUSIC_POLICIES, None]}, "broll_share": {"type": ["number", "null"]},
    "chronology": {"type": ["string", "null"], "enum": ["keep", "free", None]}, "remove_interviewer": {"type": ["string", "null"], "enum": ["auto", "always", "never", None]},
    "voice_over": {"type": ["boolean", "null"]}, "target_seconds": {"type": ["number", "null"]}},
    "required": list(SETTING_KEYS)}


def regex_settings(notes: str) -> dict:
    """What the words of the notes plainly say (the fallback when the AI call fails). Only non-null settings are returned."""
    t = str(notes or "")
    out: dict = {}
    if SERMON_RE.search(t):
        out["profile"] = "seminar"
    elif INTERVIEW_RE.search(t):
        out["profile"] = "interview_clip"
    elif RECAP_RE.search(t):
        out["profile"] = "event_recap"
    if PACE_FAST_RE.search(t):
        out["pace"] = "fast"
    elif PACE_SLOW_RE.search(t):
        out["pace"] = "slow"
    if NO_MUSIC_RE.search(t):
        out["music"] = "none"
    if NO_CAPTIONS_RE.search(t):
        out["captions"] = "none"
    if NO_GRAPHICS_RE.search(t):
        out["graphics"] = "none"
    if IN_ORDER_RE.search(t):
        out["chronology"] = "keep"
    if BOTH_SPEAKERS_RE.search(t):
        out["remove_interviewer"] = "never"
    return out


def clean_settings(data: dict | None) -> dict:
    """Only valid, non-null settings from an AI answer."""
    out: dict = {}
    d = data or {}
    allowed = {"profile": PROFILES, "pace": PACE_NAMES, "captions": CAPTION_STYLES, "graphics": GRAPHICS_POLICIES, "music": MUSIC_POLICIES,
               "chronology": ("keep", "free"), "remove_interviewer": ("auto", "always", "never")}
    for k, ok in allowed.items():
        if d.get(k) in ok:
            out[k] = d[k]
    if _num(d.get("broll_share")) and 0 <= float(d["broll_share"]) <= 1:
        out["broll_share"] = float(d["broll_share"])
    if isinstance(d.get("voice_over"), bool):
        out["voice_over"] = d["voice_over"]
    if _num(d.get("target_seconds")) and 5 <= float(d["target_seconds"]) <= 7200:
        out["target_seconds"] = float(d["target_seconds"])
    return out


def from_notes(ai, notes: str, kind: str | None = None) -> dict:
    """The settings a person's notes state (ONE cheap call, task `style_from_notes`), only the ones they state. When the call fails the words of the
    notes are read by pattern and the result carries `_fallback: True` (so a caller can fall back to its own length reader)."""
    notes = str(notes or "").strip()
    if not notes:
        return {}
    try:
        res = ai.complete_json("style_from_notes", STYLE_SYSTEM, notes[:3000], STYLE_SCHEMA)
        return clean_settings(res.data)
    except Exception as e:  # noqa: BLE001 - the pattern reading below
        log.warning(f"reading the editing settings from the notes failed ({e}); using the words of the notes")
        return {**regex_settings(notes), "_fallback": True}


def to_overrides(settings: dict | None) -> dict:
    """Flat settings (from_notes) -> the nested overrides resolve() takes. `profile` is not an override: the caller picks the profile."""
    s = {k: v for k, v in (settings or {}).items() if not str(k).startswith("_")}
    o: dict = {}
    if "pace" in s:
        o["pace"] = {"name": s["pace"]}
    if "captions" in s:
        o["captions"] = {"style": s["captions"]}
    if "graphics" in s:
        o["graphics"] = {"policy": s["graphics"]}
    if "music" in s:
        o["music"] = {"policy": s["music"]}
    if "broll_share" in s:
        o["broll"] = {"share": s["broll_share"], "per_s": None}              # (a stated share replaces the profile's own figure)
    if "chronology" in s:
        o["chronology"] = s["chronology"]
    if "remove_interviewer" in s:
        o["speakers"] = {"remove_interviewer": s["remove_interviewer"]}
    if "voice_over" in s:
        o["voice_over"] = s["voice_over"]
    if "target_seconds" in s:
        o["length"] = {"target_s": s["target_seconds"]}
    return o


def build(kind: str, notes_settings: dict | None = None, points_settings: dict | None = None, options: dict | None = None, is_clip: bool = False,
          kind_chosen: bool = False, extra: dict | None = None) -> dict:
    """The style object of one video from the layers: defaults < points < notes < options. A profile named by the person's options (`kind`) wins;
    else the notes' profile; else the points'; else the kind the AI read."""
    ns, ps = notes_settings or {}, points_settings or {}
    profile = from_kind(kind, is_clip) if kind_chosen else (ns.get("profile") or ps.get("profile") or from_kind(kind, is_clip))
    base = kind_overrides(kind, is_clip) if profile == from_kind(kind, is_clip) else {}
    return resolve(profile, layer(to_overrides(ps), to_overrides(ns), to_overrides(options), {k: v for k, v in (extra or {}).items() if k != "profile"}), base)


def flat_from_nested(partial: dict | None) -> dict:
    """A nested partial style ({"pace": {"name": "fast"}, "music": {"policy": "none"}}) -> the flat settings of from_notes (what a person states)."""
    p = partial or {}
    out: dict = {}
    for flat, path in (("pace", ("pace", "name")), ("captions", ("captions", "style")), ("graphics", ("graphics", "policy")), ("music", ("music", "policy")),
                       ("broll_share", ("broll", "share")), ("chronology", ("chronology",)), ("remove_interviewer", ("speakers", "remove_interviewer")),
                       ("voice_over", ("voice_over",)), ("target_seconds", ("length", "target_s"))):
        cur: Any = p
        for part in path:
            cur = cur.get(part) if isinstance(cur, dict) else None
        if cur is not None:
            out[flat] = cur
    if p.get("profile") in PROFILES:
        out["profile"] = p["profile"]
    return clean_settings(out)


def apply_partial(style: dict, partial: dict) -> tuple[dict, list[str]]:
    """A partial style laid over an object: (the new object, plain-words problems). Nothing changes when there are problems. A new `profile` starts
    from that profile's defaults with everything already stated kept."""
    if partial is not None and not isinstance(partial, dict):
        return style, ["the style must be an object of settings (for example {\"pace\": {\"name\": \"fast\"}})"]
    partial = {k: v for k, v in (partial or {}).items() if not str(k).startswith("_")}
    prof = partial.get("profile") if partial.get("profile") in PROFILES else None
    body = {k: v for k, v in partial.items() if k != "profile"}
    wrong = [f"\"{k}\" needs an object of settings, not {v!r}" for k, v in body.items() if isinstance(style.get(k), dict) and not isinstance(v, dict)]
    if wrong:
        return style, wrong
    if prof and prof != style.get("profile"):
        over: dict = {}
        for dotted in style.get("_set") or []:
            cur = over
            parts = dotted.split(".")
            for part in parts[:-1]:
                cur = cur.setdefault(part, {})
            cur[parts[-1]] = get(style, dotted)
        new = resolve(prof, layer(over, body))
    else:
        new = _merge({k: v for k, v in style.items() if k != "_set"}, body)
        new["_set"] = sorted(set(style.get("_set") or []) | set(_leaves(body)))
    problems = validate(new)
    return (style, problems) if problems else (new, [])


def set_pace(style: dict, name: str, paces: dict | None = None) -> float | None:
    """Change the pace word of a style (a change the person asked for) and say the gap that goes with it (None = unknown word)."""
    if name not in PACE_NAMES:
        return None
    style.setdefault("pace", {})["name"] = name
    st = [k for k in (style.get("_set") or []) if k != "pace.name"] + ["pace.name"]
    style["_set"] = sorted(st)
    if paces and name in paces:
        return float(paces[name]["gap"])
    return None


# ------------------------------------------------------------------------------------------------------------------ the brief
def fit_brief(brief: str, max_chars: int) -> str:
    """The brief cut to `max_chars` at the end of a whole paragraph or whole line, never in the middle of a rule (a first paragraph that is too long
    by itself is cut at a word)."""
    brief = str(brief or "")
    if len(brief) <= max_chars:
        return brief
    out = ""
    for para in brief.split("\n\n"):
        sep = "\n\n" if out else ""
        if len(out) + len(sep) + len(para) <= max_chars:
            out += sep + para
            continue
        piece = ""                                                           # (a paragraph of rules: as many whole lines as fit)
        for ln in para.splitlines():
            trial = piece + ("\n" if piece else "") + ln
            if len(out) + len(sep) + len(trial) > max_chars:
                break
            piece = trial
        if piece and len(para.splitlines()) > 1:
            out += sep + piece
        break
    return out if out else brief[:max_chars].rsplit(" ", 1)[0]


def capabilities() -> dict:
    """What an outside AI may ask for, so it plans inside the limits without trial and error (the MCP `capabilities` tool)."""
    from . import graphics
    try:
        from .localproj import CHECK_CODES as codes                          # (WP5's validator codes, once they exist)
    except Exception:  # noqa: BLE001
        codes = []
    schema = {k: (v if not isinstance(v, dict) else {a: b for a, b in v.items()}) for k, v in DEFAULTS["longform_talk"].items()}
    return {"profiles": list(PROFILES), "style_schema": schema, "defaults": {p: DEFAULTS[p] for p in PROFILES},
            "pace_names": list(PACE_NAMES), "caption_styles": list(CAPTION_STYLES), "graphics_policies": list(GRAPHICS_POLICIES),
            "music_policies": list(MUSIC_POLICIES), "graphics_themes": dict(graphics.STYLE_NAMES), "limits": dict(LIMITS), "checks": list(codes)}
