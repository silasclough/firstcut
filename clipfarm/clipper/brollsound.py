"""Whose sound is in the video (WP9, from a real defect: 2026-11-06, a clip filmed as B-roll that had people talking in it was used for what it
said, and its own camera-microphone sound, words legible but horrible, ended up in the video; the person never asked for that).

Rules, checked in code:
  1. A B-roll shot's own sound never plays unless the person asked for natural sound (notes like "keep the natural sound", or plan["broll_sound"]
     naming the files). The speaker's sound keeps playing under every cutaway: build_parts only swaps the PICTURE.
  2. Talk is not taken from a file that is B-roll (role broll / voiceover_broll with speech): its chatter and its camera mic are not the voice of
     the video. Those stretches are left out (with a note) when the project has real talking footage; a project with nothing else keeps them.
  3. A stretch whose sound grades "bad" (soundcheck) is never chosen when a cleaner source exists (a recorder or another camera: the multicam director
     and clean_voices already prefer them); when none exists the quality check says so, with the time: "this shot's own sound is poor".
  4. Nat sound that IS wanted starts a little before the picture (a J cut) and tails after it (an L cut), 20 dB under the voice.

    allowed_files(notes, plan)         -> "*" | set of file names whose own sound may play under their B-roll
    broll_speech_spans(files)          -> [(start, end, file name)] virtual-time stretches of files whose speech is not the video's voice
    drop_broll_speech(intervals, files, ...) -> (intervals, [plain notes])
    poor_sound_flags(parts, files, grades) -> [{start, end, file, why}]  timeline stretches whose chosen sound grades bad
    jl_mix(cutaway, scene, bf, style, allowed) -> a_mix dict for the piece | None
"""

from __future__ import annotations

import re

from ..core.log import get_logger

log = get_logger("clipper.brollsound")

KEEP_SOUND = re.compile(r"\b(keep (the )?(natural |original |ambient )?(sound|audio)|with (the )?(natural |original |ambient )?(sound|audio)|"
                        r"natural sound|ambient sound|nat sound|use (the )?(b-?roll |clip'?s )?(own )?(sound|audio))\b", re.I)
NOT_VOICE_ROLES = ("broll", "voiceover_broll")
SPEECH_MIN = 0.05                                    # share of a file that is speech: under this it is ambience, not talk
MIN_LEFT_S = 0.5


NO_SOUND = re.compile(r"\b(?:no(?!,)|without|never|mute|muted|silence|silent|disable|remove|drop|turn off|switch off|strip)\b(?:(?!\b(?:keep|use|play|want|with)\b)[^.;!?\n]){0,40}?\b(?:natural|nat|ambient|original|b-?roll|clip'?s?|their)\b[^.;!?\n]{0,20}?\b(?:sound|audio|noise)\b|"
                      r"\b(?:don'?t|do not|dont|doesn'?t|never|no need to|not)\b[^.;!?\n]{0,25}?\b(?:keep|use|want|need|include|play|add|have)\b[^.;!?\n]{0,25}?\b(?:natural|nat|ambient|original|b-?roll|clip'?s?)?\b[^.;!?\n]{0,12}?\b(?:sound|audio)\b|"
                      r"\b(?:mute|silence)\s+(?:the\s+|all\s+(?:the\s+)?)?b-?roll\b|\bb-?roll\b[^.;!?\n]{0,12}\b(?:muted|silent|without (?:its )?(?:own )?(?:sound|audio))\b", re.I)


def allowed_files(notes: str, plan: dict | None = None):
    """"*" when the notes ask for natural sound (and do not refuse it), else the set of file names plan["broll_sound"] lists (empty = nobody's: the normal case)."""
    text = str(notes or "")
    refused = bool(NO_SOUND.search(text))
    if KEEP_SOUND.search(text) and not refused:
        return "*"
    if refused:
        return set()                                                      # ("no natural sound": nothing plays, not even a file the plan names)
    names = (plan or {}).get("broll_sound") or []
    return {str(n) for n in names if n} if isinstance(names, (list, tuple, set)) else set()


def _facts(f: dict) -> dict:
    x = f.get("facts")
    return x if isinstance(x, dict) else {}


def _speech_ratio(f: dict) -> float:
    try:
        return float(((_facts(f).get("speech") or {}).get("ratio")) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def is_voice_source(f: dict) -> bool:
    """True when this file may supply the video's voice: not a file known to be B-roll (a talking-head clip is never classed B-roll: WP2)."""
    role = _facts(f).get("role")
    if role in NOT_VOICE_ROLES:
        return False
    if f.get("broll") and role is None:                       # (no footage record: the old flag still says "named B-roll")
        return False
    return True


def broll_speech_spans(files: list[dict]) -> list[tuple[float, float, str]]:
    return [(float(f["start"]), float(f["end"]), f.get("name", "")) for f in files
            if not f.get("photo") and not f.get("multicam") and f.get("audio") and not is_voice_source(f) and _speech_ratio(f) >= SPEECH_MIN]


def drop_broll_speech(intervals, files: list[dict], exact: bool = False, mont: bool = False, keep_sound_ok=frozenset()):
    """The kept stretches without anything taken from a B-roll file's chatter. Never when the person listed exact clips or this is a montage, and never
    when it would leave nothing (a folder of only such files keeps them: the person put them there)."""
    spans = broll_speech_spans(files)
    if exact or mont or not spans or not any(is_voice_source(f) and f.get("audio") and not f.get("photo") for f in files):
        return list(intervals), []
    if keep_sound_ok == "*":
        return list(intervals), []
    skip = [(a, b) for a, b, n in spans if n not in keep_sound_ok]
    out: list[tuple[float, float]] = []
    cut_s = 0.0
    for a, b in intervals:
        pieces = [(a, b)]
        for x, y in skip:
            nxt = []
            for p, q in pieces:
                if q <= x or p >= y:
                    nxt.append((p, q))
                    continue
                if p < x:
                    nxt.append((p, x))
                if q > y:
                    nxt.append((y, q))
                cut_s += min(q, y) - max(p, x)
            pieces = nxt
        out += [(p, q) for p, q in pieces if q - p >= MIN_LEFT_S]
    if not out or cut_s < 0.2:
        return list(intervals), []
    names = sorted({n for _, _, n in spans})[:3]
    return out, [f"{cut_s:.0f} s of talking taken from B-roll shots left out ({', '.join(names)}): their own microphone is not the voice of the video"]


def poor_sound_flags(parts: list[dict], files: list[dict], grades) -> list[dict]:
    """Timeline stretches whose chosen sound grades bad: [{start, end, file, why}]. grades: soundcheck.Grades (or anything with grade(path, local))."""
    out: list[dict] = []
    by_path = {f.get("path"): f for f in files}
    for p in parts:
        a = p.get("a")
        if not a or not a.get("path"):
            continue
        try:
            g = grades.grade(a["path"], max(0.0, float(a.get("ss", 0.0))))
        except Exception:  # noqa: BLE001 - a check never stops a render
            continue
        if g.get("grade") == "bad":
            name = (by_path.get(a["path"]) or {}).get("name") or str(a["path"]).rsplit("/", 1)[-1]
            t0, t1 = float(p.get("out", 0.0)), float(p.get("out", 0.0)) + float(p.get("d", 0.0))
            if out and out[-1]["file"] == name and t0 - out[-1]["end"] < 0.5:
                out[-1]["end"] = t1
            else:
                out.append({"start": t0, "end": t1, "file": name, "why": str(g.get("why") or "muffled or far from the microphone")})
    return out


def usable_nat_sound(bf: dict) -> bool:
    """A B-roll file with real ambience: it has sound, almost no speech, and is not music."""
    f = _facts(bf)
    if not bf.get("audio") or f.get("role") in ("music", "audio"):
        return False
    return _speech_ratio(bf) < SPEECH_MIN


def jl_mix(cutaway: dict, scene: dict, bf: dict, style: dict | None, allowed) -> dict | None:
    """The sound a B-roll shot brings under the voice, or None. Only when the person asked (allowed), the file has usable ambience, and the style
    gives a lead: starts jl_cut_s before the picture, 20 dB under the voice."""
    name = bf.get("name")
    if not (allowed == "*" or (isinstance(allowed, (set, frozenset)) and name in allowed)):
        return None
    if not usable_nat_sound(bf):
        return None
    lead = float(((style or {}).get("broll") or {}).get("jl_cut_s", 0.3))
    return {"path": bf["path"], "ss": max(0.0, float(cutaway["src"]) - float(bf["start"]) - lead), "lead": lead, "tail": lead, "gain_db": -20.0}


def apply_jl(parts: list[dict]) -> None:
    """Turns the `a_mix_c` marks build_parts left on B-roll pieces into `a_mix` lists the cutter mixes in ([{path, ss, delay, len, gain_db}]):
    the shot's sound under its own piece, a lead before it (J cut: starts in the piece before) and a tail after it (L cut: runs into the next one)."""
    for k, p in enumerate(parts):
        m = p.pop("a_mix_c", None)
        if not m:
            continue
        g = float(m.get("gain_db", -20.0))
        p.setdefault("a_mix", []).append({"path": m["path"], "ss": float(m["ss"]), "delay": 0.0, "len": float(p["d"]), "gain_db": g})
        lead, tail = float(m.get("lead", 0.3)), float(m.get("tail", 0.3))
        if m.get("first") and k > 0 and lead > 0.05 and float(m["ss"]) - lead >= 0:
            q = parts[k - 1]
            ln = min(lead, float(q["d"]))
            q.setdefault("a_mix", []).append({"path": m["path"], "ss": float(m["ss"]) - ln, "delay": max(0.0, float(q["d"]) - ln), "len": ln, "gain_db": g})
        if m.get("last") and k + 1 < len(parts) and tail > 0.05:
            q = parts[k + 1]
            q.setdefault("a_mix", []).append({"path": m["path"], "ss": float(m["ss"]) + float(p["d"]), "delay": 0.0, "len": min(tail, float(q["d"])), "gain_db": g})
