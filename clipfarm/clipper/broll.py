"""B-roll: footage with no dialogue. The editor can't read it from a transcript, so the AI LOOKS at it.

    describe()          a vision model reads contact sheets of every mostly-visual file and writes what each stretch shows
                        (cached in broll.json), so the rest of the pipeline can reason about pictures in words.
    plan_cutaways()     the editing AI places short B-roll cutaways over the talking, where what is being said matches what the
                        footage shows (the speaker's voice keeps playing; only the picture changes).

Files named like "B-roll ..." (or in a folder called B-roll) are the cutaway pool. Other mostly-visual files stay in the story
in the order they were shot, as before.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..core.log import get_logger
from . import moments as M

log = get_logger("clipper.broll")
BROLL_NAME = re.compile(r"b[\s_-]?roll", re.I)
SPARSE_WPS = 0.35                    # same cut-off as story.py: fewer words per second than this = mostly pictures
MAX_SHEETS = 2                       # contact sheets (12 frames each) per file - keeps the vision call small
CUT_MIN, CUT_MAX = 1.5, 4.5          # cutaway length, seconds
SPLIT_OVER = 12.0                    # a described scene longer than this is cut into parts, so ONE long shot can supply many cutaways
PART_S = 8.0                         # ...about this long each (a cutaway is 1.5-4.5 s of one part)
MAX_PARTS = 40
INTRO_FREE = 3.0                     # the first seconds stay on the speaker (the hook)

SYSTEM = (
    "You describe raw footage for a video editor who cannot see it. You get contact sheets: a grid of frames, each labelled "
    "with its time in seconds. Split the footage into scenes (a continuous stretch showing one thing) and for each say, in "
    "one concrete sentence, what it shows (subject, action, setting) - no guessing about meaning. Rate how usable it is as "
    "b-roll from 1 to 10 (sharp, steady, well lit, interesting = high; blurry, shaky, dark, empty, accidental = low).")
SCHEMA = {"type": "object", "properties": {"scenes": {"type": "array", "items": {"type": "object", "properties": {
    "start_s": {"type": "number"}, "end_s": {"type": "number"}, "what": {"type": "string"}, "quality": {"type": "number"}},
    "required": ["start_s", "end_s", "what", "quality"]}}}, "required": ["scenes"]}

PLAN_SYSTEM = (
    "You are a video editor adding B-roll cutaways over a talking video. The speaker's voice keeps playing; for a moment the "
    "picture switches to a B-roll scene that illustrates exactly what they are saying at that moment. Only cut away where a "
    "scene genuinely matches the words (a place, an object, an action that is mentioned), and use B-roll generously: something new "
    "should change on screen at least every 12 seconds, so look for a cutaway about every 15 to 20 seconds, only where it genuinely matches. Scenes marked (still photo) "
    "are photographs (they will be animated): use every one of them somewhere it fits. Never in the first seconds (the hook stays on "
    "the speaker), never two within a couple of seconds of each other, each scene at most once. Each cutaway is 1.5 to 4.5 seconds.")
PLAN_SCHEMA = {"type": "object", "properties": {"cutaways": {"type": "array", "items": {"type": "object", "properties": {
    "at": {"type": "number"}, "scene": {"type": "integer"}, "length": {"type": "number"}, "why": {"type": "string"}},
    "required": ["at", "scene", "length"]}}}, "required": ["cutaways"]}


def sheets_for(dur: float) -> int:
    """Contact sheets (12 frames each) to look at for a file this long: a frame about every 5 s up to 8 sheets, never fewer than MAX_SHEETS."""
    import math
    return max(MAX_SHEETS, min(8, math.ceil(dur / 60.0)))


def split_long(scenes: list[dict]) -> list[dict]:
    """A 9-minute shot is one 'scene' to a vision model, and a scene is used at most once, from its start - so one long shot gave ONE cutaway.
    Cut every long scene into ~8 s parts that are placed independently (each part is a different stretch of the footage)."""
    out: list[dict] = []
    for sc in scenes:
        ln = sc["end"] - sc["start"]
        if ln <= SPLIT_OVER or sc.get("photo") or sc.get("part"):
            out.append(sc)
            continue
        n = min(MAX_PARTS, max(2, round(ln / PART_S)))
        step = ln / n
        for k in range(n):
            out.append({**sc, "start": round(sc["start"] + k * step, 2), "end": round(sc["start"] + (k + 1) * step, 2), "part": [k + 1, n],
                        "what": f"{sc['what']} [part {k + 1} of {n} of one long shot]"})
    return out


def is_named(name: str) -> bool:
    return bool(BROLL_NAME.search(name or ""))


def visual_files(files: list[dict], words: list[dict]) -> list[int]:
    """Indices of files that are mostly pictures (named B-roll, or hardly any speech)."""
    out = []
    for k, f in enumerate(files):
        dur = max(1.0, f["end"] - f["start"])
        n = sum(1 for w in words if f["start"] <= w["start"] < f["end"])
        if is_named(f.get("name", "")) or n / dur < SPARSE_WPS:
            out.append(k)
    return out


def _valid_scenes(raw: list[dict], lo: float, hi: float) -> list[dict]:
    out = []
    for s in raw or []:
        try:
            a, b = max(lo, float(s["start_s"])), min(hi, float(s["end_s"]))
            q = max(1.0, min(10.0, float(s.get("quality", 5))))
        except (KeyError, TypeError, ValueError):
            continue
        what = " ".join(str(s.get("what", "")).split())[:200]
        if b - a >= 0.5 and what:
            out.append({"start": round(a, 2), "end": round(b, 2), "what": what, "quality": q})
    out.sort(key=lambda s: s["start"])
    return out


def describe(engine, src: Path, files: list[dict], words: list[dict], d: Path, progress=None) -> dict[str, list[dict]]:
    """{file name: [scene...]} for the mostly-visual files (times are in the joined source video). Cached in broll.json."""
    cache = d / "broll.json"
    have: dict[str, list[dict]] = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [k for k in visual_files(files, words) if files[k]["name"] not in have]
    if not todo:
        return {k: split_long(v) for k, v in have.items()}
    from .glasses import contact_sheets
    router, _where = engine.vision_router()
    for n, k in enumerate(todo):
        f = files[k]
        dur = f["end"] - f["start"]
        if progress:
            progress(f"Looking at {f['name']} ({n + 1} of {len(todo)})")
        n_sheets = sheets_for(dur)
        step = max(1.0, dur / (12 * n_sheets))
        try:
            sheets = contact_sheets(src, f["start"], f["end"], step)[:n_sheets]
            if not sheets:
                continue
            times = ", ".join(f"{t:.1f}" for _, ts in sheets for t in ts)
            prompt = (f"File: {f['name']} (covers {f['start']:.1f}-{f['end']:.1f} s). {len(sheets)} contact sheet(s); frame times "
                      f"in seconds: {times}. Describe the scenes. Use these absolute times.")
            res = router.complete_json("broll_describe", SYSTEM, prompt, SCHEMA, images=[b for b, _ in sheets])
            scenes = _valid_scenes(res.data.get("scenes"), f["start"], f["end"])
        except Exception as e:  # noqa: BLE001 - one file the AI can't read is simply not described
            log.warning(f"describing {f['name']} failed: {e}")
            continue
        if scenes:
            have[f["name"]] = scenes
            cache.write_text(json.dumps(have, indent=1))
    return {k: split_long(v) for k, v in have.items()}


def fill_filter(src: Path, at: float, W: int, H: int) -> str:
    """ffmpeg filters that make a cutaway FILL the frame. Footage of another shape was letterboxed into the joined video
    (black bars); find the picture inside the bars and zoom it to cover the whole frame instead."""
    import subprocess

    from ..core import ffmpeg
    try:
        r = subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "info", "-ss", f"{at + 0.3:.2f}", "-t", "1.2", "-i", str(src), "-vf",
                            "cropdetect=limit=24:round=2:reset=0", "-an", "-f", "null", "-"], capture_output=True, text=True,
                           timeout=120)
        m = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", r.stderr)
        if m:
            cw, ch, cx, cy = (int(v) for v in m[-1])
            if cw >= 64 and ch >= 64 and (cw * ch) < 0.92 * W * H:
                return f"crop={cw}:{ch}:{cx}:{cy},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}"
    except (subprocess.SubprocessError, OSError, ValueError):
        pass
    return f"scale={W}:{H}"


PICK_SYSTEM = (
    "You are a video editor choosing which stretches of silent footage (no one talking) deserve to play as their OWN shots in the "
    "video, in the order they were filmed. You see each scene's description and quality. Pick scenes that are relevant to what the "
    "video is about (follow the creator's notes), clear, steady and interesting, and that add something the talking doesn't "
    "(showing the product, the place, the action). Skip duplicates, accidents, blurry or dark footage and anything off-topic. "
    "Stay within the time budget. The scenes you don't pick will still be available as cutaways over the talking.")
PICK_SCHEMA = {"type": "object", "properties": {"keep": {"type": "array", "items": {"type": "integer"}}, "why": {"type": "string"}},
               "required": ["keep", "why"]}
STANDALONE_MAX_S = 8.0


def pick_standalone(llm, scenes: list[dict], brief: str, summary: str, budget_s: float) -> list[int]:
    """Indices (into `scenes`) of the described scenes to play as their own shots."""
    if not scenes or budget_s < 2:
        return []
    lines = [f"  S{i}: {s['what']} ({min(s['end'] - s['start'], STANDALONE_MAX_S):.0f} s, quality {s['quality']:.0f}/10)" for i, s in enumerate(scenes)]
    prompt = ((f"The creator's notes: {brief}\n" if brief else "") + (f"What the video says: {summary}\n" if summary else "")
              + f"Time budget for these shots: about {budget_s:.0f} seconds.\nScenes (in filming order):\n" + "\n".join(lines[:400])
              + "\n\nReturn the numbers of the scenes to keep.")
    try:
        res = llm.complete_json("visual_pick", PICK_SYSTEM, prompt, PICK_SCHEMA)
        keep = [int(i) for i in res.data.get("keep") or [] if 0 <= int(i) < len(scenes)]
    except Exception as e:  # noqa: BLE001 - fall back to the best-looking ones
        log.warning(f"picking silent shots failed ({e}); taking the best-rated")
        keep = [i for i in sorted(range(len(scenes)), key=lambda i: -scenes[i]["quality"]) if scenes[i]["quality"] >= 7]
    out, total = [], 0.0
    for i in dict.fromkeys(keep):                                   # in the AI's order, no repeats, inside the budget
        ln = min(scenes[i]["end"] - scenes[i]["start"], STANDALONE_MAX_S)
        if total + ln > budget_s * 1.1:
            continue
        out.append(i)
        total += ln
    return sorted(out)


def pool_files(files: list[dict], scenes: dict[str, list[dict]], words: list[dict]) -> set[int]:
    """Files used as cutaway material: named B-roll AND described, as long as something else carries the story."""
    pool = {k for k, f in enumerate(files) if is_named(f.get("name", "")) and scenes.get(f["name"])}
    return pool if len(pool) < len(files) else set()


AD_PLAN = (" THIS IS AN ADVERTISEMENT: the picture should be mostly B-roll with the speaker's voice running underneath - show the store, "
           "the products, the people and the experience that the words talk about. Cover most of the video with B-roll, change shot every 1.5 "
           "to 3 seconds, prefer the best-looking, highest-quality shots, and show the speaker's face only when a line really needs it.")


def plan_cutaways(llm, sentences: list[M.Sentence], pool: list[dict], brief: str = "", talk_s: float = 0.0, per_s: float = 18.0,
                  ad: bool = False) -> list[dict]:
    """-> [{"at": timeline s, "length": s, "src": source s, "what": ..., "why": ...}] (sorted, non-overlapping). The AI is asked again, once, when it
    places far fewer than the length calls for (a planning call is not repeatable: the same video once got 9 shots and the next try 3)."""
    out = _plan_once(llm, sentences, pool, brief, talk_s, per_s, ad)
    cap = max(1, int(talk_s / per_s)) if talk_s else 3
    if len(out) < max(2, cap // 2) and len(pool) > len(out):
        again = _plan_once(llm, sentences, pool, brief, talk_s, per_s, ad,
                           more=f"Your last answer placed only {len(out)}: place more, spread over the WHOLE video including the first half.")
        if len(again) > len(out):
            out = again
    return out


def _plan_once(llm, sentences: list[M.Sentence], pool: list[dict], brief: str = "", talk_s: float = 0.0, per_s: float = 18.0,
                ad: bool = False, more: str = "") -> list[dict]:
    """-> [{"at": timeline s, "length": s, "src": source s, "what": ..., "why": ...}] (sorted, non-overlapping)."""
    if not sentences or not pool:
        return []
    cap = max(1, int(talk_s / per_s)) if talk_s else 3
    cmin, cmax, gap = (1.2, 3.2, 0.3) if ad else (CUT_MIN, CUT_MAX, 2.0)
    lines = [f"  [{s.start:.1f}-{s.end:.1f}] {s.text}" for s in sentences]
    scenes = [f"  S{i}: {p['what']} ({p['end'] - p['start']:.1f} s long, quality {p['quality']:.0f}/10)" for i, p in enumerate(pool)]
    prompt = ((f"The creator's notes: {brief}\n" if brief else "")
              + f"Transcript with timeline seconds:\n" + "\n".join(lines[:400]) + "\n\nAvailable B-roll scenes:\n" + "\n".join(scenes)
              + f"\n\nPlace about {cap} cutaways (at most {cap}; fewer than {max(2, cap // 2)} is too few for this length). 'at' is the timeline second "
              "where the cutaway starts (inside a sentence)." + (f" {more}" if more else ""))
    try:
        res = llm.complete_json("broll_plan", PLAN_SYSTEM + (AD_PLAN if ad else ""), prompt, PLAN_SCHEMA)
    except Exception as e:  # noqa: BLE001 - no cutaways is a fine result
        log.warning(f"planning cutaways failed: {e}")
        return []
    spans = [(s.start, s.end) for s in sentences]
    out, used = [], set()
    for c in sorted(res.data.get("cutaways") or [], key=lambda c: float(c.get("at", 0))):
        try:
            at, i, ln = float(c["at"]), int(c["scene"]), float(c["length"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= i < len(pool)) or i in used or at < (0.8 if ad else INTRO_FREE) or len(out) >= cap:
            continue
        sc = pool[i]
        ln = max(cmin, min(cmax, ln, sc["end"] - sc["start"]))
        if sc["end"] - sc["start"] < cmin or not any(a - 0.2 <= at <= b for a, b in spans):
            continue
        if out and at < out[-1]["at"] + out[-1]["length"] + gap:
            continue
        used.add(i)
        out.append({"at": round(at, 2), "length": round(ln, 2), "src": round(sc["start"], 2), "scene": i,
                    "what": sc["what"], "why": str(c.get("why", ""))[:160], "file": sc.get("file"), "photo": sc.get("photo")})
    return out


def fill_ad(cutaways: list[dict], sentences: list[M.Sentence], pool: list[dict], out_len: float, share: float = 0.65,
            hold: float = 1.3, length: float = 2.4, allowed: set[int] | None = None, max_new: int = 99) -> list[dict]:
    """An ad is mostly pictures of the business. The AI's own cutaways stay; the gaps between them are filled, round after round, with
    unused scenes (best quality first, a different file each time where it can) until `share` of the video shows B-roll. The speaker's
    voice keeps playing under all of it; each cutaway starts inside a spoken sentence, and the speaker is seen for `hold` s between them."""
    spans = [(s.start, s.end) for s in sentences]
    used = {c["scene"] for c in cutaways}
    left = sorted((i for i in range(len(pool)) if i not in used and pool[i]["end"] - pool[i]["start"] >= 1.2 and (allowed is None or i in allowed)),
                  key=lambda i: -pool[i]["quality"])
    out = sorted(cutaways, key=lambda c: c["at"])

    def covered() -> float:
        return sum(c["length"] for c in out)
    last_file = out[-1].get("file") if out else None
    t = 0.8
    added = 0
    while covered() < share * out_len and left and t < out_len - 1.5 and added < max_new:
        nxt = next((c for c in out if c["at"] + c["length"] > t - 1e-6), None)
        if nxt and nxt["at"] <= t + 0.05:
            t = nxt["at"] + nxt["length"] + hold                              # inside one already placed: hop over it
            continue
        room = (nxt["at"] - hold - t) if nxt else out_len - 0.4 - t
        if room < 1.2 or not any(a - 0.2 <= t <= b for a, b in spans):
            t += 0.5
            continue
        pick = next((i for i in left if pool[i].get("file") != last_file), left[0])
        left.remove(pick)
        sc = pool[pick]
        ln = round(min(length, room, sc["end"] - sc["start"]), 2)
        out.append({"at": round(t, 2), "length": ln, "src": round(sc["start"], 2), "scene": pick, "what": sc["what"],
                    "why": "more pictures of the business", "file": sc.get("file"), "photo": sc.get("photo")})
        added += 1
        out.sort(key=lambda c: c["at"])
        last_file = sc.get("file")
        t += ln + hold
    return out


SUITABLE_SCHEMA = {"type": "object", "properties": {"ok": {"type": "array", "items": {"type": "integer"}}}, "required": ["ok"]}


def ad_suitable(llm, pool: list[dict], brief: str) -> set[int]:
    """Which scenes belong in THIS business's ad (its shop, products, staff, customers being helped) - not unrelated scenery."""
    lines = [f"  S{i}: {p['what']}" for i, p in enumerate(pool)]
    prompt = ((f"The ad's brief: {brief[:1500]}\n" if brief else "") + "Scenes:\n" + "\n".join(lines[:300])
              + "\n\nReturn the numbers of the scenes that clearly show this business, its products, its people or its customers being helped. "
                "Leave out scenery, strangers, and anything that could be from any other video.")
    try:
        res = llm.complete_json("broll_suitable", "You choose which footage belongs in an advertisement for one specific business.", prompt, SUITABLE_SCHEMA)
        return {int(i) for i in res.data.get("ok") or [] if 0 <= int(i) < len(pool)}
    except Exception as e:  # noqa: BLE001 - without the check, nothing extra is added (the AI's own picks still stand)
        log.warning(f"checking which B-roll suits the ad failed: {e}")
        return set()


PICK_ONE_SCHEMA = {"type": "object", "properties": {"scene": {"type": "integer"}}, "required": ["scene"]}   # (its own name: reusing
#                                                     PICK_SCHEMA here replaced pick_standalone's schema, so no silent shot was ever picked)


def pick_for(llm, spoken: str, pool: list[dict], free: list[int], brief: str = "") -> int | None:
    """The scene (an index into pool, from `free`) that really fits what is being said right now, or None if nothing does."""
    cands = free[:40]
    if not spoken.strip() or not cands:
        return None
    prompt = ((f"The creator's notes: {brief[:600]}\n" if brief else "") + f"What the speaker is saying right now: \"{spoken[:500]}\"\n\nScenes:\n"
              + "\n".join(f"  S{i}: {pool[i]['what']}" for i in cands)
              + "\n\nReturn the number of the ONE scene that genuinely matches or illustrates what is being said (a place, a person, an object or "
                "feeling that is mentioned). If none really fits, return -1. Never force a match.")
    try:
        res = llm.complete_json("broll_pick", "You choose B-roll for one moment of a video.", prompt, PICK_ONE_SCHEMA)
        i = int(res.data.get("scene", -1))
        return i if i in cands else None
    except Exception as e:  # noqa: BLE001
        log.warning(f"choosing B-roll for a moment failed: {e}")
        return None
