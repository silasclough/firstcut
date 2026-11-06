"""Explainer videos: a 4-8 minute 16:9 video that explains one topic from Wikipedia, narrated, with a picture or card for every line.

    topic -> Wikipedia text (the only facts allowed) -> script (beats of one spoken passage each + the picture each needs)
          -> a second pass that checks every claim against the source -> voice (edge-tts, with word times) -> a picture per beat
          (free-licence photo from Wikimedia Commons with a slow move, or a full-screen card drawn natively) -> music bed -> one mp4

Honesty rules: the writer may use ONLY the source text; an unsupported claim is rewritten once and dropped if it still fails; every photo
is public domain / CC0 / CC BY (see stock.license_ok) and credited in credits.txt next to the video, with the Wikipedia articles used.

    build(cfg, topic, minutes=6, out_dir=None, titles=None, theme=None, log_fn=print) -> {"video", "credits", "script", ...}
"""

from __future__ import annotations

import json
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from .. import stock
from ..core import ffmpeg, tts
from ..core import wiki as WK
from ..core.log import get_logger
from . import gfxcards as GC
from . import gfxplan as GP
from . import music as MU
from . import photos as PH
from .localai import LocalAI

log = get_logger("clipper.explainer")
W, H, FPS = 1920, 1080, 30
PAD = 0.30                                    # seconds of air after each spoken line
CHAPTER_S = 2.4
MAX_ANIMATIONS = 30                          # custom animations per video (each takes a few minutes to design and draw, three at a time)
WORDS_PER_MIN = 150
EFFECTS = ["zoom_in", "pan_right", "zoom_out", "pan_left"]
LOOKS = [("clean", "dots"), ("blueprint", "grid"), ("sky", "blocks"), ("slate", "stripe"), ("mint", "rule"), ("board", "frame")]     # (theme, page) - one per section, in turn

SCHEMA = {"type": "object", "properties": {
    "title": {"type": "string"},
    "beats": {"type": "array", "items": {"type": "object", "properties": {
        "section": {"type": "string"},
        "narration": {"type": "string"},
        "visual": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["photo", "slide", "stat", "flow", "bars", "timeline", "animation"]},
            "concept": {"type": "string"},
            "image": {"type": "string"},
            "title": {"type": "string"},
            "bullets": {"type": "array", "items": {"type": "string"}},
            "value": {"type": "string"},
            "label": {"type": "string"},
            "steps": {"type": "array", "items": {"type": "string"}},
            "labels": {"type": "array", "items": {"type": "string"}},
            "values": {"type": "array", "items": {"type": "string"}},
            "unit": {"type": "string"},
            "events": {"type": "array", "items": {"type": "object", "properties": {"date": {"type": "string"}, "label": {"type": "string"}}, "required": ["date", "label"]}}},
            "required": ["kind"]}},
        "required": ["narration", "visual"]}}}, "required": ["title", "beats"]}

CHECK_SCHEMA = {"type": "object", "properties": {"problems": {"type": "array", "items": {"type": "object", "properties": {
    "beat": {"type": "integer"}, "claim": {"type": "string"}, "why": {"type": "string"}}, "required": ["beat", "claim"]}}}, "required": ["problems"]}
FIX_SCHEMA = {"type": "object", "properties": {"beats": {"type": "array", "items": {"type": "object", "properties": {
    "beat": {"type": "integer"}, "narration": {"type": "string"}}, "required": ["beat", "narration"]}}}, "required": ["beats"]}

SYSTEM = (
    "You write {minutes}-minute explainer videos for a curious general audience with no background, voiced by a calm documentary narrator. "
    "You are given SOURCE TEXT: the ONLY place a fact may come from. Every claim, name, number and date must be in it; if a detail is not there, "
    "leave it out - never guess, never add from memory. Where the sources only say 'about' or 'often', say so. "
    "Write for the EAR: short declarative sentences (max 22 words), one idea each, plain words, numbers as people say them ('nineteen forty-five', "
    "'three point five million'), no parentheses, no abbreviations or symbols, no emoji, never 'in this video' or 'like and subscribe'. "
    "Structure: the FIRST beat is the hook - the single most surprising or strange true thing in the sources, at most 20 words, said as a flat statement that makes "
    "the viewer want the rest (never a definition, never 'X is a ...', never a question the video does not answer), and its 'section' stays EMPTY. The SECOND beat "
    "pays it off with the one-sentence plain answer. Then 4 or 5 sections; each answers the next question a smart newcomer would ask, building from the simplest "
    "picture to how it really works to why it matters. Put a short section title (2-5 words) in 'section' of each section's first beat: it is ONLY a label for the "
    "chapter list. NEVER say it, never write the word 'section' or the title into 'narration': the narration must read as one continuous spoken story. End with a short "
    "takeaway of what the viewer now understands. "
    "Each beat is ONE spoken passage of 16-32 words (6-12 seconds) that moves the explanation forward. About {words} words in total ({beats} beats). "
    "For each beat choose the picture: "
    "'photo' = one of the IMAGES listed below; give its id in 'image' ONLY if the picture really shows what the line says (read the descriptions); "
    "you may use an image twice if it fits, never in neighbouring beats; "
    "'slide' = a card with 'title' (max 5 words) and 2-4 'bullets' (max 6 words each), ONLY when the narration lists things; "
    "'stat' = a big number card with 'value' written in DIGITS, at most 8 characters, like '1939', '6,000' or '30%' (no words), and 'label' (max 10 words), "
    "ONLY when the beat's point is one number; "
    "ANIMATED DIAGRAMS (use them: they carry the explanation): "
    "'flow' = a process drawn step by step: 'title' (max 5 words) and 3-5 'steps' (max 4 words each) in the order the narration says them; "
    "'bars' = a comparison drawn as growing bars: 'title', 2-6 'labels' (max 3 words) and matching 'values' as plain numbers in digits (no words), 'unit' (max 5 words) - ONLY numbers the sources give; "
    "'timeline' = 3-5 dated events drawn along a line: 'title' and 'events' [{{date: '1939' or '1950s', label: max 6 words}}] - ONLY dates the sources give. "
    "'animation' = a custom animated diagram drawn for this line, for a mechanism or a structure that moves and that no photo shows: 'concept' = ONE sentence saying exactly "
    "what is drawn and how it moves, using only things the sources describe (e.g. 'a selector fork slides a clutch sleeve sideways until its teeth lock into the side of a gear'); "
    "at most 5 per video, each about a different thing, only where the line describes something physical or a step-by-step mechanism; "
    "BALANCE (this is checked): at least 40 percent of the beats are photos whenever the image list has pictures that fit (read all the descriptions: a gearbox, a "
    "cutaway or a diagram of the thing being described fits); at most 4 flows - ONLY for a real process or sequence of events, never for a list of reasons; "
    "bars only when the sources give numbers to compare; a timeline only when they give dates; slides for lists (at most 8); stat for one striking number. "
    "Never more than two cards in a row: put a photo between them. Never the same kind three times in a row."
)


# --------------------------------------------------------------------------------------------- sources
def find_title(topic: str) -> str | None:
    d = WK.api({"action": "query", "list": "search", "srsearch": topic, "srlimit": 3})
    hits = d.get("query", {}).get("search", [])
    return hits[0]["title"] if hits else None


def sources(topic: str, titles: list[str] | None = None, max_chars: int = 14000) -> list[dict]:
    names = list(titles or [])
    if not names:
        t = find_title(topic)
        if not t:
            raise ValueError(f"no Wikipedia article found for {topic!r}")
        names = [t]
    out = []
    for n in names:
        ex = WK.extract(n, max_chars=max_chars)
        if ex and len(ex["text"]) > 800:
            out.append(ex)
    if not out:
        raise ValueError(f"the Wikipedia article(s) {names} have too little text to explain from")
    return out


def source_text(srcs: list[dict]) -> str:
    return "\n\n".join(f"## {s['title']}\n{s['text']}" for s in srcs)


# --------------------------------------------------------------------------------------------- script
def write_script(ai: LocalAI, topic: str, srcs: list[dict], minutes: float, menu: list[dict], package: dict | None = None, say=print, rounds: int = 1) -> dict:
    """The spoken, YouTube-style script (plan -> draft -> retention editor -> rewrite, see scriptwriter.py), cleaned. The plain one-shot writer is the fallback."""
    from . import scriptwriter as SW
    try:
        script = SW.write(ai, topic, source_text(srcs), minutes, menu, SCHEMA, package, WORDS_PER_MIN, rounds=rounds, say=say)
        if len(script.get("beats", [])) < 8:
            raise RuntimeError("the script came back too short")
    except Exception as e:  # noqa: BLE001
        log.warning(f"story-style script failed ({e}); using the plain writer")
        say(f"script: the story writer failed ({str(e)[:100]}), using the plain one")
        return _clean_script(_write_script_plain(ai, topic, srcs, minutes, menu))
    return _clean_script(script)


def _write_script_plain(ai: LocalAI, topic: str, srcs: list[dict], minutes: float, menu: list[dict]) -> dict:
    words = int(minutes * WORDS_PER_MIN)
    beats = max(12, int(minutes * 60 / 9))
    res = ai.complete_json("flow_explainer_script", SYSTEM.format(minutes=int(round(minutes)), words=words, beats=beats),
                           f"TOPIC: {topic}\n\nIMAGES (id: description):\n" + "\n".join(f"{m['id']}: {m['desc']}" for m in menu)
                           + f"\n\nSOURCE TEXT:\n{source_text(srcs)}", SCHEMA)
    return res.data


def _clean_script(script: dict) -> dict:
    script["beats"] = [b for b in script.get("beats", []) if len(b.get("narration", "").split()) >= 4]
    for b in script["beats"]:                                                  # the section title is a label: it must never be spoken ("Section what it is. A QR code is ...")
        sec = (b.get("section") or "").strip()
        n = re.sub(r"^\s*(section|chapter)\b\s*[:.,\-]?\s*", "", b["narration"], flags=re.I)
        if sec and n.lower().startswith(sec.lower()):
            n = re.sub(r"^[\s:.,\-]+", "", n[len(sec):])
        if len(n.split()) >= 4:
            b["narration"] = n
    if script["beats"]:
        script["beats"][0]["section"] = ""                                     # the hook plays first, with no title card
    anims = 0
    for b in script["beats"]:                                                  # an animation takes minutes to draw: at most MAX_ANIMATIONS per video, the rest become a photo or a card
        v = b.get("visual") if isinstance(b.get("visual"), dict) else {}
        if v.get("kind") == "animation":
            anims += 1
            if anims > MAX_ANIMATIONS:
                b["visual"] = {"kind": "photo"}
    MAX_STATS, MAX_SLIDES = 4, 5                                               # a deck of number cards and bullet lists is a slideshow: past these, the line is drawn instead
    stats = slides = 0
    seen_values: set[str] = set()
    for b in script["beats"]:
        v = b.get("visual") if isinstance(b.get("visual"), dict) else {}
        k = v.get("kind")
        val = re.sub(r"\W+", "", str(v.get("value") or "")).lower()
        stats += 1 if k == "stat" else 0
        slides += 1 if k == "slide" else 0
        too_many = (k == "stat" and (stats > MAX_STATS or (val and val in seen_values))) or (k == "slide" and slides > MAX_SLIDES)
        if k == "stat" and val:
            seen_values.add(val)
        if too_many and anims < MAX_ANIMATIONS:
            b["visual"] = {"kind": "animation", "concept": "Show what this line says as a bold animated diagram, with things moving and changing: " + b["narration"]}
            anims += 1
    flows = 0
    for b in script["beats"]:                                                  # a flow is for a real process: past four, the steps become a plain slide
        if not isinstance(b.get("visual"), dict) or not b["visual"].get("kind"):
            b["visual"] = {"kind": "photo"}                                    # a beat the model sent without a picture plan: a photo if one fits, else a card
        v = b["visual"]
        if v.get("kind") == "flow" and len([x for x in v.get("steps") or [] if x]) >= 3:     # (only flows that will really be drawn count)
            flows += 1
            if flows > 4:
                b["visual"] = {"kind": "slide", "title": v.get("title", ""), "bullets": v.get("steps", [])[:4]}
    return script


def check_script(ai: LocalAI, script: dict, srcs: list[dict], rounds: int = 2, say=print) -> dict:
    """Every claim against the source. A beat that still fails after a rewrite is dropped, never kept."""
    src = source_text(srcs)
    for r in range(rounds + 1):
        numbered = "\n".join(f"[{i}] {b['narration']}" for i, b in enumerate(script["beats"]))
        res = ai.complete_json(
            "flow_explainer_check",
            "You are a strict fact checker. Compare each numbered narration line with the SOURCE TEXT. List every claim, name, number or date in a line "
            "that the source does NOT support or contradicts (paraphrase is fine, invention is not). Plain explanations of how something works, analogies and scenes introduced with 'imagine' or 'picture' are fine; flag only specific numbers, names, dates and claims of record the source lacks. If every line is supported, return an empty list.",
            f"SOURCE TEXT:\n{src}\n\nNARRATION:\n{numbered}", CHECK_SCHEMA)
        bad = {p["beat"]: p for p in res.data.get("problems", []) if 0 <= p.get("beat", -1) < len(script["beats"])}
        if not bad:
            say(f"fact check: all {len(script['beats'])} lines supported" + (f" (after {r} rewrite round{'s' if r != 1 else ''})" if r else ""))
            return script
        say(f"fact check round {r + 1}: {len(bad)} line(s) with unsupported claims")
        if r == rounds:
            keep = [b for i, b in enumerate(script["beats"]) if i not in bad]
            say(f"dropping {len(bad)} line(s) that still fail: " + "; ".join(p["claim"][:60] for p in bad.values()))
            script["beats"] = keep
            script["dropped"] = [p["claim"] for p in bad.values()]
            return script
        fix = ai.complete_json(
            "flow_explainer_fix",
            "Rewrite ONLY the listed narration lines so they say nothing the SOURCE TEXT does not support. Keep each line 16-32 words, spoken style, "
            "and keep its role in the story. Return the new line for each listed number.",
            "SOURCE TEXT:\n" + src + "\n\nLINES TO FIX:\n" + "\n".join(
                f"[{i}] {script['beats'][i]['narration']}\n    problem: {p['claim']} ({p.get('why', '')})" for i, p in bad.items()), FIX_SCHEMA)
        for f in fix.data.get("beats", []):
            if f["beat"] in bad and len(f["narration"].split()) >= 4:
                script["beats"][f["beat"]]["narration"] = f["narration"]
    return script


# --------------------------------------------------------------------------------------------- pictures
SKIP_FILE = re.compile(r"logo|flag of|coat of arms|signature|icon|symbol|ambox|question book|wikiquote|wikisource|commons-|edit-|portal|disambig|stub|"
                       r"padlock|crystal clear|nuvola|red pog|folder|increase|decrease|steady|\bmap\b", re.I)
GOOD_LICENSE = re.compile(r"public ?domain|cc0|\bcc[ -]by\b|\bcc[ -]by[ -]sa\b|\bpd\b|attribution", re.I)
BAD_LICENSE = re.compile(r"\b(nc|nd)\b|non-?commercial|no-?deriv|fair use|copyrighted|all rights", re.I)


def _info_batch(titles: list[str]) -> list[dict]:
    out = []
    for i in range(0, len(titles), 40):
        d = WK.api_commons({"action": "query", "titles": "|".join(titles[i:i + 40]), "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 1920})
        for p in d.get("query", {}).get("pages", []):
            info = (p.get("imageinfo") or [{}])[0]
            meta = info.get("extmetadata") or {}
            lic = (meta.get("LicenseShortName") or {}).get("value", "")
            mime, w, h = info.get("mime"), info.get("width") or 0, info.get("height") or 0
            title = p.get("title", "")
            if not info.get("thumburl") or not GOOD_LICENSE.search(lic) or BAD_LICENSE.search(lic) or SKIP_FILE.search(title):
                continue
            if mime in ("image/jpeg", "image/png"):
                if w < 900 or h < 500:
                    continue
            elif mime != "image/svg+xml":
                continue
            desc = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", (meta.get("ImageDescription") or {}).get("value", ""))).strip()
            artist = re.sub(r"<[^>]+>", "", (meta.get("Artist") or {}).get("value", "")).strip() or "Wikimedia Commons"
            out.append({"title": title, "mime": mime, "w": w, "h": h, "url": info["thumburl"], "license": lic, "artist": artist[:80],
                        "page": info.get("descriptionurl", ""), "desc": (desc or re.sub(r"^File:|\.\w+$", "", title))[:140]})
    return out


def image_menu(topic: str, srcs: list[dict], limit: int = 40, cache_file: Path | None = None) -> list[dict]:
    if cache_file and cache_file.exists() and len(json.loads(cache_file.read_text())) >= 12:       # Commons is slow and strict: one lookup per topic
        return json.loads(cache_file.read_text())
    items = _image_menu(topic, srcs, limit)
    if cache_file and items:
        cache_file.write_text(json.dumps(items))
    return items


def _wikidata_category(qid: str) -> list[str]:
    import time as _t
    for k in range(4):
        try:
            r = requests.get("https://www.wikidata.org/w/api.php", params={"action": "wbgetclaims", "entity": qid, "property": "P373", "format": "json"},
                             headers={"User-Agent": WK._ua()}, timeout=30)
            if r.status_code == 429:
                _t.sleep(8 * (k + 1))
                continue
            wd = r.json()
            return ["Category:" + c["mainsnak"]["datavalue"]["value"] for c in wd.get("claims", {}).get("P373", []) if c.get("mainsnak", {}).get("datavalue")]
        except Exception as e:  # noqa: BLE001
            log.warning(f"wikidata category lookup failed: {e}")
            _t.sleep(3)
    return []


def _image_menu(topic: str, srcs: list[dict], limit: int = 40) -> list[dict]:
    """Pictures that are really about the topic: the images used in the article(s) first, then the Commons category named like the topic.
    Free licences only (public domain, CC0, CC BY, CC BY-SA); each gets an id the writer can pick by its description."""
    titles: list[str] = []
    for s in srcs:
        d = WK.api({"action": "query", "prop": "images", "titles": s["title"], "imlimit": "max"})
        for p in d.get("query", {}).get("pages", []):
            titles += [im["title"] for im in p.get("images", [])]
    cats = []
    for s in srcs:                                                              # the Commons category the article itself points to (Wikidata P373)
        d = WK.api({"action": "query", "prop": "pageprops", "ppprop": "wikibase_item", "titles": s["title"], "redirects": 1})
        qid = next((p.get("pageprops", {}).get("wikibase_item") for p in d.get("query", {}).get("pages", [])), None)
        if qid:
            cats += _wikidata_category(qid)
    if not cats:                                                                # no category on Wikidata: the closest named ones on Commons
        for q in [s["title"] for s in srcs] + [topic]:
            d = WK.api_commons({"action": "query", "list": "search", "srsearch": q, "srnamespace": 14, "srlimit": 2})
            cats += [h["title"] for h in d.get("query", {}).get("search", [])]
            if cats:
                break
    for c in list(dict.fromkeys(cats))[:2]:
        d = WK.api_commons({"action": "query", "list": "categorymembers", "cmtitle": c, "cmtype": "file", "cmlimit": 60})
        titles += [m["title"] for m in d.get("query", {}).get("categorymembers", [])]
        sub = WK.api_commons({"action": "query", "list": "categorymembers", "cmtitle": c, "cmtype": "subcat", "cmlimit": 8})
        for sc in sub.get("query", {}).get("categorymembers", [])[:6]:            # one level down: the specific kinds (manual, CVT, cutaways ...)
            try:
                d2 = WK.api_commons({"action": "query", "list": "categorymembers", "cmtitle": sc["title"], "cmtype": "file", "cmlimit": 20})
            except Exception as e:  # noqa: BLE001
                log.warning(f"subcategory {sc['title']}: {e}")
                continue
            titles += [m["title"] for m in d2.get("query", {}).get("categorymembers", [])]
    titles = list(dict.fromkeys(titles))
    items = _info_batch(titles[:220])
    order = {t: i for i, t in enumerate(titles)}
    items.sort(key=lambda x: order.get(x["title"], 999))
    seen: dict[str, int] = {}
    kept = []
    for it in items:                                                            # a category of one exhibition has dozens of near-identical captions: two each
        k = it["desc"][:40].lower()
        seen[k] = seen.get(k, 0) + 1
        if seen[k] <= 2:
            kept.append(it)
    items = kept[:limit]
    for i, it in enumerate(items, 1):
        it["id"] = f"i{i}"
    return items


def fetch_image(it: dict, cache: Path, tries: int = 5) -> Path | None:
    """Download one chosen picture (1920 px thumbnail), slowly and politely: Wikimedia answers 'too many requests' to bursts."""
    f = cache / (re.sub(r"[^a-z0-9]+", "-", it["title"].lower())[:70] + ".jpg")
    if f.exists():
        return f
    import io
    import time as _t

    from PIL import Image
    for k in range(tries):
        try:
            r = requests.get(it["url"], headers={"User-Agent": WK._ua()}, timeout=60)
            if r.status_code == 429:
                _t.sleep(4 * (k + 1))
                continue
            r.raise_for_status()
            im = Image.open(io.BytesIO(r.content))
            if im.mode in ("RGBA", "LA", "P"):                                  # diagrams (svg, png) come on a transparent ground: put them on white
                im = im.convert("RGBA")
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[3])
                im = bg
            im.convert("RGB").save(f, quality=93)
            _t.sleep(1.0)
            return f
        except Exception as e:  # noqa: BLE001
            log.warning(f"image download failed {it['title']}: {e}")
            _t.sleep(2 * (k + 1))
    return None


def card_spec(beat: dict, dur: float, words: list[dict]) -> dict | None:
    v = beat.get("visual") if isinstance(beat.get("visual"), dict) else {}
    kind = v.get("kind")
    if kind == "stat" and v.get("value") and v.get("label") and len(v["value"]) <= 8 and re.search(r"\d", v["value"]):
        return {"kind": "bigstat", "value": v["value"], "label": v["label"], "len": dur}
    if kind == "flow" and v.get("title") and len([x for x in v.get("steps") or [] if x]) >= 3:
        steps = [x for x in v["steps"] if x][:5]
        spec = {"kind": "flow", "title": v["title"], "steps": steps, "len": dur}
        times = GP._row_times(steps, words, 0.0, horizon=dur)
        if times:
            spec["times"] = times
        return spec
    if kind == "bars" and v.get("title") and len(v.get("labels") or []) >= 2 and len(v.get("values") or []) >= 2:
        return {"kind": "bars", "title": v["title"], "labels": v["labels"], "values": v["values"], "unit": v.get("unit") or "", "len": dur}
    if kind == "timeline" and v.get("title") and len(v.get("events") or []) >= 3:
        return {"kind": "line", "title": v["title"], "events": v["events"][:5], "len": dur}
    if kind == "slide" and v.get("title") and len(v.get("bullets") or []) >= 2:
        bullets = [b for b in v["bullets"] if b][:4]
        spec = {"kind": "fullslide", "title": v["title"], "bullets": bullets, "len": dur}
        times = GP._row_times(bullets, words, 0.0, horizon=dur)
        if times:
            spec["times"] = times
        return spec
    return None


def to_mp4(src: Path, out: Path, seconds: float) -> Path:
    """A card (transparent .mov) flattened onto its own page colour - the slide-in leaves the edges clear for a moment, and black there looks broken."""
    import subprocess
    probe = out.with_suffix(".png")
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-ss", f"{max(0.0, seconds - 0.4):.2f}", "-i", str(src), "-frames:v", "1", str(probe)], check=False)
    colour = "0xEFEFED"
    try:
        from PIL import Image
        r, g, b, a = Image.open(probe).convert("RGBA").getpixel((6, 6))
        colour = "0x%02X%02X%02X" % (r, g, b)
    except Exception:  # noqa: BLE001
        pass
    probe.unlink(missing_ok=True)
    ffmpeg.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c={colour}:s={W}x{H}:r={FPS}", "-i", str(src), "-filter_complex",
                f"[1:v]scale={W}:{H},fps={FPS}[c];[0:v][c]overlay=shortest=1,format=yuv420p[o]", "-map", "[o]", "-t", f"{seconds:.3f}", "-an", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", str(out)],
               timeout=600, what="encoding a card")
    return out


# --------------------------------------------------------------------------------------------- voice cache + animations
def voice_line(tts_cfg: dict, text: str, d: Path, vcache: Path, prev: str | None, nxt: str | None) -> "tts.TTSResult":
    """One narration line; a line already recorded in the same voice (a re-render of the same script) is reused, so a paid voice is never paid for twice."""
    import hashlib
    ec = tts_cfg.get("elevenlabs") or {}
    chain = ",".join(tts_cfg.get("provider_chain", []))
    key = hashlib.sha1(f"{chain}|{ec.get('voice_id')}|{ec.get('model_id')}|{(tts_cfg.get('edge') or {}).get('voice')}|{text}|{prev}|{nxt}".encode()).hexdigest()[:20]
    meta, audio = vcache / f"{key}.json", vcache / f"{key}.mp3"
    if meta.exists() and audio.exists():
        m = json.loads(meta.read_text())
        return tts.TTSResult(audio, m["words"], ffmpeg.duration(audio), m["provider"], m["voice"])
    res = tts.synthesize(text, tts_cfg, d, previous_text=prev, next_text=nxt)
    shutil.copy(res.audio, audio)
    meta.write_text(json.dumps({"words": res.words, "provider": res.provider, "voice": res.voice}))
    return tts.TTSResult(audio, res.words, res.duration, res.provider, res.voice)


def animate(cfg, ai, tier: str, concept: str, narration: str, words: list[dict], dur: float, look: tuple[str, str], work: Path, source_text: str, say=print, hook: bool = False) -> dict | None:
    """One topic-specific animation clip of exactly `dur` seconds. tier: svg (HTML/SVG rendered by Hyperframes), library (building blocks), auto (svg, then library).
    A finished clip is kept in `work` and reused when the same concept and length come again."""
    import hashlib
    key = hashlib.sha1(f"{concept}|{narration}|{dur:.2f}|{look}|{'hook' if hook else ''}|bold2".encode()).hexdigest()[:12]
    order = ["svg", "library"] if tier == "auto" else [tier]
    for t in order:
        d = work / t
        done = d / "animation.mp4"
        stamp = d / "key.txt"
        if done.exists() and stamp.exists() and stamp.read_text() == key:
            return {"file": done, "tier": t, "cached": True}
        if t == "svg":
            from . import animate_hf as AM
        else:
            from . import animate_lib as AM
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        r = AM.make(cfg, ai, concept, narration, words, dur, look, d, source_text, say=say, **({"hook": True} if hook and t == "svg" else {}))
        if r:
            stamp.write_text(key)
            return {**r, "tier": t}
    return None


def statement_card(narration: str, dur: float, max_words: int = 20) -> dict:
    """When a line has no picture and its animation could not be made: the line's own words, large, on a page of their own (never cut mid-word, never repeated)."""
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", " ".join(narration.split())) if x.strip()]
    words: list[str] = []
    for sent in sents:
        w = sent.split()
        if words and len(words) + len(w) > max_words:
            break
        words += w
        if len(words) >= 8:
            break
    text = " ".join(words[:max_words]) or " ".join(narration.split()[:max_words])
    return {"kind": "fullquote", "text": text, "len": dur, "mark": False}


def anim_cues(ai, work: Path, concept: str, narration: str, words: list[dict], dur: float) -> list[dict]:
    """Sound-effect cues for one animation scene (kept next to the clip, so a re-render does not ask again)."""
    import hashlib
    from . import soundfx as SFX
    key = hashlib.sha1(f"{concept}|{narration}|{dur:.2f}".encode()).hexdigest()[:12]
    f = work / "cues.json"
    if f.exists():
        try:
            c = json.loads(f.read_text())
            if c.get("key") == key:
                return c["cues"]
        except ValueError:
            pass
    cues = SFX.cues_for(ai, concept, narration, words, dur)
    f.write_text(json.dumps({"key": key, "cues": cues}))
    return cues


CARD_CUES = {"stat": [(0.5, "ding", 0.8)], "bigstat": [(0.5, "ding", 0.8)], "flow": [(0.15, "whoosh", 0.5)], "bars": [(0.3, "tap", 0.6)], "line": [(0.15, "whoosh", 0.5)],
             "fullslide": [(0.1, "pop", 0.6)], "slide": [(0.1, "pop", 0.6)], "chapter": [(0.1, "whoosh", 0.5)]}


def add_mascot(cfg, segs: list[dict], voiced: list, out_dir: Path, say=print) -> list[dict]:
    """Cogsworth (the channel's mascot) pops up in the corner of the long segments, where one picture has been on screen for a while, and says the line (mouth follows the voice)."""
    from . import mascot as MC
    size = int(cfg.get("explainer.mascot_size", 340))
    chosen = MC.pick_segments(segs, min_dur=float(cfg.get("explainer.mascot_min_s", 5.0)), min_gap_s=float(cfg.get("explainer.mascot_gap_s", 16.0)),
                              skip_first_s=float(cfg.get("explainer.mascot_skip_first_s", 2.5)), every_s=float(cfg.get("explainer.mascot_every_s", 18.0)))
    out = list(segs)
    for k, start in chosen.items():
        s = segs[k]
        try:
            frames = out_dir / "beats" / f"mascot{k:03d}"
            if frames.exists():
                shutil.rmtree(frames)
            n = MC.render(voiced[s["beat"]].audio, s["dur"] - start, frames, size=size, audio_from=start)
            if n < 10:
                continue
            f = out_dir / "beats" / f"seg{k:03d}_m.mp4"
            MC.composite(s["file"], frames, start, f, size=size)
            out[k] = {**s, "file": f, "mascot": start}
            shutil.rmtree(frames, ignore_errors=True)
        except Exception as e:  # noqa: BLE001 - the video is fine without him
            log.warning(f"mascot for segment {k} failed: {e}")
    say(f"mascot: in {sum(1 for s in out if s.get('mascot') is not None)} segments")
    return out


def all_cues(segs: list[dict]) -> list[dict]:
    """Every sound effect of the video on one timeline: the scene cues (animations) plus a small default per card, offset by where each segment starts."""
    out, t0 = [], 0.0
    for s in segs:
        if s.get("kind") == "animation":
            out += [{**c, "t": round(t0 + c["t"], 3)} for c in s.get("cues") or []]
        else:
            for dt, kind, g in CARD_CUES.get(s.get("kind") or "", []):
                if dt < s["dur"]:
                    out.append({"t": round(t0 + dt, 3), "kind": kind, "gain": g})
        if s.get("mascot") is not None:
            out.append({"t": round(t0 + float(s["mascot"]), 3), "kind": "pop", "gain": 0.7})                  # Cogsworth pops in
        t0 += s["dur"]
    return sorted(out, key=lambda c: c["t"])


# --------------------------------------------------------------------------------------------- chapter cards
CARD_BG, CARD_INK, CARD_MUTED, CARD_ACC = (11, 18, 30), (236, 242, 250), (150, 166, 188), (74, 168, 255)


def _card_font(size: int, bold: bool = True):
    from PIL import ImageFont
    for path, idx in (("/System/Library/Fonts/Helvetica.ttc", 1 if bold else 0), ("/System/Library/Fonts/HelveticaNeue.ttc", 1 if bold else 0), ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0)):
        try:
            return ImageFont.truetype(path, size, index=idx)
        except OSError:
            continue
    return ImageFont.load_default()


def chapter_card_png(label: str, sub: str, number: int, total: int, photo: Path | None, out: Path, W: int = 1920, H: int = 1080) -> Path:
    """The title card that opens a part of a list video: the part's number, its name big, a few plain words under it, and a picture of the thing on the right."""
    from PIL import Image, ImageDraw, ImageOps
    im = Image.new("RGB", (W, H), CARD_BG)
    d = ImageDraw.Draw(im)
    px0, py0, pw, ph = 1010, 150, 820, 780                                   # the picture panel
    if photo and Path(photo).exists():
        pic = ImageOps.fit(Image.open(photo).convert("RGB"), (pw, ph), Image.LANCZOS, centering=(0.5, 0.5))
        im.paste(pic, (px0, py0))
    else:
        d.rectangle((px0, py0, px0 + pw, py0 + ph), fill=(20, 40, 70))
    d.rectangle((px0, py0, px0 + pw, py0 + ph), outline=(60, 80, 110), width=2)
    x0, maxw = 110, 820
    d.text((x0, 300), f"{number:02d} / {total}", fill=CARD_ACC, font=_card_font(44))
    size = 140
    while size > 64 and d.textlength(label, font=_card_font(size)) > maxw:
        size -= 6
    f = _card_font(size)
    d.text((x0, 380), label, fill=CARD_INK, font=f)
    ly = 380 + int(size * 1.18)
    d.rectangle((x0, ly, x0 + 150, ly + 6), fill=CARD_ACC)
    if sub:
        d.text((x0, ly + 40), sub, fill=CARD_MUTED, font=_card_font(46, bold=False))
    im.save(out, quality=95)
    return out


def chapter_segment(png: Path, out: Path, dur: float, W: int, H: int, fps: int) -> Path:
    """The card as video: a slow push-in and a short fade up from the dark."""
    frames = int(dur * fps)
    ffmpeg.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-loop", "1", "-i", str(png), "-t", f"{dur:.3f}", "-vf",
                f"scale={W * 2}:{H * 2},zoompan=z='1+0.025*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={W}x{H}:fps={fps},fade=t=in:st=0:d=0.25,format=yuv420p",
                "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", str(out)], timeout=300, what="a chapter card")
    return out


# --------------------------------------------------------------------------------------------- build
def build(cfg, topic: str, minutes: float = 6.0, out_dir: Path | None = None, titles: list[str] | None = None, theme: str | None = None,
          say=print, script_file: Path | None = None, animation_tier: str | None = None, progress=None, package: dict | None = None, free_voice: bool = False) -> dict:
    """progress(fraction 0..1, label) is called as the build advances (the Long-form page shows it as a bar)."""
    progress = progress or (lambda f, label: None)
    progress(0.02, "Finding sources and pictures")
    animation_tier = animation_tier or str(cfg.get("explainer.animation_tier", "auto"))
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:50]
    out_dir = Path(out_dir or Path(cfg.root) / "data" / "explainer" / slug)
    cache = Path(cfg.root) / "data" / "explainer" / "_images"
    for d in (out_dir, out_dir / "beats", cache):
        d.mkdir(parents=True, exist_ok=True)
    ai = LocalAI(cfg)
    ok, why = ai.available()
    if not ok:
        raise RuntimeError(f"no AI to write the script: {why}")

    srcs = sources(topic, titles, max_chars=min(14000, 64000 // max(1, len(titles or [])))) if titles and len(titles) > 1 else sources(topic, titles)       # a list of articles shares one source budget
    say(f"sources: {', '.join(s['title'] for s in srcs)} ({sum(len(s['text']) for s in srcs)} characters)")
    menu = image_menu(topic, srcs, cache_file=out_dir / "menu.json")
    progress(0.06, "Writing the script")
    say(f"images: {len(menu)} free-licence pictures about the topic to choose from")
    if script_file and Path(script_file).exists():
        script = json.loads(Path(script_file).read_text())
        say(f"script: reusing {script_file}")
    else:
        script = write_script(ai, topic, srcs, minutes, menu, package=package, say=say, rounds=int(cfg.get("explainer.script_rounds", 1)))
        say(f"script: {len(script['beats'])} beats, {sum(len(b['narration'].split()) for b in script['beats'])} words, title {script['title']!r}")
        script = check_script(ai, script, srcs, say=say)
    (out_dir / "script.json").write_text(json.dumps(script, indent=2))
    progress(0.14, "Recording the voice")

    # voice: one recording per beat so each picture is timed to its own line
    tts_cfg = cfg.get("tts") or {}
    if free_voice:                                                             # a draft: a free voice until the owner has approved the script (the paid voice is only spent after that)
        tts_cfg = {**tts_cfg, "provider_chain": [p for p in (tts_cfg.get("provider_chain") or []) if p != "elevenlabs"] or ["edge"]}
    cat = tts.cfg_for_category(tts_cfg, "news")
    beats = script["beats"]
    voiced = []
    try:                                                                       # a paid voice: say where the month's credits stand before spending more
        from ..core import elevenlabs_voice as EV
        b = EV.budget()
        if b:
            say(f"voice credits: {b['used']} of {b['limit']} used ({b['pct']}%), resets {b['resets']}" + ("  - RUNNING LOW" if b["warn"] else ""))
    except Exception:  # noqa: BLE001
        pass
    vcache = out_dir / "voice_cache"
    vcache.mkdir(exist_ok=True)
    for i, b in enumerate(beats):
        d = out_dir / "beats" / f"{i:02d}"
        res = voice_line(cat, b["narration"], d, vcache, beats[i - 1]["narration"] if i else None, beats[i + 1]["narration"] if i + 1 < len(beats) else None)
        voiced.append(res)
        progress(0.14 + 0.16 * (i + 1) / len(beats), f"Recording the voice ({i + 1} of {len(beats)} lines)")
    say(f"voice: {len(voiced)} lines, {sum(v.duration for v in voiced):.0f}s of speech ({voiced[0].provider}/{voiced[0].voice})")

    # pictures
    credits: list[dict] = []
    plan: list[dict] = []                      # one entry per segment in order: {"kind", "dur", "silent", "src": Path}
    chapter_no = 0
    chapter_at: list[tuple[int, str]] = []
    look_of = lambda n_: LOOKS[max(0, n_ - 1) % len(LOOKS)] if n_ else LOOKS[0]  # noqa: E731
    jobs: list[tuple] = []
    for i, (b, v) in enumerate(zip(beats, voiced)):
        sec = (b.get("section") or "").strip()
        if sec:                                                              # a chapter marker (for the video's chapter list), not a slide: the story never stops for a title card
            chapter_no += 1
            chapter_at.append((i, sec))
        dur = v.duration + PAD
        vis = b.get("visual") if isinstance(b.get("visual"), dict) else {}
        if vis.get("kind") == "animation" and (vis.get("concept") or "").strip():
            jobs.append(("animation", {"concept": vis["concept"].strip(), "narration": b["narration"], "words": v.words}, dur, False, i, look_of(chapter_no)))
            continue
        spec = card_spec(b, dur, v.words) if vis.get("kind") not in (None, "photo") else None
        if spec:
            jobs.append(("card", spec, dur, False, i, look_of(chapter_no)))
            continue
        pick = next((m for m in menu if m["id"] == (vis.get("image") or "").strip()), None)
        photo = None
        if pick:
            f = fetch_image(pick, cache)
            photo = {**pick, "file": f} if f else None
        if photo:
            if not any(c["title"] == photo["title"] for c in credits):
                credits.append(photo)
            if dur > float(cfg.get("explainer.long_photo_s", 9.0)) and not vis.get("hold") and sum(1 for j_ in jobs if j_[0] == "animation") < MAX_ANIMATIONS:
                # a still photo for this long loses the viewer: draw it as a moving diagram instead (the photo stays as the fallback if the drawing cannot be made well)
                jobs.append(("animation", {"concept": "Show what this line describes as a bold animated diagram (the photo shows: " + str(photo.get("desc") or photo.get("title") or "")[:140] + "): " + b["narration"],
                                           "narration": b["narration"], "words": v.words, "photo": photo}, dur, False, i, look_of(chapter_no)))
            else:
                jobs.append(("photo", photo, dur, False, i, None))
        elif sum(1 for j_ in jobs if j_[0] == "animation") < MAX_ANIMATIONS:   # no picture fits this line: draw one (an animation of the line), never a repeated photo or a card of words
            jobs.append(("animation", {"concept": "Draw the idea of this line as a simple animated diagram, showing what it describes happening: " + b["narration"], "narration": b["narration"],
                                       "words": v.words}, dur, False, i, look_of(chapter_no)))
        elif credits:                                                      # (animation budget used up) show a picture already used, with a different move
            jobs.append(("photo", credits[len(jobs) % len(credits)], dur, False, i, None))
        else:
            jobs.append(("card", statement_card(b["narration"], dur), dur, False, i, look_of(chapter_no)))

    jobs = [(kd, pl, du, si, bi, (LOOKS[1] if bi < 2 and lk is not None else lk)) for kd, pl, du, si, bi, lk in jobs]       # the opening shots on the dark blueprint page: the boldest look

    def make(n_job: tuple[int, tuple]) -> dict:
        k, (kind, payload, dur, silent, i, look) = n_job
        out = out_dir / "beats" / f"seg{k:03d}.mp4"
        if kind == "animation":
            got = anim_done.get(k)
            if got:
                ffmpeg.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-i", str(got["file"]), "-t", f"{dur:.3f}", "-vf", f"scale={W}:{H},fps={FPS},format=yuv420p", "-an", "-c:v", "libx264",
                            "-preset", "veryfast", "-crf", "17", str(out)], timeout=300, what="fitting an animation")
                return {"file": out, "dur": dur, "silent": silent, "beat": i, "kind": "animation", "cues": got.get("cues") or []}
            if payload.get("photo"):                                              # the drawing failed: the photo it replaced
                PH.bake(str(payload["photo"]["file"]), EFFECTS[k % len(EFFECTS)], dur, W, H, FPS, out)
                return {"file": out, "dur": dur, "silent": silent, "beat": i, "kind": "photo"}
            kind, payload = "card", statement_card(payload["narration"], dur)
        if kind == "photo":
            PH.bake(str(payload["file"]), EFFECTS[k % len(EFFECTS)], dur, W, H, FPS, out)
        else:
            mov = out_dir / "beats" / f"seg{k:03d}.mov"
            th, bg = (theme, None) if theme else (look or LOOKS[0])
            payload = {**payload, "bg": bg}
            r = GC.render(cfg, payload, W, H, mov, theme=th)
            if not r and payload.get("times"):                                 # reveal times that do not fit the card: show the bullets on the default schedule
                r = GC.render(cfg, {k: v for k, v in payload.items() if k != "times"}, W, H, mov, theme=th)
            if not r and payload.get("bullets"):                               # still no: fewer, shorter bullets
                r = GC.render(cfg, {**{k: v for k, v in payload.items() if k != "times"}, "bullets": [b[:40] for b in payload["bullets"][:3]]}, W, H, mov, theme=th)
            if not r:
                log.warning(f"card did not render, using a plain title card: {payload}")
                r = GC.render(cfg, {"kind": "chapter", "number": "", "title": str(payload.get("title") or payload.get("label") or "")[:40], "len": payload.get("len", 4.0), "bg": bg},
                              W, H, mov, theme=th)
            if not r:
                raise RuntimeError(f"card {payload.get('kind')} did not render: {payload}")
            to_mp4(mov, out, dur)
            mov.unlink(missing_ok=True)
        return {"file": out, "dur": dur, "silent": silent, "beat": i, "kind": (payload.get("kind") if kind == "card" else kind)}

    anim_done: dict[int, dict] = {}
    anim_report = []
    src_text = source_text(srcs)
    n_anim = max(1, sum(1 for j in jobs if j[0] == "animation"))
    n_done = [0]
    alock = threading.Lock()

    def draw(item):
        k, (kind, payload, dur, silent, i, look) = item
        with alock:
            progress(0.30 + 0.30 * n_done[0] / n_anim, f"Drawing the animations ({n_done[0] + 1} of {n_anim})")
        say(f"animation for line {i}: {payload['concept'][:80]}")
        th, bg = (theme, look[1]) if theme else (look or LOOKS[0])
        r = animate(cfg, ai, animation_tier, payload["concept"], payload["narration"], payload["words"], dur, (th, bg), out_dir / "anim" / f"{i:02d}", src_text, say=say, hook=i < 2)
        if r and bool(cfg.get("explainer.sfx", True)):
            try:
                r["cues"] = anim_cues(ai, out_dir / "anim" / f"{i:02d}", payload["concept"], payload["narration"], payload["words"], dur)
            except Exception as e_:  # noqa: BLE001 - no sound effects for this scene
                log.warning(f"sound cues for line {i} failed: {e_}")
        with alock:
            n_done[0] += 1
            anim_report.append({"line": i, "concept": payload["concept"], "tier": (r or {}).get("tier"), "ok": bool(r), "attempts": (r or {}).get("attempts"),
                                "seconds": (r or {}).get("seconds"), "qa_ok": (r or {}).get("qa_ok"), "cached": (r or {}).get("cached", False)})
            if r:
                anim_done[k] = r
    with ThreadPoolExecutor(int(cfg.get("explainer.animation_workers", 3) or 3)) as ex_:       # three at a time: the AI design calls wait on the network, the renders are short
        list(ex_.map(draw, [(k, j) for k, j in enumerate(jobs) if j[0] == "animation"]))
    if anim_report:
        (out_dir / "animations.json").write_text(json.dumps(anim_report, indent=1))
        say(f"animations: {sum(1 for a in anim_report if a['ok'])} of {len(anim_report)} made")
    say(f"pictures: {len(jobs)} segments ({sum(1 for j in jobs if j[0] == 'photo')} photos, {sum(1 for j in jobs if j[0] != 'photo')} cards)")
    lock_, made_ = threading.Lock(), [0]

    def make_counted(item):
        r_ = make(item)
        with lock_:
            made_[0] += 1
            progress(0.60 + 0.28 * made_[0] / len(jobs), f"Making the pictures ({made_[0]} of {len(jobs)})")
        return r_
    progress(0.60, "Making the pictures")
    with ThreadPoolExecutor(3) as ex:
        segs = list(ex.map(make_counted, enumerate(jobs)))

    # a title card with a picture of the thing at the start of every part that asks for one (beat["chapter"] = {label, sub, image})
    ch_beats = [i for i, b in enumerate(beats) if isinstance(b.get("chapter"), dict) and b["chapter"].get("label")]
    if ch_beats:
        card_len = float(cfg.get("explainer.chapter_card_s", 2.6))
        by_beat = {sg["beat"]: n_ for n_, sg in enumerate(segs) if sg.get("beat") is not None}
        carded: list[dict] = []
        cards_at: dict[int, dict] = {}
        for n_, bi in enumerate(ch_beats, 1):
            spec = beats[bi]["chapter"]
            pick = next((m for m in menu if m["id"] == (spec.get("image") or "").strip()), None)
            fpic = fetch_image(pick, cache) if pick else None
            if pick and fpic and not str(pick.get("license", "")).lower().startswith("own") and not any(c["title"] == pick["title"] for c in credits):
                credits.append(pick)
            png = chapter_card_png(spec["label"], spec.get("sub") or "", n_, len(ch_beats), fpic, out_dir / "beats" / f"chap{n_:02d}.png")
            mp4 = chapter_segment(png, out_dir / "beats" / f"chap{n_:02d}.mp4", card_len, W, H, FPS)
            cards_at[bi] = {"file": mp4, "dur": card_len, "silent": True, "beat": None, "kind": "chapter", "card_for": bi}
        for sg in segs:
            if sg.get("beat") in cards_at:
                carded.append(cards_at.pop(sg["beat"]))
            carded.append(sg)
        segs = carded
        say(f"chapter cards: {len(ch_beats)}")

    # sources card at the end
    end_len = 5.0
    arts = [s["title"] for s in srcs]
    endspec = {"kind": "fullslide", "bg": LOOKS[0][1], "title": "Sources", "bullets": [f"Wikipedia: {a}" for a in arts][:3] + ["Photos: Wikimedia Commons"], "len": end_len}
    mov = out_dir / "beats" / "end.mov"
    if GC.render(cfg, endspec, W, H, mov, theme=theme or LOOKS[0][0]):
        segs.append({"file": to_mp4(mov, out_dir / "beats" / "end.mp4", end_len), "dur": end_len, "silent": True, "beat": None})

    progress(0.90, "Mixing the sound")
    if bool(cfg.get("explainer.mascot", True)):
        segs = add_mascot(cfg, segs, voiced, out_dir, say)
    # audio: narration lines with their air, silence under the cards
    wavs = []
    for k, s in enumerate(segs):
        wav = out_dir / "beats" / f"a{k:03d}.wav"
        if s["silent"] or s["beat"] is None:
            ffmpeg.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", f"{s['dur']:.3f}", str(wav)],
                       timeout=60, what="silence")
        else:
            ffmpeg.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-i", str(voiced[s["beat"]].audio), "-af", f"apad=pad_dur={PAD + 0.2},atrim=0:{s['dur']:.3f}",
                        "-ar", "48000", "-ac", "2", str(wav)], timeout=120, what="narration line")
        wavs.append(wav)
    lst = out_dir / "audio.txt"
    lst.write_text("".join(f"file '{w.resolve()}'\n" for w in wavs))
    voice_wav = out_dir / "voice.wav"
    ffmpeg.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-af", "loudnorm=I=-16:TP=-1.5:LRA=9", "-ar", "48000", str(voice_wav)],
               timeout=300, what="joining narration")
    total = sum(s["dur"] for s in segs)

    vlst = out_dir / "video.txt"
    vlst.write_text("".join(f"file '{s['file'].resolve()}'\n" for s in segs))
    silent_mp4 = out_dir / "video_only.mp4"
    ffmpeg.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(vlst), "-c", "copy", str(silent_mp4)], timeout=300, what="joining pictures")

    progress(0.95, "Putting video and sound together")
    # music bed
    track = None
    if str(cfg.get("explainer.music", "none")) == "library":                      # off by default: a talking explainer is better with no music than with stock library music
        try:
            track = MU.pick(cfg, mood="chill", seed=slug)
        except Exception as e:  # noqa: BLE001
            log.warning(f"no music: {e}")
    final = out_dir / f"{slug}.mp4"
    cmd = [ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-i", str(silent_mp4), "-i", str(voice_wav)]
    cues = all_cues(segs) if bool(cfg.get("explainer.sfx", True)) else []
    sfx_wav = None
    if cues:
        from . import soundfx as SFX
        sfx_wav = SFX.render(cues, total, out_dir / "sfx.wav", level_db=float(cfg.get("explainer.sfx_level_db", -17)))
        (out_dir / "sfx_cues.json").write_text(json.dumps(cues, indent=1))
        say(f"sound effects: {len(cues)} cues")
        cmd += ["-i", str(sfx_wav)]
    if track:
        mi = 3 if sfx_wav else 2
        cmd += ["-stream_loop", "-1", "-i", str(track.path), "-filter_complex",
                (f"[{mi}:a]volume=-26dB,afade=t=in:d=2,afade=t=out:st={max(0, total - 3):.2f}:d=3[m];" + ("[1:a][2:a][m]amix=inputs=3" if sfx_wav else "[1:a][m]amix=inputs=2")
                 + ":duration=first:normalize=0[a]"), "-map", "0:v", "-map", "[a]"]
    elif sfx_wav:
        cmd += ["-filter_complex", "[1:a][2:a]amix=inputs=2:duration=first:normalize=0[a]", "-map", "0:v", "-map", "[a]"]
    else:
        cmd += ["-map", "0:v", "-map", "1:a"]
    cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", f"{total:.2f}", "-movflags", "+faststart", str(final)]
    ffmpeg.run(cmd, timeout=900, what="the final mix")

    cr = out_dir / "credits.txt"
    lines = [f"{script['title']}", "", "Text: Wikipedia (CC BY-SA 4.0), retold in our own words:"] + [f"  {s['title']}  {s['url']}" for s in srcs] + ["", "Pictures (Wikimedia Commons):"]
    lines += [f"  {c['title']} - {c['artist']} - {c['license']} - {c['page']}" for c in credits]
    if track:
        lines += ["", f"Music: {track.title} ({track.license})"]
    cr.write_text("\n".join(lines) + "\n")
    chapters, t_ = [], 0.0
    starts = {}
    for s_ in segs:
        if s_.get("card_for") is not None:
            starts.setdefault(s_["card_for"], t_)
        elif s_["beat"] is not None and s_["beat"] not in starts:
            starts[s_["beat"]] = t_
        t_ += s_["dur"]
    for bi, title in chapter_at:
        if bi in starts:
            chapters.append((starts[bi], title))
    say(f"done: {final} ({total / 60:.1f} min)")
    progress(1.0, "Done")
    return {"chapters": chapters, "video": final, "credits": cr, "script": out_dir / "script.json", "minutes": total / 60, "beats": len(beats), "photos": len(credits)}
