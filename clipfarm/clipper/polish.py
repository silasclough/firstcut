"""Whole-video mode: captions + music + dead-air trimming, nothing else - plus a Final Cut Pro project.

For "just caption it and add music" and for larger projects you finish in Final Cut:
  1. dead space: silences longer than the pace setting are cut (talking videos only - footage that is
     mostly visual is kept whole apart from silent lead-in / tail)
  2. original framing and resolution, burned-in word-by-word captions sized for the video's shape
  3. music picked from what is SAID (the AI reads the transcript -> mood + genre), ducked under speech
  4. outputs: clip_01.mp4 (the finished video), .srt, .txt (credits) and fcp_project.zip:
        <Title>/<Title>.fcpxml, Media/<source>, Music/<track> + license, Captions/<Title>.srt
     The zip's project points at clipper.fcp_unpack_dir/<Title>/ (unzip it there, e.g. ~/Movies/Clipper);
     the archive copy on the drive points at itself. Either way FCP's "Relink Files" can fix moved media.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import zipfile
from fractions import Fraction
from pathlib import Path

from ..core import ffmpeg
from ..core.captions import clean_words, coerce_words, group_words, render_captions
from ..core.log import get_logger
from . import fcpxml as FX
from . import music as MU
from .library import safe

log = get_logger("clipper.polish")


STANDARD_RATES = [Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001), Fraction(30),
                  Fraction(50), Fraction(60000, 1001), Fraction(60)]


def _fps(vs: dict) -> Fraction:
    """The stream's frame rate snapped to a standard one (phones record ~29.98 variable rate; FCP wants 29.97/30)."""
    for k in ("r_frame_rate", "avg_frame_rate"):
        try:
            f = Fraction(vs.get(k) or "0/0")
        except (ValueError, ZeroDivisionError):
            continue
        if 10 <= f <= 130:
            best = min(STANDARD_RATES, key=lambda r: abs(float(r) - float(f)))
            if abs(float(best) - float(f)) / float(best) < 0.01:
                return best
    return Fraction(30)


def keep_intervals(words: list[dict], total: float, gap: float) -> list[tuple[float, float]]:
    """Talking video: drop silences > gap. Mostly-visual video: keep everything but silent lead-in/tail."""
    from .engine import repace
    if not words:
        return [(0.0, total)]
    if len(words) / max(1.0, total) < 0.3:
        return [(max(0.0, words[0]["start"] - 1.0) if words[0]["start"] > 3 else 0.0, total)]
    return repace(words, 0.0, total, gap)


def select_expr(intervals: list[tuple[float, float]]) -> str:
    return "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in intervals)


def caption_groups(cap_words: list[dict], ccfg: dict) -> list[tuple[float, float, str]]:
    words = clean_words(coerce_words(cap_words), float(ccfg.get("min_word_s", 0.12)))
    groups = group_words(words, int(ccfg.get("max_words", 3)), int(ccfg.get("max_chars", 18)),
                         float(ccfg.get("pause_break_s", 0.35)))
    return [(g.start, g.end, g.text) for g in groups]


def planner_ai(cfg, engine):
    """The AI for planning questions: this machine's ChatGPT plan / OpenAI when it runs local projects (never the local Ollama), else the creative router."""
    if cfg.get("clipper.ai_videos.local_projects", False):
        from .localai import LocalAI
        return LocalAI(cfg)
    return engine.creative_router()


def looks_like_filename(title: str, files: list[dict] | None = None) -> bool:
    t = (title or "").strip().lower()
    stems = {Path(str(f.get("name", ""))).stem.lower() for f in (files or [])}
    return (not t) or t in stems or bool(re.fullmatch(r"(img|mvi|dsc|pxl|vid|mov|gopr|gx\d*|c|clip|clipper video|video)[ _-]?[\d_ -]{2,}(\.\w+)?", t)) or t in ("clipper video", "untitled", "video") or bool(re.fullmatch(r"[\w\- ]+\.(mov|mp4|m4v|mkv|webm)", t))


TITLE_SYSTEM = ("You name a video from its transcript. Write ONE short, specific, engaging title: 3 to 8 words, plain words, no quotation marks, no emoji, no all caps, "
                "no clickbait that the video does not deliver. It should say what the video is about or its funniest/strongest point.")


def fresh_title(ai, words: list[dict]) -> str | None:
    text = " ".join(w["text"] for w in words)[:4000]
    try:
        t = ai.complete_json("video_title", TITLE_SYSTEM, f"Transcript:\n{text}",
                             {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}).data.get("title", "")
    except Exception as e:  # noqa: BLE001 - keep whatever name there is
        log.warning(f"could not make up a title: {e}")
        return None
    words_ = " ".join(str(t).replace('"', "").split()).split()
    t = " ".join(words_[:8]).rstrip(" ,;:-")[:70]                                   # 3-8 words, as asked
    return t if len(t.split()) >= 2 else None


def run(engine, job_id: str, src: Path, words: list[dict], override: list[tuple[float, float]] | None = None, mc: dict | None = None) -> dict:
    from .engine import CAPTION_STYLES, PACES, remap_words
    cfg, store = engine.cfg, engine.store
    d = store.dir(job_id)
    job = store.get(job_id)
    opts = job["options"]
    title = job.get("title") or "Clipper video"
    info = ffmpeg.probe(src)
    vs = ffmpeg.video_stream(info) or {}
    from .glasses import _display_dims
    W, H = (x // 2 * 2 for x in _display_dims(vs))
    fps = _fps(vs)
    has_audio = ffmpeg.audio_stream(info) is not None
    total = ffmpeg.duration(src)

    # 1. dead space (and, in long-video mode, the story: which parts to keep, in the order they happened)
    pace = opts.get("pace") if opts.get("pace") in PACES else "natural"
    story_plan = None
    files_meta = (job.get("info") or {}).get("files") or []
    broll_scenes: dict = {}
    if opts.get("project") and files_meta:                        # AI videos: look at the dialogue-free footage (B-roll)
        from . import broll
        engine._progress(job_id, "finding", "Looking at the B-roll", 0.42)
        try:
            broll_scenes = broll.describe(engine, src, files_meta, words, d,
                                          lambda m: engine._progress(job_id, "finding", m, 0.42))
            pool = broll.pool_files(files_meta, broll_scenes, words)
            for k, f in enumerate(files_meta):
                f["broll"] = k in pool
        except Exception as e:  # noqa: BLE001 - the video is made without cutaways
            log.warning(f"B-roll step failed: {e}")
            broll_scenes = {}
    if opts.get("mode") == "long":
        from . import story
        engine._progress(job_id, "finding", "Editing the story (keeping it in order)", 0.45)
        files = files_meta
        cap = 7200.0 if opts.get("project") else 170.0                  # AI-video projects are real long-form; everything else fits a Short
        target = float(opts.get("long_minutes") or 0) * 60 or min(total * (0.85 if opts.get("project") else 0.75), cap)
        story_plan = story.plan(engine, words, files, total, title, target, float(PACES[pace]["gap"]),
                                (opts.get("prompt") or "").strip())
        intervals = story_plan.intervals
        if story_plan.title and story_plan.title != "A day out":
            title = story_plan.title[:90]
            store.update(job_id, title=title)
    else:
        intervals = keep_intervals(words, total, float(PACES[pace]["gap"]))
    if override:                                              # an edit you asked for in plain words (cuts, a hook moved up front)
        intervals = override
    out_len = sum(b - a for a, b in intervals)
    (d / "plan.json").write_text(json.dumps({"intervals": [[round(a, 2), round(b, 2)] for a, b in intervals], "total": round(total, 2)}))
    cap_words = remap_words(words, intervals, 1.0)
    if looks_like_filename(title, files_meta) and cap_words and not opts.get("title_set"):          # nobody named it: make a good one up from what is said
        made = fresh_title(planner_ai(cfg, engine), cap_words)
        if made:
            title = made
            store.update(job_id, title=title)
            log.info(f"{job_id}: titled '{title}'")
    cutaways: list[dict] = []
    pool_scenes = [{**s, "file": f["name"]} for f in files_meta if f.get("broll") for s in broll_scenes.get(f["name"], [])
                   if s["quality"] >= 4]
    if pool_scenes and cap_words:
        from . import broll
        from . import moments as M
        engine._progress(job_id, "finding", "Placing B-roll cutaways", 0.5)
        cutaways = broll.plan_cutaways(engine.creative_router(), M.sentences_from_words(cap_words), pool_scenes,
                                       (opts.get("prompt") or "").strip(), out_len)
        log.info(f"{job_id}: {len(cutaways)} B-roll cutaways planned")
    (d / "plan.json").write_text(json.dumps({"intervals": [[round(a, 2), round(b, 2)] for a, b in intervals], "total": round(total, 2),
                                             "cutaways": cutaways}))

    log.info(f"{job_id}: graphics option {opts.get('graphics')!r}, kind {opts.get('kind')!r}, {len(cap_words)} words")
    if opts.get("graphics") == "all" and not opts.get("gfx_cards") and cap_words:        # every kind of graphic, chosen from what is SAID (the same planner the local-folder projects use)
        try:
            from . import gfxplan, graphics as GRP
            from . import moments as M
            from .localproj import PROFILES
            kind = opts.get("kind") if opts.get("kind") in PROFILES else "story"
            engine._progress(job_id, "finding", "Choosing the on-screen graphics from the transcript", 0.52)
            gai = planner_ai(cfg, engine)
            cards = gfxplan.plan(gai, M.sentences_from_words(cap_words), (opts.get("prompt") or "").strip(), PROFILES[kind], out_len,
                                 hook_end=3.2 if title else 0.0, words=cap_words, cfg=cfg)
            opts["gfx_cards"] = cards
            (d / "graphics_check.md").write_text(gfxplan.check_notes(cards, M.sentences_from_words(cap_words)))          # what every graphic shows, and what you said for it: for a human to check
            opts["gfx_credits"] = sorted({i["credit"] for g in cards for i in (g.get("items") or []) if i.get("credit")})
            if not opts.get("gfx_theme"):
                opts["gfx_theme"] = GRP.match_theme(GRP.video_luma(str(src), total), kind, job_id)
            log.info(f"{job_id}: graphics: {len(cards)} planned ({', '.join(g['kind'] for g in cards)}), look {opts['gfx_theme']}")
        except Exception as e:  # noqa: BLE001 - the video is made without them
            log.warning(f"{job_id}: planning the graphics failed: {e}")

    # 2. captions sized for the frame (built for 1080-wide vertical video)
    style = opts.get("caption_style", "bold")
    style = "bold" if style == "mix" else style
    landscape = W > H
    k = (H / 1080) * 0.8 if landscape else W / 1080
    ccfg = {**(cfg.get("captions", {}) or {}), **{a: b for a, b in CAPTION_STYLES.get(style, {}).items() if a != "label"}}
    ccfg["font_size"] = int(float(ccfg.get("font_size", 92)) * k)
    ccfg["stroke_width"] = max(2, int(float(ccfg.get("stroke_width", 9)) * k))
    ccfg["box_height"] = int(float(ccfg.get("box_height", 420)) * k)
    ccfg["center_y"] = 0.84 if landscape else 0.72
    if landscape:
        ccfg["max_words"], ccfg["max_chars"] = max(int(ccfg.get("max_words", 3)), 4), max(int(ccfg.get("max_chars", 18)), 26)
    vcfg = {**(cfg.get("video", {}) or {}), "width": W, "height": H, "fps": float(fps)}
    vcfg["videotoolbox_bitrate"] = f"{max(4, round(8 * W * H / (1080 * 1920)))}M"
    if out_len > 90:                                          # long videos: 4 Mbit/s keeps them small enough to upload (a 4-min one was 278 MB)
        vcfg["videotoolbox_bitrate"] = "4M"
        vcfg["x264_crf"] = 24
    work = d / "polish_work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    caps = None
    if style != "none" and len(cap_words) >= 1:
        engine._progress(job_id, "rendering", "Making captions", 0.55)
        caps = render_captions(cap_words, out_len, work, ccfg, vcfg)

    gfx = None                                                # the animated graphics, rendered 2 at a time
    pts: list[tuple[dict, Path]] = []
    glass_items: list[dict] = []                              # where each graphic has glass (the picture behind it is blurred)
    cards_done: list[dict] = []                               # the planned graphics that rendered: Final Cut gets these as clips
    title_clips: dict = {}                                    # hook / key point -> the panel it was drawn in: its editable Final Cut title is placed there
    bug = None                                                # the corner logo, when one is asked for
    if not opts.get("no_overlays"):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from . import graphics
        specs = []                                            # (kind, text, seconds, file, extra, the thing it belongs to)
        if opts.get("graphics") in ("title", "all") and title and not opts.get("folder_title"):     # (a title graphic of your own replaces it)
            specs.append(("title", str(opts.get("hook_text") or title), 3.2, work / "title.mov", {"theme": opts.get("gfx_theme")}, "hook"))
        for k, c in enumerate(opts.get("gfx_cards") or []):  # slides, charts, stats, quotes, lower thirds planned from the transcript
            specs.append(("card", str(c.get("title") or c.get("name") or c.get("kind")), float(c["len"]), work / f"card{k}.mov",
                          {"theme": opts.get("gfx_theme"), "spec": c}, c))
        for k, c in enumerate(opts.get("callouts") or []):    # key points: animated text at the right moments
            specs.append(("point", str(c["text"]), float(c["len"]), work / f"point{k}.mov", {"theme": opts.get("gfx_theme")}, c))
        for k, g in enumerate(opts.get("folder_gfx") or []):  # the graphics in your footage folder (foldergfx: a lower third, a title, a see-through video)
            specs.append(("folder", str(g.get("label") or g.get("name")), float(g["len"]), work / f"folder{k}.mov", {"item": g},
                          {"at": float(g["at"]), "len": float(g["len"]), "kind": "folder", "style": g.get("role")}))
        e = opts.get("endcard")                               # the ad's last seconds: logo, name, call to action
        if e and e.get("text"):
            specs.append(("end", str(e["text"]), float(e["len"]), work / "endcard.mov",
                          {"logo": e.get("logo"), "business": e.get("business"), "tagline": e.get("tagline"), "theme": opts.get("gfx_theme")}, {"at": e["at"], "len": e["len"]}))
        if specs:
            engine._progress(job_id, "rendering", f"Animating {len(specs)} graphics", 0.57)
            def card_job(kind, text, secs, f, extra, who):
                zones = []                                    # where the people are while this graphic is up: it is placed (or shrunk, or dropped) to stay clear
                if kind != "end" and cfg.get("clipper.ai_videos.graphics_avoid_faces", True):
                    from . import facezones
                    at = float(who["at"]) if isinstance(who, dict) else 0.0
                    zones = facezones.zones(src, at, secs, engine.models_dir, W / H)
                if kind != "end" and caps:
                    zones = [*zones, (0.0, max(0.0, caps["y"] / H - 0.01), 1.0, 1.0)]                      # never over the captions
                if kind != "end":
                    log.info(f"graphic {kind} at {float(who['at']) if isinstance(who, dict) else 0:.1f}s: keep clear of {[tuple(round(v, 2) for v in z) for z in zones]}")
                if kind == "folder":
                    from . import foldergfx
                    return foldergfx.layer(extra["item"], secs, W, H, float(fps), f, zones), []
                return render_card(cfg, kind, text, secs, f, extra, W, H, (extra or {}).get("spec"), zones)
            with ThreadPoolExecutor(max_workers=2) as ex:
                futs = {ex.submit(card_job, kind, text, secs, f, extra, who): (kind, who, f, secs)
                        for kind, text, secs, f, extra, who in specs}
                for fu in as_completed(futs):
                    kind, who, f, secs = futs[fu]
                    mov, rects = fu.result()
                    if not mov:
                        continue
                    if kind == "title":
                        gfx = mov
                        if rects:
                            glass_items.append({"at": 0.0, "len": 3.2, "rects": rects})
                        title_clips["hook"] = {"rect": _rect_of(rects)}
                        cards_done.append({"at": 0.0, "len": 3.2, "src": Path(mov), "kind": "hook", "style": "hook"})      # (the exact look, for Final Cut)
                    else:
                        pts.append((who, mov))
                        if rects:
                            glass_items.append({"at": float(who["at"]), "len": float(secs), "rects": rects})
                        if kind == "point":
                            title_clips[round(float(who["at"]), 2)] = {"rect": _rect_of(rects)}
                            cards_done.append({"at": float(who["at"]), "len": float(secs), "src": Path(mov), "kind": "point", "style": "point"})
                        if kind in ("card", "folder"):
                            cards_done.append({"at": float(who["at"]), "len": float(secs), "src": Path(mov), "kind": who.get("kind"), "style": who.get("style")})
        bug = opts.get("logo_bug")                            # your logo small in a corner (only when your notes ask for it)
        if bug and bug.get("logo") and Path(bug["logo"]).is_file() and float(bug.get("len") or 0) > 1.0:
            try:
                from . import foldergfx
                bug["frame"] = str(foldergfx.bug_frame(bug["logo"], W, H, work / "logo_bug.png"))
                mov = foldergfx.still_clip(Path(bug["frame"]), float(bug["len"]), float(fps), work / "logo_bug.mov")
                if mov:                                       # (Final Cut gets it as a clip of its own above the picture)
                    cards_done.append({"at": float(bug["at"]), "len": float(bug["len"]), "src": mov, "kind": "folder", "style": "logo"})
            except Exception as e:  # noqa: BLE001
                log.warning(f"the corner logo could not be drawn: {e}")
                bug = None
        else:
            bug = None

    # 3. music from what is said
    track, vibe = None, {}
    if opts.get("music_file"):                                # a track already chosen (your own song, or one picked to cut to the beat)
        track = MU.resolve(cfg, opts["music_file"])
        vibe = opts.get("music_vibe") or {"mood": "", "genre": "", "why": "chosen", "by": "you"}
    elif opts.get("music"):
        engine._progress(job_id, "rendering", "Choosing music from what's said", 0.58)
        if opts["music"] == "auto":                           # (only then is an AI asked anything)
            router, _where = engine.router_for_now()
            vibe = MU.choose_vibe(router, " ".join(w["text"] for w in words), MU.genres(cfg),
                                  title + (f" (the user's brief: {opts['prompt']})" if opts.get("prompt") else ""))
        else:
            vibe = {"mood": opts["music"], "genre": "", "why": "chosen by you", "by": "you"}
        track = MU.pick(cfg, mood=vibe["mood"], genre=vibe.get("genre", ""), seed=job_id)

    music_start = 0.0
    music_src, music_ss, music_xf = None, 0.0, 0.0
    if track:
        music_start = float(opts["music_start"]) if opts.get("music_start") is not None else MU.start_offset(track, out_len, job_id)
        pieces = MU.loop_plan(ffmpeg.duration(track.path), music_start, out_len, MU.LOOP_XF)
        if len(pieces) > 1:                                   # a song shorter than the video: it starts again, crossfaded (no jump)
            music_src, music_xf = work / "music_loop.wav", MU.LOOP_XF
            MU.looped(track.path, pieces, MU.LOOP_XF, music_src)
        else:
            music_src, music_ss = track.path, music_start

    # 4. render: select the kept stretches, captions on top, music under
    engine._progress(job_id, "rendering", f"Rendering the full video ({out_len / 60:.0f} min)", 0.6)
    expr = select_expr(intervals)
    inputs = ["-i", str(src)]
    reordered = any(intervals[i][1] > intervals[i + 1][0] + 0.01 for i in range(len(intervals) - 1))
    whole = len(intervals) == 1 and intervals[0][0] <= 0.05 and abs(intervals[0][1] - total) <= 0.6
    if reordered:                                             # parts out of source order: cut each out and join them in the new order
        g = []
        for i, (a, b) in enumerate(intervals):
            g.append(f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[v{i}]")
        g.append("".join(f"[v{i}]" for i in range(len(intervals))) + f"concat=n={len(intervals)}:v=1:a=0[jv]")
        g.append(f"[jv]fps={float(fps):.5f},scale={W}:{H},setsar=1,format=yuv420p[base]")
    elif whole:                                               # nothing is cut out: the select / aselect filters would only get in the way
        g = [f"[0:v]fps={float(fps):.5f},scale={W}:{H},setsar=1,format=yuv420p[base]"]
    else:
        g = [f"[0:v]fps={float(fps):.5f},select='{expr}',setpts=N/({float(fps):.5f}*TB),"
             f"scale={W}:{H},setsar=1,format=yuv420p[base]"]
    cur = "base"
    for i, c in enumerate(cutaways):                          # B-roll cutaways: picture swaps, the speaker's voice keeps playing
        t0, ln, a = c["at"], c["length"], c["src"]
        from .broll import fill_filter
        g.append(f"[0:v]trim=start={a:.3f}:end={a + ln:.3f},setpts=PTS-STARTPTS+{t0:.3f}/TB,fps={float(fps):.5f},"
                 f"{fill_filter(src, a, W, H)},setsar=1,format=yuv420p[cw{i}]")
        g.append(f"[{cur}][cw{i}]overlay=0:0:enable='between(t,{t0:.3f},{t0 + ln:.3f})':eof_action=pass[cv{i}]")
        cur = f"cv{i}"
    if glass_items and not opts.get("no_glass"):              # liquid glass: the picture behind each panel is blurred, in the panel's shape
        budget = 8 if W >= 3000 else 22                       # every panel is one more pass over every frame: a 4K video with a dozen cards stalled, so the biggest panels get the glass
        ranked = sorted((r for gi in glass_items for r in gi["rects"]), key=lambda r: -int(r["w"]) * int(r["h"]))[:budget]
        keep = {id(r) for r in ranked}
        glass_items = [{**gi, "rects": [r for r in gi["rects"] if id(r) in keep]} for gi in glass_items]
        gk = 0
        for gi in glass_items:
            for r in gi["rects"]:
                mi = None
                if int(r.get("r") or 0) > 0 and int(r["w"]) < W:
                    inputs += ["-loop", "1", "-t", f"{gi['len'] + 0.6:.2f}", "-i", str(write_mask(work / f"mask{gk}.png", int(r["w"]) // 2 * 2, int(r["h"]) // 2 * 2, int(r["r"])))]
                    mi = inputs.count("-i") - 1
                lines, cur = glass_filters(cur, gk, float(gi["at"]), float(gi["len"]), r, mi, W, H, float(fps))
                g += lines
                gk += 1
    if caps:
        inputs += ["-f", "concat", "-safe", "0", "-i", str(caps["ffconcat"])]
        ci = inputs.count("-i") - 1
        g.append(f"[{ci}:v]format=rgba[cap];[{cur}][cap]overlay=0:{caps['y']}:format=auto:eof_action=repeat,format=yuv420p[wc]")
        cur = "wc"
    items = ([(0.0, 3.2, gfx)] if gfx else []) + [(float(c["at"]), float(c["len"]), mov) for c, mov in pts]
    ov_file = overlay_track(items, out_len, W, H, float(fps), work / "overlay.mov") if items else None
    if ov_file:                                               # ONE transparent track holding every graphic at its moment
        inputs += ["-i", str(ov_file)]
        gi = inputs.count("-i") - 1
        g.append(f"[{gi}:v]format=rgba[ov];[{cur}][ov]overlay=0:0:format=auto:eof_action=pass,format=yuv420p[wt]")
        cur = "wt"
    if not opts.get("no_overlays") and bug and bug.get("frame"):            # the corner logo: one still picture, shown from after your title to the end card
        b0, b1 = float(bug["at"]), float(bug["at"]) + float(bug["len"])
        inputs += ["-loop", "1", "-framerate", f"{float(fps):.3f}", "-t", f"{out_len:.3f}", "-i", bug["frame"]]
        bi = inputs.count("-i") - 1
        g.append(f"[{bi}:v]format=rgba[bug];[{cur}][bug]overlay=0:0:format=auto:enable='between(t,{b0:.3f},{b1:.3f})',format=yuv420p[wb]")
        cur = "wb"
    lufs = float(opts.get("loudness_lufs") or cfg.get("video.loudness_lufs", -14))     # (an ad / promo asks for louder)
    (work / "graph_v.txt").write_text(";\n".join(g))
    out, part = d / "clip_01.mp4", d / "clip_01.part.mp4"
    audio_out = has_audio or bool(track)
    t0 = time.monotonic()
    limit = max(900.0, out_len * 8)
    vfile, afile = work / "picture.mp4", work / "sound.m4a"
    if audio_out:                                             # the SOUND is made on its own (its own ffmpeg runs), the PICTURE on its own, then joined
        engine._progress(job_id, "rendering", "Mixing the sound", 0.6)
        sfx_file = None
        if cfg.get("clipper.sfx.enabled", True):                  # a whoosh as each full-screen card arrives and leaves
            from . import gfxplan, sfx
            sfx_file = sfx.track([c for c in cards_done if c.get("kind") in gfxplan.FULL], out_len, work / "sfx.wav")
        make_sound(src, intervals, reordered, whole, expr, has_audio, music_src, music_ss, out_len, MU.levels(cfg, cap_words, out_len),
                   lufs, work, afile, vcfg, limit, float((cfg.get("clipper.music", {}) or {}).get("min_gap_lu", 16)), sfx=sfx_file)
    vbase = [ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", *inputs, "-/filter_complex", str(work / "graph_v.txt"), "-map", f"[{cur}]", "-an",
             "-t", f"{out_len:.3f}"]
    try:
        ffmpeg.run_watched([*vbase, *ffmpeg.video_encode_args(vcfg), str(vfile)], stall_s=90, timeout=limit, what="the picture")   # a healthy render runs faster than real time
    except ffmpeg.FFmpegError as e:
        if ffmpeg.pick_encoder(vcfg) == "libx264":
            raise
        log.warning(f"the hardware encoder failed ({str(e)[-160:]}); rendering the picture in software")      # busy, or no GUI session
        ffmpeg.run_watched([*vbase, *ffmpeg.video_encode_args(ffmpeg.software_cfg(vcfg)), str(vfile)], stall_s=90, timeout=limit * 2,
                           what="the picture (software)")
    mux = [ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-i", str(vfile)] + (["-i", str(afile)] if audio_out else []) + \
          ["-map", "0:v"] + (["-map", "1:a"] if audio_out else []) + ["-c", "copy", "-metadata", f"title={title}", "-movflags", "+faststart", str(part)]
    ffmpeg.run(mux, timeout=300.0, what="joining picture and sound")
    part.replace(out)
    ffmpeg.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-ss", f"{min(3.0, out_len / 3):.2f}", "-i", str(out),
                "-frames:v", "1", "-vf", "scale=360:-2", "-q:v", "4", str(d / "clip_01.jpg")], what="thumbnail")
    if caps:
        shutil.copy2(caps["srt"], d / "clip_01.srt")
    credits = []
    if track:
        credits.append(track.credit)
    credits += list(opts.get("gfx_credits") or [])
    src_meta = job.get("source") or {}
    if src_meta.get("credit"):
        credits.append(src_meta["credit"])
    (d / "clip_01.txt").write_text("\n\n".join(credits) + ("\n" if credits else ""))

    groups = caption_groups(cap_words, ccfg) if caps else []
    subs = caption_groups(cap_words, ccfg)                               # Final Cut always gets the words as real captions (fcpxml.subs)
    fcp_meta = None
    if opts.get("fcp"):
        engine._progress(job_id, "rendering", "Building the Final Cut Pro project", 0.95)
        fcp_meta = write_bundle(cfg, d, title, src, info, fps, W, H, intervals, groups, track,
                                MU.duck_points(cap_words, out_len, -14.0 if len(cap_words) / max(1, out_len) >= 0.6 else -8.0,
                                               -24.0 if len(cap_words) / max(1, out_len) >= 0.6 else -16.0),
                                music_start if track else 0.0, ccfg, cutaways=cutaways, mc=mc, titles=fcp_titles(opts, title, title_clips), music_xf=music_xf,
                                overlays=cards_done, subs=subs)
    clip = {"index": 1, "file": out.name, "thumb": "clip_01.jpg", "srt": "clip_01.srt" if caps else None,
            "title": title, "hook": "", "caption": (d / "clip_01.txt").read_text().strip(), "score": None,
            "reason": story_plan.summary if story_plan else "whole video",
            "picked_by": "long video (story edit)" if story_plan else "whole video", "source_start": 0.0, "source_end": round(total, 2),
            "duration": round(out_len, 2), "cuts": len(intervals) - 1, "layout": "original", "caption_style": style,
            "pace": pace, "hook_title": False, "profile": "polish", "polish": True,
            "trimmed_s": round(total - out_len, 1), "fcp": fcp_meta,
            "story": ({"summary": story_plan.summary, "files": story_plan.kept_files} if story_plan else None),
            "cutaways": cutaways, "broll": {n: len(s) for n, s in broll_scenes.items()},
            "music": ({"title": track.title, "artist": track.artist, "license": track.license, "mood": vibe.get("mood"),
                       "genre": vibe.get("genre"), "why": vibe.get("why"), "file": track.path.name} if track else None),
            "rendered_at": int(time.time()), "render_s": round(time.monotonic() - t0, 1)}
    store.set_clip(job_id, clip)
    return clip


SANITIZE = "aeval=exprs='if(isnan(val(0))+isinf(val(0)),0,val(0))|if(isnan(val(1))+isinf(val(1)),0,val(1))':c=same"   # a broken sample -> silence


def _voice_graph(intervals, reordered: bool, whole: bool, expr: str) -> list[str]:
    """Filter lines giving [ac]: the kept stretches of the source's sound, in play order."""
    if reordered:                                             # parts out of source order: the same trims as the picture
        return [f"[0:a]atrim=start={x:.3f}:end={y:.3f},asetpts=PTS-STARTPTS[a{i}]" for i, (x, y) in enumerate(intervals)] + \
               ["".join(f"[a{i}]" for i in range(len(intervals))) + f"concat=n={len(intervals)}:v=0:a=1[ac]"]
    return ["[0:a]anull[ac]" if whole else f"[0:a]aselect='{expr}',asetpts=N/SR/TB[ac]"]


def gain_to(path, target: float) -> float:
    """The fixed gain (dB) that brings a recording to `target` LUFS (0 for silence)."""
    i = MU.integrated(path)
    return 0.0 if i is None or i < -70 else max(-30.0, min(30.0, target - i))


def make_sound(src: Path, intervals, reordered: bool, whole: bool, expr: str, has_audio: bool, music_src, music_ss: float, out_len: float,
               music_lufs: float, lufs: float, work: Path, afile: Path, vcfg: dict, limit: float, min_gap_lu: float = 16.0, sfx: Path | None = None) -> None:
    """The finished video's sound, made in measured steps - no dynamic normaliser (it pumped the level up and down over a long video):
      1. the voice: the kept stretches of the source, brought to -16 LUFS by ONE fixed gain (measured first)
      2. the music bed under it, if any: levelled, faded in and out, ducked while someone speaks, then checked to really sit at least
         `min_gap_lu` under the voice (measured, not assumed)
      3. the mix brought to `lufs` by one fixed gain, with a peak limiter for the few peaks. A broken sample (NaN / infinity, which one
         filter can produce) is silenced first - one such sample used to make the encoder stop the whole render."""
    ffb = ffmpeg.bin_path("ffmpeg")

    def run(cmd: list, what: str) -> None:
        ffmpeg.run_watched(cmd, stall_s=90, timeout=limit, what=what)
    voice = None
    if has_audio:
        voice = work / "voice_raw.wav"
        (work / "graph_voice.txt").write_text(";\n".join(_voice_graph(intervals, reordered, whole, expr) + [
            f"[ac]aresample=48000,aformat=sample_fmts=flt:channel_layouts=stereo,{SANITIZE}[vout]"]))
        run([ffb, "-y", "-v", "error", "-i", str(src), "-/filter_complex", str(work / "graph_voice.txt"), "-map", "[vout]", "-vn",
             "-t", f"{out_len:.3f}", "-c:a", "pcm_f32le", str(voice)], "the voice")
    raw = voice
    if music_src is not None:
        bed = (f"aresample=48000,aformat=channel_layouts=stereo,atrim=0:{out_len:.3f},asetpts=PTS-STARTPTS,loudnorm=I={music_lufs}:TP=-3:LRA=11,"
               f"aresample=48000,afade=t=in:st=0:d=0.8,afade=t=out:st={max(0.0, out_len - 1.5):.3f}:d=1.5")
        raw = work / "sound_raw.wav"
        if voice is not None:
            sv, sm = work / "stem_voice.wav", work / "stem_music.wav"
            vg = gain_to(voice, -16.0)
            graph = (f"[0:a]volume={vg:.2f}dB,asplit=2[vstem][sc];[1:a]{bed}[mus];"
                     "[mus][sc]sidechaincompress=threshold=0.02:ratio=8:attack=20:release=400:makeup=1[mstem]")
            run([ffb, "-y", "-v", "error", "-i", str(voice), "-ss", f"{music_ss:.3f}", "-i", str(music_src), "-filter_complex", graph,
                 "-map", "[vstem]", "-t", f"{out_len:.3f}", "-c:a", "pcm_f32le", str(sv),
                 "-map", "[mstem]", "-t", f"{out_len:.3f}", "-c:a", "pcm_f32le", str(sm)], "the voice and the music")
            gap = MU.meter_gap(sv, sm)                        # how far under the voice the music really sits (LU), measured not assumed
            adj = min(0.0, gap - (min_gap_lu + 2.0)) if gap is not None else 0.0
            log.info(f"music sits {gap if gap is None else round(gap, 1)} LU under the voice; adjusting by {adj:.1f} dB")
            run([ffb, "-y", "-v", "error", "-i", str(sv), "-i", str(sm), "-filter_complex",
                 f"[1:a]volume={adj:.2f}dB[m];[0:a][m]amix=inputs=2:duration=first:normalize=0[mix]", "-map", "[mix]",
                 "-t", f"{out_len:.3f}", "-c:a", "pcm_f32le", str(raw)], "mixing the voice and the music")
        else:
            run([ffb, "-y", "-v", "error", "-ss", f"{music_ss:.3f}", "-i", str(music_src), "-filter_complex", f"[0:a]{bed}[mus]",
                 "-map", "[mus]", "-t", f"{out_len:.3f}", "-c:a", "pcm_f32le", str(raw)], "the music")
    gain = gain_to(raw, lufs)
    chain = f"{SANITIZE},volume={gain:.2f}dB,alimiter=limit={10 ** (-1.5 / 20):.4f}:attack=5:release=50:level=disabled,{SANITIZE}"
    if sfx is not None and Path(sfx).exists():                    # the whooshes of the full-screen cards go on top of the finished mix (quiet: they sit under the voice)
        run([ffb, "-y", "-v", "error", "-i", str(raw), "-i", str(sfx), "-filter_complex",
             f"[0:a]{chain}[m];[1:a]aresample=48000,aformat=channel_layouts=stereo[s];[m][s]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.89:level=disabled[o]",
             "-map", "[o]", "-t", f"{out_len:.3f}", *ffmpeg.audio_encode_args(vcfg), str(afile)], "the sound")
        return
    run([ffb, "-y", "-v", "error", "-i", str(raw), "-af", chain,
         "-t", f"{out_len:.3f}", *ffmpeg.audio_encode_args(vcfg), str(afile)], "the sound")


def _rect_of(rects) -> tuple | None:
    """(x, y, w, h) of the panel a lower third was drawn in, from what render_card returned."""
    pn = getattr(rects, "panel", None) or (rects[0] if rects else None)
    return (pn["x"], pn["y"], pn["w"], pn["h"]) if pn else None


class _PanelRects(list):
    """The glass rectangles of a card, and (.panel) the rectangle of its panel even when the look has no glass."""
    panel: dict | None = None


def render_card(cfg, kind: str, text: str, secs: float, f: Path, extra: dict | None, W: int, H: int, spec: dict | None = None,
                avoid: list | None = None) -> tuple[Path | None, list]:
    """One animated graphic -> (a transparent .mov, the glass rectangles to blur behind it). The native renderer (gfxcards: no browser, any machine)
    draws everything but the ad's end card; Hyperframes (if installed) is the fallback and still makes the end card."""
    from . import gfxcards, graphics
    theme = (extra or {}).get("theme")
    if spec and spec.get("kind") == "animation":                  # a custom diagram drawn for this moment (HTML/SVG): slow, so it is its own path
        return render_animation(cfg, spec, secs, f, W, H, theme), []
    if kind != "end" and cfg.get("clipper.ai_videos.graphics_engine", "native") == "native":
        sp = {**(spec or {"kind": "hook" if kind == "title" else "point", "text": text}), "len": secs}
        r = gfxcards.render(cfg, sp, W, H, f, theme, avoid=avoid)
        if r:
            glass = r["glass"]
            if r.get("panel"):                                    # (where a lower third really sits: Final Cut's editable title goes there)
                px, py, pw, ph = r["panel"]
                glass = _PanelRects(glass)
                glass.panel = {"x": px, "y": py, "w": pw, "h": ph}
            return r["file"], glass
    if kind == "card":
        return None, []
    made = graphics.render_title(cfg, text, secs, W, H, f, 300.0, kind, extra)
    if made is None and kind == "end":                            # no Hyperframes on this Mac: the end card is drawn here instead
        from . import foldergfx
        ex = extra or {}
        made = foldergfx.end_overlay(ex.get("logo"), ex.get("business") or text, ex.get("tagline") or "", secs, W, H, f)
    return made, []


def render_animation(cfg, spec: dict, secs: float, f: Path, W: int, H: int, theme: str | None) -> Path | None:
    """A topic-specific animated diagram for a talking-head video: the AI draws it from the speaker's own words (explainer.animate: HTML/SVG rendered by Hyperframes,
    or the building blocks when that cannot be made), then it is laid out as a full-screen overlay with a short fade in and out. None = the video goes without it."""
    import subprocess

    from . import explainer as EX
    from .localai import LocalAI
    look = (theme if theme in ("clean", "blueprint", "sky", "slate", "mint", "board") else "clean", {"blueprint": "grid", "sky": "blocks", "slate": "stripe", "mint": "rule"}.get(theme or "", "dots"))
    try:
        r = EX.animate(cfg, LocalAI(cfg), str(cfg.get("explainer.animation_tier", "svg")), spec["concept"], spec.get("narration", ""), spec.get("timed") or [], secs, look,
                       f.parent / f"{f.stem}_anim", spec.get("context", ""), say=log.info)
    except Exception as e:  # noqa: BLE001 - a diagram is never worth losing a video over
        log.warning(f"animation failed: {e}")
        return None
    if not r:
        return None
    fade = 0.35
    cmd = [ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-i", str(r["file"]), "-t", f"{secs:.3f}", "-vf",
           f"scale={W}:{H},fps=30,format=argb,fade=t=in:st=0:d={fade}:alpha=1,fade=t=out:st={max(0.0, secs - fade):.3f}:d={fade}:alpha=1", "-an", "-c:v", "qtrle", "-pix_fmt", "argb", str(f)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not f.exists():
        log.warning(f"animation could not be laid out: {p.stderr[-300:]}")
        return None
    return f


def glass_filters(cur: str, k: int, at: float, ln: float, r: dict, mask_idx: int | None, W: int, H: int, fps: float) -> tuple[list[str], str]:
    """The picture behind a glass panel, blurred and brightened (or, for a full-frame rectangle, blurred and dimmed), in the panel's rounded shape and
    only while the graphic is on screen. -> (filter lines, the new label of the picture). mask_idx = the input holding the rounded-corner mask."""
    x, y = max(0, int(r["x"])), max(0, int(r["y"]))
    w, h = min(W - x, int(r["w"])) // 2 * 2, min(H - y, int(r["h"])) // 2 * 2
    if w < 8 or h < 8:
        return [], cur
    t0, t1 = at + float(r.get("t0") or 0.0), at + ln                              # (a tile that pops in late gets its glass late too)
    dim = float(r.get("dim") or 0.0)
    light = r.get("tone") == "light"                                          # a light look frosts the picture bright; a dark look tints it dark
    if dim:
        tone = f"eq=brightness={dim * 0.45:.3f}:saturation=0.8" if light else f"eq=brightness={-dim * 0.7:.3f}:saturation=0.85"
    else:
        tone = "eq=brightness=0.13:saturation=1.1" if light else "eq=brightness=0.05:saturation=1.3"
    f = 4 if W >= 3000 else (2 if W >= 1700 else 1)                              # a blur is soft anyway: blur a small copy and scale it back (a 4K full-frame blur stalled the render)
    shrink = f"scale={w // f}:{h // f}:flags=area," if f > 1 else ""
    grow = f",scale={w}:{h}:flags=bilinear" if f > 1 else ""
    out = [f"[{cur}]split=2[ga{k}][gb{k}]",
           f"[gb{k}]trim=start={t0:.3f}:end={t1:.3f},setpts=PTS-STARTPTS+{t0:.3f}/TB,crop={w}:{h}:{x}:{y},{shrink}gblur=sigma={(22 if not dim else 30) / f:.1f}:steps=2,{tone}{grow},format=rgba[gc{k}]"]
    src = f"gc{k}"
    if mask_idx is not None:
        out.append(f"[{mask_idx}:v]format=gray,scale={w}:{h},fps={fps:.3f},setpts=PTS-STARTPTS+{t0:.3f}/TB[gm{k}]")
        out.append(f"[gc{k}][gm{k}]alphamerge[gd{k}]")
        src = f"gd{k}"
    out.append(f"[{src}]fade=t=in:st={t0:.3f}:d=0.45:alpha=1,fade=t=out:st={max(t0, t1 - 0.35):.3f}:d=0.35:alpha=1[ge{k}]")
    out.append(f"[ga{k}][ge{k}]overlay={x}:{y}:enable='between(t,{t0:.3f},{t1:.3f})':eof_action=pass[gf{k}]")
    return out, f"gf{k}"


def write_mask(path: Path, w: int, h: int, radius: int) -> Path:
    from PIL import Image, ImageDraw
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).rounded_rectangle((0, 0, w - 1, h - 1), radius=max(1, radius), fill=255)
    img.save(path)
    return path


def overlay_track(items: list[tuple[float, float, Path]], total: float, W: int, H: int, fps: float, out: Path) -> Path | None:
    """Join every graphic (hook card, key points, end card), each at its moment, into ONE transparent video as long as the programme.
    The final render then needs a single overlay input - separate timed inputs make ffmpeg stall once one has ended and the next has
    not begun."""
    items = sorted((a, d, p) for a, d, p in items if p and Path(p).exists())
    if not items:
        return None
    cmd = [ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error"]
    labels, parts, cursor, n = [], [], 0.0, 0

    def add_gap(seconds: float):
        nonlocal n
        cmd.extend(["-f", "lavfi", "-i", f"color=c=black@0.0:s={W}x{H}:r={fps:.3f}:d={seconds:.3f},format=rgba"])      # rgba AT the source: it keeps the transparency
        parts.append(f"[{n}:v]format=rgba,setsar=1,setpts=PTS-STARTPTS[v{n}]")
        labels.append(f"[v{n}]")
        n += 1
    for at, ln, mov in items:
        if at < cursor - 0.02:
            continue                                                 # overlapping graphics: the later one is dropped
        if at - cursor > 0.04:
            add_gap(at - cursor)
        cmd.extend(["-t", f"{ln:.3f}", "-i", str(mov)])
        parts.append(f"[{n}:v]fps={fps:.3f},scale={W}:{H},format=rgba,setsar=1,trim=duration={ln:.3f},setpts=PTS-STARTPTS[v{n}]")
        labels.append(f"[v{n}]")
        n += 1
        cursor = at + ln
    if total - cursor > 0.04:
        add_gap(total - cursor)
    fc = ";".join(parts) + ";" + "".join(labels) + f"concat=n={len(labels)}:v=1:a=0[ov]"
    cmd += ["-filter_complex", fc, "-map", "[ov]", "-c:v", "qtrle", "-pix_fmt", "argb", "-r", f"{fps:.3f}", str(out)]
    try:
        ffmpeg.run(cmd, timeout=900.0, what="joining the graphics into one track")
    except ffmpeg.FFmpegError as e:
        log.warning(f"the graphics track couldn't be built ({str(e)[:200]}); making the video without graphics")
        return None
    return out


def fcp_titles(opts: dict, title: str, clips: dict | None = None) -> list[dict]:
    """The graphics of the finished video as Final Cut titles (so they can be edited there): hook card, key points, end card.
    `clips`: the ones that were also rendered as a clip (their exact look goes to Final Cut as that clip: the editable title is then switched
    off, and placed on the panel the renderer drew)."""
    clips = clips or {}
    theme = opts.get("gfx_theme")
    out: list[dict] = []
    if opts.get("graphics") in ("title", "all") and (opts.get("hook_text") or title) and not opts.get("folder_title"):
        c = clips.get("hook")
        out.append({"at": 0.0, "len": 3.2, "text": str(opts.get("hook_text") or title), "kind": "hook", "theme": theme,
                    "rect": c.get("rect") if c else None, "off": bool(c)})
    for c in opts.get("callouts") or []:
        k = clips.get(round(float(c["at"]), 2))
        out.append({"at": float(c["at"]), "len": float(c["len"]), "text": str(c["text"]), "kind": "point", "theme": theme,
                    "rect": k.get("rect") if k else None, "off": bool(k)})
    e = opts.get("endcard")
    if e and e.get("text"):
        out.append({"at": float(e["at"]), "len": float(e["len"]), "text": str(e["text"]), "kind": "end"})
    return out


def write_bundle(cfg, d: Path, title: str, src: Path, info: dict, fps: Fraction, W: int, H: int,
                 intervals, groups, track, duck, music_start: float, ccfg: dict, dest_root: Path | None = None,
                 cutaways: list | None = None, mc: dict | None = None, titles: list | None = None, music_xf: float = 0.0,
                 overlays: list | None = None, subs: list | None = None) -> dict:
    """Write the FCP project folder into the job (fcp/) and zip it; dest_root = where it will live for the
    paths inside (default clipper.fcp_unpack_dir, i.e. where you unzip it on your Mac)."""
    name = safe(title, 60)
    bundle = d / "fcp" / name
    shutil.rmtree(d / "fcp", ignore_errors=True)
    media = None
    if mc:                                                  # built from the ORIGINAL files (multicam or not): nothing is copied
        bundle.mkdir(parents=True)
    else:
        (bundle / "Media").mkdir(parents=True)
        media = bundle / "Media" / f"source{src.suffix.lower()}"
        try:
            os.link(src, media)                             # same disk: no extra copy
        except OSError:
            shutil.copy2(src, media)
    music_file = None
    if track:
        (bundle / "Music").mkdir()
        music_file = bundle / "Music" / track.path.name
        shutil.copy2(track.path, music_file)
        (bundle / "Music" / (track.path.stem + " - LICENSE.txt")).write_text(
            f"{track.title} - {track.artist}\nLicense: {track.license}\nCredit line (put in your description):\n{track.credit}\n")
    ovs = []
    if overlays:                                            # the slides / charts / quotes: transparent clips, laid over the picture in Final Cut
        (bundle / "Graphics").mkdir()
        for k, o in enumerate(overlays):
            if o.get("src") and Path(o["src"]).exists():
                shutil.copy2(o["src"], bundle / "Graphics" / f"graphic_{k + 1:02d}.mov")
                ovs.append({"at": o["at"], "len": o["len"], "name": f"graphic_{k + 1:02d}.mov"})
    if groups or subs:
        (bundle / "Captions").mkdir()
        if groups and (d / "clip_01.srt").exists():
            shutil.copy2(d / "clip_01.srt", bundle / "Captions" / f"{name}.srt")
        else:                                                           # (captions off in the video: Final Cut still gets them)
            (bundle / "Captions" / f"{name}.srt").write_text(FX.srt(subs))
    root = Path(dest_root) if dest_root else Path(cfg.get("clipper.fcp_unpack_dir", "~/Movies/Clipper")).expanduser()
    meta = write_fcpxml(bundle, name, root / name, info, fps, W, H, intervals, groups, track, duck, music_start, ccfg,
                        media.name if media else None, music_file.name if music_file else None, cutaways, mc, titles, music_xf, ovs, subs=subs)
    zpath = d / "fcp_project.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for f in sorted(bundle.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(bundle.parent).as_posix())
    meta["zip"] = zpath.name
    meta["unpack_to"] = str(root / name)
    return meta


def write_fcpxml(bundle: Path, name: str, lives_at: Path, info: dict, fps: Fraction, W: int, H: int, intervals,
                 groups, track, duck, music_start: float, ccfg: dict, media_name: str | None, music_name: str | None,
                 cutaways: list | None = None, mc: dict | None = None, titles: list | None = None, music_xf: float = 0.0,
                 ovs: list | None = None, subs: list | None = None) -> dict:
    vs = ffmpeg.video_stream(info) or {}
    au = ffmpeg.audio_stream(info) or {}
    src_media = None if mc else FX.Media(str(lives_at / "Media" / media_name), float(info.get("format", {}).get("duration") or 0),
                                         has_audio=bool(au), width=W, height=H, fps=_fps(vs),
                                         audio_rate=int(au.get("sample_rate") or 48000), audio_channels=int(au.get("channels") or 2))
    bed = None
    if track and music_name:
        bed = FX.MusicBed(str(lives_at / "Music" / music_name), ffmpeg.duration(track.path), track.title,
                          start=music_start, keyframes=duck, crossfade=music_xf)
    style = FX.CaptionStyle(size=float(ccfg.get("font_size", 80)), uppercase=bool(ccfg.get("uppercase", True)),
                            y=-(float(ccfg.get("center_y", 0.72)) - 0.5))
    overlays = [{"at": o["at"], "len": o["len"], "path": str(lives_at / "Graphics" / o["name"]), "w": W, "h": H, "fps": float(fps)} for o in (ovs or [])]
    if mc:
        xml = FX.build_multicam(name, W, H, fps, mc["group"], mc["items"], captions=groups, music=bed, style=style, cutaways=mc.get("cutaways"),
                                titles=titles, overlays=overlays, subs=subs)
        n_clips = len(mc["items"])
    else:
        xml = FX.build(name, src_media, intervals, W, H, fps, captions=groups, music=bed, style=style,
                       cutaways=[(c["at"], c["src"], c["length"]) for c in (cutaways or [])], titles=titles, overlays=overlays, subs=subs)
        n_clips = len(intervals)
    path = bundle / f"{name}.fcpxml"
    path.write_text(xml)
    errors = FX.validate(path)
    if errors:
        log.warning(f"FCPXML validation: {errors[:3]}")
    return {"folder": bundle.name, "fcpxml": path.name, "clips": n_clips, "captions": len(groups), "subs": len(FX.caption_spans(subs)),
            "multicam": bool(mc and mc.get("group")),
            "music": bool(bed), "valid": not errors, "errors": errors[:5]}


def archive_bundle(cfg, store, job_id: str) -> None:
    """Copy the FCP project into the job's archive folder on the drive, re-pointed at that location."""
    from urllib.parse import quote
    job = store.get(job_id)
    clip = (job.get("clips") or [{}])[0]
    meta = clip.get("fcp") or {}
    d = store.dir(job_id)
    src_bundle = d / "fcp" / meta.get("folder", "") if meta.get("folder") else None
    if not job.get("library") or not src_bundle or not src_bundle.is_dir():
        return
    from .library import drive_ready
    if not drive_ready(Path(job["library"]).parent):
        log.warning("archive drive not mounted; the Final Cut project copy will be made later")
        return
    dest = Path(job["library"]) / "Final Cut Project"
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(src_bundle, dest, copy_function=shutil.copy2)
    xml_file = dest / meta.get("fcpxml", "")
    if xml_file.is_file() and meta.get("unpack_to"):
        xml_file.write_text(xml_file.read_text().replace(quote(meta["unpack_to"]), quote(str(dest))))
