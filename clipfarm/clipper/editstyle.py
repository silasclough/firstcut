"""Edit style for My Clips: what is added on top of a finished reel.

    standard   nothing (today's reel, untouched)
    youtube    every kind of animated graphic - stats, quotes, slides, charts, flags, company logos, photos - in a bold look matched to the footage
    business   calm: slides, charts, stats, in a light or dark look matched to the footage

It is a pass over the FINISHED clip: the graphics are planned from what is said in the clip (and fact-checked against it), drawn by the same native
engine as the AI Videos page, kept off every face, and composited onto the video; the sound is copied untouched.
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

from ..core import ffmpeg
from ..core.log import get_logger

log = get_logger("clipper.editstyle")
STYLES = ("standard", "auto", "youtube", "business")
PROFILE_OF = {"youtube": "promo", "business": "business"}        # (a reel is short: the promo profile's budget, with the YouTube look)


SCHEMA = {"type": "object", "properties": {"style": {"type": "string", "enum": ["none", "business", "youtube"]}, "why": {"type": "string"},
                                           "caption_style": {"type": "string", "enum": ["karaoke", "phrase"]}}, "required": ["style"]}
SYSTEM = (
    "You decide whether a short vertical clip is INFORMATIONAL. Informational = the speaker is teaching or explaining something: facts, numbers, names of companies, people or places, "
    "history, science, news, business, how-tos, a list of points, an explanation of why something happened. If it is informational, answer 'business' (calm: podcasts, interviews, "
    "explainers, history, documentary-style talk) or 'youtube' (energetic: fast tips, tech and news commentary, hot takes) and it gets graphics. "
    "If it is NOT informational - streamer or gameplay footage with banter, reactions, comedy, pranks, a personal or emotional story, vibes, a moment where the person or the "
    "picture is the whole point - answer 'none'. Judge by what is SAID, not by the genre of the channel. When unsure, 'none'. "
    "Also answer caption_style: karaoke (one to three words at a time) for educational or step-by-step talk; phrase (three to five words) for stories, emotion, comedy and interviews.")


def decide_full(router, source: dict | None, title: str, transcript: str) -> dict:
    """-> {"style": "none" | "business" | "youtube", "why": ..., "caption_style": "karaoke" | "phrase" | None}. Streamer-kind sources never get graphics (no AI asked) and are always
    phrase captions; everything else: is it informational, and does it want karaoke or phrase captions?"""
    s = source or {}
    meta = " ".join(str(s.get(k) or "") for k in ("title", "topic", "channel", "uploader"))
    if s.get("kind") in ("streamer", "twitch"):
        return {"style": "none", "why": "streamer footage: no graphics", "caption_style": "phrase"}
    try:
        res = router.complete_json("edit_style", SYSTEM, f"Source video: {meta}\nClip title: {title}\nWhat is said in the clip:\n{transcript[:1500]}", SCHEMA).data
        cs = res.get("caption_style") if res.get("caption_style") in ("karaoke", "phrase") else None
        return {"style": res.get("style") if res.get("style") in ("none", "business", "youtube") else "none", "why": str(res.get("why") or "")[:160], "caption_style": cs}
    except Exception as e:  # noqa: BLE001 - not sure = no graphics
        log.warning(f"edit style: could not decide ({e}); no graphics")
        return {"style": "none", "why": "could not decide", "caption_style": None}


def decide(router, source: dict | None, title: str, transcript: str) -> tuple[str, str]:
    """For the 'auto' edit style: -> ("none" | "business" | "youtube", why). Streamer-kind sources never get graphics (no AI asked); everything else: is it informational?."""
    r = decide_full(router, source, title, transcript)
    return r["style"], r["why"]


def caption_mood(router, source: dict | None, title: str, transcript: str) -> str | None:
    """'karaoke' | 'phrase' | None (the AI did not say) for the clip's captions: the same question `decide` asks, kept for the caption look."""
    return decide_full(router, source, title, transcript).get("caption_style")


def apply(engine, job_id: str, out: Path, cap_words: list[dict], out_len: float, style: str, W: int, H: int, fps: float,
          hook_title: bool = True, source: dict | None = None, title: str = "", cap_y: float | None = None, hook_end: float | None = None) -> dict | None:
    """Add the style's graphics to the finished clip `out` in place. -> {"cards": n, "theme": ..., "credits": [...]} or None if nothing was added."""
    if style == "auto" and out_len >= 12 and cap_words:
        style, why = decide(engine.router_for_now()[0], source, title, " ".join(w["text"] for w in cap_words))
        log.info(f"edit style auto for {job_id}: {style} ({why})")
        if style == "none":
            return {"cards": 0, "style": "none", "why": why}
    if style not in PROFILE_OF or out_len < 12 or not cap_words:
        return None
    from . import facezones, gfxplan, graphics, localproj, polish
    cfg = engine.cfg
    try:
        router, _where = engine.router_for_now()
        prof = dict(localproj.PROFILES[PROFILE_OF[style]])
        if style == "business":
            prof["kinds"] = [k for k in prof["kinds"] if k != "hub"]
        cards = gfxplan.plan(router, localproj.seg(cap_words), "", prof, out_len, hook_end=(float(hook_end) if hook_end is not None else 2.8 if hook_title else 0.0), words=cap_words, cfg=cfg)
    except Exception as e:  # noqa: BLE001 - a style never stops a reel from being made
        log.warning(f"edit style {style}: planning failed ({e}); keeping the plain reel")
        return None
    if not cards:
        return None
    luma = graphics.video_luma(str(out), out_len)
    theme = graphics.match_theme(luma, "youtube" if style == "youtube" else "business", job_id)
    work = out.parent / f"{out.stem}_style"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    items, glass, sfx_events = [], [], []
    for k, c in enumerate(cards):
        zones = facezones.zones(out, float(c["at"]), float(c["len"]), engine.models_dir, W / H) if cfg.get("clipper.ai_videos.graphics_avoid_faces", True) else []
        if cap_y:
            zones = [*zones, (0.0, max(0.0, cap_y - 0.06), 1.0, min(1.0, cap_y + 0.10))]                 # the captions: no graphic goes over them
        ex = {"theme": theme, "spec": c}
        if cap_y:
            ex["caption_top"] = max(0.0, cap_y - 0.06) * H                                                  # (a lower third sits just above the captions)
        if cfg.get("clipper.graphics.safe_zones", True):
            ex["safe"] = True
        if cfg.get("clipper.ai_videos.graphics_avoid_faces", True) and cfg.get("clipper.graphics.timed_zones", True):
            try:
                ex["timed"] = facezones.timed_zones(out, float(c["at"]), float(c["len"]), engine.models_dir, W, H)
            except Exception:  # noqa: BLE001
                pass
        mov, rects = polish.render_card(cfg, "card", str(c.get("title") or c["kind"]), float(c["len"]), work / f"card{k}.mov", ex, W, H, c, zones)
        if not mov:
            continue
        items.append((float(c["at"]), float(c["len"]), Path(mov)))
        if c["kind"] in gfxplan.FULL:
            sfx_events.append({"at": float(c["at"]), "len": float(c["len"]), "style": c.get("style")})
        if rects:
            glass.append({"at": float(c["at"]), "len": float(c["len"]), "rects": rects})
    track = polish.overlay_track(items, out_len, W, H, fps, work / "overlay.mov") if items else None
    if not track:
        shutil.rmtree(work, ignore_errors=True)
        return None
    inputs, g, cur, gk = ["-i", str(out)], [], "0:v", 0
    g.append("[0:v]format=yuv420p[base]")
    cur = "base"
    for gi in glass:
        for r in gi["rects"]:
            mi = None
            if int(r.get("r") or 0) > 0 and int(r["w"]) < W:
                inputs += ["-loop", "1", "-t", f"{gi['len'] + 0.6:.2f}", "-i", str(polish.write_mask(work / f"mask{gk}.png", int(r["w"]) // 2 * 2, int(r["h"]) // 2 * 2, int(r["r"])))]
                mi = inputs.count("-i") - 1
            lines, cur = polish.glass_filters(cur, gk, float(gi["at"]), float(gi["len"]), r, mi, W, H, fps)
            g += lines
            gk += 1
    inputs += ["-i", str(track)]
    oi = inputs.count("-i") - 1
    g.append(f"[{oi}:v]format=rgba[ov];[{cur}][ov]overlay=0:0:format=auto:eof_action=pass,format=yuv420p[final]")
    (work / "graph.txt").write_text(";\n".join(g))
    part = out.with_name(out.stem + ".style.mp4")
    vcfg = dict(cfg.get("video", {}) or {})
    sfx_file = None
    if sfx_events and cfg.get("clipper.sfx.enabled", True) and ffmpeg.audio_stream(ffmpeg.probe(out)) is not None:
        from . import sfx
        sfx_file = sfx.track(sfx_events, out_len, work / "sfx.wav")
    if sfx_file:                                                                         # the whooshes of the full-screen cards, quietly under the voice
        inputs += ["-i", str(sfx_file)]
        si = inputs.count("-i") - 1
        g.append(f"[0:a][{si}:a]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.89:level=disabled[aout]")
        (work / "graph.txt").write_text(";\n".join(g))
    amap = ["-map", "[aout]", *ffmpeg.audio_encode_args(vcfg)] if sfx_file else ["-map", "0:a?", "-c:a", "copy"]

    def build(vc: dict) -> list[str]:
        return [ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", *inputs, "-/filter_complex", str(work / "graph.txt"), "-map", "[final]", *amap,
                *ffmpeg.video_encode_args(vc), "-movflags", "+faststart", "-t", f"{out_len:.3f}", str(part)]
    cmd = build(vcfg)
    try:
        t0 = time.monotonic()
        try:
            ffmpeg.run(cmd, timeout=1200.0, what="edit style graphics")
        except ffmpeg.FFmpegError:
            ffmpeg.run(build(ffmpeg.software_cfg(vcfg)), timeout=1800.0, what="edit style graphics (software)")
        if not ffmpeg.has_video(part):
            raise RuntimeError("empty result")
        part.replace(out)
        log.info(f"edit style {style}: {len(items)} graphics in {time.monotonic() - t0:.0f}s ({theme})")
    except Exception as e:  # noqa: BLE001
        log.warning(f"edit style {style}: compositing failed ({str(e)[-200:]}); keeping the plain reel")
        part.unlink(missing_ok=True)
        return None
    finally:
        shutil.rmtree(work, ignore_errors=True)
    credits = sorted({i["credit"] for c in cards for i in (c.get("items") or []) if i.get("credit")})
    return {"cards": len(items), "style": style, "theme": theme, "kinds": [c["kind"] for c in cards], "credits": credits}
