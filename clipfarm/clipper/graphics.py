"""Animated graphics with Hyperframes (HeyGen, Apache-2.0): HTML + GSAP rendered to video, here as a transparent title card
that pops in over the first seconds of an AI video. Runs the locally installed CLI (clipper.ai_videos.hyperframes_dir,
default ~/clipfarm/tools/ai-video; install: scripts/install_ai_video.sh). Anything that goes wrong returns None and the video
is made without it - graphics are never worth losing a video over.
"""

from __future__ import annotations

import html
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..core.log import get_logger

log = get_logger("clipper.graphics")
DEFAULT_DIR = "~/clipfarm/tools/ai-video"
GSAP = "https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"


def cli(cfg=None) -> Path | None:
    base = Path(str((cfg.get("clipper.ai_videos.hyperframes_dir") if cfg else None) or DEFAULT_DIR)).expanduser()
    exe = base / "node_modules" / ".bin" / "hyperframes"
    return exe if exe.exists() else None


THEMES: dict[str, dict] = {
    # every video keeps ONE theme (chosen from its name, or when you ask for different graphics); different videos get different looks
    "clean": {"font": "Inter,ui-sans-serif,system-ui,sans-serif", "weight": 800, "case": "none", "track": "-0.02em", "radius": 0.36, "accent": "#ffd23f",
              "hook_bg": "#ffffff", "hook_fg": "#111111", "pt_bg": "rgba(14,16,22,.88)", "pt_fg": "#ffffff", "bar": True, "enter": "slide", "shadow": True},
    "bold": {"font": "'Arial Black','Helvetica Neue',Impact,sans-serif", "weight": 900, "case": "uppercase", "track": "0.01em", "radius": 0.06, "accent": "#ff3b30",
             "hook_bg": "#ff3b30", "hook_fg": "#ffffff", "pt_bg": "#111111", "pt_fg": "#ffffff", "bar": True, "enter": "wipe", "shadow": True},
    "minimal": {"font": "'Helvetica Neue',Helvetica,Arial,sans-serif", "weight": 600, "case": "none", "track": "0.01em", "radius": 0.0, "accent": "#ffffff",
                "hook_bg": "transparent", "hook_fg": "#ffffff", "pt_bg": "transparent", "pt_fg": "#ffffff", "bar": "under", "enter": "rise", "shadow": False},
    "glass": {"font": "-apple-system,'SF Pro Display','Helvetica Neue',sans-serif", "weight": 700, "case": "none", "track": "-0.01em", "radius": 0.5,
              "accent": "#5ee6c8", "hook_bg": "rgba(255,255,255,.2)", "hook_fg": "#ffffff", "pt_bg": "rgba(20,24,34,.55)", "pt_fg": "#ffffff", "bar": "dot",
              "enter": "rise", "shadow": True},
    # modern light and dark looks for the native cards (gfxcards): glass_light = frosted white glass, slate = a flat near-black card, clean = a white card
    "glass_light": {"font": "-apple-system,'SF Pro Display','Helvetica Neue',sans-serif", "weight": 700, "case": "none", "track": "-0.01em", "radius": 0.5,
                    "accent": "#0a84ff", "hook_bg": "rgba(255,255,255,.55)", "hook_fg": "#10131a", "pt_bg": "rgba(255,255,255,.55)", "pt_fg": "#10131a", "bar": "dot",
                    "enter": "rise", "shadow": True},
    "slate": {"font": "-apple-system,'SF Pro Display','Helvetica Neue',sans-serif", "weight": 700, "case": "none", "track": "-0.01em", "radius": 0.4, "accent": "#ff9f0a",
              "hook_bg": "#14161c", "hook_fg": "#ffffff", "pt_bg": "#14161c", "pt_fg": "#ffffff", "bar": True, "enter": "slide", "shadow": True},
    # monotone pastel looks: white and ONE colour, light and soft (the accent colour is the whole palette)
    "pastel_orange": {"font": "'Avenir Next','Helvetica Neue',sans-serif", "weight": 700, "case": "none", "track": "0em", "radius": 0.5, "accent": "#ff8a4c",
                      "hook_bg": "#fff1e8", "hook_fg": "#7a3410", "pt_bg": "#fff1e8", "pt_fg": "#7a3410", "bar": "dot", "enter": "rise", "shadow": True},
    "pastel_green": {"font": "'Avenir Next','Helvetica Neue',sans-serif", "weight": 700, "case": "none", "track": "0em", "radius": 0.5, "accent": "#33b98a",
                     "hook_bg": "#e9f9f2", "hook_fg": "#0f5a41", "pt_bg": "#e9f9f2", "pt_fg": "#0f5a41", "bar": "dot", "enter": "rise", "shadow": True},
    "pastel_blue": {"font": "'Avenir Next','Helvetica Neue',sans-serif", "weight": 700, "case": "none", "track": "0em", "radius": 0.5, "accent": "#4a8fff",
                    "hook_bg": "#eaf2ff", "hook_fg": "#143a7a", "pt_bg": "#eaf2ff", "pt_fg": "#143a7a", "bar": "dot", "enter": "rise", "shadow": True},
    "pastel_pink": {"font": "'Avenir Next','Helvetica Neue',sans-serif", "weight": 700, "case": "none", "track": "0em", "radius": 0.5, "accent": "#ff6fa5",
                    "hook_bg": "#ffeef5", "hook_fg": "#7a1f45", "pt_bg": "#ffeef5", "pt_fg": "#7a1f45", "bar": "dot", "enter": "rise", "shadow": True},
    # a dark "diagram board" look: near-black boxes with thick white outlines, monospaced capitals, yellow and red accents (explainer videos)
    "board": {"font": "Menlo,'SF Mono',Consolas,monospace", "weight": 800, "case": "uppercase", "track": "0.02em", "radius": 0.12, "accent": "#ffd23f",
              "hook_bg": "#141414", "hook_fg": "#ffffff", "pt_bg": "#141414", "pt_fg": "#ffffff", "bar": True, "enter": "wipe", "shadow": True},
    # flat looks for the full-screen explainer cards: page / ink / dim / dot colours are used by gfxcards._page
    "blueprint": {"font": "'Helvetica Neue',Helvetica,Arial,sans-serif", "weight": 800, "case": "none", "track": "-0.01em", "radius": 0.2, "accent": "#4fd1ff",
                  "hook_bg": "#0e203a", "hook_fg": "#ebf4ff", "pt_bg": "#0e203a", "pt_fg": "#ebf4ff", "bar": True, "enter": "wipe", "shadow": False,
                  "page": [14, 32, 58], "ink": [235, 244, 255], "dim": [110, 140, 178], "dot": [34, 62, 100]},
    "sky": {"font": "'Helvetica Neue',Helvetica,Arial,sans-serif", "weight": 800, "case": "none", "track": "-0.01em", "radius": 0.3, "accent": "#1f6feb",
            "hook_bg": "#d6e8fc", "hook_fg": "#0d2040", "pt_bg": "#d6e8fc", "pt_fg": "#0d2040", "bar": True, "enter": "slide", "shadow": False,
            "page": [214, 232, 252], "ink": [13, 32, 64], "dim": [120, 150, 190], "dot": [188, 213, 243]},
    "mint": {"font": "'Helvetica Neue',Helvetica,Arial,sans-serif", "weight": 800, "case": "none", "track": "-0.01em", "radius": 0.3, "accent": "#12a06c",
             "hook_bg": "#def3e8", "hook_fg": "#0a3224", "pt_bg": "#def3e8", "pt_fg": "#0a3224", "bar": True, "enter": "rise", "shadow": False,
             "page": [222, 243, 232], "ink": [10, 50, 36], "dim": [120, 168, 146], "dot": [190, 225, 207]},
    "mono": {"font": "'SF Mono',Menlo,Consolas,monospace", "weight": 700, "case": "uppercase", "track": "0.04em", "radius": 0.12, "accent": "#39ff88",
             "hook_bg": "#0b0f14", "hook_fg": "#39ff88", "pt_bg": "#0b0f14", "pt_fg": "#e8fff1", "bar": True, "enter": "wipe", "shadow": True},
    # added 2026-11-05 for FirstCut's style picker: a news-desk look, warm frosted glass, and a fifth pastel
    "broadcast": {"font": "'Helvetica Neue',Helvetica,Arial,sans-serif", "weight": 800, "case": "uppercase", "track": "0.02em", "radius": 0.04, "accent": "#0b3d91",
                  "hook_bg": "#0b3d91", "hook_fg": "#ffffff", "pt_bg": "#0b3d91", "pt_fg": "#ffffff", "bar": True, "enter": "wipe", "shadow": True},
    "glass_amber": {"font": "-apple-system,'SF Pro Display','Helvetica Neue',sans-serif", "weight": 700, "case": "none", "track": "-0.01em", "radius": 0.5,
                    "accent": "#ffb340", "hook_bg": "rgba(40,28,10,.45)", "hook_fg": "#ffffff", "pt_bg": "rgba(40,28,10,.45)", "pt_fg": "#ffffff", "bar": "dot",
                    "enter": "rise", "shadow": True},
    "bold_white": {"font": "'Arial Black','Helvetica Neue',Impact,sans-serif", "weight": 900, "case": "uppercase", "track": "0.01em", "radius": 0.06,
                   "accent": "#ffffff", "hook_bg": "#ffffff", "hook_fg": "#111111", "pt_bg": "#ffffff", "pt_fg": "#111111", "bar": True, "enter": "wipe",
                   "shadow": True},
    "pastel_yellow": {"font": "'Avenir Next','Helvetica Neue',sans-serif", "weight": 700, "case": "none", "track": "0em", "radius": 0.5, "accent": "#f2b705",
                      "hook_bg": "#fff8e0", "hook_fg": "#6b4e00", "pt_bg": "#fff8e0", "pt_fg": "#6b4e00", "bar": "dot", "enter": "rise", "shadow": True},
}

# What the style picker calls each look, in the order it shows them
STYLE_NAMES = {"glass": "Liquid glass", "glass_light": "Liquid glass, light", "glass_amber": "Liquid glass, warm", "clean": "Clean white",
               "slate": "Slate", "minimal": "Minimal", "bold": "Bold red", "bold_white": "Bold white", "broadcast": "Bold blue (broadcast)", "board": "Diagram board", "mono": "Terminal",
               "blueprint": "Blueprint", "sky": "Sky", "mint": "Mint", "pastel_blue": "Pastel blue", "pastel_green": "Pastel green",
               "pastel_orange": "Pastel orange", "pastel_pink": "Pastel pink", "pastel_yellow": "Pastel yellow"}


def pick_theme(seed: str, avoid: str | None = None) -> str:
    """A theme for this video: stable for the same seed, so one video never mixes styles; `avoid` asks for a different one."""
    import hashlib
    names = sorted(THEMES)
    pool = [n for n in names if n != avoid] or names
    return pool[int(hashlib.sha1(str(seed).encode()).hexdigest()[:8], 16) % len(pool)]


def video_luma(path: str, duration: float, samples: int = 6) -> float | None:
    """How bright the picture is, 0 (black) to 1 (white): the mean of a few small frames spread over the video. None if it can't be read."""
    import subprocess

    import numpy as np
    vals = []
    for k in range(samples):
        t = duration * (k + 0.5) / samples
        try:
            raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(path), "-frames:v", "1", "-vf", "scale=64:36,format=gray", "-f", "rawvideo", "-"],
                                 capture_output=True, timeout=60).stdout
            if len(raw) == 64 * 36:
                vals.append(float(np.frombuffer(raw, dtype=np.uint8).mean()) / 255.0)
        except (OSError, subprocess.SubprocessError):
            continue
    return round(sum(vals) / len(vals), 3) if vals else None


# Which looks suit which kind of video, for bright and for dark footage. Light footage gets light, airy looks and dark footage gets dark ones, so the graphics
# belong to the picture instead of fighting it; the kind of video decides how loud they are (YouTube gets the bold ones).
PASTELS = ["pastel_orange", "pastel_green", "pastel_blue", "pastel_pink"]
LOOKS = {"bright": {"business": ["glass_light", "clean", "pastel_blue", "pastel_green"], "story": ["glass_light", "minimal", "clean", "pastel_orange", "pastel_pink"],
                    "youtube": ["bold", "board", "glass_light", "pastel_orange", "pastel_blue"], "promo": ["bold", "glass_light", "pastel_pink", "board"],
                    "ad": ["bold", "clean", "glass_light", "pastel_orange"]},
         "dark": {"business": ["glass", "slate", "pastel_blue", "pastel_green"], "story": ["glass", "minimal", "slate", "pastel_pink"],
                  "youtube": ["bold", "board", "glass", "slate", "mono", "pastel_orange"], "promo": ["bold", "board", "glass", "mono"], "ad": ["bold", "slate", "glass", "board"]}}


def match_theme(luma: float | None, kind: str, seed: str, avoid: str | None = None) -> str:
    """The look for this video's graphics, matched to how bright the picture is and what kind of video it is (stable for the same seed)."""
    import hashlib
    family = LOOKS["bright" if (luma is not None and luma >= 0.42) else "dark"]
    pool = [t for t in family.get(kind, family["story"]) if t != avoid] or family.get(kind, family["story"])
    return pool[int(hashlib.sha1(f"{seed}:{kind}".encode()).hexdigest()[:8], 16) % len(pool)]


def _layout(W: int, H: int) -> dict:
    """Where graphics sit. The picture is a person's face: nothing goes over the middle. Everything is a LOWER THIRD, bottom left, above
    the captions (portrait: higher, clear of the app's buttons)."""
    portrait = H >= W
    base = min(W, H)
    return {"portrait": portrait, "left": int(W * (0.06 if not portrait else 0.07)), "max_w": int(W * (0.55 if not portrait else 0.84)),
            "size": int(base * (0.056 if not portrait else 0.072)), "pad": int(base * (0.02 if not portrait else 0.026)),
            "bottom": int(H * (0.22 if not portrait else 0.33))}


def _theme(name: str | None) -> dict:
    return THEMES.get(name or "", THEMES["clean"])


def card_html(text: str, seconds: float, W: int, H: int, kind: str = "title", theme: str | None = None) -> str:
    """A lower-third card (the hook, or a key point), as wide as its words, in the theme's style."""
    T = _theme(theme)
    hook = kind == "title"
    text = " ".join(str(text).split())[:(90 if hook else 60)]
    L = _layout(W, H)
    size = int(L["size"] * (1.0 if hook else 0.82) * (1.0 if len(text) <= (30 if hook else 24) else 0.86 if len(text) <= 52 else 0.74))
    out_at = max(0.7, seconds - 0.4)
    bg, fg = (T["hook_bg"], T["hook_fg"]) if hook else (T["pt_bg"], T["pt_fg"])
    rad = int(size * T["radius"])
    shadow = f"box-shadow:0 {int(size * .12)}px {int(size * .4)}px rgba(0,0,0,.35);" if T["shadow"] and bg != "transparent" else ""
    glass = "backdrop-filter:blur(14px);border:1px solid rgba(255,255,255,.25);" if bg.startswith("rgba") and T["bar"] == "dot" else ""
    txs = "text-shadow:0 2px 14px rgba(0,0,0,.65);" if bg == "transparent" else ""
    deco_css = {True: f"#bar{{width:{max(6, int(size * .2))}px;background:{T['accent']};flex:none}}",
                "dot": f"#bar{{width:{int(size * .3)}px;height:{int(size * .3)}px;border-radius:50%;background:{T['accent']};flex:none;align-self:center;margin-left:{int(L['pad'] * 1.2)}px}}",
                "under": f"#bar{{position:absolute;left:0;bottom:-{int(size * .18)}px;height:{max(4, int(size * .09))}px;width:100%;background:{T['accent']};transform-origin:left}}"
                }.get(T["bar"], "")
    enter = {"slide": 'tl.fromTo("#c",{opacity:0,x:-60},{opacity:1,x:0,duration:.4,ease:"power3.out"},0.05);'
                      f'tl.to("#c",{{opacity:0,x:-40,duration:.3}},{out_at});',
             "wipe": 'tl.fromTo("#c",{clipPath:"inset(0 100% 0 0)",opacity:1},{clipPath:"inset(0 0% 0 0)",duration:.38,ease:"power2.out"},0.05);'
                     f'tl.to("#c",{{clipPath:"inset(0 0 0 100%)",duration:.3,ease:"power2.in"}},{out_at});',
             "rise": 'tl.fromTo("#c",{opacity:0,y:36},{opacity:1,y:0,duration:.45,ease:"power3.out"},0.05);'
                     + ('tl.fromTo("#bar",{scaleX:0},{scaleX:1,duration:.5,ease:"power2.out"},0.25);' if T["bar"] == "under" else "")
                     + f'tl.to("#c",{{opacity:0,y:-18,duration:.3}},{out_at});'}[T["enter"]]
    deco = '<div id="bar"></div>' if T["bar"] else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="UTF-8"/><meta name="viewport" content="width={W}, height={H}"/>
<script src="{GSAP}"></script>
<style>*{{margin:0;padding:0;box-sizing:border-box}}
html,body{{width:{W}px;height:{H}px;overflow:hidden;background:transparent}}
#root{{width:100%;height:100%;position:relative;font-family:{T["font"]}}}
#c{{position:absolute;left:{L["left"]}px;bottom:{L["bottom"]}px;width:fit-content;max-width:{L["max_w"]}px;display:flex;align-items:stretch;
border-radius:{rad}px;background:{bg};color:{fg};font-weight:{T["weight"]};text-transform:{T["case"]};font-size:{size}px;line-height:1.1;
letter-spacing:{T["track"]};{shadow}{glass}{txs}{"overflow:hidden;" if T["bar"] is True else ""}}}
#tx{{position:relative;padding:{int(L["pad"] * (0.9 if T["bar"] == "under" else 0.8))}px {int(L["pad"] * 1.3)}px}}
{deco_css}
</style></head><body>
<div id="root" data-composition-id="main" data-start="0" data-duration="{seconds}" data-width="{W}" data-height="{H}">
<div id="c" class="clip" data-start="0" data-duration="{seconds}" data-track-index="0">{deco if T["bar"] != "under" else ""}<div id="tx">{html.escape(text)}{deco if T["bar"] == "under" else ""}</div></div></div>
<script>
const tl = gsap.timeline({{ paused: true }});
{enter}
window.__timelines["main"] = tl; tl.seek(0);
</script></body></html>
"""


def title_html(text: str, seconds: float, W: int, H: int, theme: str | None = None) -> str:
    return card_html(text, seconds, W, H, "title", theme)


def point_html(text: str, seconds: float, W: int, H: int, theme: str | None = None) -> str:
    return card_html(text, seconds, W, H, "point", theme)


def end_html(business: str, tagline: str, logo: str | None, seconds: float, W: int, H: int, theme: str | None = None) -> str:
    """The last seconds of an ad: the logo (only if there is one), the name and a short call to action, centred over a soft dark wash."""
    T = _theme(theme)
    business, tagline = " ".join(str(business).split())[:40], " ".join(str(tagline).split())[:48]
    base = min(W, H)
    big, small = int(base * (0.095 if not logo else 0.075)), int(base * 0.042)
    logo_h = int(H * 0.30)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="UTF-8"/><meta name="viewport" content="width={W}, height={H}"/>
<script src="{GSAP}"></script>
<style>*{{margin:0;padding:0;box-sizing:border-box}}
html,body{{width:{W}px;height:{H}px;overflow:hidden;background:transparent}}
#root{{width:100%;height:100%;position:relative;font-family:{T["font"]}}}
#wash{{position:absolute;inset:0;background:radial-gradient(ellipse at center, rgba(8,10,16,.70), rgba(8,10,16,.88))}}
#col{{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:{int(base * 0.028)}px;text-align:center;color:#fff}}
#logo{{max-height:{logo_h}px;max-width:{int(W * 0.5)}px;object-fit:contain;{"" if logo else "display:none;"}}}
#biz{{font-weight:{max(800, T["weight"])};text-transform:{T["case"]};font-size:{big}px;letter-spacing:{T["track"]};line-height:1.05;max-width:{int(W * 0.8)}px}}
#rule{{width:{int(W * 0.12)}px;height:{max(4, int(base * 0.007))}px;background:{T["accent"]};border-radius:4px}}
#tag{{font-weight:700;text-transform:{T["case"]};font-size:{small}px;color:{T["accent"] if T["accent"] != "#ffffff" else "#ffd23f"};letter-spacing:.03em;max-width:{int(W * 0.8)}px}}
</style></head><body>
<div id="root" data-composition-id="main" data-start="0" data-duration="{seconds}" data-width="{W}" data-height="{H}">
<div id="wash" class="clip" data-start="0" data-duration="{seconds}" data-track-index="0"></div>
<div id="col" class="clip" data-start="0" data-duration="{seconds}" data-track-index="1"><img id="logo" src="{logo or ""}"/><div id="biz">{html.escape(business)}</div><div id="rule"></div><div id="tag">{html.escape(tagline)}</div></div></div>
<script>
const tl = gsap.timeline({{ paused: true }});
tl.fromTo("#wash", {{opacity:0}}, {{opacity:1, duration:.4}}, 0);
tl.fromTo("#logo", {{opacity:0, y:30, scale:.9}}, {{opacity:1, y:0, scale:1, duration:.5, ease:"power3.out"}}, 0.15);
tl.fromTo("#biz", {{opacity:0, y:30}}, {{opacity:1, y:0, duration:.5, ease:"power3.out"}}, 0.35);
tl.fromTo("#rule", {{scaleX:0}}, {{scaleX:1, duration:.4, ease:"power2.out"}}, 0.55);
tl.fromTo("#tag", {{opacity:0, y:20}}, {{opacity:1, y:0, duration:.45, ease:"power3.out"}}, 0.65);
window.__timelines["main"] = tl; tl.seek(0);
</script></body></html>
"""


def render_title(cfg, text: str, seconds: float, W: int, H: int, out: Path, timeout: float = 300.0, kind: str = "title",
                 extra: dict | None = None) -> Path | None:
    exe = cli(cfg)
    if exe is None or not str(text).strip():
        return None
    W, H = int(W) // 2 * 2, int(H) // 2 * 2
    scale = 1080 / min(W, H)                       # compose at ~1080 and let ffmpeg scale: keeps renders quick on big frames
    cw, ch = int(round(W * scale / 2) * 2), int(round(H * scale / 2) * 2)
    with tempfile.TemporaryDirectory(prefix="hf-title-") as tmp:
        proj = Path(tmp)
        if kind == "end":
            ex = extra or {}
            logo = None
            if ex.get("logo") and Path(ex["logo"]).is_file():
                logo = "logo" + Path(ex["logo"]).suffix.lower()
                shutil.copy2(ex["logo"], proj / logo)
            (proj / "index.html").write_text(end_html(ex.get("business", text), ex.get("tagline", ""), logo, seconds, cw, ch, (extra or {}).get("theme")))
        else:
            (proj / "index.html").write_text((point_html if kind == "point" else title_html)(text, seconds, cw, ch, (extra or {}).get("theme")))
        (proj / "hyperframes.json").write_text('{"name": "title"}')
        env = {**os.environ, "PATH": os.pathsep.join(["/opt/homebrew/bin", "/usr/local/bin", os.environ.get("PATH", "")]),
               "HYPERFRAMES_SKIP_SKILLS": "1", "HYPERFRAMES_NO_TELEMETRY": "1", "DO_NOT_TRACK": "1"}
        mov = proj / "title.mov"
        try:
            r = subprocess.run(["nice", "-n", "10", str(exe), "render", str(proj), "--format", "mov", "--workers", "2", "--quiet",
                                "-o", str(mov)], capture_output=True, text=True, timeout=timeout, env=env, cwd=str(exe.parent.parent.parent))
            if r.returncode or not mov.exists():
                log.warning(f"hyperframes title failed: {(r.stderr or r.stdout)[-300:]}")
                return None
            shutil.copy2(mov, out)
        except (subprocess.SubprocessError, OSError) as e:
            log.warning(f"hyperframes title failed: {e}")
            return None
    return out
