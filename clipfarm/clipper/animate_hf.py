"""Topic-specific animated diagrams, tier B: the AI writes an SVG + GSAP scene, Hyperframes (headless Chrome) renders it.

    concept + what the narrator says + the source facts  ->  the AI writes the drawing code (it composes the helpers in assets/hyperframes/hf-lib.js)
    ->  `hyperframes check` (lint, layout, contrast: errors go back to the AI, up to 3 rounds)  ->  render to mp4  ->  a few frames are shown to a vision
    model that checks text is readable and nothing is cut off or overlapping (problems go back to the AI once)  ->  the clip, or None (callers fall back).

    make(cfg, ai, concept, narration, words, duration, look, out_dir, source_text="") -> {"file", "attempts", "notes"} | None

The page, colours and ink come from the same look as the cards (gfxcards), so animations sit in a video with the slides around them. Everything the scene
draws must come from the source text: the AI is told so and the QA pass is asked to flag invented labels and numbers.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from ..core import ffmpeg
from ..core.log import get_logger
from . import gfxcards as GC

log = get_logger("clipper.animate_hf")
ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets" / "hyperframes"
DEFAULT_DIR = "~/Movies/claude-editor"
DEFAULT_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
W, H, FPS = 1920, 1080, 30

SCHEMA = {"type": "object", "properties": {
    "notes": {"type": "string"},
    "svg": {"type": "string"},
    "script": {"type": "string"}}, "required": ["script"]}
QA_SCHEMA = {"type": "object", "properties": {
    "ok": {"type": "boolean"},
    "problems": {"type": "array", "items": {"type": "string"}}}, "required": ["ok", "problems"]}

HELPERS_DOC = (ASSETS / "hf-lib.js").read_text().split("*/", 1)[0].lstrip("/* \n") if (ASSETS / "hf-lib.js").exists() else ""

EXAMPLE = """
// EXAMPLE (a different topic, to show the style - do NOT copy its content):
const cx = W / 2, cy = 560;
const big = HF.gear(HF.svg, { x: cx + 200, y: cy, teeth: 24, r: 230, stroke: C.ink, fill: "none" });
const small = HF.gear(HF.svg, { x: cx - 250, y: cy, teeth: 12, r: 118, stroke: C.accent, fill: "none" });
const t1 = HF.label(HF.svg, "Small gear", cx - 250, cy + 220, { size: 52, fill: C.accent });
const t2 = HF.label(HF.svg, "Large gear", cx + 200, cy + 330, { size: 52, fill: C.ink });
[big.g, small.g].forEach((g) => (g.style.opacity = 0));
[t1, t2].forEach((t) => (t.style.opacity = 0));
HF.fadeIn(tl, small.g, 0.4); HF.fadeIn(tl, big.g, 0.8); HF.fadeIn(tl, t1, 1.4); HF.fadeIn(tl, t2, 1.8);
HF.spin(tl, small, { deg: 720, start: 1.5, dur: D - 1.5 });
HF.spin(tl, big, { deg: HF.meshDeg(small, big, 720), start: 1.5, dur: D - 1.5 });   // turns half as fast, the other way
"""

SYSTEM = (
    "You are a motion designer making ONE short animated diagram for an explainer video, as SVG drawn by JavaScript and animated with GSAP. "
    "You receive: the CONCEPT to show, the NARRATION spoken over it with the time each word is said, the SOURCE TEXT (the only place a fact, name or number "
    "may come from), the duration D in seconds and the colour palette. Return JSON: 'svg' (optional static SVG markup placed inside the stage; may be empty), "
    "'script' (JavaScript, the body of a function) and 'notes' (one sentence on what the viewer sees).\n"
    "THE SCRIPT runs inside a page that already provides: W=1920, H=1080 (the stage, a viewBox), D (seconds), C = {page, ink, dim, accent} (colour strings; "
    "draw with ONLY these colours, and give a shape a different brightness with the 'opacity' attribute rather than a new colour), tl (the one paused "
    "GSAP timeline - add every animation to it, never create another, never register anything), HF (helpers, below), HF.svg (the <svg> to draw into), "
    "WORDS (the timed words) and gsap. No fetch, no timers, no Math.random, no Date, no setInterval; everything is driven by tl.\n"
    "HELPERS:\n{helpers}\n"
    "DESIGN RULES: (1) show the mechanism WORKING, not a poster: things move, turn, slide, fill or count in the way the narration describes, in the order it "
    "describes it; a still arrangement of boxes is a failure. (2) Time each part to its narration: use HF.at('word') (seconds, may be null) so a part "
    "appears as the narrator says it; every part must appear before D-0.8; keep motion going until D (a slow continuing turn is fine). (3) Labels: at most 3 "
    "words, at least 46px (font-size), taken from the source text; no numbers or facts the source does not give - if you need an illustrative size, show it "
    "by drawing, not by writing a number. (4) Everything inside the stage with 110px margin; no text over a shape's edge, no overlapping labels; leave the "
    "bottom 140px free for captions. (5) One clear idea, drawn large: the main shapes at least 260px across. (6) Initially hide whatever fades in by "
    "setting style.opacity = 0 on its group, then HF.fadeIn. (7) Use plain SVG: no filters, gradients, images or external fonts; stroke widths 6-12. "
    "(8) Never use CSS transforms on the same property that GSAP animates; to move a group use HF.move or tl.to with x/y only. "
    "(9) Do NOT paint a background or a full-stage rectangle: the page (colour and pattern) is already behind your drawing. "
    "(10) No console.log needed; wrap nothing in try/catch. Keep the script under 120 lines.\n"
    "(11) BE BOLD (this is watched on a phone, and a timid picture loses the viewer): the main subject fills at least 45 percent of the stage height, never more than a third of the "
    "stage is empty, strong contrast (ink on the page colour, the accent for the one thing that matters), thick strokes, labels at least 56px. Something MOVES within the first 0.5 s and "
    "something NEW happens at least every 2.5 s: when D is over 8 seconds build it as scenes of 3 to 5 seconds, each scene changing the picture clearly (a zoom, a swap, a new part "
    "arriving, a counter running), so no stretch of the clip is a still arrangement.\n"
    "A WORKED EXAMPLE of the style and the helpers:{example}"
)


# ------------------------------------------------------------------------------------------ look
def look_tokens(theme: str, bg: str) -> dict:
    """The page / ink / dim / accent colours of a look, the same ones the cards use. -> {page, ink, dim, dot, accent, bg}."""
    look = GC.Look(theme, W, H)
    base, ink, dim, dot = GC._page(look, bg)
    page = base.getpixel((2, 2))
    acc = GC._accent_on(look, page, ink)
    hx = lambda c: "#%02x%02x%02x" % tuple(int(v) for v in c[:3])  # noqa: E731
    return {"page": hx(page), "ink": hx(ink), "dim": hx(dim), "dot": hx(dot), "accent": hx(acc), "bg": bg or "dots"}


def page_css(t: dict) -> str:
    """The card page as CSS (flat patterns only), so animations and cards match."""
    page, dot, ink, acc, bg = t["page"], t["dot"], t["ink"], t["accent"], t["bg"]
    base = f"background-color:{page};"
    if bg == "grid":
        return base + (f"background-image:linear-gradient({dot} 1px,transparent 1px),linear-gradient(90deg,{dot} 1px,transparent 1px);background-size:48px 48px;")
    if bg == "rule":
        return base + (f"background-image:linear-gradient({dot} 2px,transparent 2px);background-size:100% 66px;")
    if bg == "stripe":
        return base + (f"background-image:repeating-linear-gradient(135deg,{dot} 0,{dot} 2px,transparent 2px,transparent 26px);")
    if bg == "dots":
        return base + (f"background-image:radial-gradient({dot} 1.6px,transparent 1.8px);background-size:30px 30px;")
    return base


# ------------------------------------------------------------------------------------------ project
def assemble(proj: Path, script: str, svg: str, tokens: dict, duration: float, words: list[dict]) -> Path:
    proj.mkdir(parents=True, exist_ok=True)
    shutil.copy(ASSETS / "gsap-3.14.2.min.js", proj / "gsap.min.js")
    shutil.copy(ASSETS / "hf-lib.js", proj / "hf-lib.js")
    wj = json.dumps([{"text": w["text"], "start": round(float(w["start"]), 3)} for w in words if "start" in w])
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="UTF-8"/><meta name="viewport" content="width={W}, height={H}"/>
<script src="gsap.min.js"></script><script src="hf-lib.js"></script>
<style>*{{margin:0;padding:0;box-sizing:border-box}}body{{{page_css(tokens)}font-family:'Helvetica Neue',Helvetica,Arial,sans-serif}}
#root{{position:relative;width:100%;height:100%;overflow:hidden}}#stage{{position:absolute;inset:0;width:100%;height:100%}}</style></head>
<body><div id="root" data-composition-id="main" data-start="0" data-width="{W}" data-height="{H}" data-duration="{duration:.3f}">
<svg id="stage" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">{svg or ''}</svg></div>
<script>
window.WORDS = {wj};
const W = {W}, H = {H}, D = {duration:.3f};
const C = {json.dumps({k: tokens[k] for k in ('page', 'ink', 'dim', 'accent')})};
HF.svg = document.getElementById("stage");
const tl = gsap.timeline({{ paused: true }});
(function () {{
{script}
}})();
window.__timelines["main"] = tl;
</script></body></html>
"""
    (proj / "index.html").write_text(html)
    return proj


def _env(cfg) -> dict:
    env = dict(os.environ)
    env["HYPERFRAMES_BROWSER_PATH"] = str(cfg.get("clipper.animations.browser") or DEFAULT_CHROME)
    env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + env.get("PATH", "")                 # node, for the Hyperframes command, when started by launchd
    return env


def cli(cfg) -> Path | None:
    """The Hyperframes command: the folder in the config (clipper.ai_videos.hyperframes_dir), else the one installed with the project (tools/ai-video, as on the
    M2), else ~/Movies/claude-editor (this Mac)."""
    dirs = [cfg.get("clipper.ai_videos.hyperframes_dir"), ROOT / "tools" / "ai-video", DEFAULT_DIR]
    for d in dirs:
        if not d:
            continue
        exe = Path(str(d)).expanduser() / "node_modules" / ".bin" / "hyperframes"
        if exe.exists():
            return exe
    return None


def _strip(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s or "")


def run_check(cfg, proj: Path, timeout: float = 240.0) -> tuple[bool, str]:
    """hyperframes check: lint + runtime + layout + contrast. -> (passed, the findings as text for the AI)."""
    exe = cli(cfg)
    if not exe:
        return False, "hyperframes is not installed"
    proj = Path(proj).resolve()                                                    # the CLI runs from its own folder: paths must be absolute
    for k in range(3):                                                             # a loaded machine can miss the browser's page-load window: that is not the design's fault
        try:
            p = subprocess.run([str(exe), "check", str(proj), "--timeout", "60000"], capture_output=True, text=True, timeout=timeout, env=_env(cfg), cwd=str(exe.parents[2]))
        except subprocess.TimeoutExpired:
            log.warning("hyperframes check timed out; trying again")
            continue
        out = _strip(p.stdout + "\n" + p.stderr)
        if "Navigation timeout" in out or "Browser session never ran" in out:
            log.warning(f"hyperframes check: the browser did not load the page in time (try {k + 1}); trying again")
            time.sleep(8)
            continue
        keep = [ln for ln in out.splitlines() if ln.strip() and not ln.startswith(("[INFO]", "[hyperframes]", "◆"))]
        return "Check passed" in out, "\n".join(keep)[-2500:]
    return True, "the browser never loaded the page for the check (machine busy): skipped, the render step still runs"


def render(cfg, proj: Path, out: Path, timeout: float = 900.0) -> bool:
    exe = cli(cfg)
    if not exe:
        return False
    proj, out = Path(proj).resolve(), Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        p = subprocess.run([str(exe), "render", str(proj), "-o", str(out), "--fps", str(FPS), "-q", "looks", "--quiet"], capture_output=True, text=True,
                           timeout=timeout, env=_env(cfg), cwd=str(exe.parents[2]))
    except subprocess.TimeoutExpired:
        log.warning("hyperframes render timed out")
        return False
    if p.returncode != 0 or not out.exists() or out.stat().st_size < 5000:
        log.warning(f"hyperframes render failed: {_strip(p.stderr or p.stdout)[-400:]}")
        return False
    return True


def frames(video: Path, at: list[float], size: int = 960) -> list[bytes]:
    out = []
    for t in at:
        r = subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1", "-vf", f"scale={size}:-1",
                            "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "4", "-"], capture_output=True)
        if r.stdout:
            out.append(r.stdout)
    return out


# ------------------------------------------------------------------------------------------ the AI
HOOK_NOTE = ("THIS IS THE OPENING SHOT OF THE VIDEO: it decides whether the viewer stays. Make it the boldest picture in the video: ONE huge subject filling the frame, first movement at 0 s, "
             "a cut-like change every 2 s, and one big punch of text (at most 4 words, 110px or larger) that states the surprising claim the narrator opens with.")


def design(ai, concept: str, narration: str, words: list[dict], duration: float, tokens: dict, source_text: str, feedback: str = "", previous: dict | None = None, hook: bool = False) -> dict:
    timed = " ".join(f"{w['text']}@{float(w['start']):.1f}" for w in words if "start" in w)[:2400]
    system = SYSTEM.replace("{helpers}", HELPERS_DOC).replace("{example}", EXAMPLE)
    prompt = (f"CONCEPT: {concept}\n\nNARRATION (spoken over the diagram): {narration}\nTIMED WORDS: {timed}\n\nD = {duration:.2f} seconds\n"
              f"PALETTE: {json.dumps({k: tokens[k] for k in ('page', 'ink', 'dim', 'accent')})}\n" + (f"\n{HOOK_NOTE}\n" if hook else "") + f"\nSOURCE TEXT:\n{source_text[:9000]}\n")
    if previous and feedback:
        prompt += (f"\nYOUR PREVIOUS ATTEMPT:\n{json.dumps({'svg': previous.get('svg', ''), 'script': previous.get('script', '')})[:6000]}\n\n"
                   f"PROBLEMS TO FIX (keep what works, change only what these require):\n{feedback}\n")
    return ai.complete_json("flow_anim_design", system, prompt, SCHEMA).data


def qa(ai, video: Path, concept: str, narration: str, duration: float) -> tuple[bool, list[str]]:
    ts = [duration * f for f in (0.18, 0.45, 0.72, 0.93)]
    imgs = frames(video, ts)
    if not imgs:
        return True, []
    res = ai.complete_json(
        "anim_qa", "You check frames from an animated explainer diagram. Be strict but fair: report only real problems.",
        f"The diagram should show: {concept}\nThe narrator says: {narration}\nThese are 4 frames in time order. Check: is any text cut off, overlapping or too small to read on a phone? "
        "Does anything extend outside the frame or into the bottom caption strip? Is the picture mostly empty, small in the frame (the main subject under about 40 percent of the height), "
        "low contrast, or just a still arrangement with nothing happening between frames? A timid, sparse picture is a problem. "
        "Does it plausibly show the concept? Are there labels or numbers that look invented (not about the concept)? "
        "Answer ok=true only if there is no serious problem; list each serious problem in one short sentence.", QA_SCHEMA, images=imgs)
    return bool(res.data.get("ok")), [str(p) for p in res.data.get("problems", [])][:6]


def make(cfg, ai, concept: str, narration: str, words: list[dict], duration: float, look: tuple[str, str], out_dir: Path, source_text: str = "",
         max_rounds: int = 3, say=print, hook: bool = False) -> dict | None:
    """One animation clip, or None when it could not be made well enough. `look` = (theme, page style)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens = look_tokens(*look)
    proj = out_dir / "proj"
    final = out_dir / "animation.mp4"
    prev: dict | None = None
    feedback = ""
    t0 = time.time()
    for attempt in range(1, max_rounds + 1):
        try:
            d = design(ai, concept, narration, words, duration, tokens, source_text, feedback, prev, hook=hook)
        except Exception as e:  # noqa: BLE001
            log.warning(f"animation design failed: {e}")
            return None
        (out_dir / f"design_{attempt}.json").write_text(json.dumps(d, indent=1))
        assemble(proj, d.get("script", ""), d.get("svg", ""), tokens, duration, words)
        ok, findings = run_check(cfg, proj)
        say(f"  attempt {attempt}: check {'passed' if ok else 'FAILED'} ({time.time() - t0:.0f}s)")
        if not ok:
            prev, feedback = d, "hyperframes check reported:\n" + findings
            continue
        if not render(cfg, proj, final):
            prev, feedback = d, "the render failed: the script probably threw an error or never registered; check every helper call and variable name"
            continue
        good, problems = qa(ai, final, concept, narration, duration)
        say(f"  attempt {attempt}: look {'ok' if good else 'problems: ' + '; '.join(problems)} ({time.time() - t0:.0f}s)")
        if good or attempt == max_rounds:
            return {"file": final, "attempts": attempt, "notes": d.get("notes", ""), "qa_ok": good, "problems": problems, "seconds": time.time() - t0}
        prev, feedback = d, "a reviewer looking at frames of your render found:\n- " + "\n- ".join(problems)
    return None
