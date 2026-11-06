"""Final Cut Pro project export (FCPXML 1.11) for Clipper edits, so every decision stays editable in FCP.

    <project folder>/
        <name>.fcpxml          import in Final Cut Pro (File > Import > XML...)
        Media/source.<ext>     the original video (not re-encoded)
        Music/<track>.mp3      + <track>.txt with its license / credit line
        Captions/<name>.srt    the captions as subtitles too

Timeline: one asset-clip per kept stretch of the source (every dead-air cut is a real edit you can
extend), the music as a connected clip under the first one (spanning the whole timeline) with its ducking
as volume keyframes, each caption group as a Basic Title (font / size / colour / outline editable), and
the optional hook title on top. All times are snapped to the sequence's frame grid, as FCP requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from urllib.parse import quote
from xml.sax.saxutils import escape as _escape


def escape(s) -> str:
    """XML text and attribute safe: a double quote inside name=\"...\" would end the attribute."""
    return _escape(str(s), {'"': "&quot;"})

BASIC_TITLE_UID = ".../Titles.localized/Bumper:Opener.localized/Basic Title.localized/Basic Title.moti"
TITLE_POSITION_KEY = "9999/999166631/999166633/1/100/101"     # Basic Title > Position
DTD_DIR = Path("/Applications/Final Cut Pro.app/Contents/Frameworks/Interchange.framework/Versions/A/Resources")
VERSION = "1.11"


STANDARD = [Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001), Fraction(30), Fraction(50), Fraction(60000, 1001),
            Fraction(60), Fraction(48000, 1001), Fraction(48), Fraction(120000, 1001), Fraction(120)]


def std_rate(fr) -> Fraction:
    """A clip's exact frame rate as Final Cut knows it: 23.976023... -> 24000/1001. (Rounding to 23.976 first gave 2997/125, a rate
    Final Cut does not recognise as 23.98.)"""
    if isinstance(fr, Fraction) and fr in STANDARD:
        return fr
    f = float(fr or 30.0)
    best = min(STANDARD, key=lambda r: abs(float(r) - f))
    return best if abs(float(best) - f) / float(best) < 0.002 else Fraction(f).limit_denominator(1001)


class Clock:
    """Seconds -> FCPXML rational time on the sequence's frame grid."""

    def __init__(self, fps: Fraction):
        self.frame = 1 / Fraction(fps)

    def frames(self, seconds: float) -> int:
        return int(round(Fraction(seconds).limit_denominator(10 ** 6) / self.frame))

    def t(self, seconds: float) -> str:
        v = self.frames(seconds) * self.frame
        return f"{v.numerator}/{v.denominator}s" if v.denominator != 1 else f"{v.numerator}s"

    def at(self, frames: int) -> str:
        v = frames * self.frame
        return f"{v.numerator}/{v.denominator}s" if v.denominator != 1 else f"{v.numerator}s"


def file_url(path: Path | str) -> str:
    return "file://" + quote(str(Path(path).resolve() if not str(path).startswith("/") else path))


def graphic_position(text: str, kind: str, seq_w: int, seq_h: int) -> tuple[tuple[float, float], float]:
    """Where the finished video shows a graphic (graphics._layout: a lower third, bottom left, above the captions), as a Final Cut title
    Position (pixels from the centre, y up) and font size, so the project looks like the video. The end card is centred."""
    from .graphics import _layout
    L = _layout(seq_w, seq_h)
    if kind == "end":
        return (0.0, 0.0), L["size"] * 1.4
    size = L["size"] * (1.0 if kind == "hook" else 0.82)
    width = min(L["max_w"], len(str(text)) * size * 0.56 + 2.6 * L["pad"])
    return (-seq_w / 2 + L["left"] + width / 2, -seq_h / 2 + L["bottom"] + size * 0.75), size


def _title_lines(clk, ref: str, lane: int, offset_s: str, dur_s: str, text: str, kind: str, seq_w: int, seq_h: int, style: "CaptionStyle", ts: int) -> list[str]:
    """One editable Basic Title in Final Cut for a graphic of the finished video (hook card, key point, end card), where the video has it."""
    pos, size = graphic_position(text, kind, seq_w, seq_h)
    if kind == "hook":
        color, bg, name = "0.07 0.07 0.07 1", ' backgroundColor="1 1 1 1"', "Hook title"
    elif kind == "end":
        color, bg, name = "1 1 1 1", ' backgroundColor="0.05 0.06 0.09 0.82"', "End card"
    else:
        color, bg, name = "1 1 1 1", ' backgroundColor="0.055 0.063 0.086 0.9"', "Key point"
    return [f'              <title ref="{ref}" lane="{lane}" offset="{offset_s}" name="{name}" duration="{dur_s}">',
            f'                <param name="Position" key="{TITLE_POSITION_KEY}" value="{_fmt(pos[0])} {_fmt(pos[1])}"/>',
            f'                <text><text-style ref="ts{ts}">{escape(text)}</text-style></text>',
            f'                <text-style-def id="ts{ts}"><text-style font="Arial Black" fontSize="{_fmt(size)}" fontColor="{color}"{bg} bold="1" alignment="center"/></text-style-def>',
            "              </title>"]


@dataclass
class Media:
    path: str                                # absolute path as FCP will see it
    duration: float
    has_video: bool = True
    has_audio: bool = True
    width: int = 1920
    height: int = 1080
    fps: Fraction = Fraction(30)
    audio_rate: int = 48000
    audio_channels: int = 2


@dataclass
class MusicBed:
    path: str
    duration: float
    name: str
    start: float = 0.0                       # where in the track playback starts
    keyframes: list[tuple[float, float]] = field(default_factory=list)   # (timeline s, dB)
    crossfade: float = 0.0                   # a song shorter than the video repeats, each repeat crossfaded this long (as in the video)


def _gain_at(keyframes: list[tuple[float, float]], t: float, default: float = 0.0) -> float:
    if not keyframes:
        return default
    if t <= keyframes[0][0]:
        return keyframes[0][1]
    for (t0, v0), (t1, v1) in zip(keyframes, keyframes[1:]):
        if t0 <= t <= t1:
            return v0 + (v1 - v0) * ((t - t0) / (t1 - t0) if t1 > t0 else 0.0)
    return keyframes[-1][1]


def _music_lines(clk: "Clock", music: "MusicBed", ref: str, offset_f: int, total_f: int) -> list[str]:
    """The music bed as connected clips under the first piece of the timeline: one clip if the song is long enough, else the song again
    and again (alternating lanes so the repeats can overlap), crossfaded and ducked exactly like the finished video's sound."""
    from .music import loop_plan
    total = total_f * float(clk.frame)
    out: list[str] = []
    pieces = loop_plan(music.duration, music.start, total, music.crossfade)
    for k, (at, src, ln) in enumerate(pieces):
        fade_in = music.crossfade if k else 0.0
        fade_out = music.crossfade if k + 1 < len(pieces) else 0.0
        times = sorted({at, at + ln, *(t for t, _ in music.keyframes if at < t < at + ln),
                        *([at + fade_in] if fade_in else []), *([at + ln - fade_out] if fade_out else [])})

        def gain(t: float) -> float:
            g = _gain_at(music.keyframes, t)
            if fade_in and t < at + fade_in:
                g += -60.0 * (1 - (t - at) / fade_in)
            if fade_out and t > at + ln - fade_out:
                g += -60.0 * (1 - (at + ln - t) / fade_out)
            return max(-96.0, g)
        name = f"Music: {escape(music.name)}" + (f" ({k + 1})" if len(pieces) > 1 else "")
        out.append(f'              <asset-clip ref="{ref}" lane="{-1 - (k % 2)}" offset="{clk.at(offset_f + clk.frames(at))}" name="{name}" '
                   f'start="{clk.t(src)}" duration="{clk.at(max(1, clk.frames(ln)))}" audioRole="music">')
        if music.keyframes or fade_in or fade_out:
            out += ['                <adjust-volume>', '                  <param name="amount">', '                    <keyframeAnimation>']
            out += [f'                      <keyframe time="{clk.t(src + (t - at))}" value="{_fmt(gain(t))}dB"/>' for t in times]
            out += ['                    </keyframeAnimation>', '                  </param>', '                </adjust-volume>']
        out.append("              </asset-clip>")
    return out


@dataclass
class CaptionStyle:
    font: str = "Arial Black"
    size: float = 80
    color: str = "1 1 1 1"
    highlight: str = "1 0.91 0.1 1"
    stroke: str = "0 0 0 1"
    stroke_width: float = 6
    y: float = -0.25                          # vertical position as a fraction of frame height (centre = 0)
    uppercase: bool = True


def _fmt(n: float) -> str:
    return f"{n:.4f}".rstrip("0").rstrip(".")


CAPTION_ROLE = "iTT?captionFormat=ITT.en"     # Final Cut's own caption role (the captions lane; View › Captions turns them on)


def caption_spans(subs: list[tuple[float, float, str]] | None) -> list[tuple[float, float, str]]:
    """Real captions never overlap (Final Cut refuses two iTT captions at once): each one ends where the next begins."""
    out = []
    xs = sorted((float(a), float(b), str(t).strip()) for a, b, t in (subs or []) if str(t).strip() and float(b) > float(a))
    for k, (a, b, t) in enumerate(xs):
        if k + 1 < len(xs):
            b = min(b, xs[k + 1][0])
        if b - a >= 0.1:
            out.append((a, b, t))
    return out


def srt(subs: list[tuple[float, float, str]] | None) -> str:
    """The captions as an .srt file (the Final Cut bundle's Captions folder: import it, or use the real captions in the project)."""
    def ts(t: float) -> str:
        ms = int(round(max(0.0, t) * 1000))
        return f"{ms // 3600000:02d}:{ms % 3600000 // 60000:02d}:{ms % 60000 // 1000:02d},{ms % 1000:03d}"
    return "\n".join(f"{k}\n{ts(a)} --> {ts(b)}\n{t}\n" for k, (a, b, t) in enumerate(caption_spans(subs), 1))


def _caption_lines(lane: int, offset_s: str, dur_s: str, text: str, ts: int) -> list[str]:
    """A REAL caption (not a title): the editor fixes a word in Final Cut's caption inspector, exports SRT/iTT, or burns it in."""
    return [f'              <caption lane="{lane}" offset="{offset_s}" name="{escape(text[:40])}" duration="{dur_s}" role="{CAPTION_ROLE}">',
            f'                <text placement="bottom"><text-style ref="ts{ts}">{escape(text)}</text-style></text>',
            f'                <text-style-def id="ts{ts}"><text-style font=".AppleSystemUIFont" fontSize="13" fontFace="Regular" '
            f'fontColor="1 1 1 1" backgroundColor="0 0 0 1"/></text-style-def>',
            "              </caption>"]


PIP_SCALE = 0.3                                 # the speaker's window: 30% of the frame
PIP_MARGIN = 0.04                               # from the frame's edges, as a fraction of its height


def layout_geometry(layout: str, seq_w: int, seq_h: int) -> dict:
    """Where each picture of a layout sits, in Final Cut's own units (transform position: percent of the frame height from the centre;
    trim: percent of the original frame height). pip: the speaker small, top right, over the full B-roll. split (a vertical video):
    the speaker on top, the B-roll below, each the middle half of its own picture."""
    if layout == "pip":
        a = seq_w / seq_h
        pw, ph = PIP_SCALE * a, PIP_SCALE                                   # (in frame heights)
        x = (a / 2 - PIP_MARGIN - pw / 2) * 100
        y = (0.5 - PIP_MARGIN - ph / 2) * 100
        return {"inset": {"position": f"{_fmt(x)} {_fmt(y)}", "scale": f"{_fmt(PIP_SCALE)} {_fmt(PIP_SCALE)}"}, "broll": None}
    if layout == "split":
        return {"inset": {"position": "0 25", "trim": (25, 25)}, "broll": {"position": "0 -25", "trim": (25, 25)}}
    return {"inset": None, "broll": None}


def _adjust_lines(g: dict | None, conform: bool, indent: str) -> list[str]:
    """adjust-crop / adjust-conform / adjust-transform in the order the DTD wants."""
    out = []
    if g and g.get("trim"):
        top, bottom = g["trim"]
        out.append(f'{indent}<adjust-crop mode="trim"><trim-rect top="{_fmt(top)}" bottom="{_fmt(bottom)}"/></adjust-crop>')
    if conform:
        out.append(f'{indent}<adjust-conform type="fill"/>')
    if g and (g.get("position") or g.get("scale")):
        out.append(f'{indent}<adjust-transform position="{g.get("position", "0 0")}" scale="{g.get("scale", "1 1")}"/>')
    return out


def event_name(name: str) -> str:
    """The Final Cut event a project lands in: the video's name without its version, so v1, v2... share one event (it said "Clipper")."""
    import re
    return re.sub(r"\s+v\d+$", "", str(name or "").strip()) or "FirstCut"


def build(name: str, source: Media, segments: list[tuple[float, float]], seq_w: int, seq_h: int, fps: Fraction,
          captions: list[tuple[float, float, str]] | None = None, music: MusicBed | None = None,
          hook: tuple[str, float] | None = None, style: CaptionStyle | None = None,
          source_colorspace: str = "1-1-1 (Rec. 709)", cutaways: list[tuple[float, float, float]] | None = None,
          titles: list[dict] | None = None, overlays: list[dict] | None = None, subs: list[tuple[float, float, str]] | None = None) -> str:
    """segments: kept (source_in, source_out) stretches in order. captions: (timeline_start, timeline_end, text) drawn as titles (only
    when the video has them burned in). subs: the same words as REAL captions, always (Final Cut shows them when its captions are on).
    cutaways: (timeline_start, source_in, length) B-roll pictures laid over the talking (audio muted), on lane 1;
    captions and the hook move up a lane to stay on top of them."""
    clk = Clock(fps)
    style = style or CaptionStyle()
    captions = captions or []
    cutaways = sorted(cutaways or [])
    titles = sorted(titles or [], key=lambda t: t["at"])
    cap_lane, hook_lane = (2, 3) if cutaways else (1, 2)
    subs = caption_spans(subs)
    src_fps = Clock(source.fps)
    out: list[str] = []
    A = out.append
    A('<?xml version="1.0" encoding="UTF-8"?>')
    A("<!DOCTYPE fcpxml>")
    A(f'<fcpxml version="{VERSION}">')
    A("  <resources>")
    A(f'    <format id="r1" frameDuration="{clk.at(1)}" width="{seq_w}" height="{seq_h}" colorSpace="1-1-1 (Rec. 709)"/>')
    A(f'    <format id="r2" frameDuration="{src_fps.at(1)}" width="{source.width}" height="{source.height}" '
      f'colorSpace="{source_colorspace}"/>')
    A(f'    <asset id="r3" name="{escape(Path(source.path).stem)}" start="0s" duration="{src_fps.t(source.duration)}" '
      f'hasVideo="{int(source.has_video)}" format="r2" videoSources="{int(source.has_video)}" '
      f'hasAudio="{int(source.has_audio)}" audioSources="{int(source.has_audio)}" '
      f'audioChannels="{source.audio_channels}" audioRate="{source.audio_rate}">')
    A(f'      <media-rep kind="original-media" src="{escape(file_url(source.path))}"/>')
    A("    </asset>")
    if music:
        A(f'    <asset id="r4" name="{escape(music.name)}" start="0s" duration="{clk.t(music.duration)}" hasAudio="1" '
          f'audioSources="1" audioChannels="2" audioRate="44100">')
        A(f'      <media-rep kind="original-media" src="{escape(file_url(music.path))}"/>')
        A("    </asset>")
    if captions or hook or titles:
        A(f'    <effect id="r5" name="Basic Title" uid="{BASIC_TITLE_UID}"/>')
    A("  </resources>")

    # timeline layout: frame-exact offsets so consecutive clips butt together
    clips, pos = [], 0
    for a, b in segments:
        start_f, len_f = clk.frames(a), max(1, clk.frames(b) - clk.frames(a))
        clips.append((pos, start_f, len_f))
        pos += len_f
    total_f = pos

    A("  <library>")
    A(f'    <event name="{escape(event_name(name))}">')
    A(f'      <project name="{escape(name)}">')
    A(f'        <sequence format="r1" duration="{clk.at(total_f)}" tcStart="0s" tcFormat="NDF" '
      f'audioLayout="stereo" audioRate="48k">')
    A("          <spine>")
    ts = 0
    for i, (off_f, start_f, len_f) in enumerate(clips):
        A(f'            <asset-clip ref="r3" offset="{clk.at(off_f)}" name="{escape(Path(source.path).stem)} {i + 1}" '
          f'start="{clk.at(start_f)}" duration="{clk.at(len_f)}" format="r2" tcFormat="NDF" audioRole="dialogue">')
        if seq_w != source.width or seq_h != source.height:
            A('              <adjust-conform type="fill"/>')
        if i == 0 and music:                          # connected under clip 1, running the whole timeline
            out.extend(_music_lines(clk, music, "r4", start_f, total_f))
        for (c_at, c_src, c_len) in cutaways:                 # B-roll that starts inside this clip rides on it (picture only)
            c_f = clk.frames(c_at)
            if not (off_f <= c_f < off_f + len_f):
                continue
            A(f'              <asset-clip ref="r3" lane="1" offset="{clk.at(start_f + (c_f - off_f))}" name="B-roll" '
              f'start="{clk.at(clk.frames(c_src))}" duration="{clk.at(max(1, clk.frames(c_len)))}" format="r2" tcFormat="NDF">')
            if seq_w != source.width or seq_h != source.height:
                A('                <adjust-conform type="fill"/>')
            A('                <adjust-volume amount="-96dB"/>')
            A("              </asset-clip>")
        if i == 0 and hook:
            text, secs = hook
            ts += 1
            A(f'              <title ref="r5" lane="{hook_lane}" offset="{clk.at(start_f)}" name="Hook title" '
              f'duration="{clk.t(min(secs, total_f * float(clk.frame)))}">')
            A(f'                <param name="Position" key="{TITLE_POSITION_KEY}" value="0 {_fmt(seq_h * 0.36)}"/>')
            A(f'                <text><text-style ref="ts{ts}">{escape(text)}</text-style></text>')
            A(f'                <text-style-def id="ts{ts}"><text-style font="Arial Black" fontSize="{_fmt(style.size * 0.8)}" '
              f'fontColor="0.04 0.04 0.04 1" backgroundColor="1 1 1 1" alignment="center"/></text-style-def>')
            A("              </title>")
        seg_a, seg_b = off_f, off_f + len_f
        for t in titles:                              # hook / key points / end card that START inside this clip ride on it
            t_f = clk.frames(t["at"])
            if seg_a <= t_f < seg_b:
                ts += 1
                out.extend(_title_lines(clk, "r5", hook_lane + (0 if t["kind"] == "hook" else 1), clk.at(start_f + (t_f - seg_a)),
                                        clk.t(t["len"]), t["text"], t["kind"], seq_w, seq_h, style, ts))
        for (c0, c1, text) in captions:               # captions that START inside this clip ride on it
            c0f, c1f = clk.frames(c0), clk.frames(c1)
            if not (seg_a <= c0f < seg_b):
                continue
            dur_f = max(1, min(c1f, seg_b + 2 * clk.frames(1.0)) - c0f)
            ts += 1
            words = text.upper() if style.uppercase else text
            A(f'              <title ref="r5" lane="{cap_lane}" offset="{clk.at(start_f + (c0f - seg_a))}" name="Caption" '
              f'duration="{clk.at(dur_f)}">')
            A(f'                <param name="Position" key="{TITLE_POSITION_KEY}" value="0 {_fmt(style.y * seq_h)}"/>')
            A(f'                <text><text-style ref="ts{ts}">{escape(words)}</text-style></text>')
            A(f'                <text-style-def id="ts{ts}"><text-style font="{escape(style.font)}" '
              f'fontSize="{_fmt(style.size)}" fontColor="{style.color}" bold="1" strokeColor="{style.stroke}" '
              f'strokeWidth="{_fmt(style.stroke_width)}" alignment="center"/></text-style-def>')
            A("              </title>")
        for (c0, c1, text) in subs:                   # real captions that START inside this clip ride on it
            c0f = clk.frames(c0)
            if seg_a <= c0f < seg_b:
                ts += 1
                out.extend(_caption_lines(hook_lane + 3, clk.at(start_f + (c0f - seg_a)), clk.at(max(1, clk.frames(c1) - c0f)), text, ts))
        A("            </asset-clip>")
    A("          </spine>")
    A("        </sequence>")
    A("      </project>")
    A("    </event>")
    A("  </library>")
    A("</fcpxml>")
    return "\n".join(out) + "\n"


def validate(xml_path: Path) -> list[str]:
    """Check against Final Cut Pro's own DTD when FCP is installed on this Mac (empty list = valid)."""
    import shutil
    import subprocess
    import tempfile
    dtd = DTD_DIR / f"FCPXMLv{VERSION.replace('.', '_')}.dtd"
    if not dtd.exists():                                  # (Final Cut installed under another name, e.g. "Final Cut Pro Creator Studio", or only Motion)
        import glob
        name = f"Contents/Frameworks/Interchange.framework/Versions/A/Resources/{dtd.name}"
        dtd = Path(next(iter(sorted(glob.glob(f"/Applications/Final Cut Pro*.app/{name}")) + sorted(glob.glob(f"/Applications/Motion*.app/{name}"))), str(dtd)))
    if not dtd.exists() or not shutil.which("xmllint"):
        return []
    with tempfile.TemporaryDirectory() as tmp:            # xmllint can't load a DTD from a path with spaces
        local = Path(tmp) / dtd.name
        shutil.copy2(dtd, local)
        r = subprocess.run(["xmllint", "--noout", "--dtdvalid", str(local), str(Path(xml_path).resolve())],
                           capture_output=True, text=True, cwd=tmp)
    return [ln for ln in r.stderr.splitlines() if ln.strip()] if r.returncode else []


def colorspace_of(video_stream: dict) -> str:
    """FCP colorSpace string for a source (iPhone HDR is HLG)."""
    trc = str(video_stream.get("color_transfer", ""))
    if trc == "arib-std-b67":
        return "9-18-9 (Rec. 2020 HLG)"
    if trc == "smpte2084":
        return "9-16-9 (Rec. 2020 PQ)"
    if str(video_stream.get("color_primaries", "")) == "bt2020":
        return "9-1-9 (Rec. 2020)"
    return "1-1-1 (Rec. 709)"


# ----------------------------------------------------------------------------------------- multicam
def _res_id(prefix: str, n: int) -> str:
    return f"{prefix}{n}"




def build_multicam(name: str, seq_w: int, seq_h: int, fps: Fraction, group: dict | None, items: list[dict],
                   captions: list[tuple[float, float, str]] | None = None, music: MusicBed | None = None,
                   hook: tuple[str, float] | None = None, style: CaptionStyle | None = None,
                   cutaways: list[dict] | None = None, source_colorspace: str = "1-1-1 (Rec. 709)", titles: list[dict] | None = None,
                   overlays: list[dict] | None = None, subs: list[tuple[float, float, str]] | None = None) -> str:
    """A Final Cut project built from the ORIGINAL files. With a multicam shoot (`group`) the main timeline is a REAL multicam clip:
    every camera is an angle of one multicam media, synced where the sound lined up, and each angle switch is an ordinary cut of that
    clip - so any switch can be changed afterwards with the angle viewer. Without one (group None), it is plain clips of the original
    files, one per kept stretch, each extendable. `group` = multicam.detect_group(...) (angles with files at "t0" on the shared
    timeline; recorders = audio only).

    items (the program, in order):
      {"kind": "mc", "out", "dur", "t": shared-timeline second where this piece starts, "video": angle id, "audio": angle id or "Recorder"}
      {"kind": "clip", "out", "dur", "path", "start": seconds into that file, "w", "h", "fps", "audio": bool, "tc", "tc_df",
       "aud": {"path", "start", "tc", "dur", "ch"} when a synced recording is the sound (the camera's own sound is then silenced)}
    cutaways: [{"at", "len", "path", "start", "w", "h", "fps", "layout"?}] picture-only B-roll over the program (lane 1; titles move up).
      layout "pip": the speaker's own picture comes back small, top right, over the B-roll (a connected clip of the same source, lane 2);
      "split" (vertical): the speaker on the top half, the B-roll on the bottom half. Both stay ordinary, editable clips.
    subs: the words as REAL captions (always; Final Cut shows them when View › Captions is on)."""
    clk = Clock(fps)
    style = style or CaptionStyle()
    captions = captions or []
    cutaways = sorted(cutaways or [], key=lambda c: c["at"])
    safe = []
    for c in cutaways:                                           # a B-roll shot must lie INSIDE its file, or Final Cut refuses it ("no respective media")
        fd = c.get("file_dur")
        if fd:
            c = {**c, "len": min(c["len"], fd - c["start"] - 0.05)}
        if c["len"] >= 0.3 and c["start"] >= 0:
            safe.append(c)
    cutaways = safe
    titles = sorted(titles or [], key=lambda t: t["at"])
    laid = any(c.get("layout") for c in cutaways)
    cap_lane, hook_lane = ((3, 4) if laid else (2, 3)) if cutaways else (1, 2)    # (a layout's speaker window is lane 2)
    subs = caption_spans(subs)
    angles = (group or {}).get("angles") or []
    recorders = (group or {}).get("recorders") or []
    tcs: dict[str, float] = {}                                   # each file's embedded start timecode (cameras that record time of day)
    clocks: dict[str, Clock] = {}                                # each file's own frame grid
    dfs: dict[str, bool] = {}
    out: list[str] = []
    A = out.append
    fmt_ids: dict[tuple, str] = {}
    assets: dict[str, str] = {}
    asset_lines: list[str] = []

    def fmt(w, h, fr) -> str:
        key = (int(w), int(h), std_rate(fr))
        if key not in fmt_ids:
            fmt_ids[key] = f"f{len(fmt_ids) + 2}"
        return fmt_ids[key]

    def tcf(path: str) -> str:
        return "DF" if dfs.get(path) else "NDF"

    def asset(path: str, dur: float, w: int, h: int, fr: float, has_video: bool = True, has_audio: bool = True, tc: float = 0.0,
              df: bool = False, channels: int = 2) -> str:
        if path in assets:
            return assets[path]
        aid = f"a{len(assets) + 1}"
        assets[path] = aid
        tcs[path] = float(tc or 0.0)
        dfs[path] = bool(df)
        sf = clocks[path] = Clock(std_rate(fr)) if (has_video and fr) else clk
        f = fmt(w, h, fr) if has_video else None
        fmt_attr = f' format="{f}"' if f else ""
        asset_lines.append(f'    <asset id="{aid}" name="{escape(Path(path).stem)}" start="{sf.t(tc or 0.0)}" duration="{sf.t(dur)}" '
                           f'hasVideo="{int(has_video)}"{fmt_attr} videoSources="{int(has_video)}" '
                           f'hasAudio="{int(has_audio)}" audioSources="{int(has_audio)}" audioChannels="{int(channels or 2)}" audioRate="48000">')
        asset_lines.append(f'      <media-rep kind="original-media" src="{escape(file_url(path))}"/>')
        asset_lines.append("    </asset>")
        return aid

    # the multicam media: one angle per camera (+ one for the recorder), every file at its place on the shared timeline
    mc_lines: list[str] = []

    def angle_clips(files: list[dict], clip_line) -> None:
        """The clips of ONE angle. An angle is a SERIAL container (Final Cut's own definition: "elements ordered serially in time"): its
        first clip starts at the angle's start whatever `offset` says, so a camera that began recording later is put in place by an explicit
        <gap> of that length - and so is any hole between two files of the same camera. Every position is whole frames."""
        cursor = 0
        for c in sorted(files, key=lambda c: c["t0"]):
            at = max(cursor, clk.frames(c["t0"]))                    # (two files of one camera can't overlap: a frame of rounding is closed up)
            if at > cursor:
                mc_lines.append(f'          <gap name="Gap" offset="{clk.at(cursor)}" start="0s" duration="{clk.at(at - cursor)}"/>')
            mc_lines.append(clip_line(c, clk.at(at)))
            cursor = at + max(1, clk.frames(c["dur"]))

    for ang in angles:
        mc_lines.append(f'        <mc-angle name="{escape(ang["id"])}" angleID="{escape(ang["id"])}">')

        def video_line(c, offset):
            aid = asset(c["path"], c["dur"], c["w"], c["h"], c["fps"], tc=c.get("tc", 0.0), df=c.get("tc_df", False))
            return (f'          <asset-clip ref="{aid}" offset="{offset}" name="{escape(Path(c["path"]).stem)}" '
                    f'start="{clocks[c["path"]].t(c.get("tc", 0.0))}" duration="{clk.t(c["dur"])}" format="{fmt(c["w"], c["h"], c["fps"])}" '
                    f'tcFormat="{tcf(c["path"])}"/>')
        angle_clips(ang["files"], video_line)
        mc_lines.append("        </mc-angle>")
    if recorders:
        mc_lines.append('        <mc-angle name="Recorder" angleID="Recorder">')

        def audio_line(r, offset):
            aid = asset(r["path"], r["dur"], 0, 0, 0, has_video=False, tc=r.get("tc", 0.0), channels=r.get("ch", 2))
            return (f'          <asset-clip ref="{aid}" offset="{offset}" name="{escape(Path(r["path"]).stem)}" '
                    f'start="{clk.t(r.get("tc", 0.0))}" duration="{clk.t(r["dur"])}" tcFormat="NDF"/>')
        angle_clips(recorders, audio_line)
        mc_lines.append("        </mc-angle>")
    # anything else used by the program
    for it in items:
        if it["kind"] == "clip":
            asset(it["path"], it.get("file_dur") or (it["start"] + it["dur"] + 1), it["w"], it["h"], it["fps"], has_audio=bool(it.get("audio", True)),
                  tc=it.get("tc", 0.0), df=it.get("tc_df", False))
            if it.get("aud"):
                au = it["aud"]
                asset(au["path"], au.get("dur") or (au["start"] + it["dur"] + 1), 0, 0, 0, has_video=False, tc=au.get("tc", 0.0),
                      channels=au.get("ch") or 2)
    for c in cutaways:
        asset(c["path"], c.get("file_dur") or (c["start"] + c["len"] + 1), c["w"], c["h"], c["fps"], has_audio=False, tc=c.get("tc", 0.0),
              df=c.get("tc_df", False))
    overlays = sorted(overlays or [], key=lambda o: o["at"])                    # slides / charts / quotes: transparent clips of their own length
    for o in overlays:
        asset(o["path"], o["len"] + 0.05, o["w"], o["h"], o["fps"], has_audio=False)

    total = sum(max(1, clk.frames(i["dur"])) for i in items)
    A('<?xml version="1.0" encoding="UTF-8"?>')
    A("<!DOCTYPE fcpxml>")
    A(f'<fcpxml version="{VERSION}">')
    A("  <resources>")
    A(f'    <format id="r1" frameDuration="{clk.at(1)}" width="{seq_w}" height="{seq_h}" colorSpace="1-1-1 (Rec. 709)"/>')
    for (w, h, fr), fid in fmt_ids.items():
        A(f'    <format id="{fid}" frameDuration="{Clock(fr).at(1)}" width="{w}" height="{h}" colorSpace="{source_colorspace}"/>')
    out.extend(asset_lines)
    if angles:
        A(f'    <media id="mc1" name="{escape(name)} multicam">')
        A('      <multicam format="r1" tcStart="0s" tcFormat="NDF">')
        out.extend(mc_lines)
        A("      </multicam>")
        A("    </media>")
    if music:
        A(f'    <asset id="am" name="{escape(music.name)}" start="0s" duration="{clk.t(music.duration)}" hasAudio="1" '
          f'audioSources="1" audioChannels="2" audioRate="44100">')
        A(f'      <media-rep kind="original-media" src="{escape(file_url(music.path))}"/>')
        A("    </asset>")
    if captions or hook or titles:
        A(f'    <effect id="rt" name="Basic Title" uid="{BASIC_TITLE_UID}"/>')
    A("  </resources>")
    A("  <library>")
    A(f'    <event name="{escape(event_name(name))}">')
    A(f'      <project name="{escape(name)}">')
    A(f'        <sequence format="r1" duration="{clk.at(total)}" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">')
    A("          <spine>")
    pos = 0
    ts = 0
    for idx, it in enumerate(items):
        d_f = max(1, clk.frames(it["dur"]))
        off_f = pos
        pos += d_f
        if it["kind"] == "mc" and angles:
            t_f = clk.frames(it["t"])
            A(f'            <mc-clip ref="mc1" offset="{clk.at(off_f)}" name="{escape(name)}" start="{clk.at(t_f)}" duration="{clk.at(d_f)}">')
            vid = it["video"]
            vang = next((a for a in angles if a["id"] == vid), None)
            ref = vang["files"][0] if vang and vang["files"] else None
            conform = bool(ref and (ref["w"] != seq_w or ref["h"] != seq_h))

            def video_source(enable):                               # Final Cut's own order and shape: audio source first, the video one carries the conform
                A(f'              <mc-source angleID="{escape(vid)}" srcEnable="{enable}">')
                if conform:
                    A('                <adjust-conform type="fill"/>')
                if it.get("zoom"):                                  # a punch-in that hides a jump cut (the picture's own transform lives in the angle's source)
                    A(f'                <adjust-transform scale="{_fmt(it["zoom"])} {_fmt(it["zoom"])}"/>')
                A("              </mc-source>")
            if it["audio"] == vid:
                video_source("all")
            else:
                A(f'              <mc-source angleID="{escape(it["audio"])}" srcEnable="audio"/>')
                video_source("video")
            start_inner = t_f
        else:
            aid = assets[it["path"]]
            st_f = clk.frames(it["start"] + tcs.get(it["path"], 0.0))
            A(f'            <asset-clip ref="{aid}" offset="{clk.at(off_f)}" name="{escape(Path(it["path"]).stem)}" start="{clk.at(st_f)}" '
              f'duration="{clk.at(d_f)}" format="{fmt(it["w"], it["h"], it["fps"])}" tcFormat="{tcf(it["path"])}">')
            if it["w"] != seq_w or it["h"] != seq_h:
                A('              <adjust-conform type="fill"/>')
            if it.get("zoom"):
                A(f'              <adjust-transform scale="{_fmt(it["zoom"])} {_fmt(it["zoom"])}"/>')
            if it.get("aud") and it.get("audio", True):
                A('              <adjust-volume amount="-96dB"/>')                 # the synced recording below is the sound
            start_inner = st_f
        if idx == 0 and music:
            out.extend(_music_lines(clk, music, "am", start_inner, total))
        if it["kind"] == "clip" and it.get("aud"):                              # the clean recording, in sync, under this clip
            au = it["aud"]
            A(f'              <asset-clip ref="{assets[au["path"]]}" lane="-3" offset="{clk.at(start_inner)}" name="{escape(Path(au["path"]).stem)}" '
              f'start="{clk.t(au["start"] + float(au.get("tc") or 0.0))}" duration="{clk.at(d_f)}" audioRole="dialogue"/>')
        if idx == 0 and hook:
            text, secs = hook
            ts += 1
            A(f'              <title ref="rt" lane="{hook_lane}" offset="{clk.at(start_inner)}" name="Hook title" '
              f'duration="{clk.t(min(secs, total * float(clk.frame)))}">')
            A(f'                <param name="Position" key="{TITLE_POSITION_KEY}" value="0 {_fmt(seq_h * 0.36)}"/>')
            A(f'                <text><text-style ref="ts{ts}">{escape(text)}</text-style></text>')
            A(f'                <text-style-def id="ts{ts}"><text-style font="Arial Black" fontSize="{_fmt(style.size * 0.8)}" '
              f'fontColor="0.04 0.04 0.04 1" backgroundColor="1 1 1 1" alignment="center"/></text-style-def>')
            A("              </title>")
        for c in cutaways:                                      # picture-only B-roll that starts inside this piece
            c_f = clk.frames(c["at"])
            if off_f <= c_f < off_f + d_f:
                g = layout_geometry(c.get("layout") or "", seq_w, seq_h)
                A(f'              <asset-clip ref="{assets[c["path"]]}" lane="1" offset="{clk.at(start_inner + (c_f - off_f))}" name="B-roll" '
                  f'start="{clk.at(clk.frames(c["start"] + tcs.get(c["path"], 0.0)))}" duration="{clk.at(max(1, clk.frames(c["len"])))}" '
                  f'format="{fmt(c["w"], c["h"], c["fps"])}" tcFormat="{tcf(c["path"])}">')
                out.extend(_adjust_lines(g["broll"], c["w"] != seq_w or c["h"] != seq_h, "                "))
                A('                <adjust-volume amount="-96dB"/>')
                A("              </asset-clip>")
        for c in cutaways:                                      # a layout: the speaker's own picture over the B-roll, for the part of
            if not c.get("layout"):                             #  the shot inside this piece (each piece is its own moment of the speaker)
                continue
            g = layout_geometry(c["layout"], seq_w, seq_h)["inset"]
            a_f, b_f = max(off_f, clk.frames(c["at"])), min(off_f + d_f, clk.frames(c["at"]) + max(1, clk.frames(c["len"])))
            if b_f - a_f < 1 or not g:
                continue
            at_s, dur_s = clk.at(start_inner + (a_f - off_f)), clk.at(b_f - a_f)
            if it["kind"] == "mc" and angles:
                A(f'              <mc-clip ref="mc1" lane="2" offset="{at_s}" name="Speaker ({c["layout"]})" start="{at_s}" duration="{dur_s}">')
                A(f'                <mc-source angleID="{escape(it["video"])}" srcEnable="video">')
                out.extend(_adjust_lines(g, bool(ref and (ref["w"] != seq_w or ref["h"] != seq_h)), "                  "))
                A("                </mc-source>")
                A("              </mc-clip>")
            elif it["kind"] != "mc":
                A(f'              <asset-clip ref="{assets[it["path"]]}" lane="2" offset="{at_s}" name="Speaker ({c["layout"]})" start="{at_s}" '
                  f'duration="{dur_s}" format="{fmt(it["w"], it["h"], it["fps"])}" tcFormat="{tcf(it["path"])}">')
                out.extend(_adjust_lines(g, it["w"] != seq_w or it["h"] != seq_h, "                "))
                A('                <adjust-volume amount="-96dB"/>')
                A("              </asset-clip>")
        for o in overlays:                                      # a graphic card over the picture (above the titles)
            o_f = clk.frames(o["at"])
            if off_f <= o_f < off_f + d_f:
                A(f'              <asset-clip ref="{assets[o["path"]]}" lane="{hook_lane + 2}" offset="{clk.at(start_inner + (o_f - off_f))}" name="Graphic" '
                  f'start="0s" duration="{clk.at(max(1, clk.frames(o["len"])))}" format="{fmt(o["w"], o["h"], o["fps"])}" tcFormat="NDF">')
                if o["w"] != seq_w or o["h"] != seq_h:
                    A('                <adjust-conform type="fill"/>')
                A("              </asset-clip>")
        for t in titles:                                        # hook / key points / end card
            t_f = clk.frames(t["at"])
            if off_f <= t_f < off_f + d_f:
                ts += 1
                out.extend(_title_lines(clk, "rt", hook_lane + (0 if t["kind"] == "hook" else 1), clk.at(start_inner + (t_f - off_f)),
                                        clk.t(t["len"]), t["text"], t["kind"], seq_w, seq_h, style, ts))
        for (c0, c1, text) in captions:
            c0f, c1f = clk.frames(c0), clk.frames(c1)
            if not (off_f <= c0f < off_f + d_f):
                continue
            dur_f = max(1, min(c1f, off_f + d_f + 2 * clk.frames(1.0)) - c0f)
            ts += 1
            words = text.upper() if style.uppercase else text
            A(f'              <title ref="rt" lane="{cap_lane}" offset="{clk.at(start_inner + (c0f - off_f))}" name="Caption" duration="{clk.at(dur_f)}">')
            A(f'                <param name="Position" key="{TITLE_POSITION_KEY}" value="0 {_fmt(style.y * seq_h)}"/>')
            A(f'                <text><text-style ref="ts{ts}">{escape(words)}</text-style></text>')
            A(f'                <text-style-def id="ts{ts}"><text-style font="{escape(style.font)}" fontSize="{_fmt(style.size)}" fontColor="{style.color}" '
              f'bold="1" strokeColor="{style.stroke}" strokeWidth="{_fmt(style.stroke_width)}" alignment="center"/></text-style-def>')
            A("              </title>")
        for (c0, c1, text) in subs:                             # real captions that START inside this piece ride on it
            c0f = clk.frames(c0)
            if off_f <= c0f < off_f + d_f:
                ts += 1
                out.extend(_caption_lines(hook_lane + 3, clk.at(start_inner + (c0f - off_f)), clk.at(max(1, clk.frames(c1) - c0f)), text, ts))
        A("            </mc-clip>" if it["kind"] == "mc" and angles else "            </asset-clip>")
    A("          </spine>")
    A("        </sequence>")
    A("      </project>")
    A("    </event>")
    A("  </library>")
    A("</fcpxml>")
    return "\n".join(out) + "\n"
