"""Transitions: where a good editor would use one, decided from the pieces of the cut (pure functions, no ffmpeg here).

The engine's cuts are hard cuts and stay hard cuts. A transition is used only where it says something:
  * fade in from black at the very start (not when a title card opens the video: the card is the opening),
  * fade out to black at the very end,
  * a short cross dissolve where the story jumps a long way (another place / another recording / far apart in the same one),
  * a B-roll shot of 4 s or more dissolves in and out over the speaker (the one-sided pair counts as ONE transition).
Never inside a continuous talking passage, never on a jump cut that tightens speech, never two within 12 s of each other, none in an
advertisement or promo, none in a video under 30 s, none in a video that must stay exactly as it is ("Add B-roll to a video").

Only things Final Cut can hold as ordinary, editable parts of the project are written (fcpxml.py):
  * "dissolve": a stock Cross Dissolve <transition> on the spine, centred on the cut, with Final Cut's own Audio Crossfade,
  * "broll": the B-roll clip is extended half a transition each side and its opacity ramps in and out (a dissolve over the speaker),
  * "fade_in" / "fade_out": opacity ramps over black on the first / last clip.
The MP4 shows the same transition of the same length at the same place (localproj.assemble and polish.run use render_parts()/the fade list).

A dissolve centred on a cut needs "handles": media past the end of the piece before it and before the start of the piece after it. The pieces
come from long recordings, so the handles are real footage; a boundary without them (a clip that starts at second 0, B-roll at the end of its
file) simply gets no transition.
"""
from __future__ import annotations

import copy
from fractions import Fraction

DISSOLVE_S = 0.5            # a cross dissolve (centred on the cut)
FADE_IN_S = 0.6
FADE_OUT_S = 1.0
MIN_SPACING_S = 12.0        # at most one transition event per this many seconds
JUMP_S = 45.0               # the story jumps this far (seconds of source) or to another recording: a topic / place change
BROLL_MIN_S = 4.0           # a B-roll shot this long or longer dissolves in and out
MIN_PIECE_S = 1.2           # a piece must keep this much of itself after a dissolve takes half a transition off each end
MIN_VIDEO_S = 30.0
EDGE_S = 3.0                # no dissolve this close to the fades at the ends
QUIET_STYLES = ("ad", "promo")
MODES = ("auto", "none")        # (and "full": cross dissolves too, for whoever sets it by hand)


def mode_of(value) -> str:
    """The option as 'auto' or 'none' (anything else the person never chose is automatic)."""
    v = str(value or "auto").lower()
    return "none" if v in ("none", "off", "no", "false", "0") else "full" if v == "full" else "auto"      # ("full" = with cross dissolves: not offered in the page)


def half_frames(rate) -> int:
    """Half a dissolve in frames: a dissolve is always 2 * this many frames, centred on the cut."""
    return max(1, round(DISSOLVE_S / 2 * float(rate)))


def _cut(p: dict) -> bool:
    return bool((p.get("v") or {}).get("cut"))


def _layout(p: dict) -> bool:
    return bool((p.get("v") or {}).get("layout")) or bool(p.get("v0") and (p.get("v") or {}).get("layout"))


def joined(a: dict, b: dict) -> bool:
    """True when piece b carries on the SAME sound as piece a (the same test as localproj.joined)."""
    x, y = a.get("a"), b.get("a")
    return bool(x and y and x["path"] == y["path"] and abs(x["ss"] + a["d"] - y["ss"]) < 0.05)


def _room_after(p: dict, secs: float, durs: dict) -> bool:
    """Is there `secs` more picture after the end of this piece (its handle for the dissolve)?"""
    v = p.get("v") or {}
    dur = durs.get(v.get("path"))
    if dur is None:
        dur = (p.get("item") or {}).get("file_dur")
    return dur is None or float(v.get("ss") or 0.0) + float(p["d"]) + secs <= float(dur) - 0.05


def _room_before(p: dict, secs: float) -> bool:
    return float((p.get("v") or {}).get("ss") or 0.0) - secs >= 0.0


def _boundary_ok(a: dict, b: dict, h: int, rate, durs: dict) -> bool:
    """Both pieces long enough to give half a dissolve, no layout, a picture on both sides, and real media for the handles."""
    keep = max(2, round(MIN_PIECE_S * float(rate)))
    if int(a.get("n") or 0) < h + keep or int(b.get("n") or 0) < h + keep:
        return False
    if _layout(a) or _layout(b) or (a.get("v") or {}).get("nopic") or (b.get("v") or {}).get("nopic"):
        return False
    secs = h / float(rate)
    return _room_after(a, secs, durs) and _room_before(b, secs)


def plan(parts: list[dict], rate, *, style: str = "story", mode: str = "auto", hook_at_start: bool = False,
         durs: dict | None = None, topics=(), outro: bool = False, dissolves: bool = False) -> list[dict]:
    """The transitions of this cut, in time order. `parts` = localproj's pieces after quantize(): each with d (seconds), n (frames), out (timeline
    second), v (picture: path, ss, cut for B-roll, layout), a (sound: path, ss), item. `topics` = timeline seconds where the story changes subject
    (optional). Returns dicts: kind dissolve | broll | fade_in | fade_out, at (timeline second), n / dur (frames / seconds of the whole
    transition), plus k (index of the piece AFTER the cut) for the part-level ones.
    The user almost never wants transitions: by default (dissolves=False) the only ones are a fade out at the end and, when the video opens on
    a picture rather than on someone already talking, a fade in at the start. Cross dissolves are off unless asked for (dissolves=True)."""
    if mode_of(mode) == "none" or not parts:
        return []
    dissolves = dissolves or mode_of(mode) == "full"
    total = float(parts[-1]["out"]) + float(parts[-1]["d"])
    if str(style or "story") in QUIET_STYLES or total < MIN_VIDEO_S:
        return []
    durs = durs or {}
    h = half_frames(rate)
    n = 2 * h
    d = float(Fraction(n) / Fraction(rate).limit_denominator(1001))
    topics = [float(t) for t in (topics or ())]
    lo, hi = FADE_IN_S + EDGE_S, total - FADE_OUT_S - EDGE_S
    events: list[dict] = []
    # B-roll shots: a run of consecutive B-roll pieces of one file is one shot
    k = 0
    while dissolves and k < len(parts):
        if not _cut(parts[k]):
            k += 1
            continue
        j = k
        while (j + 1 < len(parts) and _cut(parts[j + 1]) and parts[j + 1]["v"]["path"] == parts[k]["v"]["path"]
               and abs(float(parts[j]["v"]["ss"]) + float(parts[j]["d"]) - float(parts[j + 1]["v"]["ss"])) < 0.05):
            j += 1
        shot = sum(float(parts[i]["d"]) for i in range(k, j + 1))
        at_in, at_out = float(parts[k]["out"]), float(parts[j]["out"]) + float(parts[j]["d"])
        if (shot >= BROLL_MIN_S and k > 0 and j + 1 < len(parts) and not _cut(parts[k - 1]) and not _cut(parts[j + 1])
                and lo <= at_in and at_out <= hi
                and _boundary_ok(parts[k - 1], parts[k], h, rate, durs) and _boundary_ok(parts[j], parts[j + 1], h, rate, durs)):
            events.append({"kind": "broll", "k": k, "k_out": j + 1, "at": at_in, "at_out": at_out, "shot_len": round(shot, 3),
                           "n": n, "dur": d, "audio": "keep", "prio": 1, "why": f"a {shot:.0f} s B-roll shot dissolves in and out"})
        k = j + 1
    # the story jumps: another recording, or far apart in the same one
    for k in range(1, len(parts) if dissolves else 0):
        a, b = parts[k - 1], parts[k]
        t = float(b["out"])
        if _cut(a) or _cut(b) or not (lo <= t <= hi) or joined(a, b):
            continue
        sa, sb = a.get("a"), b.get("a")
        far = ((a["v"].get("path") != b["v"].get("path"))
               or (sa and sb and abs(float(sa["ss"]) + float(a["d"]) - float(sb["ss"])) >= JUMP_S)
               or (not (sa and sb) and abs(float(a["v"]["ss"]) + float(a["d"]) - float(b["v"]["ss"])) >= JUMP_S))
        topic = any(abs(t - x) < 0.3 for x in topics)
        if not (far or topic):
            continue
        if outro and k == len(parts) - 1:                       # (the closing card starts at its first frame: no media before it to dissolve from)
            continue
        if _boundary_ok(a, b, h, rate, durs):
            events.append({"kind": "dissolve", "k": k, "at": t, "n": n, "dur": d, "audio": "crossfade", "prio": 0,
                           "why": "the story changes subject" if topic and not far else "the story jumps to another place or time"})
    # spacing: topic jumps first, then B-roll; one event per MIN_SPACING_S (a B-roll pair is one event spanning its shot)
    chosen: list[dict] = []
    for ev in sorted(events, key=lambda e: (e["prio"], e["at"])):
        a0, a1 = ev["at"], ev.get("at_out", ev["at"])
        if all(a0 - MIN_SPACING_S >= (c.get("at_out", c["at"])) or a1 + MIN_SPACING_S <= c["at"] for c in chosen):
            chosen.append(ev)
    out = sorted(chosen, key=lambda e: e["at"])
    for ev in out:
        ev.pop("prio", None)
    # the ends: a fade out; a fade in only when the video opens on a picture (B-roll), not on a title card and not on someone already talking
    nf_out = max(2, round(FADE_OUT_S * float(rate)))
    fps = Fraction(rate).limit_denominator(1001)
    if not hook_at_start and _cut(parts[0]):
        nf_in = max(2, round(FADE_IN_S * float(rate)))
        out.insert(0, {"kind": "fade_in", "at": 0.0, "n": nf_in, "dur": float(Fraction(nf_in) / fps), "why": "opens from black"})
    out.append({"kind": "fade_out", "at": total - float(Fraction(nf_out) / fps), "n": nf_out, "dur": float(Fraction(nf_out) / fps), "why": "ends to black"})
    return out


# ------------------------------------------------------------------------------------------------------ the MP4 side
def _shift(src, delta: float):
    """A picture / sound source moved by `delta` seconds (its clean copy along with it)."""
    if not isinstance(src, dict):
        return src
    out = dict(src)
    if "ss" in out:
        out["ss"] = float(out["ss"]) + delta
    if out.get("clean_ss") is not None:
        out["clean_ss"] = float(out["clean_ss"]) + delta
    return out


def _moved(p: dict, delta: float) -> dict:
    q = copy.copy(p)
    for key in ("v", "a", "v0"):
        if p.get(key):
            q[key] = _shift(p[key], delta)
    return q


def render_parts(parts: list[dict], events: list[dict], rate) -> list[dict]:
    """The pieces the MP4 is cut from: every dissolve takes half a transition off the end of the piece before and the start of the piece after,
    and a transition piece {"xf": {"a", "b", "audio"}} (two pieces of 2h frames, each carrying on past its edge: the handles) stands in the
    middle. The total length is the same; the picture is blended by localproj.assemble. B-roll pairs make two of them (in and out)."""
    rate_f = Fraction(rate).limit_denominator(1001)
    at: dict[int, dict] = {}
    for ev in events:
        if ev["kind"] == "dissolve":
            at[ev["k"]] = ev
        elif ev["kind"] == "broll":
            at[ev["k"]] = ev
            at[ev["k_out"]] = ev
    out: list[dict] = []
    for i, p in enumerate(parts):
        q = dict(p)
        ev_in, ev_next = at.get(i), at.get(i + 1)            # (a boundary before this piece trims its start, one after trims its end)
        h_in = ev_in["n"] // 2 if ev_in else 0
        h_out = ev_next["n"] // 2 if ev_next else 0
        if ev_in:
            q = _moved(q, h_in / float(rate_f))
            q["fade_in"] = False
        if ev_next:
            q["fade_out"] = False
        q["n"] = int(p["n"]) - h_in - h_out
        q["d"] = float(Fraction(q["n"]) / rate_f)
        if ev_in:                                            # the transition between the piece before and this one
            a, b = parts[i - 1], p
            h = h_in
            tail = _moved(a, (int(a["n"]) - h) / float(rate_f))
            head = _moved(b, -h / float(rate_f))
            for x in (tail, head):
                x["n"] = 2 * h
                x["d"] = float(Fraction(2 * h) / rate_f)
                x["fade_in"] = x["fade_out"] = False
            same = joined(a, b)
            out.append({"xf": {"a": tail, "b": head, "audio": "keep" if same else ev_in.get("audio", "crossfade")}, "n": 2 * h,
                        "d": float(Fraction(2 * h) / rate_f), "v": head["v"], "a": None, "item": b.get("item")})
        out.append(q)
    return out


def fade_list(events: list[dict]) -> list[dict]:
    """The fades at the ends, as the final render (polish.run) applies them to the whole picture."""
    return [{"kind": e["kind"], "at": round(float(e["at"]), 4), "dur": round(float(e["dur"]), 4)} for e in events if e["kind"] in ("fade_in", "fade_out")]


def opacity_at(events: list[dict], t: float) -> float:
    """The picture's opacity at timeline second t, from the fades (1.0 in the middle of the video)."""
    v = 1.0
    for e in events:
        if e["kind"] == "fade_in" and t < e["at"] + e["dur"]:
            v = min(v, max(0.0, (t - e["at"]) / e["dur"]))
        elif e["kind"] == "fade_out" and t > e["at"]:
            v = min(v, max(0.0, 1.0 - (t - e["at"]) / e["dur"]))
    return v
