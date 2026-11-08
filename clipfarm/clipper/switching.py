"""Multicam switching rules that hold whatever the director model says (WP7).

The director (a vision model) proposes which angle to show; local models ignore instructions, so the rules are code:
  - an angle changes only at a switch point: the end of a sentence, a change of speaker, or the end of a thought (a sentence end followed by a pause)
  - never strictly inside a word
  - every angle is held at least `min_hold_s`
  - an angle that does not show the speaker (face-on-camera ratio under 0.3) is not used for that stretch unless it is the wide shot
  - a reaction cut to the listener is short (1.5-3 s) and only during a pause
Pure functions on word dicts ({"text", "start", "end", "who"?}).
"""

from __future__ import annotations

import re

SENT_END = re.compile(r"[.!?][\"')\]]*$")
THOUGHT_PAUSE = 0.6
ON_CAMERA = 0.3                    # an angle shows the speaker when the speaker's face is on it in at least this share of samples
REACTION_S = (1.5, 3.0)
REACTION_PAUSE = 0.5
SNAP_TOLERANCE = 5.0               # the director's switch time moves to the nearest allowed point within this many seconds, else it is dropped


def switch_points(words: list[dict], angle_ratio: dict[str, float] | None = None, wide: set[str] | None = None) -> list[dict]:
    """Where an angle may change: [{"t", "why": sentence_end|speaker_change|thought_end, "allowed": [angle ids] | None (all)}], in time order.
    angle_ratio: {angle id: share of samples in which the speaker's face is on that angle}; `wide` angles are always allowed."""
    pts: list[dict] = []
    ws = [w for w in words if isinstance(w, dict) and "start" in w and "end" in w]
    allowed = None
    if angle_ratio:
        allowed = sorted({a for a, r in angle_ratio.items() if r >= ON_CAMERA} | set(wide or ()))
        allowed = allowed or None
    for i, w in enumerate(ws):
        nxt = ws[i + 1] if i + 1 < len(ws) else None
        ends = bool(SENT_END.search(str(w.get("text") or "")))
        change = bool(nxt and w.get("who") and nxt.get("who") and w["who"] != nxt["who"])
        if not (ends or change):
            continue
        gap = (nxt["start"] - w["end"]) if nxt else 9.0
        t = round(float(w["end"]) + min(0.15, max(0.0, gap) / 2), 3)            # in the gap after the word, never inside the next one
        why = "speaker_change" if change else ("thought_end" if ends and gap >= THOUGHT_PAUSE else "sentence_end")
        pts.append({"t": t, "why": why, "allowed": allowed})
    return pts


def inside_a_word(t: float, words: list[dict], margin: float = 0.02) -> bool:
    return any(isinstance(w, dict) and w.get("start", 0) + margin < t < w.get("end", 0) - margin for w in words or [])


def snap(t: float, points: list[dict], tol: float = SNAP_TOLERANCE) -> dict | None:
    """The nearest switch point to a requested time, within `tol` seconds."""
    best = min(points, key=lambda p: abs(p["t"] - t), default=None)
    return best if best is not None and abs(best["t"] - t) <= tol else None


def apply(segs: list[dict], points: list[dict], words: list[dict], min_hold: float, stretches: list[tuple[float, float]] | None = None) -> list[dict]:
    """Make the director's segments obey the rules. segs: [{"start","end","angle"}] contiguous within each stretch. Every internal boundary moves to the
    nearest allowed switch point (or the segment merges into its neighbour); an angle not allowed at its start falls back to the previous one; holds
    shorter than `min_hold` merge into the previous segment. The first start and last end of each run never move."""
    if not segs:
        return segs
    out: list[dict] = []
    for s in sorted(segs, key=lambda x: x["start"]):
        s = dict(s)
        gap_to_prev = out and abs(out[-1]["end"] - s["start"]) < 0.05
        if not out or not gap_to_prev:                                         # a new run (after a gap between stretches)
            out.append(s)
            continue
        if out[-1]["angle"] == s["angle"]:
            out[-1]["end"] = s["end"]
            continue
        p = snap(s["start"], points) if points else {"t": s["start"], "allowed": None}
        if p is None or inside_a_word(p["t"], words) or p["t"] <= out[-1]["start"] + 1e-6 or p["t"] >= s["end"] - 1e-6:
            out[-1]["end"] = s["end"]                                          # no allowed place to switch: stay on the angle that is up
            continue
        if p.get("allowed") and s["angle"] not in p["allowed"]:
            out[-1]["end"] = s["end"]                                          # that angle does not show the speaker here
            continue
        out[-1]["end"] = p["t"]
        s["start"] = p["t"]
        out.append(s)
    i = 1
    while i < len(out):                                                        # holds
        a, b = out[i - 1], out[i]
        contiguous = abs(a["end"] - b["start"]) < 0.05
        if contiguous and b["end"] - b["start"] < min_hold:
            a["end"] = b["end"]
            out.pop(i)
            continue
        i += 1
    merged: list[dict] = []
    for s in out:
        if merged and merged[-1]["angle"] == s["angle"] and abs(merged[-1]["end"] - s["start"]) < 0.05:
            merged[-1]["end"] = s["end"]
        else:
            merged.append(s)
    return merged


def reaction_cuts(words: list[dict], listener_energy: list[tuple[float, float]], listener_angle: str, speaker_angle: str,
                  min_hold: float = 5.0, last_switch: float = -99.0) -> list[dict]:
    """Short cuts to the listener: during a pause of at least 0.5 s where the listener's face moves more than its own median in the 1 s around it.
    -> [{"start", "end", "angle", "reaction": True}] of 1.5-3 s, never closer than `min_hold` to another switch or to each other."""
    if not listener_energy:
        return []
    es = sorted(e for _, e in listener_energy)
    med = es[len(es) // 2]
    out: list[dict] = []
    ws = [w for w in words if isinstance(w, dict) and "start" in w and "end" in w]
    for a, b in zip(ws, ws[1:]):
        pause = b["start"] - a["end"]
        if pause < REACTION_PAUSE:
            continue
        t = a["end"] + 0.05
        near = [e for tt, e in listener_energy if t - 0.5 <= tt <= t + 1.0]
        if not near or max(near) <= max(med * 1.5, med + 1e-6):
            continue
        length = min(REACTION_S[1], max(REACTION_S[0], pause + 0.8))
        if t - max(last_switch, out[-1]["end"] if out else -99.0) < min_hold:
            continue
        out.append({"start": round(t, 2), "end": round(t + length, 2), "angle": listener_angle, "back_to": speaker_angle, "reaction": True})
    return out


def wide_return(points: list[dict], step_starts: list[float], wide_angle: str, covered_after: list[float]) -> list[dict]:
    """The wide as the return shot: at each outline step start and after any B-roll (seminar): -> [{"t", "angle"}] snapped to switch points."""
    out = []
    for t in sorted(set(step_starts) | set(covered_after)):
        p = snap(t, points, 3.0)
        out.append({"t": p["t"] if p else t, "angle": wide_angle})
    return out


def insert_reactions(cuts: list[dict], reactions: list[dict]) -> list[dict]:
    """Put short listener cuts into the director's list: each reaction splits the segment it falls in (speaker, listener, speaker again). A reaction
    that would leave less than a second of the speaker's angle on either side, or that runs over a segment edge, is left out."""
    out = [dict(c) for c in cuts]
    for r in sorted(reactions, key=lambda x: x["start"]):
        i = next((k for k, c in enumerate(out) if c["start"] + 1.0 <= r["start"] and r["end"] <= c["end"] - 1.0 and c["angle"] != r["angle"]), None)
        if i is None:
            continue
        c = out[i]
        tail = {"start": r["end"], "end": c["end"], "angle": c["angle"]}
        c["end"] = r["start"]
        out[i:i + 1] = [c, {"start": r["start"], "end": r["end"], "angle": r["angle"], "reaction": True}, tail]
    return out
