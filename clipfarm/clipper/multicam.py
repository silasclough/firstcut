"""Multicam: several cameras (and maybe a recorder) filming the same thing, lined up by SOUND.

Clocks are not trusted. Every camera heard the same room, so the loudness pattern of the speech is the same on all of them even if
the microphones sound nothing alike; that pattern is matched between files wherever they overlap, which gives each file's place on
ONE shared timeline (the "master timeline", second 0 = the earliest moment any file covers).

    detect angles     which files are the same camera? folder / file names ("Cam A", "Angle 2"), else the camera's own signature
    place_files       sync every file with sound onto the master timeline (overlap of 10 s or more is enough)
    build_group       the placed files -> angles (cameras) + the best sound sources, as a "group" the editor treats like one long file
    choose_angles     the AI director: which angle to show when (only switching when it helps, never too quickly)
    cut_list          the chosen angles for the kept stretches, as a flat list the renderer and the Final Cut export both use

The Final Cut export (fcpxml_multicam in fcpxml.py) writes a real multicam clip, so you can change any angle switch afterwards.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..core.log import get_logger
from . import audiosync as AS

log = get_logger("clipper.multicam")
HEAD_S = 600.0                       # how much of each file's start is compared (a constant offset per file; drift over an hour is ~0.2 s)
MIN_OVERLAP_S = 10.0
MIN_Z = 7.0
SHORT_CLIP_S = 45.0                  # a matched clip shorter than this is B-roll that happened to hear the room, not an angle
FRAME = AS.HOP / AS.SR               # 10 ms

ANGLE_RE = re.compile(r"(?:^|[\s_\-./(\[])(?:cam(?:era)?|angle|cam-?)\s*[_\-.]?\s*([a-z0-9]{1,2})(?=$|[\s_\-./)\]])|"
                      r"(?:^|[\s_\-./(\[])([a-z0-9])\s*[_\-]?\s*cam(?:era)?(?=$|[\s_\-./)\]])", re.I)


# ----------------------------------------------------------------------------------------- which camera is which
def angle_label(rel: str) -> str | None:
    """'Interview/Cam A/C0001.MP4' -> 'A'. Looks at folder names first, then the file name."""
    parts = rel.replace("\\", "/").split("/")
    for part in list(reversed(parts[:-1])) + [parts[-1]]:
        m = ANGLE_RE.search(part)
        if m:
            return (m.group(1) or m.group(2)).upper()
    return None


def signature(f: dict) -> tuple:
    """What a camera stamps on its files: the letters of the file name, the picture size, the frame rate."""
    prefix = re.sub(r"[^A-Za-z]+", "", Path(f["name"]).stem)[:6].upper()
    family = (((f.get("facts") or {}).get("camera") or {}).get("family")) or ""          # (a GoPro and a Sony that both wrote 1080p25 files are two cameras)
    return (prefix, f.get("w"), f.get("h"), round(float(f.get("fps") or 0)), family)


def assign_angles(files: list[dict]) -> dict[str, str]:
    """{file name: angle id} for the video files. Explicit labels (folder / name) win; the rest are grouped by camera signature."""
    out: dict[str, str] = {}
    sigs: dict[tuple, str] = {}
    n = 0
    for f in files:
        lab = angle_label(f.get("rel") or f["name"])
        if lab:
            out[f["name"]] = f"Cam {lab}"
    for f in files:
        if f["name"] in out:
            continue
        s = signature(f)
        if s not in sigs:
            n += 1
            sigs[s] = f"Camera {n}"
        out[f["name"]] = sigs[s]
    return out


# ----------------------------------------------------------------------------------------- partial-overlap matching
def corr_partial(env_a: np.ndarray, env_v: np.ndarray, min_overlap: int = int(MIN_OVERLAP_S / FRAME)) -> tuple[int, float, int] | None:
    """Best place of clip `v` on timeline `a`, overlap allowed to be partial (v may start before a or end after it).
    -> (lag in frames: v[0] sits at a[lag], possibly negative; z-score; overlap frames)."""
    la, lv = len(env_a), len(env_v)
    if la < min_overlap or lv < min_overlap:
        return None
    n = 1 << int(np.ceil(np.log2(la + lv)))
    c = np.fft.irfft(np.fft.rfft(env_a, n) * np.conj(np.fft.rfft(env_v, n)), n)
    lags = np.arange(-(lv - 1), la)
    vals = np.concatenate([c[n - (lv - 1):], c[:la]])
    a_lo, a_hi = np.maximum(0, lags), np.minimum(la, lags + lv)
    v_lo, v_hi = np.maximum(0, -lags), np.minimum(lv, la - lags)
    sqa = np.concatenate([[0.0], np.cumsum(env_a ** 2)])
    sqv = np.concatenate([[0.0], np.cumsum(env_v ** 2)])
    overlap = a_hi - a_lo
    ok = overlap >= min_overlap
    if not ok.any():
        return None
    ea = np.sqrt(np.maximum(sqa[a_hi] - sqa[a_lo], 1e-9))
    ev = np.sqrt(np.maximum(sqv[v_hi] - sqv[v_lo], 1e-9))
    ncc = np.where(ok, vals / (ea * ev), -1.0)
    k = int(np.argmax(ncc))
    valid = ncc[ok]
    rest = np.delete(ncc, slice(max(0, k - 40), k + 40))
    rest = rest[rest > -1.0]
    if len(rest) < 20:
        return None
    z = float((ncc[k] - rest.mean()) / (rest.std() + 1e-9))
    return int(lags[k]), z, int(overlap[k])


def corr_peaks(env_a: np.ndarray, env_v: np.ndarray, k: int = 5, min_overlap: int = int(MIN_OVERLAP_S / FRAME)) -> list[tuple[int, float, int]]:
    """The k best distinct places of clip `v` on timeline `a` (like corr_partial, but every strong candidate, best first). A loud pause or a
    short accidental overlap can outscore the truth, so callers check each candidate against the real audio (see verify)."""
    la, lv = len(env_a), len(env_v)
    if la < min_overlap or lv < min_overlap:
        return []
    n = 1 << int(np.ceil(np.log2(la + lv)))
    c = np.fft.irfft(np.fft.rfft(env_a, n) * np.conj(np.fft.rfft(env_v, n)), n)
    lags = np.arange(-(lv - 1), la)
    vals = np.concatenate([c[n - (lv - 1):], c[:la]])
    a_lo, a_hi = np.maximum(0, lags), np.minimum(la, lags + lv)
    v_lo, v_hi = np.maximum(0, -lags), np.minimum(lv, la - lags)
    sqa = np.concatenate([[0.0], np.cumsum(env_a ** 2)])
    sqv = np.concatenate([[0.0], np.cumsum(env_v ** 2)])
    overlap = a_hi - a_lo
    ok = overlap >= min_overlap
    if not ok.any():
        return []
    ncc = np.where(ok, vals / (np.sqrt(np.maximum(sqa[a_hi] - sqa[a_lo], 1e-9)) * np.sqrt(np.maximum(sqv[v_hi] - sqv[v_lo], 1e-9))), -1.0)
    base = ncc[ok]
    out: list[tuple[int, float, int]] = []
    work = ncc.copy()
    for _ in range(k):
        j = int(np.argmax(work))
        if work[j] <= -1.0:
            break
        out.append((int(lags[j]), float((ncc[j] - base.mean()) / (base.std() + 1e-9)), int(overlap[j])))
        work[max(0, j - 200):j + 200] = -1.0
    return out


def phat_peak(path_a, ta: float, path_b, tb: float, secs: float = 6.0, sr: int = 16000) -> tuple[float, float] | None:
    """Do these two recordings carry the same sound at these moments? -> (lag in seconds, sharpness of the match) by GCC-PHAT, or None."""
    try:
        a = AS.decode(path_a, max(0.0, ta), secs, sr)
        b = AS.decode(path_b, max(0.0, tb), secs, sr)
    except Exception:  # noqa: BLE001
        return None
    n = min(len(a), len(b))
    if n < sr * 2:
        return None
    a, b = a[:n], b[:n]
    X = np.fft.rfft(a, 2 * n) * np.conj(np.fft.rfft(b, 2 * n))
    X /= np.abs(X) + 1e-9
    c = np.fft.irfft(X, 2 * n)
    c = np.concatenate([c[-n:], c[:n]])
    k = int(np.argmax(c))
    return (k - n) / sr, float(c[k] / (np.median(np.abs(c)) + 1e-12))


def verify(placed: dict, it: dict, abs_start: int, probes: int = 3) -> float:
    """How well a candidate position agrees with the actual audio of the files already placed: the summed sharpness of the matches found at
    a few moments spread over their overlap (0 if fewer than two moments agree on a lag near zero)."""
    s0, s1 = abs_start * FRAME, abs_start * FRAME + it["dur"]
    best = 0.0
    for name, p in placed.items():
        if not p.get("path"):
            continue
        p0 = p["abs"] * FRAME
        lo, hi = max(s0, p0), min(s1, p0 + p["dur"])
        if hi - lo < 14.0:
            continue
        good, score = 0, 0.0
        for f in (0.2, 0.5, 0.8)[:probes]:
            t = lo + 3.0 + (hi - lo - 12.0) * f
            r = phat_peak(p["path"], t - p0, it["path"], t - s0)
            if r and abs(r[0]) <= 0.08 and r[1] >= 10.0:
                good += 1
                score += r[1]
        if good >= 2:
            best = max(best, score)
    return best


class Timeline:
    """The shared timeline's loudness curve, grown as files are placed on it. Positions are ABSOLUTE frames (the anchor starts at 0);
    `origin` is the absolute frame of the curve's index 0 (it moves left when a file starts before everything placed so far)."""

    def __init__(self, env: np.ndarray):
        self.origin = 0
        self.sum = env.astype(np.float64).copy()
        self.cnt = np.ones(len(env))

    def curve(self) -> np.ndarray:
        return self.sum / np.maximum(self.cnt, 1.0)

    def add(self, env: np.ndarray, abs_start: int) -> None:
        idx = abs_start - self.origin
        if idx < 0:                                               # grow to the left
            self.sum = np.concatenate([np.zeros(-idx), self.sum])
            self.cnt = np.concatenate([np.zeros(-idx), self.cnt])
            self.origin += idx
            idx = 0
        if idx + len(env) > len(self.sum):                        # grow to the right
            extra = idx + len(env) - len(self.sum)
            self.sum = np.concatenate([self.sum, np.zeros(extra)])
            self.cnt = np.concatenate([self.cnt, np.zeros(extra)])
        self.sum[idx:idx + len(env)] += env
        self.cnt[idx:idx + len(env)] += 1


def place_files(items: list[dict], progress=None) -> dict[str, dict]:
    """Line every file up on one timeline.

    items: [{"name","path","dur","anchor": bool}] - files that have sound. The anchor (a recorder if there is one, else the longest
    file) is the first reference. -> {name: {"t0": seconds on the master timeline where the file begins, "z": confidence}} for the files
    that matched; files that match nothing are left out (they are not part of this shoot). Times are shifted so the earliest is 0."""
    if not items:
        return {}
    anchor = next((i for i in items if i.get("anchor")), None) or max(items, key=lambda i: i["dur"])
    tl = Timeline(AS.envelope(AS.decode(anchor["path"])))
    placed: dict[str, dict] = {anchor["name"]: {"abs": 0, "z": 99.0, "overlap_s": anchor["dur"], "path": anchor["path"], "dur": anchor["dur"]}}
    pending = [i for i in items if i["name"] != anchor["name"]]
    envs: dict[str, np.ndarray] = {}
    for _round in range(6):                                       # a file that only overlaps a LATER-placed file matches in a later round
        curve = tl.curve()
        found = []
        for k, it in enumerate(pending):
            if progress:
                progress(k / max(1, len(pending)), f"Lining up {it['name']} with the other cameras")
            if it["name"] not in envs:
                envs[it["name"]] = AS.envelope(AS.decode(it["path"], 0.0, min(it["dur"], HEAD_S)))
            best = None
            for lag, z, ov in corr_peaks(curve, envs[it["name"]]):         # every strong candidate is checked against the real audio
                if z < MIN_Z * 0.6:
                    continue
                score = verify(placed, it, tl.origin + lag)
                if score > 0 and (best is None or score > best[0]):
                    best = (score, tl.origin + lag, z, ov)
            if best:
                found.append((it, best[1], best[2], best[3]))
        if not found:
            break
        for it, abs_start, z, ov in sorted(found, key=lambda x: -x[2]):
            tl.add(envs[it["name"]], abs_start)
            placed[it["name"]] = {"abs": abs_start, "z": round(z, 1), "overlap_s": round(ov * FRAME, 1), "path": it["path"], "dur": it["dur"]}
        done = {it["name"] for it, *_ in found}
        pending = [i for i in pending if i["name"] not in done]
        if not pending:
            break
    lo = min(v["abs"] for v in placed.values())
    return {n: {"t0": round((v["abs"] - lo) * FRAME, 3), "z": v["z"], "overlap_s": v["overlap_s"]} for n, v in placed.items()}


def refine_offsets(items: list[dict], placed: dict[str, dict], probes: int = 5) -> dict[str, dict]:
    """place_files() lines files up to ~10 ms from their loudness curves. Here every file is checked against the real audio of the file
    it overlaps most (already refined; the anchor first) at `probes` moments spread over the overlap, by GCC-PHAT: the moments that agree
    give its offset to a fraction of a millisecond. The MEDIAN is used, so a little clock drift over a long take is split evenly instead
    of all landing at the end ("drift_ms" = how far the moments disagreed)."""
    by_name = {i["name"]: i for i in items}
    out = {n: dict(v) for n, v in placed.items()}
    order = sorted(out, key=lambda n: (-out[n]["z"], n))                    # the anchor (z 99) is the first reference
    done = order[:1]
    for n in order[1:]:
        it = by_name.get(n)
        best = None
        for r in done:
            ri = by_name.get(r)
            if not it or not ri:
                continue
            lo = max(out[n]["t0"], out[r]["t0"])
            hi = min(out[n]["t0"] + it["dur"], out[r]["t0"] + ri["dur"])
            if hi - lo > 20.0 and (best is None or hi - lo > best[1]):
                best = (r, hi - lo, lo, hi)
        if best:
            r, _ov, lo, hi = best
            lags = []
            for k in range(probes):
                t = lo + 3.0 + (hi - lo - 12.0) * (k + 1) / (probes + 1)
                res = phat_peak(by_name[r]["path"], t - out[r]["t0"], it["path"], t - out[n]["t0"])
                if res and abs(res[0]) <= 0.12 and res[1] >= 8.0:
                    lags.append(res[0])
            if len(lags) >= 2:
                out[n]["t0"] = out[n]["t0"] + float(np.median(lags))
                out[n]["drift_ms"] = round((max(lags) - min(lags)) * 1000, 1)
        done.append(n)
    lo = min(v["t0"] for v in out.values())
    for v in out.values():
        v["t0"] = round(v["t0"] - lo, 4)
    return out


# ----------------------------------------------------------------------------------------- the group
ANGLE_ROLES = ("a_roll", "multicam_angle")
ON_CAMERA_MIN = 0.3                  # a file whose speaker is on camera at least this share of the time can be an angle even if its role is unsure


def angle_candidates(files: list[dict]) -> tuple[list[dict], str]:
    """-> (the files that may be camera angles, a plain note when the face test could not be used). A file the footage record calls B-roll can never
    be an angle, even if its microphone heard the room; with a record, an angle is a_roll / multicam_angle or has the speaker on camera. When fewer
    than two files pass that, the sound test alone decides (every non-B-roll file with sound), and the note says so."""
    base = [f for f in files if f["audio"] and f["dur"] >= 15 and not f.get("broll")]
    withrec = [f for f in base if (f.get("facts") or {}).get("role")]
    if not withrec:
        return base, ""                                                   # (no footage record: the sound test of before)
    notbroll = [f for f in base if (f.get("facts") or {}).get("role") != "broll"]

    def on_cam(f: dict) -> float:
        return float((((f.get("facts") or {}).get("faces") or {}).get("on_camera_speaker_ratio")) or 0.0)
    passing = [f for f in notbroll if (f.get("facts") or {}).get("role") in ANGLE_ROLES or on_cam(f) >= ON_CAMERA_MIN]
    if len(passing) >= 2:
        return passing, ""
    return notbroll, "fewer than two files show the speaker on camera: the angles are found by sound alone"


def angle_shows(clips: list[dict], by_name: dict[str, dict]) -> str:
    """What an angle shows, from the footage record's measured faces: 'a medium close-up of one person', 'a wide shot of the room' ('' = unknown)."""
    from .footage import shot_size_from_ratio
    ratios, counts = [], []
    for c in clips:
        fa = (by_name.get(c["name"]) or {}).get("facts") or {}
        fc = fa.get("faces") or {}
        if fc.get("measured") is False and not fc.get("largest_ratio_median"):
            continue
        if "largest_ratio_median" in fc:
            ratios.append(float(fc.get("largest_ratio_median") or 0.0))
            counts.append(int(fc.get("count_median") or 0))
    if not ratios:
        return ""
    r, n = float(np.median(ratios)), int(np.median(counts))
    if n == 0 or r <= 0:
        return "a wide shot of the room (no face large enough to see)"
    size = (shot_size_from_ratio(r) or "").replace("_", " ")
    who = "one person" if n == 1 else f"{n} people"
    return f"a {size} of {who}" if size else f"{who} on screen"


def detect_group(files: list[dict], audios: list[dict], progress=None) -> dict | None:
    """Find out whether this footage is a multicam shoot. -> a group, or None.

    files: the scanned videos (with "rel", "w", "h", "fps", "audio", "dur", "path"); audios: separate audio recordings.
    A group needs 2+ different cameras that really overlap in sound. Everything else stays an ordinary file."""
    cand, cand_note = angle_candidates(files)                       # (the footage record decides who may be an angle; the sound test then finds who overlaps)
    by_name = {f["name"]: f for f in files}
    angle_of = assign_angles(cand)
    if len(cand) < 2 and not audios:
        return None                                              # two cameras of the SAME model look identical: only the sound can tell
    items = [{"name": "rec:" + a["name"], "path": a["path"], "dur": a["dur"], "anchor": i == 0} for i, a in enumerate(sorted(audios, key=lambda a: -a["dur"]))]
    items += [{"name": f["name"], "path": f["path"], "dur": f["dur"]} for f in cand]
    placed = place_files(items, progress)
    try:
        placed = refine_offsets(items, placed)                           # sample-accurate sync (lip sync to a fraction of a frame)
    except Exception as e:  # noqa: BLE001 - the loudness match alone is accurate to ~10 ms
        log.warning(f"refining the sync failed ({e}); keeping the first match")
    vids = [f for f in cand if f["name"] in placed]
    if len(vids) < 2 and not (vids and audios):
        return None                                              # nothing overlapped: not a multicam shoot
    long_vids = [f for f in vids if f["dur"] >= SHORT_CLIP_S] or vids
    short = sorted(f["name"] for f in vids if f not in long_vids)
    angles: dict[str, list[dict]] = {}
    for f in sorted(long_vids, key=lambda f: placed[f["name"]]["t0"]):
        angles.setdefault(angle_of[f["name"]], []).append({"name": f["name"], "path": f["path"], "t0": placed[f["name"]]["t0"], "dur": f["dur"],
                                                          "w": f["w"], "h": f["h"], "fps": f["fps"], "z": placed[f["name"]]["z"], "tc": f.get("tc", 0.0),
                                                          "tc_df": f.get("tc_df", False)})
    angles = split_overlaps(angles)                              # one camera can only show ONE clip at a time: overlapping clips are other cameras
    if len(angles) < 2:
        return None                                              # every clip came from one camera, one after another
    recs = [{"name": a["name"], "path": a["path"], "t0": placed["rec:" + a["name"]]["t0"], "dur": a["dur"], "tc": a.get("tc", 0.0),
             "ch": a.get("ch", 2)} for a in audios if "rec:" + a["name"] in placed]
    end = max([c["t0"] + c["dur"] for cl in angles.values() for c in cl] + [r["t0"] + r["dur"] for r in recs])
    short_ok = [n for n in short if (by_name.get(n, {}).get("facts") or {}).get("role") not in ANGLE_ROLES]          # (a short a_roll clip stays a talking clip, not B-roll)
    return {"angles": [{"id": k, "files": v, "shows": angle_shows(v, by_name)} for k, v in sorted(angles.items())], "recorders": recs, "dur": round(end, 3),
            "files": sorted(f["name"] for f in long_vids), "short": short_ok, "short_a_roll": sorted(set(short) - set(short_ok)),
            "unmatched": sorted(f["name"] for f in cand if f["name"] not in placed), **({"note": cand_note} if cand_note else {})}


def split_overlaps(angles: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Clips of the same camera that overlap in time can't share an angle: the later ones go to extra angles ('Cam A 2', ...)."""
    out: dict[str, list[dict]] = {}
    for aid, clips in angles.items():
        tracks: list[list[dict]] = []
        for c in sorted(clips, key=lambda c: c["t0"]):
            for t in tracks:
                if t[-1]["t0"] + t[-1]["dur"] <= c["t0"] + 0.05:
                    t.append(c)
                    break
            else:
                tracks.append([c])
        for k, t in enumerate(tracks):
            if k == 0:
                out[aid] = t
            else:
                n = 2
                while f"Camera {n}" in out or f"Camera {n}" in angles:
                    n += 1
                out[f"Camera {n}" if aid.startswith("Camera ") else f"{aid} {k + 1}"] = t
    return out


def cover(clips: list[dict], t: float) -> dict | None:
    """The clip of one angle (or recorder) that covers master time t."""
    return next((c for c in clips if c["t0"] - 1e-6 <= t < c["t0"] + c["dur"] - 1e-6), None)


def angle_files(group: dict, angle_id: str) -> list[dict]:
    return next((a["files"] for a in group["angles"] if a["id"] == angle_id), [])


RECORDER_BONUS = 3.0                       # a recorder is usually the mic on the person: it wins a near tie


def sound_at(group: dict, t: float) -> dict | None:
    """Which file provides the sound at master time t. With measured sound (group["sound_r"], rate_sources): the cleanest recording that
    covers t, recorders included (a recorder in a pocket or across the room loses to a camera's clean mic). Without: a recorder if one
    covers it, else the camera whose own recording is cleanest."""
    rq = group.get("sound_r") or {}
    cands = [r for r in group["recorders"] if r["t0"] - 1e-6 <= t < r["t0"] + r["dur"] - 1e-6] + \
            [c for c in (cover(a["files"], t) for a in group["angles"]) if c is not None]          # (every recorder that heard it)
    if rq and len({float((rq.get(c["path"]) or {}).get("score", 0.0)) for c in cands}) > 1:      # (all the same: nothing measured)
        best, bs = None, None
        for c in cands:
            sc = float((rq.get(c["path"]) or {}).get("score", 0.0)) + (RECORDER_BONUS if c in group["recorders"] else 0.0)
            if bs is None or sc > bs + 1e-9:
                best, bs = c, sc
        return {"path": best["path"], "local": t - best["t0"], "clip": best}
    r = cover(group["recorders"], t)
    if r:
        return {"path": r["path"], "local": t - r["t0"], "clip": r}
    q = group.get("sound_q") or {}
    best = None
    for a in group["angles"]:                                   # the CLEANEST recording wins (measured, see rate_sound), then the longest
        c = cover(a["files"], t)
        if c and (best is None or (q.get(c["path"], 0.0), c["dur"]) > (q.get(best["path"], 0.0), best["dur"])):
            best = c
    return {"path": best["path"], "local": t - best["t0"], "clip": best} if best else None


def rate_sound(group: dict, meter) -> dict:
    """Score every camera's own recording for cleanliness (how far its speech stands above its own background, in dB) so one angle's
    audio is used throughout instead of whichever camera happens to be showing. Stored in group["sound_q"]."""
    import numpy as np
    q = group.setdefault("sound_q", {})
    for a in group["angles"]:
        for c in a["files"]:
            if c["path"] in q:
                continue
            try:
                e, _ = meter.curve(c["path"])
                q[c["path"]] = round(float(np.percentile(e, 90) - np.percentile(e, 10)), 2) if len(e) > 200 else 0.0
            except Exception as ex:  # noqa: BLE001 - unrated = lowest priority
                log.warning(f"couldn't rate the sound of {c['path']}: {ex}")
                q[c["path"]] = 0.0
    return group


def rate_sources(group: dict, rate=None) -> dict:
    """Every recording of the shoot - cameras AND recorders - scored by what an editor hears (soundcheck: speech over the room,
    consonants over the hiss, level), into group["sound_r"] = {path: {"score", "grade"}}. sound_at() then picks the cleanest."""
    from . import soundcheck
    rate = rate or soundcheck.rate_file
    rq = group.setdefault("sound_r", {})
    for c in [c for a in group["angles"] for c in a["files"]] + list(group.get("recorders") or []):
        if c["path"] in rq:
            continue
        try:
            rq[c["path"]] = rate(c["path"], float(c.get("dur") or 0.0))
        except Exception as ex:  # noqa: BLE001 - unrated = the lowest priority
            log.warning(f"couldn't rate the sound of {c['path']}: {ex}")
            rq[c["path"]] = {"score": 0.0, "grade": "bad"}
    return group


def sound_spans(group: dict, min_s: float = 1.0) -> list[tuple[dict, float, float]]:
    """Which file provides the sound for each stretch of the shared timeline - exactly the choice sound_at() makes for the finished
    video (the recorder, else the camera whose own sound is cleanest) - as [(clip, from, to)] in time order. The transcript is made from
    the same sound, so it hears the cleanest recording too."""
    clips = [c for a in group["angles"] for c in a["files"]] + list(group.get("recorders") or [])
    cuts = sorted({0.0, float(group.get("dur") or 0.0)} | {float(c["t0"]) for c in clips} | {float(c["t0"]) + float(c["dur"]) for c in clips})
    out: list[tuple[dict, float, float]] = []
    for x0, x1 in zip(cuts, cuts[1:]):
        if x1 - x0 < 1e-6:
            continue
        s = sound_at(group, (x0 + x1) / 2)
        if not s:
            continue
        c = s["clip"]
        if out and out[-1][0]["path"] == c["path"] and abs(out[-1][0]["t0"] - c["t0"]) < 1e-6 and abs(out[-1][2] - x0) < 1e-6:
            out[-1] = (out[-1][0], out[-1][1], x1)
        else:
            out.append((c, x0, x1))
    if len(out) > 1:                                    # a sliver (a camera running a moment past the recorder) is not worth a transcript
        kept = [s for s in out if s[2] - s[1] >= min_s] or out
        out = []
        for c, x0, x1 in kept:
            if out and out[-1][0]["path"] == c["path"] and abs(out[-1][0]["t0"] - c["t0"]) < 1e-6:
                out[-1] = (out[-1][0], out[-1][1], x1)
            else:
                out.append((c, x0, x1))
    return out


def sound_clips(group: dict) -> list[dict]:
    """Every file that can provide sound, as the clips of a single timeline. (How edits made before 2026-10-13 were transcribed: the
    cameras in the order they start. Kept so those edits can be re-made exactly; new edits use sound_spans.)"""
    if group["recorders"]:
        return sorted(group["recorders"], key=lambda c: c["t0"])
    # without a recorder: walk the cameras in time order and use whichever file starts next past what is already covered
    out, end = [], -1.0
    for c in sorted((c for a in group["angles"] for c in a["files"]), key=lambda c: c["t0"]):
        if c["t0"] + c["dur"] > end + 1e-6:
            out.append(c)
            end = c["t0"] + c["dur"]
    return out


# ----------------------------------------------------------------------------------------- the director
MIN_HOLD = 6.0            # seconds: never cut away sooner than this
MAX_HOLD = 14.0           # seconds: an angle that has been up this long is allowed to change if there is anything to change to
DIRECTOR_SYSTEM = (
    "You are the director cutting a multicam interview. You see frames from each camera angle taken at the same moments (labelled with "
    "the angle name and the time). For every moment, choose the angle that best shows what matters: whoever is speaking, the "
    "reaction when it helps, a wide shot to establish or to breathe. A multicam interview normally uses ALL its angles: change angle "
    "at natural moments (a new sentence, a new point, a pause) roughly every 9 to 12 seconds (never more than 12 seconds on one angle without a "
    "change, and don't cut more often than that: constant cutting feels frantic), and stay on one angle for a whole stretch only if the other "
    "angle is unusable there. Hold every shot at least 6 seconds and don't ping-pong. "
    "The words you are directing are all spoken by the GUEST (the person being interviewed), so cut between angles that show the guest; "
    "show the interviewer only if they are the only usable angle. Avoid angles where the frame is blurry, badly framed, or the person is "
    "out of shot. Return the switches as segments covering each requested stretch.")


def director_system(min_hold: float = MIN_HOLD, lo: float = 9.0, hi: float = 12.0) -> str:
    """DIRECTOR_SYSTEM with the hold numbers of the video's style (the defaults give the text above)."""
    return (DIRECTOR_SYSTEM.replace("roughly every 9 to 12 seconds (never more than 12 seconds", f"roughly every {lo:g} to {hi:g} seconds (never more than {hi:g} seconds")
            .replace("Hold every shot at least 6 seconds", f"Hold every shot at least {min_hold:g} seconds"))


DIRECTOR_SCHEMA = {"type": "object", "properties": {"segments": {"type": "array", "items": {"type": "object", "properties": {
    "start": {"type": "number"}, "end": {"type": "number"}, "angle": {"type": "string"}, "why": {"type": "string"}},
    "required": ["start", "end", "angle"]}}}, "required": ["segments"]}


def sheets_for(group: dict, stretches: list[tuple[float, float]], per_stretch_s: float = 4.0, max_images: int = 10) -> list[tuple[bytes, str]]:
    """Contact sheets (one per angle per batch of moments) at the kept stretches: [(jpeg, caption)]."""
    from .glasses import contact_sheets
    times = []
    for a, b in stretches:
        t = a
        while t < b and len(times) < 400:
            times.append(round(t, 2))
            t += per_stretch_s
    if not times:
        return []
    out: list[tuple[bytes, str]] = []
    per = 12
    batches = [times[i:i + per] for i in range(0, len(times), per)][:max(1, min(max_images, 24 // max(1, len(group["angles"]))))]
    for ang in group["angles"]:
        for batch in batches:
            tiles = []
            for t in batch:
                c = cover(ang["files"], t)
                tiles.append((t, c))
            img = _tile_sheet(tiles)
            if img:
                out.append((img, f"angle {ang['id']}, moments {batch[0]:.0f}-{batch[-1]:.0f} s"))
    return out


def _tile_sheet(tiles: list[tuple[float, dict | None]], cols: int = 4, tile_w: int = 256) -> bytes | None:
    """One JPEG: a frame per moment from this angle (black where the angle has no footage), each labelled with its master time."""
    import subprocess
    import tempfile
    from io import BytesIO

    from PIL import Image, ImageDraw, ImageFont

    from ..core import ffmpeg
    from . import logcolor                                          # (log footage seen in normal colour)
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        for k, (t, c) in enumerate(tiles):
            if c is None:
                frames.append((t, None))
                continue
            png = Path(tmp) / f"{k}.jpg"
            subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-y", "-ss", f"{t - c['t0']:.2f}", "-i", c["path"], "-frames:v", "1",
                            "-vf", f"{logcolor.vf_prefix(c['path'])}scale={tile_w}:-2", str(png)], capture_output=True, timeout=120)
            frames.append((t, Image.open(png).convert("RGB") if png.exists() else None))
        if not any(im is not None for _, im in frames):
            return None
        th = next(im.height for _, im in frames if im is not None)
        rows = (len(frames) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tile_w + (cols - 1) * 4, rows * th + (rows - 1) * 4), (20, 20, 20))
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 20)
        except OSError:
            font = ImageFont.load_default()
        d = ImageDraw.Draw(sheet)
        for i, (t, im) in enumerate(frames):
            x, y = (i % cols) * (tile_w + 4), (i // cols) * (th + 4)
            if im is not None:
                sheet.paste(im.resize((tile_w, th)), (x, y))
            label = f"{int(t // 60)}:{t % 60:04.1f}"
            d.rectangle((x, y, x + 8 + d.textlength(label, font=font), y + 26), fill=(0, 0, 0))
            d.text((x + 4, y + 2), label, font=font, fill=(255, 235, 60))
        buf = BytesIO()
        sheet.save(buf, "JPEG", quality=70)
        return buf.getvalue()


def director_windows(stretches: list[tuple[float, float]], window_s: float = 240.0, max_n: int = 40) -> list[list[tuple[float, float]]]:
    """The kept stretches in consecutive groups of about `window_s` seconds of finished video: the director looks at one group at a time,
    so a long cut is directed all the way through (one look used to see only the first ~8 minutes)."""
    out: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []
    acc = 0.0
    for a, b in stretches:
        if cur and (acc + (b - a) > window_s or len(cur) >= max_n):
            out.append(cur)
            cur, acc = [], 0.0
        cur.append((a, b))
        acc += b - a
    if cur:
        out.append(cur)
    return out


def choose_angles(ai, group: dict, stretches: list[tuple[float, float]], brief: str = "", words: list[dict] | None = None,
                  hold: tuple[float, float, float] | None = None, points: list[dict] | None = None, wide_share: float | None = None) -> list[dict]:
    """The director: -> [{"start","end","angle"}] over the kept stretches (master seconds), after enforcing hold times."""
    ids = [a["id"] for a in group["angles"]]
    segs: list[dict] = []
    wins = director_windows(stretches)
    for k, win in enumerate(wins):
        sheets = sheets_for(group, win)
        if not sheets:
            continue
        spoken = " ".join(w["text"] for w in (words or []) if any(a <= w["start"] < b for a, b in win))[:3000]
        shows = [f"{a['id']}: {a['shows']}" for a in group["angles"] if a.get("shows")]
        prompt = ((f"The creator's notes: {brief}\n" if brief else "") + f"Angles: {', '.join(ids)}.\n"
                  + (("What the cameras show (measured): " + "; ".join(shows) + ".\n") if shows else "")
                  + (f"This is part {k + 1} of {len(wins)} of the video (the rest is directed separately).\n" if len(wins) > 1 else "")
                  + _rules_text(points, win, hold, wide_share)
                  + "Stretches to direct (seconds on the shared timeline): " + "; ".join(f"{a:.1f}-{b:.1f}" for a, b in win) +
                  f"\nWhat is said: {spoken}\nThe images are contact sheets, one per angle per batch of moments: " +
                  " | ".join(c for _, c in sheets) + ".")
        try:
            res = ai.complete_json("multicam_director", director_system(*hold) if hold else DIRECTOR_SYSTEM, prompt, DIRECTOR_SCHEMA, images=[b for b, _ in sheets])
            for s in res.data.get("segments") or []:
                if str(s.get("angle")) in ids:
                    segs.append({"start": float(s["start"]), "end": float(s["end"]), "angle": str(s["angle"])})
        except Exception as e:  # noqa: BLE001 - fall back to the main angle for this part
            log.warning(f"the director failed for part {k + 1} ({e}); using the widest angle there")
    return cut_list(group, stretches, segs, hold[0] if hold else MIN_HOLD, points=points, words=words)


def _rules_text(points: list[dict] | None, win: list[tuple[float, float]], hold, wide_share: float | None) -> str:
    """The switching rules as the director's prompt text (the code enforces them again in cut_list whatever the model answers)."""
    if not points:
        return ""
    times = [p["t"] for p in points if any(a <= p["t"] <= b for a, b in win)][:60]
    if not times:
        return ""
    mh = hold[0] if hold else MIN_HOLD
    wide = f" Return to the wide about {wide_share:.0%} of the time or when two people talk at once." if wide_share else ""
    return (f"You may only change angle at these times: {', '.join(f'{t:.1f}' for t in times)}. Hold every angle at least {mh:g} s; during one person's "
            f"monologue stay 15 to 30 s on the angle that shows them; cut to a listener for a reaction only when the speaker pauses or the listener reacts."
            + wide + "\n")


def cut_list(group: dict, stretches: list[tuple[float, float]], segs: list[dict], min_hold: float = MIN_HOLD,
             points: list[dict] | None = None, words: list[dict] | None = None) -> list[dict]:
    """The AI's segments -> a clean list over exactly the kept stretches: every moment has an angle that HAS footage there, no
    shot shorter than MIN_HOLD (it joins its neighbour), and the very long ones left alone. -> [{"start","end","angle"}]."""
    ids = [a["id"] for a in group["angles"]]
    default = max(ids, key=lambda i: sum(c["dur"] for c in angle_files(group, i)))              # the angle with the most footage
    out: list[dict] = []
    for a, b in stretches:
        cuts = sorted({a, b} | {x for s in segs for x in (s["start"], s["end"]) if a < x < b})
        for x0, x1 in zip(cuts, cuts[1:]):
            mid = (x0 + x1) / 2
            ang = next((s["angle"] for s in segs if s["start"] <= mid < s["end"]), None)
            if not ang or not cover(angle_files(group, ang), mid):
                ang = next((i for i in [default, *ids] if cover(angle_files(group, i), mid)), None)
            if ang is None:
                continue                                                                     # no camera saw this moment
            if out and out[-1]["angle"] == ang and abs(out[-1]["end"] - x0) < 1e-6:
                out[-1]["end"] = x1
            else:
                out.append({"start": x0, "end": x1, "angle": ang})
    # shots shorter than MIN_HOLD join the previous shot (unless that angle has no footage there)
    i = 0
    while i < len(out):
        s = out[i]
        if s["end"] - s["start"] < min_hold and i > 0 and abs(out[i - 1]["end"] - s["start"]) < 1e-6 \
                and cover(angle_files(group, out[i - 1]["angle"]), (s["start"] + s["end"]) / 2):
            out[i - 1]["end"] = s["end"]
            out.pop(i)
            continue
        i += 1
    if points or words:                                                        # the switching rules hold whatever the model answered (switching.py)
        from . import switching as SW
        out = SW.apply(out, points or [], words or [], min_hold)
    return out
