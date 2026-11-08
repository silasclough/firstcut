"""Where the people are, so graphics never cover a face.

For every graphic's moment the picture under it is sampled (twice a second) and the faces found (OpenCV YuNet, the same detector that frames the
vertical clips). Each face becomes a "zone": the head with room around it (hair, chin, a little shoulder), as a rectangle in 0..1 of the frame; the zones of
every sample in the graphic's time window are kept, because people move. gfxcards places each graphic where no zone is, or makes it smaller, or drops it.
"""

from __future__ import annotations

from pathlib import Path

from ..core.log import get_logger

log = get_logger("clipper.facezones")


def zone_of(face, aspect: float) -> tuple[float, float, float, float]:
    """A detected face (centre cx, cy and width w, in 0..1 of the frame; `aspect` = frame width / height) -> the head's zone."""
    bw = face.w * 1.8
    bh = face.w * aspect * 2.5                                      # a face is ~1.25x taller than wide; with hair and chin about 2.5x
    cy = face.cy - 0.04 * bh
    return (max(0.0, face.cx - bw / 2), max(0.0, cy - bh / 2), min(1.0, face.cx + bw / 2), min(1.0, cy + bh / 2))


_CACHE: dict = {}


def _samples(video: Path, start: float, dur: float, models_dir: Path, rate: float):
    """The faces of video[start : start + dur] at `rate` samples a second (kept, so the flat zones and the timed zones of one card cost one detection)."""
    key = (str(video), round(float(start), 2), round(float(dur), 2), float(rate), str(models_dir))
    if key not in _CACHE:
        from . import reframe as R
        # (960 px wide and faces down to 1.2 % of the width: in a wide shot of a stairwell every student is about 2 % wide, and the old 480 px / 3 %
        #  check saw only the nearest one, so a logo card landed on the others)
        samples, _n = R.detect_faces(Path(video), max(0.0, start), max(0.5, dur), models_dir, rate, width=960, min_w=0.012)
        if len(_CACHE) > 64:
            _CACHE.clear()
        _CACHE[key] = samples
    return _CACHE[key]


def zones(video: Path, start: float, dur: float, models_dir: Path, aspect: float, rate: float = 2.0) -> list[tuple[float, float, float, float]]:
    """The head zones of everyone seen in video[start : start + dur]. [] if nobody (or the detector is unavailable)."""
    try:
        samples = _samples(video, start, dur, models_dir, rate)
    except Exception as e:  # noqa: BLE001 - no faces known: graphics keep their usual places
        log.warning(f"face check failed: {e}")
        return []
    out = []
    for row in samples:
        for f in row:
            z = zone_of(f, aspect)
            if not any(abs(z[0] - o[0]) < 0.02 and abs(z[1] - o[1]) < 0.02 and abs(z[2] - o[2]) < 0.02 and abs(z[3] - o[3]) < 0.02 for o in out):
                out.append(z)
    return out


def timed_zones(video: Path, start: float, dur: float, models_dir: Path, W: int, H: int, rate: float = 2.0) -> list[dict]:
    """The typed zones (face, mouth_eyes, body) of everyone seen in video[start : start + dur], WITH TIME (seconds from `start`, one sample = 1/rate s), in
    pixels of a W x H frame. A presenter who walks only blocks the places and the moments they are in. [] if nobody or no detector."""
    try:
        from . import placement as PL
        samples = _samples(video, start, dur, models_dir, rate)
    except Exception as e:  # noqa: BLE001 - no faces known: graphics keep their usual places
        log.warning(f"face check failed: {e}")
        return []
    out: list[dict] = []
    for k, row in enumerate(samples):
        t0 = k / rate
        for f in row:
            out += PL.zones_from_face(f.cx, f.cy, f.w, W, H, t0, t0 + 1.0 / rate)
    return out
