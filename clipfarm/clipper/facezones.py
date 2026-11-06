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


def zones(video: Path, start: float, dur: float, models_dir: Path, aspect: float, rate: float = 2.0) -> list[tuple[float, float, float, float]]:
    """The head zones of everyone seen in video[start : start + dur]. [] if nobody (or the detector is unavailable)."""
    try:
        from . import reframe as R
        # (960 px wide and faces down to 1.2 % of the width: in a wide shot of a stairwell every student is about 2 % wide, and the old 480 px / 3 %
        #  check saw only the nearest one, so a logo card landed on the others)
        samples, _n = R.detect_faces(Path(video), max(0.0, start), max(0.5, dur), models_dir, rate, width=960, min_w=0.012)
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
