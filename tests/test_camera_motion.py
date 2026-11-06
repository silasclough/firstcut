"""Camera motion in Find footage's heat map: the smooth move (a push-in, a pan) is the best part of a shot, hand-held settling is not, a person
walking through a locked-off shot is not camera motion. Synthetic clips: a textured picture seen through a moving virtual camera."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from clipfarm.clipper import cameramotion as CM
from clipfarm.clipper import catalog as C
from clipfarm.core import ffmpeg

cv2 = pytest.importorskip("cv2")
W, H, FPS = 640, 360, 24


def _texture(seed, scale=2.0):
    rng = np.random.default_rng(seed)
    big = rng.integers(0, 255, (int(H * scale), int(W * scale), 3), dtype=np.uint8)
    big = cv2.normalize(cv2.GaussianBlur(big, (0, 0), 2.2), None, 0, 255, cv2.NORM_MINMAX)
    for _ in range(40):
        x, y = rng.integers(0, big.shape[1]), rng.integers(0, big.shape[0])
        cv2.circle(big, (int(x), int(y)), int(rng.integers(8, 40)), tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    return big


def _view(big, cx, cy, zoom):
    M = cv2.getRotationMatrix2D((cx, cy), 0.0, zoom)
    M[0, 2] += W / 2 - cx
    M[1, 2] += H / 2 - cy
    return cv2.warpAffine(big, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _write(path: Path, fn, secs=10, pix="yuv420p"):
    p = subprocess.Popen([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
                          "-i", "-", "-c:v", "libx264", "-g", "12", "-keyint_min", "12", "-sc_threshold", "0", "-pix_fmt", pix, str(path)],
                         stdin=subprocess.PIPE)
    for i in range(int(secs * FPS)):
        p.stdin.write(fn(i / FPS).tobytes())
    p.stdin.close()
    assert p.wait() == 0
    return path


def _ramp(t, a, b):
    return float(np.clip((t - a) / (b - a), 0, 1))


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("cm")
    out = {}
    big = _texture(1)
    cx, cy = big.shape[1] / 2, big.shape[0] / 2
    jit = np.random.default_rng(7).normal(0, 9, (300, 2))

    def push(t):                      # two seconds of hand-held settling, then a steady push-in to 1.5x, then still
        if t < 2.0:
            dx, dy = jit[int(t * FPS)]
            return _view(big, cx + dx, cy + dy, 1.0)
        return _view(big, cx, cy, 1.0 + 0.5 * _ramp(t, 2.0, 8.0))
    out["push"] = _write(d / "push.mp4", push)
    big3 = _texture(2, 3.0)
    out["pan"] = _write(d / "pan.mp4", lambda t: _view(big3, 400 + 40 * np.clip(t - 1.0, 0, 8.0), big3.shape[0] / 2, 1.0))
    out["shake"] = _write(d / "shake.mp4", lambda t: _view(big, cx + jit[int(t * FPS)][0], cy + jit[int(t * FPS)][1], 1.0))
    base = _view(big, cx, cy, 1.0)

    def person(t):                    # a still camera; a big bright 'person' walks across
        im = base.copy()
        x = int(60 + 90 * t)
        cv2.rectangle(im, (x, 80), (x + 110, 340), (40, 40, 230), -1)
        cv2.rectangle(im, (x + 25, 100), (x + 85, 150), (200, 220, 240), -1)
        return im
    out["person"] = _write(d / "person.mp4", person)
    out["still"] = _write(d / "still.mp4", lambda t: base)
    return out


def _profile(path):
    return CM.profile(CM.extract(ffmpeg.bin_path("ffmpeg"), path))


def test_a_push_in_is_found_where_it_starts_not_in_the_settling_before_it(clips):
    prof = _profile(clips["push"])
    a, b = CM.best_move(prof)
    assert 1.5 <= a <= 2.5 and 7.0 <= b <= 9.0                       # the move is 2 s to 8 s
    t = prof["t"]
    assert prof["cam"][(t > 6) & (t < 7)].min() > 0.9               # (high in the move)
    assert prof["cam"][(t > 0.4) & (t < 1.6)].max() < 0.5            # (not high in the shaky start)


def test_a_steady_pan_is_a_move(clips):
    a, b = CM.best_move(_profile(clips["pan"]))
    assert 0.5 <= a <= 1.5 and 8.0 <= b <= 9.8


def test_shaking_all_the_way_has_no_move_and_scores_low(clips):
    prof = _profile(clips["shake"])
    assert CM.best_move(prof) is None
    assert float(np.median(prof["cam"])) < 0.2


def test_a_person_walking_through_a_locked_off_shot_is_not_camera_motion(clips):
    prof = _profile(clips["person"])
    assert CM.best_move(prof) is None
    assert float(prof["cam"].min()) >= 0.4                           # (not marked down either)


def test_a_locked_off_shot_is_neutral(clips):
    prof = _profile(clips["still"])
    assert CM.best_move(prof) is None
    assert np.allclose(prof["cam"], CM.NEUTRAL, atol=0.05)


def test_too_little_to_judge_is_none():
    assert CM.profile(np.zeros((2, 7), np.float32)) is None
    assert CM.analyze([0.0, 0.5], np.zeros((2, 8, 8), np.uint8)).shape == (0, 7)


def _catalog(tmp_path):
    from tests.test_catalog import FakeEmbedder, _cfg
    return C.Catalog(_cfg(tmp_path, [tmp_path / "Drive"]), embedder=FakeEmbedder())


def test_indexing_keeps_the_motion_and_the_heat_map_peaks_at_the_start_of_the_move(tmp_path, clips):
    root = tmp_path / "Drive" / "Video"
    root.mkdir(parents=True)
    (root / "push.mp4").write_bytes(clips["push"].read_bytes())
    cat = _catalog(tmp_path)
    assert cat.index()["new"] == 1
    rec = list(cat.records())[0]
    assert rec["motion"] == CM.VERSION and (cat.dir / "motion" / f"{rec['id']}.npy").exists()
    h = cat.heat(rec["id"])
    assert "motion" not in h                                          # (nothing pending: it was kept at indexing)
    assert 1.5 <= h["peak_t"] <= 2.5                                  # Best part: where the push-in starts
    step = h["step"]
    heat = np.array(h["heat"])
    shaky, move = heat[: int(1.6 / step) + 1].max(), heat[int(3 / step): int(7 / step) + 1].mean()
    assert move > shaky                                               # the move is hotter than the settling before it


def test_an_older_record_gets_its_motion_in_the_background_and_says_pending(tmp_path, clips):
    root = tmp_path / "Drive" / "Video"
    root.mkdir(parents=True)
    (root / "pan.mp4").write_bytes(clips["pan"].read_bytes())
    cat = _catalog(tmp_path)
    cat.index()
    rec = list(cat.records())[0]
    (cat.dir / "motion" / f"{rec['id']}.npy").unlink()
    rec.pop("motion")                                                 # (as if catalogued before motion was kept)
    cat.save(rec)
    h = cat.heat(rec["id"])
    assert h.get("motion") == "pending"
    import time
    for _ in range(100):
        h = cat.heat(rec["id"])
        if h.get("motion") != "pending":
            break
        time.sleep(0.2)
    assert h.get("motion") is None and 0.5 <= h["peak_t"] <= 1.5


def test_a_file_without_motion_still_gets_the_old_heat_map(tmp_path, clips):
    root = tmp_path / "Drive" / "Video"
    root.mkdir(parents=True)
    (root / "still.mp4").write_bytes(clips["still"].read_bytes())
    cat = _catalog(tmp_path)
    cat.index()
    h = cat.heat(list(cat.records())[0]["id"])
    assert "peak_t" not in h and len(h["heat"]) >= 3
