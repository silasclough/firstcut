"""Log footage (S-Log3 etc.) is converted to normal colour in what FirstCut renders itself, never in the Final Cut export
(2026-11-06: "only for the MP4 clips and not when you push it to Final Cut")."""
import subprocess
from pathlib import Path

import numpy as np
import pytest

from clipfarm.clipper import logcolor as L
from clipfarm.core import ffmpeg
from test_replace_shot import made  # noqa: F401 - (a real small project, made once per test)

SONY_XML = """<?xml version="1.0" encoding="UTF-8"?>
<NonRealTimeMeta xmlns="urn:schemas-professionalDisc:nonRealTimeMeta:ver.2.00" lastUpdate="2026-11-01T10:00:00-06:00">
	<Device manufacturer="Sony" modelName="ILME-FX30" serialNo="4294967295"/>
	<AcquisitionRecord>
		<Group name="CameraUnitMetadataSet">
			<Item name="CaptureGammaEquation" value="{gamma}"/>
			<Item name="CaptureColorPrimaries" value="{prim}"/>
			<Item name="CodingEquations" value="rec709"/>
		</Group>
	</AcquisitionRecord>
</NonRealTimeMeta>"""


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEOBUILDER_HOME", str(tmp_path / "home"))
    L._mem.clear()
    L._setting.update(at=0.0, val=None)
    return tmp_path / "home"


def _clip(path: Path, seconds=1.0, color="gray"):
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=160x90:d={seconds}:r=24",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def test_a_sony_sidecar_or_the_xml_inside_the_mp4_says_s_log3(tmp_path):
    a = _clip(tmp_path / "C0001.MP4")
    (tmp_path / "C0001M01.XML").write_text(SONY_XML.format(gamma="s-log3-cine", prim="s-gamut3-cine"))
    assert L.detect(a) == {"profile": "slog3", "gamut": "sgamut3cine", "how": "sidecar"}
    b = _clip(tmp_path / "C0002.MP4")                                     # the FX30 writes the same XML into the file itself
    with b.open("ab") as f:
        f.write(SONY_XML.format(gamma="s-log3", prim="s-gamut3").encode())
    assert L.detect(b) == {"profile": "slog3", "gamut": "sgamut3", "how": "metadata"}
    c = _clip(tmp_path / "C0003.MP4")
    with c.open("ab") as f:
        f.write(SONY_XML.format(gamma="rec709", prim="rec709").encode())    # a normal picture profile: nothing to convert
    assert L.detect(c)["profile"] is None and L.vf(c) == ""
    assert L.detect(_clip(tmp_path / "phone.mp4"))["profile"] is None     # no metadata, no camera maker: never guessed


def test_the_lut_maps_grey_black_and_white_sensibly():
    v = lambda code: (code - 64) / 876                                    # 10-bit code value -> what ffmpeg hands a filter
    g = lambda x: np.array([[x, x, x]])
    assert abs(float(L.slog3(v(420))) - 0.18) < 1e-6                      # Sony's own 18 % grey point
    for prof, x in (("slog2", v(347)), ("vlog", v(433)), ("clog3", 0.345), ("dlog", v(409))):
        assert abs(float(CURVE(prof, x)) - 0.18) < 0.01, prof
    assert abs(L.convert(g(v(420)), "slog3")[0, 1] - 0.46) < 0.005        # 18 % grey -> 0.46 on screen
    assert L.convert(g(v(95)), "slog3")[0, 1] < 0.005                     # S-Log3 black stays black
    assert 0.8 < L.convert(g(v(598)), "slog3")[0, 1] < 0.92               # 90 % white is bright, not clipped
    hi = L.convert(np.stack([np.linspace(v(700), v(820), 7)] * 3, 1), "slog3")[:, 1]
    assert np.all(np.diff(hi) > 0) and hi[-1] < 1.0                       # highlights still roll on: no flat clip
    assert np.allclose(L.to_709("sgamut3cine").sum(axis=1), 1.0, atol=1e-4)   # white stays white


def CURVE(prof, x):
    return L.CURVES[prof](x)


def test_the_cube_file_and_the_filter(tmp_path, home):
    p = L.cube("slog3", "sgamut3cine")
    lines = p.read_text().splitlines()
    assert lines[1] == "LUT_3D_SIZE 33" and len(lines) == 4 + 33 ** 3 and p.parent == home / "luts"
    a = _clip(tmp_path / "C0001.MP4", color="0x8c8c8c")                     # S-Log3-ish grey
    (tmp_path / "C0001M01.XML").write_text(SONY_XML.format(gamma="s-log3-cine", prim="s-gamut3-cine"))
    f = L.vf(a)
    assert f.startswith("lut3d=file='") and "interp=tetrahedral" in f
    out = tmp_path / "f.png"
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-y", "-i", str(a), "-frames:v", "1", "-vf", f, str(out)], check=True)
    assert out.exists()                                                     # the bundled ffmpeg takes the filter as written
    (home).mkdir(parents=True, exist_ok=True)
    (home / "settings.json").write_text('{"defaults": {"long": {"log": "leave"}}}')
    L._setting.update(at=0.0, val=None)
    assert L.vf(a) == ""                                                    # Settings › How it edits › Log footage: Leave as shot


def test_the_look_test_never_fires_on_normal_footage():
    rng = np.random.default_rng(3)
    normal = [np.clip(rng.normal(0.45, 0.25, (54, 96, 3)), 0, 1) for _ in range(3)]
    logish = [np.clip(0.42 + 0.12 * rng.normal(0, 1, (54, 96, 1)) + 0.015 * rng.normal(0, 1, (54, 96, 3)), 0.2, 0.75) for _ in range(3)]
    assert not L.look(normal) and L.look(logish) and not L.look(logish[:1])


def test_the_mp4_is_converted_and_final_cut_gets_the_original(made, monkeypatch):  # noqa: F811
    """The long-form cut: every stretch of a log clip is cut through the LUT; the .fcpxml has no LUT and no colour processing."""
    from clipfarm.clipper import localproj as LP
    monkeypatch.setattr(L, "detect", lambda path, use_look=True: {"profile": "slog3", "gamut": "sgamut3cine", "how": "sidecar"})
    cmds = []
    real = LP.ffmpeg.run

    def run(cmd, *a, **k):
        cmds.append((k.get("what", ""), " ".join(map(str, cmd))))
        return real(cmd, *a, **k)
    monkeypatch.setattr(LP.ffmpeg, "run", run)
    made["runner"].rerender(made["proj"]["id"])
    assert made["store"].get(made["proj"]["job"])["status"] == "done"
    cuts = [c for w, c in cmds if w.startswith("cutting a stretch")]
    assert cuts and all("lut3d=file=" in c for c in cuts)
    fcp = max((made["folder"] / "Final Cut").rglob("*.fcpxml"), key=lambda p: p.stat().st_mtime).read_text().lower()
    assert "lut" not in fcp and "colorprocessing" not in fcp and "logprocessing" not in fcp
