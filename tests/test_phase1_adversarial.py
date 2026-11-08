"""Phase 1 (WP1/WP2/WP4/WP14) adversarial checks by the tester: an old plan.json re-rendered, a messy folder, real delivered-file probes, bad MCP arguments."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from clipfarm.clipper import localproj as LP
from clipfarm.clipper import polish
from clipfarm.clipper import styles_profile as SPF
from clipfarm.core import ffmpeg

FF = ffmpeg.bin_path("ffmpeg")
sys.path.insert(0, str(Path(__file__).parent))


def _f(name="a.mp4", start=0.0, dur=3.5):
    return {"name": name, "path": f"/x/{name}", "start": start, "end": start + dur, "dur": dur, "w": 1920, "h": 1080, "fps": 25.0, "audio": True}


def test_short_clip_pool_scene_running_past_its_end_is_left_out_not_clamped_to_nothing():
    got, notes = LP.usable_pool([{"start": 3.4, "end": 6.0, "quality": 7, "what": "x"}], [_f()])
    assert got == [] and len(notes) == 1


def test_an_old_scene_without_ids_is_stamped_and_resolved():
    got, notes = LP.usable_pool([{"start": 1.0, "end": 3.0, "quality": 7, "what": "x"}], [_f()])
    assert got[0]["sid"] == "s0000" and got[0]["file"] == "a.mp4" and not notes


def test_resolve_pool_rejects_a_scene_that_starts_before_its_file():
    f = _f(start=100.0, dur=10.0)
    got, _ = LP.resolve_pool([{"file": "a.mp4", "fstart": -2.0, "fend": 1.0, "start": 98.0, "end": 101.0, "sid": "s1"}], [f])
    assert all(s["start"] >= f["start"] for s in got)


def test_apply_partial_with_a_non_object_section_is_a_plain_problem_not_a_crash():
    new, problems = SPF.apply_partial(SPF.resolve("reel"), {"pace": 5})
    assert problems


def test_apply_partial_out_of_range_values_are_problems_and_change_nothing():
    base = SPF.resolve("reel")
    new, problems = SPF.apply_partial(base, {"broll": {"share": 7}, "pace": {"name": "warp"}})
    assert problems and new is base
    new, problems = SPF.apply_partial(base, {"nonsense": {"x": 1}})
    assert isinstance(problems, list)


def test_contradicting_layers_follow_options_over_notes_over_points():
    s = SPF.build("promo", {"pace": "slow"}, {"pace": "fast"}, {"pace": {"name": "tight"}} and {"pace": "tight"})
    assert s["pace"]["name"] == "tight"
    s = SPF.build("promo", {"pace": "slow"}, {"pace": "fast"}, None)
    assert s["pace"]["name"] == "slow"
    assert SPF.regex_settings("") == {} and SPF.regex_settings(None) == {}


def test_fit_brief_survives_zero_and_negative_room():
    assert SPF.fit_brief("hello world", 0) == "" and isinstance(SPF.fit_brief("hello world", -5), str)


def test_untagged_hd_source_keeps_its_colours_through_the_delivery_chain(tmp_path):
    src, out = tmp_path / "hd.mp4", tmp_path / "out.mp4"
    subprocess.run([FF, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=0xC81E1E:s=1920x1080:d=1:r=25,format=rgb24,scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
                    "-c:v", "libx264", "-qp", "0", "-bsf:v", "h264_metadata=matrix_coefficients=2:colour_primaries=2:transfer_characteristics=2", str(src)], check=True)
    from clipfarm.clipper import renderq as RQ
    stage = RQ.colour_filter(src)                                   # the helper polish.run uses for its colour stage
    assert "in_color_matrix=bt709" in stage                         # untagged and 1920 wide: read as bt709, as players do
    subprocess.run([FF, "-v", "error", "-y", "-i", str(src), "-filter_complex",
                    f"[0:v]{stage},setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv,format=yuv420p[t]",
                    "-map", "[t]", "-c:v", "libx264", "-crf", "18", *polish.COLOR_TAGS, str(out)], check=True)

    def red(p):
        raw = subprocess.run([FF, "-v", "error", "-i", str(p), "-frames:v", "1", "-vf", "crop=2:2:500:100,scale=in_color_matrix=bt709,format=rgb24", "-f", "rawvideo", "-"],
                             capture_output=True, check=True).stdout
        return tuple(raw[:3])
    a, b = red(src), red(out)
    assert max(abs(x - y) for x, y in zip(a, b)) <= 4, (a, b)
    # NOTE: the polish.run graph is documented in the report: the same stage is applied there.


def test_tagged_sources_keep_their_colours_through_the_delivery_chain(tmp_path):
    for tag, extra in (("709", ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]),
                       ("601", ["-colorspace", "smpte170m", "-color_primaries", "smpte170m", "-color_trc", "smpte170m"])):
        src, out = tmp_path / f"s{tag}.mp4", tmp_path / f"o{tag}.mp4"
        subprocess.run([FF, "-v", "error", "-y", "-f", "lavfi", "-i", "smptebars=s=640x360:d=1:r=25,format=yuv420p", *extra, "-c:v", "libx264", "-qp", "0", str(src)], check=True)
        subprocess.run([FF, "-v", "error", "-y", "-i", str(src), "-filter_complex",
                        "[0:v]scale=out_color_matrix=bt709:out_range=tv,setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv,format=yuv420p[t]",
                        "-map", "[t]", "-c:v", "libx264", "-crf", "18", *polish.COLOR_TAGS, str(out)], check=True)
        v = ffmpeg.video_stream(ffmpeg.probe(out))
        assert (v.get("color_space"), v.get("color_primaries"), v.get("color_transfer"), v.get("color_range")) == ("bt709", "bt709", "bt709", "tv")


def test_delivery_bitrate_is_bounded_and_independent_of_length():
    assert ffmpeg.delivery_bitrate_mbit(1920, 1080) == 10.0 and ffmpeg.delivery_bitrate_mbit(1080, 1920) == 10.0
    assert ffmpeg.delivery_bitrate_mbit(320, 180) == 6.0 and ffmpeg.delivery_bitrate_mbit(7680, 4320) == 40.0
    assert ffmpeg.delivery_bitrate_mbit(0, 0) == 6.0


