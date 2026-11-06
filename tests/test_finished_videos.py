"""Finished videos (exports, finals) are not footage for the catalogue; a shot's playable copy is made once."""
import threading
from pathlib import Path

from clipfarm.clipper import catalog as C


def test_exports_and_finals_are_not_footage():
    f = C.finished_video
    assert f(Path("Projects/Fall Recap/Final/Fall Recap.mov"))
    assert f(Path("Projects/Fall Recap/Exports/anything.mp4"))
    assert f(Path("Projects/Fall Recap/Fall Recap FINAL v2.mov"))
    assert f(Path("Projects/Fall Recap/Delivered/clip.mp4"))


def test_camera_files_are_footage_even_with_odd_names():
    f = C.finished_video
    assert not f(Path("Video/B-roll/Athletics/C5211.MP4"))
    assert not f(Path("Projects/Fall Recap/raw/final_take_v2.mp4"))
    assert not f(Path("Video/B-roll/Campus/Quad in autumn.mp4"))


def test_the_colour_change_goes_after_the_scale(tmp_path):
    cat = C.Catalog.__new__(C.Catalog)
    cat._lut = lambda p: "lut3d=file='x.cube',"
    assert cat._lut_after("a.mp4") == ",lut3d=file='x.cube'"
    cat._lut = lambda p: ""
    assert cat._lut_after("a.mp4") == ""


def test_one_lock_per_preview_name():
    cat = C.Catalog.__new__(C.Catalog)
    cat._idx_lock = threading.Lock()
    assert cat._preview_lock("a.mp4") is cat._preview_lock("a.mp4")
    assert cat._preview_lock("a.mp4") is not cat._preview_lock("b.mp4")
