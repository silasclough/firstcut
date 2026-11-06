"""A new version moves the older ones to the Trash (2026-11-06: "there's three versions of this video and it didn't delete the old ones")."""
from clipfarm.clipper import localproj as LP


def test_older_videos_and_xml_go_to_the_trash_and_final_cut_folders_stay(tmp_path, monkeypatch):
    trashed = []
    monkeypatch.setattr(LP, "to_trash", lambda paths: trashed.extend(p.name for p in paths))
    out = tmp_path / "Output"
    (tmp_path / "Final Cut" / "Talk v1").mkdir(parents=True)
    out.mkdir()
    for v in (1, 2, 3):
        (out / f"Talk v{v}.mp4").write_bytes(b"x")
        (out / f"Talk v{v}.srt").write_text("1")
        (tmp_path / f"Talk v{v}.fcpxml").write_text("<fcpxml/>")
    (out / "my own notes v2.txt").write_text("keep")                                      # not ours
    (out / ".Talk v4.mp4.part").write_bytes(b"x")                                          # (in progress)
    LP.older_versions_to_trash(tmp_path, 3)
    assert sorted(trashed) == ["Talk v1.fcpxml", "Talk v1.mp4", "Talk v1.srt", "Talk v2.fcpxml", "Talk v2.mp4", "Talk v2.srt"]
    assert (tmp_path / "Final Cut" / "Talk v1").is_dir()


def test_the_trash_is_the_real_trash_not_a_delete(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(LP.subprocess, "run", lambda cmd, **k: calls.append(cmd))
    monkeypatch.setattr(LP.shutil, "which", lambda name: "/usr/bin/trash" if name == "trash" else None)
    f = tmp_path / "Talk v1.mp4"
    f.write_bytes(b"x")
    LP.to_trash([f])
    assert calls == [["/usr/bin/trash", str(f)]] and f.exists()                             # (handed to macOS; never unlinked here)


def test_final_cut_puts_each_video_in_an_event_named_after_it():
    """It imported every project into an event called "Clipper"; now the video's name, all its versions in one event."""
    from clipfarm.clipper import fcpxml as F
    assert F.event_name("Weekend Workshop Recap v3") == "Weekend Workshop Recap"
    assert F.event_name("Talk") == "Talk" and F.event_name("") == "FirstCut"
