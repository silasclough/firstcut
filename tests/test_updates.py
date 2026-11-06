"""Automatic updates (app/VideoBuilder/updates.swift, Sparkle 2): the setting, the busy rule the Swift side uses, and the release files."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_updates_setting():
    from clipfarm.clipper import studio as SD
    assert SD.defaults()["updates"] == "auto"
    assert SD.clean({"updates": "ask"})["updates"] == "ask" and SD.clean({"updates": "off"})["updates"] == "off"
    assert "updates" not in SD.clean({"updates": "now please"})


def _blocking(what: str) -> bool:                       # the rule in updates.swift (Updates.blocking), kept in step by the test below
    w = what.lower()
    return bool(w) and "catalogu" not in w and "download" not in w


def test_the_swift_busy_rule_matches_the_engines_words():
    """An update never restarts FirstCut in the middle of a video, clips or a transcript; cataloguing and Setup's downloads carry on
    after the restart, so they don't hold it back. The engine's wording (studio_web.busy) and the Swift rule must agree."""
    swift = (ROOT / "app/VideoBuilder/updates.swift").read_text()
    assert 'return !w.isEmpty && !w.contains("catalogu") && !w.contains("download")' in swift
    web = (ROOT / "clipfarm/clipper/studio_web.py").read_text()
    says = re.findall(r'return "([^"]+)"', web[web.index("def busy"):web.index("def busy") + 1200])
    assert {s: _blocking(s) for s in says} == {"a video is being made": True, "clips are being made": True,
                                                "your footage library is being catalogued": False, "downloads are running": False,
                                                "a transcript is being written": True}
    assert _blocking("") is False


def test_release_files_are_in_place():
    v = (ROOT / "app/release/VERSION").read_text().strip()
    assert re.fullmatch(r"\d+\.\d+\.\d+", v)
    assert f"## {v}" in (ROOT / "app/release/CHANGES.md").read_text()                   # the update card's notes
    key = (ROOT / "app/release/SUPublicEDKey").read_text().strip()
    assert len(key) == 44 and key.endswith("=")                                           # (public: the private key stays in the Keychain)
    pkg = (ROOT / "scripts/package_videobuilder.sh").read_text()
    assert "SUFeedURL" in pkg and "SUPublicEDKey" in pkg and "updates.swift" in pkg and "SPARKLE_SHA=" in pkg
    for script in ("scripts/package_videobuilder.sh", "scripts/release.sh"):
        assert subprocess.run(["bash", "-n", str(ROOT / script)]).returncode == 0
