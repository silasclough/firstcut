"""Phase 1 MCP tools (capabilities, style_profile, footage_details, change_video style) with bad arguments: an error result, never a crash."""
import json
import sys

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_mcp_server import call, engine, rpc  # noqa: E402,F401  (the engine fixture)


def test_capabilities_ignores_stray_arguments(engine):
    r, text = call(engine["server"], "capabilities", junk=[1, 2], style="x")
    assert not r["isError"] and "profiles" in json.loads(text)


def test_style_profile_bad_actions_and_styles_are_errors_not_crashes(engine):
    for kw in ({"action": "explode"}, {"action": "set", "style": "fast"}, {"action": "set", "style": []}, {"action": "set", "style": None},
               {"action": "set", "style": {"pace": {"name": "warp"}}, "scope": 5}, {"project": "no such video"}):
        out = rpc(engine["server"], {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "style_profile", "arguments": {"project": "p1", **kw}}})[0]
        assert "result" in out or out["error"]["code"] in (-32603, -32602), out
        if "result" in out and kw.get("action") == "set" and not kw.get("style"):
            assert out["result"]["isError"]


def test_footage_details_on_missing_and_empty_folders(engine, tmp_path):
    for folder in (str(tmp_path / "nope"), "", str(tmp_path)):
        out = rpc(engine["server"], {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "footage_details", "arguments": {"folder": folder}}})[0]
        assert "result" in out or "error" in out
        if "result" in out:
            assert out["result"]["isError"] or out["result"]["content"]


def test_footage_details_survives_corrupt_cache_and_plan(engine, tmp_path, monkeypatch):
    from clipfarm.clipper import mcp_server as M
    (tmp_path / ".aivideo" / "facts").mkdir(parents=True)
    (tmp_path / ".aivideo" / "plan.json").write_text("{not json")
    (tmp_path / ".aivideo" / "broll.json").write_text(json.dumps({"a.mp4": {"scenes": [{"start": 1, "end": 2, "what": None}]}}))
    (tmp_path / ".aivideo" / "facts" / "junk.json").write_text("\x00\x01")
    monkeypatch.setattr(M, "_allowed", lambda p: True)
    notes = M._facts_notes(str(tmp_path), [{"name": "a.mp4", "path": str(tmp_path / "a.mp4")}])
    assert "a.mp4" in notes
