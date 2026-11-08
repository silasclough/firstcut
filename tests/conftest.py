"""Every test runs offline: nothing may reach a real AI (local Ollama, OpenAI), another machine over SSH, or the Claude CLI.

A test that did once loaded the 14B model on the M3 laptop (heavy AI must never run there) and could have spent money on OpenAI.
Connections to this machine's own test servers (127.0.0.1 on a random port) still work; anything else fails at once, so a test that
forgets to fake its AI fails loudly instead of quietly calling the real thing.
"""

import socket
import subprocess

import pytest

OLLAMA_PORT = 11434
BLOCKED_CLI = {"ssh", "scp", "rsync", "claude", "ollama", "osascript", "parakeet-mlx", "trash"}   # (trash: never the real Trash from a test)


class NetworkBlocked(OSError):
    pass


def _local(addr) -> bool:
    if isinstance(addr, tuple) and len(addr) >= 2:
        return addr[0] in ("127.0.0.1", "::1", "localhost") and addr[1] != OLLAMA_PORT
    return isinstance(addr, (str, bytes))                       # unix sockets (multiprocessing and friends)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    from clipfarm.clipper import claude_cli                    # the dev Mac has Claude switched on in its local config: tests never use it
    monkeypatch.setattr(claude_cli.ClaudeCLI, "available", lambda self: False)
    from clipfarm.clipper import net                            # the code believes it is online; a test of offline behaviour says so
    monkeypatch.setattr(net, "online", lambda *a, **k: True)
    real_connect, real_cc = socket.socket.connect, socket.create_connection

    def connect(self, addr):
        if not _local(addr):
            raise NetworkBlocked(f"tests run offline: no connection to {addr}")
        return real_connect(self, addr)

    def create_connection(address, *a, **k):
        if not _local(address):
            raise NetworkBlocked(f"tests run offline: no connection to {address}")
        return real_cc(address, *a, **k)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "create_connection", create_connection)

    def blocked(cmd) -> bool:
        c = cmd if isinstance(cmd, (list, tuple)) else [cmd]
        return any(str(x).rsplit("/", 1)[-1] in BLOCKED_CLI for x in c[:5])      # (also behind "nice -n 10 ...")

    real_run, real_popen = subprocess.run, subprocess.Popen

    def run(cmd, *a, **k):
        if blocked(cmd):
            raise NetworkBlocked(f"tests run offline: {cmd[0]} is not run")
        return real_run(cmd, *a, **k)

    class Popen(real_popen):
        def __init__(self, cmd, *a, **k):
            if blocked(cmd):
                raise NetworkBlocked(f"tests run offline: {cmd[0]} is not run")
            super().__init__(cmd, *a, **k)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "Popen", Popen)
    yield


# ---- public build: tests of parts that aren't published (written by the export; see each reason)
PUBLIC_SKIP = {'tests/test_aivideo.py::test_folder_scan_orders_by_shooting_time_skips_results_and_marks_broll': "needs scripts/ai_edit.py, the older remote folder client that isn't part of FirstCut", 'tests/test_aivideo.py::test_server_gives_the_folder_client_everything_it_needs': "needs scripts/ai_edit.py, the older remote folder client that isn't part of FirstCut", 'tests/test_aivideo.py::test_upload_resumes_after_the_server_restarts_and_only_missing_files_are_sent': "needs scripts/ai_edit.py, the older remote folder client that isn't part of FirstCut", 'tests/test_aivideo.py::test_the_video_starts_by_itself_when_the_last_file_arrives_if_asked': "needs scripts/ai_edit.py, the older remote folder client that isn't part of FirstCut", 'tests/test_aivideo.py::test_a_half_received_file_is_not_counted_as_arrived': "needs scripts/ai_edit.py, the older remote folder client that isn't part of FirstCut", 'tests/test_catalog.py::test_the_shipped_work_config_is_local_only': 'a private work-drive config', 'tests/test_catalog_edges.py::test_the_work_overlay_config_applies_from_a_relative_path': 'a private work-drive config', 'tests/test_catalog_edges.py::test_vb_talks_to_the_server_and_says_what_went_wrong': 'a private helper script', 'tests/test_clip_openings_edge.py::test_feed_and_ingest_with_clips_that_lack_title_or_qa_keys': "the posting relay isn't part of FirstCut", 'tests/test_clip_openings_edge.py::test_feed_hides_what_the_gate_refuses_but_shows_own_uploads': "the posting relay isn't part of FirstCut", 'tests/test_clip_openings_edge.py::test_own_upload_is_exempt_in_ingest_and_in_the_feed': "the posting relay isn't part of FirstCut", 'tests/test_clip_openings.py::test_the_swipe_feed_hides_what_the_posting_gate_would_refuse': "the review feed isn't part of FirstCut", 'tests/test_studio_edges.py::test_make_config_stops_on_personal_details': "checks the private pattern list, which isn't published"}


def pytest_collection_modifyitems(config, items):
    for item in items:
        why = PUBLIC_SKIP.get(item.nodeid.split("[")[0])
        if why:
            item.add_marker(pytest.mark.skip(reason=why))
