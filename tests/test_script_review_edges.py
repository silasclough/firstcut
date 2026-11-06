"""Edge cases and bad input for the plan review screen (tester's additions)."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from clipfarm.clipper import flow as FL
from clipfarm.clipper import localproj as LP
from clipfarm.clipper import moments as M
from tests.test_script_review import OUTLINE, items, sents


def ab(out, step=None):
    return [(i["a"], i["b"]) for i in out if step is None or i.get("step") == step]


# ------------------------------------------------------------------ edit_items
def test_extend_at_the_start_and_end_of_the_transcript_is_refused():
    s = sents()
    first = [{"kind": "talk", "a": 0, "b": 1, "step": 1}]
    last = [{"kind": "talk", "a": 7, "b": 8, "step": 3}]
    with pytest.raises(ValueError, match="no line"):
        FL.edit_items(s, first, {"op": "extend", "first": 0, "last": 1, "edge": "start"})
    with pytest.raises(ValueError, match="no line"):
        FL.edit_items(s, last, {"op": "extend", "first": 7, "last": 8, "edge": "end"})


def test_dropping_from_a_passage_that_was_already_split_needs_the_new_bounds():
    s = sents()
    split = FL.edit_items(s, items(), {"op": "drop", "first": 0, "last": 2, "sentence": 1})
    with pytest.raises(ValueError, match="changed"):                       # a second click on the old (0, 2) passage
        FL.edit_items(s, split, {"op": "drop", "first": 0, "last": 2, "sentence": 0})
    out = FL.edit_items(s, split, {"op": "drop", "first": 2, "last": 2, "sentence": 2})
    assert ab(out, 1) == [(0, 0)]


def test_a_line_outside_the_passage_cannot_be_dropped():
    with pytest.raises(ValueError, match="not in the passage"):
        FL.edit_items(sents(), items(), {"op": "drop", "first": 0, "last": 2, "sentence": 7})


def test_add_overlapping_a_used_line_even_partly_is_refused_and_input_is_not_mutated():
    s, its = sents(), items()
    before = json.dumps(its)
    for a, b in ((2, 5), (4, 6), (0, 0), (3, 8)):
        with pytest.raises(ValueError, match="already used"):
            FL.edit_items(s, its, {"op": "add", "step": 2, "first": a, "last": b})
    assert json.dumps(its) == before


def test_add_refuses_missing_reversed_or_unknown_lines():
    for first, last in ((None, 5), (5, None), (6, 5), (-1, 5), (5, 99)):
        with pytest.raises(ValueError):
            FL.edit_items(sents(), items(), {"op": "add", "step": 2, "first": first, "last": last})


def test_add_to_a_step_without_passages_goes_after_the_earlier_steps():
    its = [{"kind": "talk", "a": 0, "b": 1, "step": 1}, {"kind": "talk", "a": 7, "b": 8, "step": 3}]
    out = FL.edit_items(sents(), its, {"op": "add", "step": 2, "first": 3, "last": 4})
    assert [i["step"] for i in out] == [1, 2, 3]


def test_ops_ignore_non_talk_items_and_keep_them():
    its = items()
    its.insert(1, {"kind": "broll", "step": 1, "a": 0, "b": 2, "file": "x.mov"})            # same bounds as a talk passage
    out = FL.edit_items(sents(), its, {"op": "drop", "first": 0, "last": 2, "sentence": 2})
    assert any(i.get("kind") == "broll" for i in out) and ab([i for i in out if i.get("kind") == "talk"], 1) == [(0, 1)]
    only_photo = [{"kind": "photo", "step": 1}]                                              # no a/b at all
    with pytest.raises(ValueError, match="changed"):
        FL.edit_items(sents(), only_photo, {"op": "drop", "first": 0, "last": 0, "sentence": 0})
    out = FL.edit_items(sents(), only_photo, {"op": "add", "step": 1, "first": 0, "last": 0})
    assert len(out) == 2


def test_unknown_op_and_missing_op():
    for op in ({}, {"op": "reorder"}, {"op": None}):
        with pytest.raises(ValueError, match="unknown"):
            FL.edit_items(sents(), items(), op)


def test_split_halves_keep_the_step():
    out = FL.edit_items(sents(), items(), {"op": "drop", "first": 0, "last": 2, "sentence": 1})
    assert [i["step"] for i in out[:2]] == [1, 1]


# Bad types must be a plain-words ValueError (the route turns that into a 409), never a TypeError (a 500).
BAD = [
    {"op": "drop", "first": 0, "last": 2, "sentence": "1"},
    {"op": "drop", "first": 0, "last": 2, "sentence": None},
    {"op": "drop", "first": "0", "last": "2", "sentence": 1},
    {"op": "extend", "first": 3, "last": 4, "edge": "middle"},
    {"op": "extend", "first": 3, "last": 4},
    {"op": "add", "step": None, "first": 5, "last": 5},
    {"op": "add", "step": "2", "first": 5, "last": 5},
    {"op": "add", "step": 2, "first": "5", "last": "5"},
    {"op": "add", "step": 2, "first": 5.5, "last": 5.5},
    {"op": "add", "step": 99, "first": 5, "last": 5},
    {"op": "add", "step": 0, "first": 5, "last": 5},
    {"op": "add", "step": -1, "first": 5, "last": 5},
]


@pytest.mark.parametrize("op", BAD)
def test_bad_values_are_refused_with_a_value_error(op):
    try:
        out = FL.edit_items(sents(), items(), op, len(OUTLINE))
    except ValueError:
        return
    pytest.fail(f"accepted {op}: {out}")


# ------------------------------------------------------------------ script_map
def test_script_map_with_an_empty_plan_no_passages_and_no_sentences():
    m = FL.script_map(sents(), [], OUTLINE)
    assert [x["passages"] for x in m] == [[], [], []] and m[0]["n"] == 1
    assert FL.script_map([], [], OUTLINE)[1]["passages"] == []
    assert FL.script_map(sents(), items(), []) == []


def test_script_map_step_without_items_and_non_talk_items():
    its = [{"kind": "talk", "a": 0, "b": 2, "step": 1}, {"kind": "broll", "step": 2, "a": 3, "b": 4}]
    m = FL.script_map(sents(), its, OUTLINE)
    assert m[1]["passages"] == [] and m[0]["passages"][0]["after"]["sentence"] == 3            # broll does not use up line 3


def test_script_map_items_pointing_past_the_transcript_do_not_crash():
    m = FL.script_map(sents(), [{"kind": "talk", "a": 7, "b": 20, "step": 3}], OUTLINE)
    assert [ln["sentence"] for ln in m[2]["passages"][0]["lines"]] == [7, 8]


def test_script_map_kind_missing_counts_as_talk():
    m = FL.script_map(sents(), [{"a": 0, "b": 2, "step": 1}], OUTLINE)
    assert len(m[0]["passages"]) == 1


# ------------------------------------------------------------------ runner
class FakeState:
    def __init__(self, tmp):
        self.st = LP.State(tmp)

    def __getattr__(self, k):
        return getattr(self.st, k)


def _runner(tmp_path, plan=None, status="approval", notes=None):
    st = LP.State(tmp_path)
    st.put("plan.json", plan if plan is not None else {"items": items(), "awaiting_approval": True, "flow": []})
    r = LP.LocalRunner.__new__(LP.LocalRunner)
    r.review_lock = threading.Lock()
    r.running = None
    proj = {"notes": notes if notes is not None else "Script:\n- " + "\n- ".join(OUTLINE), "job": "j"}
    r._setup = lambda pid: (proj, tmp_path, st, None, None)

    class Store:
        def get(self, j):
            return {"status": status, "options": {}}
    r.store = Store()
    r.saved_inputs = lambda folder, st_, ai, tr: (None, None, None, [])
    import clipfarm.clipper.localproj as lp
    return r, st, lp


def test_review_edit_without_a_script_is_refused_and_changes_nothing(tmp_path, monkeypatch):
    r, st, lp = _runner(tmp_path, notes="just make it good")
    monkeypatch.setattr(lp, "seg", lambda words: sents())
    with pytest.raises(ValueError, match="no script"):
        r.review_edit("p", {"op": "drop", "first": 0, "last": 2, "sentence": 2})
    assert st.get("plan.json")["items"] == items() and not (st.dir / "backups").exists()


@pytest.mark.parametrize("status,plan", [("done", None), ("approval", {"items": items()}), ("approval", {})])
def test_review_edit_and_find_refuse_when_nothing_awaits_approval(tmp_path, monkeypatch, status, plan):
    r, st, lp = _runner(tmp_path, plan=plan, status=status)
    monkeypatch.setattr(lp, "seg", lambda words: sents())
    with pytest.raises(ValueError, match="no plan waiting"):
        r.review_edit("p", {"op": "drop", "first": 0, "last": 2, "sentence": 2})
    with pytest.raises(ValueError, match="no plan waiting"):
        r.review_find("p", 1, "x")


def test_review_find_rejects_steps_out_of_range(tmp_path, monkeypatch):
    r, st, lp = _runner(tmp_path)
    monkeypatch.setattr(lp, "seg", lambda words: sents())
    for step in (0, -1, 4, 10 ** 9):
        with pytest.raises(ValueError, match="no such step"):
            r.review_find("p", step, "x")


def test_a_failed_edit_leaves_no_half_written_plan(tmp_path, monkeypatch):
    r, st, lp = _runner(tmp_path)
    monkeypatch.setattr(lp, "seg", lambda words: sents())
    with pytest.raises(ValueError):
        r.review_edit("p", {"op": "extend", "first": 3, "last": 4, "edge": "start"})
    saved = st.get("plan.json")
    assert saved["items"] == items() and not saved.get("flow_edited")


def test_concurrent_edits_are_serialised_and_every_one_lands(tmp_path, monkeypatch):
    r, st, lp = _runner(tmp_path, plan={"items": [{"kind": "talk", "a": 0, "b": 0, "step": 1}], "awaiting_approval": True})
    monkeypatch.setattr(lp, "seg", lambda words: sents())
    errs = []

    def go(op):
        try:
            r.review_edit("p", op)
        except Exception as e:            # noqa: BLE001
            errs.append(e)
    ops = [{"op": "add", "step": 2, "first": i, "last": i} for i in (2, 3, 4, 5, 6)]
    ts = [threading.Thread(target=go, args=(o,)) for o in ops]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs
    got = sorted(i["a"] for i in st.get("plan.json")["items"])
    assert got == [0, 2, 3, 4, 5, 6]


def test_two_clicks_on_the_same_stale_passage_second_is_refused(tmp_path, monkeypatch):
    r, st, lp = _runner(tmp_path)
    monkeypatch.setattr(lp, "seg", lambda words: sents())
    op = {"op": "drop", "first": 0, "last": 2, "sentence": 2}
    r.review_edit("p", op)
    with pytest.raises(ValueError, match="changed"):
        r.review_edit("p", op)
    assert ab(st.get("plan.json")["items"], 1) == [(0, 1)]
    assert len(list((st.dir / "backups").glob("*before-review-edits*"))) == 1             # one backup, of the original plan
    assert json.loads(next((st.dir / "backups").glob("*before-review-edits*")).read_text())["items"] == items()


def test_approve_renders_the_edited_items(tmp_path, monkeypatch):
    r, st, lp = _runner(tmp_path)
    monkeypatch.setattr(lp, "seg", lambda words: sents())
    r.review_edit("p", {"op": "drop", "first": 0, "last": 2, "sentence": 2})
    seen = {}
    r.rerender = lambda pid: seen.update(plan=st.get("plan.json"))
    r.approve("p")
    p = seen["plan"]
    assert ab(p["items"], 1) == [(0, 1)] and "awaiting_approval" not in p and p["pending_render"]
    assert list((st.dir / "backups").glob("*approved-script-map*"))


def test_nothing_renders_during_review(tmp_path, monkeypatch):
    r, st, lp = _runner(tmp_path)
    monkeypatch.setattr(lp, "seg", lambda words: sents())
    r.rerender = lambda pid: pytest.fail("rendered")
    r.render = lambda *a, **k: pytest.fail("rendered")
    r.review_edit("p", {"op": "add", "step": 2, "first": 5, "last": 5})
    assert st.get("plan.json")["awaiting_approval"] is True


# ------------------------------------------------------------------ HTTP routes
def _server(runner, tmp, status="approval", mode="local"):
    from clipfarm.clipper import server as S

    class P:
        def get(self, pid):
            return {"id": pid, "mode": mode, "job": "j"}

        def view(self, p):
            return {"id": p["id"]}

    class St:
        def get(self, j):
            return {"status": status}
    from clipfarm.clipper.engine import Engine
    from clipfarm.clipper.store import Store
    from clipfarm.core.config import load_config
    cfg = load_config()
    cfg.root = tmp
    store = Store(tmp / "jobs")
    h = S.make_handler(Engine(cfg, store), St(), None, None, 100, P(), runner)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def _post(httpd, path, body, raw=None):
    data = raw if raw is not None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}{path}", data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}


class StubRunner:
    running = None

    def __init__(self):
        self.calls = []

    def review_edit(self, pid, op):
        self.calls.append(("edit", op))
        if op.get("op") == "boom":
            raise ValueError("plain words")

    def review_find(self, pid, step, query):
        self.calls.append(("find", step, query))
        return []


def test_routes_pass_only_known_fields_and_turn_value_errors_into_409(tmp_path):
    r = StubRunner()
    httpd = _server(r, tmp_path)
    try:
        code, body = _post(httpd, "/api/ai/projects/p/plan-edit", {"op": "drop", "first": 1, "last": 2, "sentence": 1, "evil": "x"})
        assert code == 200 and r.calls[0] == ("edit", {"op": "drop", "first": 1, "last": 2, "sentence": 1})
        assert _post(httpd, "/api/ai/projects/p/plan-edit", {"op": "boom"})[0] == 409
        assert _post(httpd, "/api/ai/projects/p/plan-find", {"step": 1, "query": "   "})[0] == 400
        assert _post(httpd, "/api/ai/projects/p/plan-find", {"step": 1})[0] == 400
        code, body = _post(httpd, "/api/ai/projects/p/plan-find", {"step": 2, "query": "x" * 100000})
        assert code == 200 and len(r.calls[-1][2]) == 300                         # query is capped
        r.running = "p"
        assert _post(httpd, "/api/ai/projects/p/plan-edit", {"op": "drop"})[0] == 409
    finally:
        httpd.shutdown()


@pytest.mark.parametrize("step", ["abc", [1], {"a": 1}, 1.5, None, "", "3"])
def test_find_route_with_odd_step_values_never_500(tmp_path, step):
    r = StubRunner()
    httpd = _server(r, tmp_path)
    try:
        code, _ = _post(httpd, "/api/ai/projects/p/plan-find", {"step": step, "query": "x"})
        assert code < 500, (step, code)
    finally:
        httpd.shutdown()


@pytest.mark.parametrize("raw", [b"not json", b"[]", b'"x"', b"null", b"123"])
def test_edit_route_with_non_object_bodies_never_500(tmp_path, raw):
    r = StubRunner()
    httpd = _server(r, tmp_path)
    try:
        code, _ = _post(httpd, "/api/ai/projects/p/plan-edit", None, raw=raw)
        assert code < 500, (raw, code)
    finally:
        httpd.shutdown()


def test_edit_route_with_a_real_runner_and_bad_types_gives_409_not_500(tmp_path, monkeypatch):
    r, st, lp = _runner(tmp_path)
    monkeypatch.setattr(lp, "seg", lambda words: sents())
    httpd = _server(r, tmp_path)
    try:
        for op in BAD:
            code, body = _post(httpd, "/api/ai/projects/p/plan-edit", op)
            assert code in (200, 400, 409), (op, code)           # 400 comes from the generic TypeError catch
    finally:
        httpd.shutdown()


def test_view_survives_a_script_map_saved_before_lines_had_times(tmp_path):
    """A project already waiting for approval when this feature shipped has lines without start/end."""
    from clipfarm.clipper import aivideo as AV
    from clipfarm.clipper.engine import Engine
    from clipfarm.clipper.store import Store
    from clipfarm.core.config import load_config
    cfg = load_config()
    cfg.root = tmp_path
    store = Store(tmp_path / "jobs")
    projects = AV.Projects(cfg, store, Engine(cfg, store))
    job = store.create({"files": []}, {})
    folder = tmp_path / "f"
    (folder / ".aivideo").mkdir(parents=True)
    old_map = [{"n": 1, "step": "x", "passages": [{"first": 0, "last": 0, "lines": [{"sentence": 0, "text": "Hi."}]}]}]
    (folder / ".aivideo" / "plan.json").write_text(json.dumps({"awaiting_approval": True, "script_map": old_map}))
    v = projects.view({"id": "p", "job": job["id"], "mode": "local", "folder": str(folder)})
    assert v["awaiting_approval"] is True and v["script_map"]
