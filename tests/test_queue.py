import json
import multiprocessing

import pytest

from clipfarm.core.queue import AlreadyRunning, InvalidTransition, JobQueue, State, can_transition


@pytest.fixture
def q(tmp_path):
    return JobQueue(tmp_path / "queue")


def test_happy_path(q):
    job = q.create("minecraft", "brand", {"seed": 1})
    assert job.state == State.IDEA
    for s in (State.SCRIPTED, State.RENDERED, State.REVIEW, State.APPROVED, State.SCHEDULED, State.POSTED):
        q.transition(job, s)
    assert q.get(job.id).state == State.POSTED
    assert [h["to"] for h in q.get(job.id).history] == ["idea", "scripted", "rendered", "review", "approved",
                                                         "scheduled", "posted"]


def test_auto_approve_skips_review(q):
    job = q.create("minecraft", "b")
    q.transition(job, State.SCRIPTED)
    q.transition(job, State.RENDERED)
    q.transition(job, State.APPROVED)
    assert job.state == State.APPROVED


@pytest.mark.parametrize("src,dst", [
    (State.IDEA, State.RENDERED), (State.IDEA, State.POSTED), (State.REVIEW, State.SCHEDULED),
    (State.POSTED, State.IDEA), (State.REJECTED, State.APPROVED), (State.SCRIPTED, State.REVIEW),
])
def test_illegal_transitions(src, dst):
    assert not can_transition(src, dst)


def test_illegal_transition_raises_and_does_not_persist(q):
    job = q.create("minecraft", "b")
    with pytest.raises(InvalidTransition):
        q.transition(job, State.POSTED)
    assert q.get(job.id).state == State.IDEA


def test_same_state_transition_is_noop(q):
    job = q.create("minecraft", "b")
    n = len(job.history)
    q.transition(job, State.IDEA)
    assert len(job.history) == n


def test_fail_and_retry_resume_from_last_good_state(q):
    job = q.create("minecraft", "b")
    q.transition(job, State.SCRIPTED)
    q.record_attempt(job, "render")
    q.fail(job, "render", "ffmpeg exploded")
    j = q.get(job.id)
    assert j.state == State.FAILED and "ffmpeg exploded" in j.error and j.last_good_state == "scripted"
    q.retry(j)
    j = q.get(job.id)
    assert j.state == State.SCRIPTED and j.error is None and j.attempts == {}


def test_retry_requires_failed(q):
    job = q.create("minecraft", "b")
    with pytest.raises(InvalidTransition):
        q.retry(job)


def test_persistence_and_crash_safety(q, tmp_path):
    job = q.create("streamer", "b", {"x": 1})
    q.transition(job, State.SCRIPTED)
    # a fresh queue object (= process restart) sees the same state
    q2 = JobQueue(tmp_path / "queue")
    assert q2.get(job.id).state == State.SCRIPTED
    # a leftover temp file from a crashed write is ignored
    (tmp_path / "queue" / f".{job.id}.json.abc.tmp").write_text("{garbage")
    assert [j.id for j in q2.all()] == [job.id]
    # a corrupt job file is logged and skipped, not fatal
    (tmp_path / "queue" / "broken.json").write_text("{not json")
    assert [j.id for j in q2.all()] == [job.id]


def test_list_filters_and_stats(q):
    a = q.create("minecraft", "b1")
    b = q.create("streamer", "b2")
    q.transition(b, State.SCRIPTED)
    assert [j.id for j in q.list(State.IDEA)] == [a.id]
    assert [j.id for j in q.list(fmt="streamer")] == [b.id]
    assert q.stats() == {"minecraft": {"idea": 1}, "streamer": {"scripted": 1}}


def test_prefix_lookup(q):
    job = q.create("minecraft", "b")
    assert q.get(job.id[:-2]).id == job.id


def test_dry_run_queue_writes_nothing(tmp_path):
    dq = JobQueue(tmp_path / "dry", dry_run=True)
    job = dq.create("minecraft", "b")
    dq.transition(job, State.SCRIPTED)
    assert dq.get(job.id).state == State.SCRIPTED
    assert not (tmp_path / "dry").exists()


def _try_lock(path, out):
    try:
        with JobQueue(path).lock("t"):
            out.put("got")
    except AlreadyRunning:
        out.put("busy")


def test_lock_is_exclusive(q, tmp_path):
    out = multiprocessing.get_context("spawn").Queue()
    with q.lock("t"):
        p = multiprocessing.get_context("spawn").Process(target=_try_lock, args=(tmp_path / "queue", out))
        p.start()
        p.join(30)
        assert out.get(timeout=5) == "busy"
    p = multiprocessing.get_context("spawn").Process(target=_try_lock, args=(tmp_path / "queue", out))
    p.start()
    p.join(30)
    assert out.get(timeout=5) == "got"


def test_job_file_is_valid_json(q, tmp_path):
    job = q.create("aigen", "b", {"ai_generated": True})
    d = json.loads((tmp_path / "queue" / f"{job.id}.json").read_text())
    assert d["state"] == "idea" and d["data"]["ai_generated"] is True
