"""The plan of a long video as something an outside AI (the MCP) can read, check, edit and send back (WP5 of docs/plans/editor-improvement-plan.md).

    view      the saved plan with readable lines (every passage has an id, its seconds and its words), the constraints with how each was understood,
              the problems `constraints.check_plan` finds, and what is unmet. Nothing is rendered.
    check     a plan an outside AI edited, checked in code: every id must come from the lists FirstCut made (sentence numbers, file names, scene ids).
    apply     the same check, then the edited plan is saved and rendered (a plan with a hard problem is refused with the reasons).
    search    a quote or a topic in the transcript, as word numbers and seconds.
    preview   up to 60 s of the finished video at 480 px, or one still at each cut: a cheap look without a full render.

All of them work on the saved files of a video (`plan.json`, the words, the footage list); none of them calls an AI except a topic search.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from ..core import ffmpeg
from ..core.log import get_logger
from . import constraints as CN
from . import transcript_search as TS

log = get_logger("clipper.planapi")
PREVIEW_MAX_S = 60.0
PREVIEW_W = 480
MAX_FRAMES = 12


def _inputs(runner, pid: str):
    from .localproj import seg
    proj, folder, st, ai, _engine = runner._setup(pid)
    job = runner.store.get(proj["job"])
    files, _group, _all_words, words = runner.saved_inputs(folder, st, ai, (job.get("options") or {}).get("transcriber", "parakeet"))
    return proj, folder, st, ai, job, files, words, seg(words)


def with_ids(plan: dict) -> dict:
    """The plan with a stable `id` (i0, i1 ...) on every item (an id that is already there is kept)."""
    items, dropped = [], 0
    for i, it in enumerate(plan.get("items") or []):
        if not isinstance(it, dict):                                 # (a damaged plan: a part that is not a part is left out, not a crash)
            dropped += 1
            continue
        it = dict(it)
        it.setdefault("id", f"i{i}")
        items.append(it)
    out = {**plan, "items": items}
    if dropped:
        out["problems"] = [*(plan.get("problems") or []), {"message": f"{dropped} damaged part(s) of the plan were left out.", "hard": False}]
    return out


def _readable(plan: dict, sents) -> list[dict]:
    by = {s.idx: s for s in sents}
    out = []
    for it in with_ids(plan)["items"]:
        row = {"id": it["id"], "kind": it.get("kind"), "satisfies": it.get("satisfies") or []}
        if it.get("kind") == "talk":
            a, b = it.get("a"), it.get("b")
            row.update({"a": a, "b": b})
            if a in by and b in by:
                row.update({"from_s": round(by[a].start, 1), "to_s": round(by[b].end, 1),
                            "text": " ".join(by[i].text for i in range(a, b + 1) if i in by)[:240]})
        elif it.get("kind") == "visual":
            row.update({"start": it.get("start"), "end": it.get("end"), "file": it.get("file")})
        out.append(row)
    return out


def view(runner, pid: str) -> dict:
    proj, folder, st, ai, job, files, words, sents = _inputs(runner, pid)
    plan = st.get("plan.json")
    if not plan:
        raise ValueError("this video has no plan yet")
    plan = with_ids(plan)
    res = CN.check_plan(plan, plan.get("constraints") or [], files, words, plan.get("pool") or [], plan.get("style_obj"), sents=sents)
    pool = [{"sid": s.get("sid"), "file": s.get("file"), "fstart": s.get("fstart"), "fend": s.get("fend"), "what": str(s.get("what") or "")[:120]}
            for s in (plan.get("pool") or [])[:300]]
    return {"status": job.get("status"), "stage": job.get("stage"), "title": plan.get("title"), "summary": plan.get("summary"), "dry_run": bool(plan.get("dry_run")),
            "awaiting_approval": bool(plan.get("awaiting_approval")), "approval": plan.get("approval"), "items": _readable(plan, sents),
            "constraints": plan.get("constraints") or [], "problems": res["problems"], "totals": res["totals"], "unmet": plan.get("unmet") or [],
            "pool": pool, "report": CN.report(plan.get("constraints") or [], res["problems"], plan.get("unmet"), plan.get("post_check"))}


def _clean_items(rows) -> list[dict]:
    """The items an outside AI sent, cleaned to the shapes the renderer reads. Raises ValueError (plain words) for anything else."""
    items = CN._items_ok(rows)
    if items is None:
        raise ValueError("the items must be a list: each {\"kind\": \"talk\", \"a\": sentence number, \"b\": sentence number} or "
                         "{\"kind\": \"visual\", \"start\": seconds, \"end\": seconds}")
    for src, it in zip(rows, items):
        if isinstance(src, dict) and src.get("id"):
            it["id"] = str(src["id"])[:40]
        if isinstance(src, dict):
            if isinstance(src.get("protect"), bool):
                it["protect"] = src["protect"]
            for k in ("t0", "t1"):                                   # (the seconds an item was planned with: an edit of something else must not lose them)
                v = CN._num(src.get(k))
                if v is not None:
                    it[k] = v
    return items


def check(runner, pid: str, edited: dict) -> dict:
    """The validator on an edited plan ({"items": [...]}; the other parts of the saved plan stay as they are)."""
    if not isinstance(edited, dict):
        raise ValueError("send the plan as an object with an `items` list")
    proj, folder, st, ai, job, files, words, sents = _inputs(runner, pid)
    stored = st.get("plan.json")
    if not stored:
        raise ValueError("this video has no plan yet")
    cand = {**stored, "items": _clean_items(edited.get("items"))}
    res = CN.validate_edited_plan(with_ids(cand), files, words, cand.get("pool") or [], sents)
    return {"ok": res["ok"], "problems": res["problems"], "totals": res["totals"]}


def apply(runner, pid: str, edited: dict, render: bool = True) -> dict:
    """Save an edited plan and render it. A plan with a hard problem is refused (nothing changes) and the reasons are returned in the error."""
    if not isinstance(edited, dict):
        raise ValueError("send the plan as an object with an `items` list")
    proj, folder, st, ai, job, files, words, sents = _inputs(runner, pid)
    stored = st.get("plan.json")
    if not stored:
        raise ValueError("this video has no plan yet")
    cand = with_ids({**stored, "items": _clean_items(edited.get("items"))})
    res = CN.validate_edited_plan(cand, files, words, cand.get("pool") or [], sents)
    if not res["ok"]:
        raise ValueError("the plan has problems that must be fixed first: " + "; ".join(p["message"] for p in res["problems"] if p["hard"])[:900])
    st.snapshot("plan.json", stored, "before-plan-from-an-ai")
    cand.pop("approval", None)
    cand.pop("dry_run", None)
    cand["awaiting_approval"], cand["pending_render"], cand["pending_note"] = True, True, "plan sent by an AI"
    cand["problems"], cand["totals"] = res["problems"], res["totals"]
    st.put("plan.json", cand)
    if render:
        runner.start(pid, "approve")
    return {"ok": True, "problems": res["problems"], "totals": res["totals"], "rendering": bool(render)}


def search(runner, pid: str, query: str, mode: str = "quote") -> dict:
    query = " ".join(str(query or "").split())
    if not query:
        raise ValueError("say what to look for")
    proj, folder, st, ai, job, files, words, sents = _inputs(runner, pid)
    hits = TS.search(words, sents, query, mode, ai if str(mode).lower() == "topic" else None)
    return {"query": query, "mode": "topic" if str(mode).lower() == "topic" else "quote", "hits": hits[:TS.MAX_HITS], "ambiguous": len(hits) > 1 and TS.far_apart(hits)}


def _final(runner, pid: str) -> Path:
    proj = runner.projects.get(pid)
    job = runner.store.get(proj["job"])
    clip = (job.get("clips") or [None])[0]
    f = runner.store.dir(job["id"]) / str((clip or {}).get("file") or "")
    if not clip or not f.is_file():
        raise ValueError("this video isn't finished yet")
    return f


def preview(runner, pid: str, start_s: float = 0.0, end_s: float | None = None, frames_only: bool = False) -> dict:
    """A cheap look at the finished video: `frames_only` -> one still at each cut (up to 12) in the range; else a 480 px MP4 of the range (at most 60 s)."""
    final = _final(runner, pid)
    dur = ffmpeg.duration(final)
    try:
        a = max(0.0, float(start_s or 0.0))
        end = None if end_s in (None, "") else float(end_s)
    except (TypeError, ValueError):
        raise ValueError("start_s and end_s must be numbers of seconds") from None
    b = min(dur, end) if end is not None else min(dur, a + 20.0)
    if b <= a:
        raise ValueError("the range is empty")
    if b - a > PREVIEW_MAX_S:
        raise ValueError(f"a preview is at most {PREVIEW_MAX_S:.0f} s; ask for a shorter range")
    out_dir = final.parent / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{int(time.time())}"
    if frames_only:
        from . import qa
        cuts = [t for t in qa.cut_times(final) if a <= t <= b][:MAX_FRAMES - 1]
        times = sorted({round(a, 2), *[round(t + 0.05, 2) for t in cuts]})[:MAX_FRAMES]
        frames = []
        for k, t in enumerate(times):
            p = out_dir / f"frame_{tag}_{k:02d}.jpg"
            subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(final), "-frames:v", "1",
                            "-vf", f"scale={PREVIEW_W}:-2", "-q:v", "4", str(p)], capture_output=True, timeout=120)
            if p.exists():
                frames.append({"at": t, "path": str(p)})
        return {"frames": frames}
    p = out_dir / f"preview_{tag}.mp4"
    subprocess.run(["nice", "-n", "10", ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-ss", f"{a:.2f}", "-t", f"{b - a:.2f}", "-i", str(final),
                    "-vf", f"scale={PREVIEW_W}:-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-c:a", "aac", "-b:a", "96k", str(p)],
                   capture_output=True, timeout=600)
    if not p.exists():
        raise ValueError("the preview could not be made")
    return {"path": str(p), "from_s": round(a, 1), "to_s": round(b, 1)}


def rerender_changed(runner, pid: str, changed: list[str] | None = None) -> dict:
    """Render the saved plan again. FirstCut re-encodes the whole video (its B-roll placement is kept when the cut did not change, so it is quicker than the
    first render); it does not re-encode only the pieces that changed, and it says so."""
    runner.start(pid, "rerender")
    return {"ok": True, "note": "The whole video is rendered again (FirstCut cannot re-encode only the changed passages).", "changed": list(changed or [])}
