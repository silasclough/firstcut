"""What the planner knew, as the extras of the Final Cut project (WP13): markers, the project note, alternate takes and the source-clips event.

Pure functions over the plan and the footage records (no files, no ffmpeg): `localproj` calls `extras(...)` once the cut is known and hands the result
to the one FCPXML writer (fcpxml.build_project). Everything here is data the render already had; nothing is invented.
"""

from __future__ import annotations

from datetime import date

from . import fcpextras as XT

ROLE_KEYWORD = {"a_roll": "A-roll", "broll": "B-roll", "voiceover_broll": "B-roll", "recorder": "Recorder", "audio": "Recorder"}


def _file_at(files: list[dict], t: float) -> dict | None:
    return next((f for f in files if not f.get("photo") and f.get("start", 0) <= t < f.get("end", 0) + 1e-6), None)


def alt_take_clips(plan: dict, files: list[dict]) -> list[dict]:
    """[{"kept": {"path", "start"}, "path", "start", "dur", "w", "h", "fps", ...}] for every take of a repeated line that the cut did NOT use
    (plan["alt_takes"] / plan["take_groups"], WP6), placed by the kept take's own file and time. A take whose time is not in a known file is left out."""
    groups = {g.get("id"): g for g in (plan.get("take_groups") or []) if isinstance(g, dict)}
    used = set()
    for it in plan.get("items") or []:
        if isinstance(it, dict) and it.get("kind") == "talk" and isinstance(it.get("a"), int) and isinstance(it.get("b"), int):
            used |= set(range(it["a"], it["b"] + 1))
    out: list[dict] = []
    for alt in plan.get("alt_takes") or []:
        g = groups.get(alt.get("group"))
        if not g:
            continue
        mine = next((t for t in g.get("takes") or [] if t.get("a") == alt.get("a") and t.get("b") == alt.get("b")), None)
        kept = next((t for t in g.get("takes") or [] if t.get("a") in used), None)
        if not mine or not kept:
            continue
        fa, fk = _file_at(files, XT.num(mine.get("start"))), _file_at(files, XT.num(kept.get("start")))
        if not fa or not fk:
            continue
        dur = XT.num(mine.get("end")) - XT.num(mine.get("start"))
        if dur < 0.3:
            continue
        out.append({"kept": {"path": fk["path"], "start": XT.num(kept["start"]) - fk["start"]},
                    "path": fa["path"], "start": XT.num(mine["start"]) - fa["start"], "dur": dur, "w": fa["w"], "h": fa["h"], "fps": fa["fps"],
                    "file_dur": fa.get("dur"), "tc": fa.get("tc", 0.0), "tc_df": fa.get("tc_df", False),
                    **{k: fa[k] for k in ("ar", "ach", "cs") if fa.get(k)},
                    "note": f"Another take of this line ({dur:.0f} s). It is switched off: the kept take plays in the cut."})
    return out[:40]


def source_clips(files: list[dict], scenes: dict | None = None) -> list[dict]:
    """One record per footage file for the source-clips event: its role and scenes as keywords, usable stretches as favourites, black / frozen
    stretches as rejects (from the footage facts, WP2). Photos, library and stock items are left out."""
    out: list[dict] = []
    for f in files:
        if f.get("photo") or f.get("library") or f.get("stock") or not f.get("path") or not f.get("w"):
            continue
        facts = f.get("facts") if isinstance(f.get("facts"), dict) else {}
        role = str(facts.get("role") or ("broll" if f.get("broll") else "a_roll" if f.get("audio") else ""))
        q = facts.get("quality") if isinstance(facts.get("quality"), dict) else {}
        rejects = []
        for key, why in (("black", "black picture"), ("frozen", "frozen picture")):
            for r in (q.get(key) if isinstance(q.get(key), list) else []):
                if isinstance(r, (list, tuple)) and len(r) >= 2:
                    rejects.append((XT.num(r[0]), XT.num(r[1]), why))
        usable = []
        for u in (facts.get("usable") if isinstance(facts.get("usable"), list) else []):
            if isinstance(u, dict):
                usable.append((XT.num(u.get("start")), XT.num(u.get("end"))))
            elif isinstance(u, (list, tuple)) and len(u) >= 2:
                usable.append((XT.num(u[0]), XT.num(u[1])))
        base = XT.num(f.get("start"), 0.0)                       # (scene times are on the shared timeline: back to this file's own seconds)
        sc = [{"start": XT.num(s["start_s"]) if "start_s" in s else XT.num(s.get("start")) - base,
               "end": XT.num(s["end_s"]) if "end_s" in s else XT.num(s.get("end")) - base, "what": s.get("what")}
              for s in ((scenes or {}).get(f.get("name")) or []) if isinstance(s, dict) and s.get("what")]
        out.append({"path": f["path"], "dur": f.get("dur") or 0.0, "w": f["w"], "h": f["h"], "fps": f.get("fps") or 30, "audio": bool(f.get("audio")),
                    "tc": f.get("tc", 0.0), "tc_df": f.get("tc_df", False), "role": role,
                    "keywords": [ROLE_KEYWORD.get(role, "")] if ROLE_KEYWORD.get(role) else [], "scenes": sc[:40],
                    "usable": [u for u in usable if u[1] > u[0]][:20], "reject": rejects[:20],
                    **{k: f[k] for k in ("ar", "ach", "cs") if f.get(k)}})
    return out


def project_note(plan: dict, style_line: str = "", today: str | None = None) -> str:
    """The sequence's note: what made this project and what to do with it. No names of people or machines."""
    bits = [f"Rough cut by FirstCut, {today or date.today().isoformat()}. B-roll choices are suggestions: each shot has a note and keywords saying why."]
    if style_line:
        bits.append(style_line)
    if plan.get("summary"):
        bits.append(str(plan["summary"])[:300])
    unmet = [str(u.get("text") or u.get("what") or "") for u in (plan.get("unmet") or []) if isinstance(u, dict)]
    if unmet:
        bits.append("Not done from your notes: " + "; ".join(x for x in unmet if x)[:240])
    bits.append("Red markers are things to check. Blue markers are chapters. Alternate takes are switched off under the take that plays.")
    return " ".join(bits)[:900]


def extras(plan: dict, files: list[dict], scenes: dict | None, start_of, remap, issues: list[dict] | None, cutaways: list[dict] | None,
           style_line: str = "", today: str | None = None) -> dict:
    """{"markers", "project_note", "alt_takes", "source_clips"}: every part is optional and a failure in one never loses the others."""
    out: dict = {}
    chapters: list[dict] = []
    try:
        from . import gfxcontent as GC
        chapters = GC.chapter_marks(plan.get("flow"), plan.get("items") or [], start_of, remap) if plan.get("flow") else []
    except Exception:  # noqa: BLE001 - no chapters is a fine result
        chapters = []
    try:
        out["markers"] = XT.markers_from_plan(plan, issues=issues, chapters=chapters, cutaways=cutaways)
    except Exception:  # noqa: BLE001
        out["markers"] = []
    for key, fn in (("alt_takes", lambda: alt_take_clips(plan, files)), ("source_clips", lambda: source_clips(files, scenes)),
                    ("project_note", lambda: project_note(plan, style_line, today))):
        try:
            out[key] = fn()
        except Exception:  # noqa: BLE001
            out[key] = [] if key != "project_note" else None
    return out
