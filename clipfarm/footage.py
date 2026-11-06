"""Background footage downloader for sources whose license allows reuse (config footage.sources).

Each downloaded file gets a sidecar <name>.footage.json recording where it came from and the
license statement it was used under, so you can always show provenance later.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .context import Context
from .core.log import get_logger
from .core.retry import call_with_retry

log = get_logger("footage")


def _existing_ids(folder: Path) -> set[str]:
    ids = set()
    for side in folder.glob("*.footage.json"):
        try:
            ids.add(json.loads(side.read_text()).get("id"))
        except Exception:  # noqa: BLE001
            pass
    return ids


def list_candidates(src: dict) -> list[dict]:
    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True, "extract_flat": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(src["url"], download=False)
    pat = re.compile(src.get("title_filter", ""), re.I) if src.get("title_filter") else None
    out = []
    for e in info.get("entries") or []:
        dur = float(e.get("duration") or 0)
        if pat and not pat.search(e.get("title") or ""):
            continue
        if src.get("max_minutes") and dur > float(src["max_minutes"]) * 60:
            continue
        if src.get("min_minutes") and dur < float(src["min_minutes"]) * 60:
            continue
        out.append({"id": e.get("id"), "title": e.get("title"), "duration": dur,
                    "url": e.get("url") or f"https://www.youtube.com/watch?v={e.get('id')}"})
    return out


def fetch(ctx: Context, only: str | None = None, max_videos: int | None = None, list_only: bool = False) -> list[Path]:
    import yt_dlp

    folder = ctx.cfg.path("minecraft_footage")
    folder.mkdir(parents=True, exist_ok=True)
    have = _existing_ids(folder)
    got: list[Path] = []
    for src in ctx.cfg.get("footage.sources", []) or []:
        if only and src.get("name") != only:
            continue
        if not src.get("license_note"):
            log.error(f"footage source {src.get('name')}: license_note is required; skipping")
            continue
        cands = [c for c in list_candidates(src) if c["id"] not in have]
        limit = int(max_videos if max_videos is not None else src.get("max_videos", 4))
        cands = cands[:limit]
        for c in cands:
            print(f"{'would download' if (list_only or ctx.dry_run) else 'downloading'}: {c['title']} "
                  f"({c['duration'] / 60:.0f} min) {c['url']}")
        if list_only or ctx.dry_run:
            continue
        for c in cands:
            stem = f"{src.get('name', 'src')}_{c['id']}"
            opts = {"quiet": True, "no_warnings": True, "noprogress": True, "continuedl": True,
                    "format": src.get("format", "bv*[vcodec^=avc1][height<=1920]/bv*[height<=1920]"),
                    "outtmpl": str(folder / f"{stem}.%(ext)s"), "retries": 5, "fragment_retries": 5}

            def download(opts=opts, url=c["url"]):
                with yt_dlp.YoutubeDL(opts) as ydl:
                    inf = ydl.extract_info(url, download=True)
                    return inf, Path(ydl.prepare_filename(inf))
            try:
                # YouTube sometimes answers 403 mid-download; a fresh attempt (new URLs) usually works,
                # and yt-dlp resumes the .part file.
                info, path = call_with_retry(download, attempts=3, base=20, cap=90, what=f"download {c['id']}")
            except Exception as e:  # noqa: BLE001 - one bad video must not stop the rest
                log.error(f"skipping {c['url']}: {str(e).splitlines()[0][:200]} (re-run later; if 403s persist: "
                          f".venv/bin/pip install -U yt-dlp)")
                continue
            (folder / f"{path.stem}.footage.json").write_text(json.dumps({
                "id": c["id"], "title": c["title"], "url": c["url"], "source": src.get("name"),
                "channel": info.get("channel"), "license_note": src["license_note"], "credit": src.get("credit", ""),
                "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "width": info.get("width"), "height": info.get("height")}, indent=2))
            log.info(f"footage saved: {path.name}")
            got.append(path)
    return got
