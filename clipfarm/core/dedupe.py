"""Dedupe of ideas/stories/clips: exact normalized hash + word-trigram Jaccard similarity."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def _h32(s: str) -> int:
    return int(hashlib.blake2b(s.encode(), digest_size=4).hexdigest(), 16)


def shingles(text: str, n: int = 3) -> set[int]:
    toks = normalize(text).split()
    if len(toks) < n:
        return {_h32(" ".join(toks))} if toks else set()
    return {_h32(" ".join(toks[i:i + n])) for i in range(len(toks) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class Deduper:
    def __init__(self, db_path: Path | str, dry_run: bool = False):
        self.dry_run = dry_run
        db_path = Path(db_path)
        if not dry_run:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path) if (not dry_run or db_path.exists()) else ":memory:")
        self.db.execute("""CREATE TABLE IF NOT EXISTS items (
            kind TEXT, job_id TEXT, created_at TEXT, norm_hash TEXT, shingles TEXT, text TEXT)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS ix_kind_hash ON items(kind, norm_hash)")
        self.db.commit()

    def find_similar(self, kind: str, text: str, threshold: float, limit: int = 3000) -> tuple[str, float] | None:
        h = hashlib.sha1(normalize(text).encode()).hexdigest()
        row = self.db.execute("SELECT job_id FROM items WHERE kind=? AND norm_hash=? LIMIT 1", (kind, h)).fetchone()
        if row:
            return row[0], 1.0
        mine = shingles(text)
        best: tuple[str, float] | None = None
        for job_id, sh in self.db.execute(
                "SELECT job_id, shingles FROM items WHERE kind=? ORDER BY created_at DESC LIMIT ?", (kind, limit)):
            score = jaccard(mine, set(json.loads(sh)))
            if score >= threshold and (best is None or score > best[1]):
                best = (job_id, score)
        return best

    def add(self, kind: str, text: str, job_id: str) -> None:
        if self.dry_run:
            return
        self.db.execute("INSERT INTO items VALUES (?,?,?,?,?,?)", (
            kind, job_id, datetime.now(timezone.utc).isoformat(),
            hashlib.sha1(normalize(text).encode()).hexdigest(), json.dumps(sorted(shingles(text))), text[:2000]))
        self.db.commit()

    def recent(self, kind: str, n: int = 30) -> list[str]:
        return [r[0] for r in self.db.execute(
            "SELECT text FROM items WHERE kind=? ORDER BY created_at DESC LIMIT ?", (kind, n))]

    def seen_key(self, kind: str, key: str) -> bool:
        return self.find_similar(kind, key, threshold=1.01) is not None  # exact-hash only

    def remove_job(self, job_id: str) -> None:
        if not self.dry_run:
            self.db.execute("DELETE FROM items WHERE job_id=?", (job_id,))
            self.db.commit()
