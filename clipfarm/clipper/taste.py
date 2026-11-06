"""Learns what you like from your swipes in the feed (and Approve / Reject in the Approval tab).

Every rating is an event: the clip's features + liked (1) or not (0). A scroll-past without watching is a
weak "not" (weight 0.25) until you rate it for real. The model is a smoothed naive-Bayes: per feature
(channel, topic, framing, length, score, music mood, title words, ...) the like-rate is compared with the
overall like-rate, shrunk towards "no opinion" while there are few ratings, and the log-odds are added up.
It is deliberately simple - it works from the first handful of ratings and can explain itself.

It is used to
  * order the feed (best guess first, with some exploring so it keeps learning),
  * pick which topics / channels the M2 clips each night (Finder), and
  * rank the moments inside a video before the clips are cut (bonus on the AI's score).

state/taste_events.json   {"<job>:<index>": {label, w, feats, t, kind}}   (rating again overwrites; undo deletes)
state/taste.json          the compiled model (the M2 copies this file before each night)
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from pathlib import Path

from ..core.state import atomic_write_json

STOP = set("the a an and or but of to in on for with at by from is are was were be been it its this that these those "
           "you your i my me we our they them he she his her as so if not no do does did have has had will would can "
           "could just than then when what why how who which about into over out up down one two more most very".split())
GROUP_W = {"channel": 1.0, "topic": 0.8, "words": 0.6, "tx": 0.6, "score": 0.7, "dur": 0.5, "src": 0.5, "mood": 0.4,
           "layout": 0.3, "style": 0.3, "kind": 0.3, "why": 0.3, "speech": 0.3, "shape": 0.3, "mmood": 0.3}
FEAT_V = 2            # bump when features() learns something new: old ratings are then recomputed (feed.relearn)
K = 3.0                       # prior strength: a feature needs a few ratings before it moves the prediction
PASSIVE_W = 0.25


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _sig(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def dur_bucket(d: float) -> str:
    return "<20" if d < 20 else "20-30" if d < 30 else "30-40" if d < 40 else "40-55" if d < 55 else "55+"


def words_of(text: str) -> list[str]:
    return sorted({w for w in re.findall(r"[a-z][a-z']{2,}", (text or "").lower()) if w not in STOP})[:14]


def top_words(text: str, n: int) -> list[str]:
    """The n most frequent meaningful words (4+ letters) of a text - what the clip is actually about."""
    counts: dict[str, int] = {}
    for w in re.findall(r"[a-z][a-z']{3,}", (text or "").lower()):
        if w not in STOP:
            counts[w] = counts.get(w, 0) + 1
    return [w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


def features(clip: dict, src: dict | None = None) -> list[str]:
    """Feature keys 'group:value' for a rendered clip (or a candidate moment dict with title/duration/score)."""
    src = src or {}
    f: list[str] = []
    if src.get("channel"):
        f.append(f"channel:{str(src['channel']).strip().lower()}")
    if src.get("topic"):
        f.append(f"topic:{str(src['topic']).strip().lower()}")
    if src.get("kind"):
        f.append(f"kind:{src['kind']}")
    if src.get("id"):
        f.append(f"src:{src['id']}")
    if clip.get("layout"):
        f.append(f"layout:{clip['layout']}")
    if clip.get("caption_style"):
        f.append(f"style:{clip['caption_style']}")
    if clip.get("duration"):
        f.append(f"dur:{dur_bucket(float(clip['duration']))}")
    if clip.get("score") is not None:
        f.append(f"score:{int(float(clip['score']))}")
    music = clip.get("music") or {}
    if music.get("mood"):
        f.append(f"mood:{music['mood']}")
    for w in words_of(f"{clip.get('title', '')} {clip.get('hook', '')}"):
        f.append(f"words:{w}")
    # what is actually SAID in the clip (the transcript), not just how it was titled
    text = clip.get("transcript") or ""
    for w in top_words(text, 25):
        f.append(f"tx:{w}")
    for w in top_words(clip.get("reason") or "", 8):
        f.append(f"why:{w}")
    if text and clip.get("duration"):
        rate = len(text.split()) / max(1.0, float(clip["duration"]))
        f.append("speech:" + ("fast" if rate > 3.2 else "slow" if rate < 2.0 else "normal"))
    if text.count("?") >= 2:
        f.append("shape:questions")
    if text.count("!") >= 3:
        f.append("shape:exclaiming")
    if "?" in (clip.get("title") or ""):
        f.append("shape:question-title")
    mm = (clip.get("moment") or {}).get("mood")
    if mm:
        f.append(f"mmood:{mm}")
    return f


class Model:
    def __init__(self, counts: dict[str, list[float]], n: int, likes: float, total: float):
        self.counts, self.n, self.likes, self.total = counts, n, likes, total
        self.base = (likes + 1) / (total + 2) if total else 0.5

    @classmethod
    def from_events(cls, events: dict, only: str | None = None, skip: tuple[str, ...] = ()) -> "Model":
        counts: dict[str, list[float]] = {}
        likes = total = 0.0
        n = 0
        for e in events.values():
            if (only and e.get("kind") != only) or e.get("kind") in skip:
                continue
            w, y = float(e.get("w", 1.0)), float(e.get("label", 0))
            total += w
            likes += w * y
            n += 1 if e.get("kind") != "skip" else 0
            for f in e.get("feats", []):
                c = counts.setdefault(f, [0.0, 0.0])
                c[0] += w * y
                c[1] += w
        return cls(counts, n, likes, total)

    def weight(self, feat: str) -> float:
        """Log-odds shift of this feature versus the overall like-rate (0 = no opinion yet)."""
        l, t = self.counts.get(feat, (0.0, 0.0))
        if not t:
            return 0.0
        rate = (l + K * self.base) / (t + K)
        return _logit(rate) - _logit(self.base)

    def predict(self, feats: list[str]) -> float:
        groups: dict[str, list[float]] = {}
        for f in feats:
            groups.setdefault(f.split(":", 1)[0], []).append(self.weight(f))
        z = _logit(self.base)
        for g, ws in groups.items():
            z += GROUP_W.get(g, 0.3) * sum(ws) / len(ws)
        return min(max(_sig(z), 0.02), 0.98)

    def explain(self, feats: list[str], k: int = 2) -> list[str]:
        ws = sorted(((self.weight(f), f) for f in feats if f.split(":", 1)[0] != "src"), reverse=True)
        return [f.split(":", 1)[1] for w, f in ws[:k] if w > 0.25]

    def summary(self) -> dict:
        rows = []
        for f, (l, t) in self.counts.items():
            if t >= 3 and not f.startswith(("src:", "score:")):
                rows.append((self.weight(f), f, round(l / t, 2), round(t, 1)))
        rows.sort(reverse=True)
        fmt = lambda r: {"feature": r[1], "like_rate": r[2], "n": r[3]}
        return {"n": self.n, "like_rate": round(self.base, 3),
                "likes": [fmt(r) for r in rows[:8] if r[0] > 0.2],
                "dislikes": [fmt(r) for r in rows[::-1][:8] if r[0] < -0.2]}


class Taste:
    """Event log + compiled model (thread-safe; the server and the M2 each have their own copy of the model)."""

    def __init__(self, state_dir: Path):
        self.dir = Path(state_dir)
        self.events_path = self.dir / "taste_events.json"
        self.model_path = self.dir / "taste.json"
        self.lock = threading.RLock()
        self._model: Model | None = None
        self._loaded = 0.0

    # ------------------------------------------------------------ events
    def _events(self) -> dict:
        try:
            return json.loads(self.events_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def record(self, key: str, label: int, feats: list[str], kind: str = "rating", weight: float = 1.0) -> None:
        with self.lock:
            ev = self._events()
            old = ev.get(key)
            if kind == "skip" and old and old.get("kind") != "skip":
                return                                         # a real rating beats a passive scroll-past
            ev[key] = {"label": int(label), "w": weight, "feats": feats, "kind": kind, "t": time.time(), "v": FEAT_V}
            self._save(ev)

    def _events_touch(self, key: str, t: float) -> None:
        """(tests) pin an event's timestamp."""
        with self.lock:
            ev = self._events()
            if key in ev:
                ev[key]["t"] = t
                self._save(ev)

    def ingest_perf(self, perf: dict, min_age_h: float = 48) -> int:
        """Turn posted-clip analytics (performance.py) into training events: a hit counts as a like for the clip's
        features, a flop as a no. Weighted a little under a direct swipe."""
        from .performance import labels
        lab = labels(perf, min_age_h)
        with self.lock:
            ev = self._events()
            for k in [k for k, e in ev.items() if e.get("kind") == "perf" and k[5:] not in lab]:
                ev.pop(k)
            for jid, (y, feats) in lab.items():
                ev[f"perf:{jid}"] = {"label": y, "w": 0.5, "feats": feats, "kind": "perf", "t": time.time(), "v": FEAT_V}
            self._save(ev)
        return len(lab)

    def forget(self, key: str) -> None:
        with self.lock:
            ev = self._events()
            if ev.pop(key, None) is not None:
                self._save(ev)

    def _save(self, ev: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.events_path, ev)
        m = Model.from_events(ev, skip=("perf",))                  # YOUR taste; the audience's is kept apart (audience())
        atomic_write_json(self.model_path, {"counts": m.counts, "n": m.n, "likes": m.likes, "total": m.total,
                                            "saved": time.time()})
        self._model, self._loaded = m, time.time()

    # ------------------------------------------------------------ model
    def model(self) -> Model:
        with self.lock:
            try:
                mt = self.model_path.stat().st_mtime
            except OSError:
                mt = 0.0
            if self._model is None or mt > self._loaded:
                try:
                    d = json.loads(self.model_path.read_text())
                    self._model = Model(d["counts"], int(d["n"]), float(d["likes"]), float(d["total"]))
                except (OSError, json.JSONDecodeError, KeyError):
                    self._model = Model.from_events(self._events(), skip=("perf",))
                self._loaded = max(mt, time.time()) if mt else time.time()
            return self._model

    # ------------------------------------------------------------ the AUDIENCE (how posted clips really performed)
    def audience(self) -> Model:
        """Model of what viewers watch, learnt only from the analytics of clips we posted."""
        with self.lock:
            try:
                mt = self.events_path.stat().st_mtime
            except OSError:
                mt = 0.0
            if getattr(self, "_aud", None) is None or mt > getattr(self, "_aud_at", -1):
                self._aud = Model.from_events(self._events(), only="perf")
                self._aud_at = mt
            return self._aud

    def audience_weight(self) -> float:
        """How much to trust viewers over your own taste: 0 until 20 mature posts (before that it is mostly noise), then growing slowly
        to 0.75 - because it is the audience, not you, that decides what grows the channel."""
        n = self.audience().n
        return 0.0 if n < 20 else min(0.75, n / (n + 30))

    def predict(self, feats: list[str]) -> float:
        """Chance this clip is a keeper, blending your taste with what the audience has actually watched."""
        pt = self.model().predict(feats)
        w = self.audience_weight()
        if w <= 0:
            return pt
        pa = self.audience().predict(feats)
        return _sig((1 - w) * _logit(pt) + w * _logit(pa))

    # ------------------------------------------------------------ honest self-check
    def progressive(self) -> list[tuple[float, int]]:
        """(predicted like-probability, what you actually did) for every rating, each predicted by a model that had
        only seen the ratings BEFORE it - how the model really performs on clips it hasn't been taught."""
        ev = sorted((e for e in self._events().values() if e.get("kind") == "rating"), key=lambda e: e["t"])
        counts: dict[str, list[float]] = {}
        likes = total = 0.0
        out = []
        for e in ev:
            if total >= 10:
                out.append((Model(counts, 0, likes, total).predict(e["feats"]), int(e["label"])))
            total += 1
            likes += e["label"]
            for f in e["feats"]:
                c = counts.setdefault(f, [0.0, 0.0])
                c[0] += e["label"]
                c[1] += 1
        return out

    def autopost_gate(self, min_ratings: int = 50, target: float = 0.8, min_samples: int = 12) -> dict:
        """Is the model good enough to post on its own? Needs `min_ratings` ratings, and a confidence level at which
        the clips it was that sure about really were liked (lower bound of the hit-rate >= target)."""
        n = sum(1 for e in self._events().values() if e.get("kind") == "rating")
        if n < min_ratings:
            return {"on": False, "n": n, "need": min_ratings, "why": f"needs {min_ratings - n} more ratings"}
        pr = self.progressive()
        for t in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
            sel = [y for p, y in pr if p >= t]
            if len(sel) >= min_samples and wilson_lower(sum(sel), len(sel)) >= target:
                return {"on": True, "n": n, "need": min_ratings, "threshold": t, "hit_rate": round(sum(sel) / len(sel), 3),
                        "samples": len(sel), "why": f"{sum(sel)} of {len(sel)} clips it was >= {int(t * 100)}% sure about were liked"}
        best = max(((len([y for p, y in pr if p >= t]), t) for t in (0.6, 0.7, 0.8, 0.9)), default=(0, 0))
        return {"on": False, "n": n, "need": min_ratings, "why": "not confident enough yet - keep rating"}

    # ------------------------------------------------------------ uses
    def trust(self) -> float:
        """0 -> 1 as ratings accumulate: how much to trust the model over the AI's own score."""
        return min(1.0, self.model().n / 40)

    def topic_order(self, topics: list[str], rng) -> list[str]:
        """Topics in the order to try them: liked ones more often, unexplored ones always get a chance."""
        m = self.model()
        if m.n < 8:
            return list(topics)
        pool = list(topics)
        weights = {}
        for t in pool:
            f = f"topic:{t.strip().lower()}"
            tot = m.counts.get(f, (0, 0))[1]
            bonus = 0.6 if tot < 3 else 0.0                    # explore topics we know little about
            weights[t] = math.exp(1.2 * m.weight(f) + bonus)
        out = []
        while pool:
            tot = sum(weights[t] for t in pool)
            r, acc = rng.random() * tot, 0.0
            pick = pool[-1]
            for t in pool:
                acc += weights[t]
                if r <= acc:
                    pick = t
                    break
            out.append(pick)
            pool.remove(pick)
        return out

    def channel_blocked(self, channel: str) -> bool:
        f = f"channel:{(channel or '').strip().lower()}"
        l, t = self.model().counts.get(f, (0.0, 0.0))
        al, at = self.audience().counts.get(f, (0.0, 0.0))
        return (t >= 6 and l / t < 0.08) or (at >= 15 and al / at < 0.1)      # you delete it, or viewers ignore it (viewers: only after 15 posts)

    def relearn(self, recompute) -> int:
        """Re-derive the features of ratings made before features() learnt new tricks. `recompute(key)` -> features or None."""
        with self.lock:
            ev = self._events()
            n = 0
            for k, e in ev.items():
                if e.get("v", 1) >= FEAT_V or e.get("kind") == "perf":
                    continue
                feats = recompute(k)
                if feats:
                    e["feats"], e["v"] = feats, FEAT_V
                    n += 1
            if n:
                self._save(ev)
            return n

    def moment_prob(self, title: str, hook: str, duration: float, score: float, src: dict, text: str = "") -> float | None:
        """How likely you are to like this candidate clip (None while the model is too young to say)."""
        if self.model().n < 60 and self.audience_weight() <= 0:
            return None
        return self.predict(features({"title": title, "hook": hook, "duration": duration, "score": score, "transcript": text}, src))

    def moment_bonus(self, title: str, hook: str, duration: float, score: float, src: dict, text: str = "") -> float:
        """Points added to a moment's AI score before the best are picked (only once there's enough to go on)."""
        m = self.model()
        if m.n < 30 and self.audience_weight() <= 0:
            return 0.0
        p = self.predict(features({"title": title, "hook": hook, "duration": duration, "score": score, "transcript": text}, src))
        return max(-1.5, min(1.5, (p - m.base) * 5.0))


def pull_perf(cfg) -> bool:
    """M2: copy the posting server's clip_perf.json (how the posted clips performed)."""
    import subprocess
    host = cfg.get("clipper.auto.push_host")
    if not host:
        t = cfg.get("handoff.target") or ""
        host = t.split(":", 1)[0] if ":" in t else None
    if not host:
        return False
    dest = cfg.path("state")
    dest.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                        f"{host}:clipfarm/data/state/clip_perf.json", str(dest / "clip_perf.json")],
                       check=True, capture_output=True, timeout=60)
        return True
    except Exception:  # noqa: BLE001 - keep what we have
        return False


def pull_queue_list(cfg) -> bool:
    """M2: copy the server's clip_queue.json (approved clips still waiting for a posting slot)."""
    import subprocess
    host = cfg.get("clipper.auto.push_host")
    if not host:
        t = cfg.get("handoff.target") or ""
        host = t.split(":", 1)[0] if ":" in t else None
    if not host:
        return False
    dest = cfg.path("state")
    dest.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                        f"{host}:clipfarm/data/state/clip_queue.json", str(dest / "clip_queue.json")],
                       check=True, capture_output=True, timeout=60)
        return True
    except Exception:  # noqa: BLE001
        return False


def pull_need(cfg) -> bool:
    """M2: copy the server's clip_need.json (how many more clips its posting slots still need)."""
    import subprocess
    host = cfg.get("clipper.auto.push_host")
    if not host:
        t = cfg.get("handoff.target") or ""
        host = t.split(":", 1)[0] if ":" in t else None
    if not host:
        return False
    dest = cfg.path("state")
    dest.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                        f"{host}:clipfarm/data/state/clip_need.json", str(dest / "clip_need.json")],
                       check=True, capture_output=True, timeout=60)
        return True
    except Exception:  # noqa: BLE001
        return False


def wilson_lower(successes: float, n: float, z: float = 1.28) -> float:
    """Lower bound (~90% one-sided) of a true rate seen as successes/n - small samples don't count as proof."""
    if n <= 0:
        return 0.0
    p = successes / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def pull(cfg) -> bool:
    """M2: fetch the posting server's current model (taste.json) before a night's run."""
    import subprocess
    host = cfg.get("clipper.auto.push_host")
    if not host:
        t = cfg.get("handoff.target") or ""
        host = t.split(":", 1)[0] if ":" in t else None
    if not host:
        return False
    dest = cfg.path("state")
    dest.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                        f"{host}:clipfarm/data/state/taste.json", str(dest / "taste.json")],
                       check=True, capture_output=True, timeout=60)
        return True
    except Exception:  # noqa: BLE001 - use the model we already have
        return False
