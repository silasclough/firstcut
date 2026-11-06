"""Stock footage for Find footage (FirstCut): search Pixabay and Pexels (videos and photos, with the person's own free keys) and
Openverse (Creative Commons photos, no key), in one normalized list, and save what the person picks to disk.

search(q, kinds, n, page) -> [{source, kind, id, what, thumb, preview, w, h, dur, page, license, credit, download, creator}]
    `id` is "<source>:<kind>:<their id>". `preview` is a small playable mp4 for videos (watched in the app only); `download` is the
    best file at or under 1080p. Results are cached for 10 minutes (the providers ask for that, and it keeps searching quick).
fetch(item, dest_dir) -> Path
    downloads `download` into dest_dir/<source>/<what - id>.<ext> with a sidecar .json (licence, credit, page). Idempotent; a failed
    download leaves nothing behind.
sources() -> which providers are connected (the keys themselves are never returned or logged).

Pixabay: content may not be hotlinked in published work, so anything used is downloaded first (fetch); previewing in the app is fine.
Openverse: only licences fine for monetised video (CC0, public domain, CC BY), with the credit line it asks for.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import requests

from ..core.log import get_logger

log = get_logger("clipper.stockfind")
UA = {"User-Agent": "FirstCut/1.0 (video editing app; stock search)"}
TIMEOUT = 20
CACHE_S = 600
_CACHE: dict = {}
_ITEMS: dict = {}                                                      # every result shown lately, by id: all the page may ask to download
OK_OPENVERSE = {"cc0", "pdm", "by"}                                    # no NC, ND or SA: fine for monetised videos


def sources() -> dict:
    return {"pixabay": bool(os.environ.get("PIXABAY_API_KEY")), "pexels": bool(os.environ.get("PEXELS_API_KEY")), "openverse": True}


def _get(url: str, **kw) -> dict:
    r = requests.get(url, timeout=kw.pop("timeout", TIMEOUT), headers={**UA, **kw.pop("headers", {})}, **kw)
    r.raise_for_status()
    return r.json()


def _words(s) -> str:
    return " ".join(str(s or "").replace(",", " ").split())[:160]


# ------------------------------------------------------------------------------------------------ the providers
def _pixabay_videos(q: str, n: int, page: int) -> list[dict]:
    key = os.environ.get("PIXABAY_API_KEY")
    if not key:
        return []
    d = _get("https://pixabay.com/api/videos/", params={"key": key, "q": q[:100], "per_page": max(3, min(n, 200)), "page": page, "safesearch": "true"})
    out = []
    for v in d.get("hits") or []:
        vv = v.get("videos") or {}
        sizes = [vv.get(k) or {} for k in ("large", "medium", "small", "tiny")]
        best = next((f for f in sizes if f.get("url") and 0 < int(f.get("height") or 0) <= 1080), None) or next((f for f in sizes if f.get("url")), None)
        small = next((f for f in reversed(sizes) if f.get("url") and int(f.get("height") or 0) >= 300), None) or best
        if not best:
            continue
        thumb = (vv.get("medium") or {}).get("thumbnail") or (vv.get("tiny") or {}).get("thumbnail") or \
            (f"https://i.vimeocdn.com/video/{v['picture_id']}_640x360.jpg" if v.get("picture_id") else "")
        out.append({"source": "pixabay", "kind": "video", "id": f"pixabay:video:{v['id']}", "what": _words(v.get("tags")), "thumb": thumb,
                    "preview": small["url"], "w": int(best.get("width") or 0), "h": int(best.get("height") or 0), "dur": float(v.get("duration") or 0),
                    "page": v.get("pageURL") or "", "license": "Pixabay Content License", "credit": f"Video by {v.get('user') or 'a Pixabay contributor'} from Pixabay",
                    "download": best["url"], "creator": v.get("user") or ""})
    return out


def _pixabay_photos(q: str, n: int, page: int) -> list[dict]:
    key = os.environ.get("PIXABAY_API_KEY")
    if not key:
        return []
    d = _get("https://pixabay.com/api/", params={"key": key, "q": q[:100], "image_type": "photo", "per_page": max(3, min(n, 200)), "page": page,
                                                 "safesearch": "true"})
    return [{"source": "pixabay", "kind": "photo", "id": f"pixabay:photo:{h['id']}", "what": _words(h.get("tags")), "thumb": h.get("webformatURL") or "",
             "preview": "", "w": int(h.get("imageWidth") or 0), "h": int(h.get("imageHeight") or 0), "dur": 0.0, "page": h.get("pageURL") or "",
             "license": "Pixabay Content License", "credit": f"Photo by {h.get('user') or 'a Pixabay contributor'} from Pixabay",
             "download": h.get("largeImageURL") or h.get("webformatURL") or "", "creator": h.get("user") or ""}
            for h in d.get("hits") or [] if h.get("webformatURL")]


def _pexels_videos(q: str, n: int, page: int) -> list[dict]:
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        return []
    d = _get("https://api.pexels.com/videos/search", params={"query": q[:100], "per_page": max(3, min(n, 80)), "page": page},
             headers={"Authorization": key})
    out = []
    for v in d.get("videos") or []:
        files = [f for f in v.get("video_files") or [] if f.get("link") and str(f.get("file_type", "video/mp4")).endswith("mp4")]
        if not files:
            continue
        files.sort(key=lambda f: int(f.get("height") or 0))
        best = ([f for f in files if int(f.get("height") or 0) <= 1080] or files)[-1]
        small = next((f for f in files if int(f.get("height") or 0) >= 360), files[0])
        who = (v.get("user") or {}).get("name") or "a Pexels contributor"
        slug = re.sub(r"[-/]+", " ", str(v.get("url") or "").rstrip("/").rsplit("/", 1)[-1])
        out.append({"source": "pexels", "kind": "video", "id": f"pexels:video:{v['id']}", "what": _words(re.sub(r"\d+$", "", slug)), "thumb": v.get("image") or "",
                    "preview": small["link"], "w": int(best.get("width") or 0), "h": int(best.get("height") or 0), "dur": float(v.get("duration") or 0),
                    "page": v.get("url") or "", "license": "Pexels License", "credit": f"Video by {who} from Pexels", "download": best["link"], "creator": who})
    return out


def _pexels_photos(q: str, n: int, page: int) -> list[dict]:
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        return []
    d = _get("https://api.pexels.com/v1/search", params={"query": q[:100], "per_page": max(3, min(n, 80)), "page": page}, headers={"Authorization": key})
    out = []
    for ph in d.get("photos") or []:
        src = ph.get("src") or {}
        who = ph.get("photographer") or "a Pexels photographer"
        out.append({"source": "pexels", "kind": "photo", "id": f"pexels:photo:{ph['id']}", "what": _words(ph.get("alt")), "thumb": src.get("medium") or src.get("small") or "",
                    "preview": "", "w": int(ph.get("width") or 0), "h": int(ph.get("height") or 0), "dur": 0.0, "page": ph.get("url") or "",
                    "license": "Pexels License", "credit": f"Photo by {who} from Pexels", "download": src.get("large2x") or src.get("large") or src.get("original") or "",
                    "creator": who})
    return out


def _openverse_photos(q: str, n: int, page: int) -> list[dict]:
    d = _get("https://api.openverse.org/v1/images/", params={"q": q[:100], "license": "cc0,pdm,by", "page_size": max(3, min(n, 20)), "page": page,
                                                              "mature": "false"})
    out = []
    for r in d.get("results") or []:
        lic = str(r.get("license") or "").lower()
        if lic not in OK_OPENVERSE or not r.get("url"):
            continue
        name = "Public domain" if lic in ("cc0", "pdm") else f"CC BY {r.get('license_version') or ''}".strip()
        credit = "" if lic in ("cc0", "pdm") else (r.get("attribution") or f"“{r.get('title') or 'Photo'}” by {r.get('creator') or 'unknown'}, {name}")
        out.append({"source": "openverse", "kind": "photo", "id": f"openverse:photo:{r['id']}", "what": _words(r.get("title")),
                    "thumb": r.get("thumbnail") or f"https://api.openverse.org/v1/images/{r['id']}/thumb/",   # (small and quick: never the
                                                                                                                 #  original, which can be 20 MB)
                    "preview": "", "w": int(r.get("width") or 0), "h": int(r.get("height") or 0), "dur": 0.0, "page": r.get("foreign_landing_url") or "",
                    "license": name, "credit": credit, "download": r["url"], "creator": r.get("creator") or ""})
    return out


PROVIDERS = [("video", _pixabay_videos), ("video", _pexels_videos), ("photo", _pixabay_photos), ("photo", _pexels_photos), ("photo", _openverse_photos)]
_openverse_photos.slow = True                # a new search takes it 7-12 s and it refuses several at once: the words only

# ------------------------------------------------------------------------------------------------ better matches
# 2026-11-06: "the stock photo and video and music is really, really bad. you search for something and it is not at all related".
# The providers rank by their own word match (Openverse: Flickr titles). So: (1) the words become 2-3 stock-friendly searches
# (one cheap AI call, cached), searched together; (2) every result's thumbnail is scored against the words with SigLIP 2, the
# picture model Find footage already uses, and the list is ordered by what the pictures SHOW, with the clearly unrelated dropped;
# (3) tiny pictures, sub-4-second clips and near-duplicates go. Without the AI or the picture model it is what it was.
_REWRITE = None                       # (q, "stock" | "music") -> [searches]; the app sets it (configure)
_EMBED = None                         # () -> a SigLIP (catalog.SigLIP: .text, .images) or None
_VEC: dict = {}                       # thumbnail url -> its picture vector (kept for the session)
_ALT: dict = {}                       # (q, kind) -> the searches it became
MIN_SIDE = {"video": 1280, "photo": 1600}   # below this a result loses points (shown only when little else matches)
TINY = 640                                  # below this (longest side) it is not shown
FLOOR = 0.07                                # SigLIP 2 cosine below this: the picture doesn't show the words (calibrated on real
                                            # Openverse searches: matches 0.08-0.18, "Portrait of the Artist" for "two people talking" 0.054)
SAME = 0.95                                 # two thumbnails this alike are the same picture (Openverse has many re-uploads)


def configure(rewrite=None, embedder=None) -> None:
    """The app's AI (for the extra searches) and picture model (for ranking). Both optional."""
    global _REWRITE, _EMBED
    if rewrite is not None:
        _REWRITE = rewrite
    if embedder is not None:
        _EMBED = embedder


REWRITE_WAIT = 4.0                    # a search waits at most this long for the rewrite (it carries on, for "More" and next time)
_PENDING: dict = {}


def searches_for(q: str, kind: str = "stock", wait: float | None = None) -> list[str]:
    """The words, then up to two stock-friendly rewrites ("two people talking" -> "two people conversation", "friends talking
    coffee"). Only the words when there is no AI, it fails (offline), or it takes longer than `wait` (it is then kept for later)."""
    import threading
    q = " ".join(str(q or "").split())[:100]
    key = (q.lower(), kind)
    if key in _ALT:
        return _ALT[key]
    if not (_REWRITE and q):
        return [q]

    def work():
        out = [q]
        try:
            got = []
            ans = _REWRITE(q, kind) or []
            if isinstance(ans, str):
                ans = [ans]                                            # (one string is one search, not letters)
            for x in ans:                                              # (a model sometimes answers ['["a", "b"]']: unpack it)
                if isinstance(x, str) and x.strip().startswith("["):
                    try:
                        got += [str(y) for y in json.loads(x)]
                        continue
                    except ValueError:
                        pass
                got.append(x)
            for x in got:
                x = " ".join(str(x or "").split())[:100]
                if x and x.lower() not in {o.lower() for o in out}:
                    out.append(x)
                if len(out) >= 3:
                    break
        except Exception as e:  # noqa: BLE001 - the words alone, as before
            log.info(f"stock search: no extra searches ({str(e)[:120]})")
        _ALT[key] = out
        _PENDING.pop(key, None)
        if len(_ALT) > 500:
            _ALT.pop(next(iter(_ALT)))
    t = _PENDING.get(key)
    if t is None:
        t = _PENDING[key] = threading.Thread(target=work, daemon=True)
        t.start()
    t.join(REWRITE_WAIT if wait is None else wait)
    return _ALT.get(key, [q])


QUERY_SYSTEM = ("You turn what a video editor typed into searches for stock footage sites (Pexels, Pixabay, Openverse) or stock music. "
                "Stock sites match short literal phrases about what is visible (or, for music, genre, mood, instruments, tempo). "
                "Give 2 alternative searches that would find what the editor means, each 2-4 words, different from each other and from "
                "the original; no quotes, no hashtags.")
QUERY_SCHEMA = {"type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
                "required": ["queries"], "additionalProperties": False}


def ai_rewriter(ai):
    """A rewrite function over LocalAI (task stock_queries: a cheap model)."""
    def go(q: str, kind: str) -> list[str]:
        what = "stock music" if kind == "music" else "stock video and photos"
        res = ai.complete_json("stock_queries", QUERY_SYSTEM, f"Searching for {what}. The editor typed: {q}", QUERY_SCHEMA)
        qs = res.data.get("queries") or []
        return [str(x) for x in ([qs] if isinstance(qs, str) else qs)][:3]
    return go


def _quality(it: dict) -> float:
    """0 for a usable result, below 0 for one that is small (or a very short clip)."""
    side = max(int(it.get("w") or 0), int(it.get("h") or 0))
    pen = 0.0
    if side and side < MIN_SIDE.get(it.get("kind"), 1600):
        pen -= 0.1 if side >= 1000 else 0.3                            # (a picture that matches still wins: a match is worth ~0.6-1.1)
    return pen


def _dupe_key(it: dict) -> str:
    what = re.sub(r"\(\d{5,}\)|\b\d{6,}\b", "", str(it.get("what") or "").lower())               # ("... Library (11056363313)")
    title = re.sub(r"[^a-z0-9]+", " ", what).strip()[:48]
    if not title:
        return "id|" + str(it.get("id") or it.get("download") or it.get("thumb") or id(it))     # (nothing to compare: keep it)
    return title + "|" + str(it.get("creator") or "").lower() + "|" + str(it.get("kind") or "") + "|" + str(it.get("duration") or "")


def _thumb_image(url: str):
    from io import BytesIO
    from PIL import Image
    r = requests.get(url, headers=UA, timeout=8)
    r.raise_for_status()
    im = Image.open(BytesIO(r.content)).convert("RGB")
    im.thumbnail((448, 448))
    return im


def rank(q: str, items: list[dict], keep_least: int = 6) -> list[dict]:
    """Order by what the thumbnails show (SigLIP), blended with the provider's own order; drop the clearly unrelated (keeping at
    least `keep_least`). Without the picture model: the provider order, small pictures last."""
    import concurrent.futures as cf
    for k, it in enumerate(items):
        it["_prov"] = 1.0 - k / max(1, len(items))                     # (the provider's order, merged)
    emb = None
    if _EMBED:
        try:
            emb = _EMBED()
        except Exception as e:  # noqa: BLE001
            log.info(f"stock ranking without pictures ({str(e)[:120]})")
    sims: dict = {}
    if emb is not None and items:
        try:
            import numpy as np
            todo = [it for it in items if it.get("thumb") and it["thumb"] not in _VEC]
            with cf.ThreadPoolExecutor(8) as ex:
                ims = dict(zip([it["thumb"] for it in todo], ex.map(lambda it: _safe(_thumb_image, it["thumb"]), todo)))
            got = [(u, im) for u, im in ims.items() if im is not None]
            if got:
                vecs = emb.images([im for _, im in got])
                for (u, _), v in zip(got, vecs):
                    _VEC[u] = v
            t = emb.text([f"a photo of {q}"])[0]
            for it in items:
                v = _VEC.get(it.get("thumb"))
                if v is not None:
                    sims[it["id"]] = float(np.dot(v, t))
            while len(_VEC) > 4000:
                _VEC.pop(next(iter(_VEC)))
        except Exception as e:  # noqa: BLE001 - ranking is a bonus
            log.warning(f"stock ranking failed: {str(e)[:200]}")
            sims = {}
    for it in items:
        s = sims.get(it["id"])
        it["match"] = round(s, 4) if s is not None else None
        it["_score"] = (s * 6.0 if s is not None else 0.0) + 0.25 * it["_prov"] + _quality(it)
    out = sorted(items, key=lambda it: -it["_score"])
    if sims:                                                          # the same picture twice (another upload of it): once
        kept, vs = [], []
        for it in out:
            v = _VEC.get(it.get("thumb"))
            if v is not None and any(float((v * w).sum()) >= SAME for w in vs):
                continue
            if v is not None:
                vs.append(v)
            kept.append(it)
        out = kept
        good = [it for it in out if it["match"] is None or it["match"] >= FLOOR]
        out = good if len(good) >= keep_least else out[:max(keep_least, len(good))]
    for k, it in enumerate(out):
        it["best"] = bool(sims) and k < 3 and (it["match"] or 0) >= FLOOR
        it.pop("_prov", None)
        it.pop("_score", None)
    return out


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception:  # noqa: BLE001
        return None


def search(q: str, kinds=("video", "photo"), n: int = 24, page: int = 1, smart: bool = True) -> list[dict]:
    """Stock shots for the words, from every connected provider (videos first when both kinds are wanted), searched as the words
    plus up to two stock-friendly rewrites, and ordered by what the pictures show (see "better matches")."""
    q = " ".join(str(q or "").split())[:100]
    if not q:
        return []
    kinds = tuple(k for k in ("video", "photo") if k in (kinds or ()))
    key = (q.lower(), kinds, int(n), int(page), tuple(sorted(sources().items())), bool(smart and (_REWRITE or _EMBED)))
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < CACHE_S:
        remember(hit[1])
        return hit[1]
    import concurrent.futures as cf
    per = max(4, n // 2)
    with cf.ThreadPoolExecutor(6) as ex:
        jobs = [(q, fn) for kind, fn in PROVIDERS if kind in kinds]    # the words themselves: at once
        futs = [ex.submit(fn, q, per, page) for _, fn in jobs]
        for alt in (searches_for(q)[1:] if smart else []):             # ...and the rewrites, as soon as they come (or not at all)
            for kind, fn in PROVIDERS:                                 # (not Openverse: a new search takes it 7-12 s and it refuses
                if kind in kinds and not getattr(fn, "slow", False):  #  several at once; its results are ranked instead)
                    jobs.append((alt, fn))
                    futs.append(ex.submit(fn, alt, max(4, n // 3), page))
        lists: list[list[dict]] = []
        for (qq, fn), fut in zip(jobs, futs):
            try:
                lists.append(fut.result())
            except Exception as e:  # noqa: BLE001 - one provider down: the others still answer
                log.warning(f"stock search {fn.__name__} failed: {str(e)[:200]}")
                lists.append([])
    out, seen, dupes = [], set(), set()
    for k in range(max((len(x) for x in lists), default=0)):          # one from each, in turn (the words' own results first)
        for lst in lists:
            if k < len(lst) and lst[k]["id"] not in seen:
                it = lst[k]
                seen.add(it["id"])
                if it.get("kind") == "video" and 0 < float(it.get("dur") or 0) < 4:
                    continue                                          # (a clip too short to use)
                side = max(int(it.get("w") or 0), int(it.get("h") or 0))
                if side and side < TINY:
                    continue
                dk = _dupe_key(it)
                if dk in dupes:
                    continue                                          # (the same picture under another id: Openverse has many)
                dupes.add(dk)
                out.append(it)
    if smart:
        out = rank(q, out)
    out = out[:n]
    _CACHE[key] = (time.time(), out)
    remember(out)
    if len(_CACHE) > 200:
        for old in sorted(_CACHE, key=lambda x: _CACHE[x][0])[:50]:
            _CACHE.pop(old, None)
    return out


def remember(items: list[dict]) -> None:
    for it in items:
        _ITEMS.pop(it["id"], None)
        _ITEMS[it["id"]] = it
    while len(_ITEMS) > 5000:                                          # (the oldest go first)
        _ITEMS.pop(next(iter(_ITEMS)))


def known(item_id) -> dict | None:
    """A result this app showed (the page sends only its id: the address it is downloaded from is never the page's)."""
    return _ITEMS.get(str(item_id or ""))


# ------------------------------------------------------------------------------------------------ saving
def _ext(item: dict) -> str:
    m = re.search(r"\.(mp4|mov|m4v|jpe?g|png|webp)(?:\?|$)", str(item.get("download") or ""), re.I)
    if m:
        return "." + m.group(1).lower().replace("jpeg", "jpg")
    return ".mp4" if item.get("kind") == "video" else ".jpg"


def name_for(item: dict) -> str:
    from .library import safe
    what = " ".join(str(item.get("what") or "").split()[:6]) or ("Stock video" if item.get("kind") == "video" else "Stock photo")
    return safe(f"{what} - {item.get('source', 'stock')} {str(item.get('id', '')).rsplit(':', 1)[-1]}", 100) + _ext(item)


def fetch(item: dict, dest_dir: Path) -> Path:
    """The file of one stock result, downloaded once into dest_dir/<source>/ (with its licence and credit next to it)."""
    if not isinstance(item, dict) or item.get("source") not in ("pixabay", "pexels", "openverse") or not str(item.get("download") or "").startswith("https://"):
        raise ValueError("that isn't a stock result FirstCut can download")
    folder = Path(dest_dir) / str(item["source"])
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / name_for(item)
    least = 10_000 if item.get("kind") == "video" else 1_000
    if out.exists() and out.stat().st_size >= least:
        return out
    part = out.with_name(out.name + ".part")
    try:
        with requests.get(item["download"], headers=UA, timeout=(15, 300), stream=True) as r:
            r.raise_for_status()
            with open(part, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        if part.stat().st_size < least:
            raise ValueError("the download came back empty")
        part.replace(out)
    except Exception:
        part.unlink(missing_ok=True)
        raise
    side = {k: item.get(k) for k in ("source", "kind", "id", "what", "page", "license", "credit", "creator", "w", "h", "dur")}
    out.with_suffix(".json").write_text(json.dumps({**side, "fetched_at": time.time()}, indent=1))
    return out


def credit_for(path) -> str | None:
    try:
        return json.loads(Path(path).with_suffix(".json").read_text()).get("credit") or None
    except (OSError, ValueError):
        return None
