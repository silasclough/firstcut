"""Reusable stock / government / public-domain VIDEO, searched by what a shot should show.

Providers (each is skipped if it fails or needs a key you have not set; every result carries the licence it was found under):

  nasa      NASA Image and Video Library - NASA's media are public domain (no key)
  archive   Internet Archive collections that are US-government or public-domain films: fedflix, prelinger, nasa (no key). Per item the
            licence URL must say public domain / CC0 / CC BY, or the collection itself must be one of those
  commons   Wikimedia Commons videos whose licence is Public domain, CC0 or CC BY (not -SA/-NC/-ND) (no key)
  pexels    Pexels videos (and photos: pexels_photos) - the Pexels licence allows commercial use, no credit needed (PEXELS_API_KEY, free)
  pixabay   Pixabay videos (and photos: pixabay_photos) - Pixabay Content licence, same idea (PIXABAY_API_KEY, free). Files are downloaded
            and kept, never hotlinked
  broll()   Pixabay + Pexels videos and photos for a long video's B-roll (landscape for 16:9, at most 1080p); fetch(into=) keeps them in its Stock folder

Never used: anything NonCommercial / NoDerivatives / ShareAlike, or with no stated licence. The licence text and the credit go with every file
(a `.stock.json` next to it) and into the post's description. Config: stock.* (providers, cache folder, size limits).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import requests

from .core.log import get_logger

log = get_logger("stock")
UA = {"User-Agent": "Clipfarm/1.0 (stock video search; contact: owner)"}
TIMEOUT = 30
_COOL: dict[str, float] = {}                 # provider -> time before which it is left alone (it said "too many requests")
PD_COLLECTIONS = {"fedflix": "Public domain (US government film, FedFlix)", "nasa": "Public domain (NASA)",
                  "prelinger": "Public domain (Prelinger Archives)"}


def license_ok(text: str) -> bool:
    """Public domain, CC0 or plain CC BY / Pexels / Pixabay / NASA. Anything with NC, ND or SA in it, or unknown, is out."""
    t = str(text or "").lower()
    if not t:
        return False
    if re.search(r"\b(nc|nd|sa)\b|non-?commercial|no-?deriv|share-?alike|by-nc|by-sa|by-nd", t):
        return False
    return bool(re.search(r"public ?domain|cc0|zero|\bcc[ -]by\b|/by/|attribution|pexels|pixabay|nasa|government|fedflix|prelinger", t))


def _get(url: str, **kw) -> requests.Response:
    r = requests.get(url, timeout=kw.pop("timeout", TIMEOUT), headers={**UA, **kw.pop("headers", {})}, **kw)
    r.raise_for_status()
    return r


def _secs(v) -> float:
    if v in (None, ""):
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    parts = [float(x) for x in str(v).split(":") if re.match(r"^\d+(\.\d+)?$", x)]
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts))) if parts else 0.0


def _clean(s: str, n: int = 400) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", str(s or "")).split())[:n]


# ------------------------------------------------------------------------------------------------ providers
def nasa(q: str, n: int) -> list[dict]:
    d = _get("https://images-api.nasa.gov/search", params={"q": q, "media_type": "video"}).json()
    out = []
    for it in (d.get("collection", {}).get("items") or [])[:n]:
        m = (it.get("data") or [{}])[0]
        out.append({"provider": "nasa", "id": m.get("nasa_id"), "title": _clean(m.get("title"), 160), "desc": _clean(m.get("description")),
                    "tags": m.get("keywords") or [], "duration": 0.0, "w": 0, "h": 0, "_assets": it.get("href"),
                    "page": f"https://images.nasa.gov/details/{m.get('nasa_id')}", "license": "Public domain (NASA)",
                    "credit": f"NASA ({m.get('center') or 'NASA'})", "creator": "NASA"})
    return [o for o in out if o["id"] and o["_assets"]]


def _nasa_file(item: dict) -> str | None:
    files = _get(item["_assets"]).json()
    mp4 = [u for u in files if str(u).lower().endswith(".mp4")]
    for want in ("~medium", "~large", "~small", "~orig", ""):
        for u in mp4:
            if want in u:
                return u.replace(" ", "%20")
    return None


def archive(q: str, n: int) -> list[dict]:
    cols = " OR ".join(PD_COLLECTIONS)
    d = _get("https://archive.org/advancedsearch.php", params={
        "q": f"({q}) AND mediatype:movies AND collection:({cols})", "fl[]": ["identifier", "title", "licenseurl", "runtime", "description", "collection",
                                                                            "creator"], "rows": n, "output": "json", "sort[]": "downloads desc"}).json()
    out = []
    for doc in d.get("response", {}).get("docs", []):
        cols_ = doc.get("collection") or []
        cols_ = [cols_] if isinstance(cols_, str) else cols_
        lic = doc.get("licenseurl") or next((PD_COLLECTIONS[c] for c in cols_ if c in PD_COLLECTIONS), "")
        if not license_ok(lic):
            continue
        out.append({"provider": "archive", "id": doc["identifier"], "title": _clean(doc.get("title"), 160), "desc": _clean(doc.get("description")),
                    "tags": cols_, "duration": _secs(doc.get("runtime")), "w": 0, "h": 0, "page": f"https://archive.org/details/{doc['identifier']}",
                    "license": lic, "credit": f"{_clean(doc.get('title'), 100)} (Internet Archive)", "creator": _clean(doc.get("creator"), 80) or "Internet Archive"})
    return out


def _archive_file(item: dict) -> str | None:
    meta = _get(f"https://archive.org/metadata/{item['id']}").json()
    files = [f for f in meta.get("files", []) if str(f.get("name", "")).lower().endswith((".mp4", ".mpeg4", ".m4v"))]
    files = [f for f in files if 0 < int(f.get("size") or 0) < 400 * 1024 * 1024]
    if not files:
        return None
    best = max(files, key=lambda f: (f.get("format", "") in ("h.264", "MPEG4", "512Kb MPEG4"), int(f.get("size") or 0) < 150 * 1024 * 1024, int(f.get("size") or 0)))
    return f"https://archive.org/download/{item['id']}/{requests.utils.quote(best['name'])}"


COMMONS_OK = re.compile(r"^(public domain|pd|cc0|cc[- ]by[- ][\d.]+)", re.I)


def commons(q: str, n: int) -> list[dict]:
    d = _get("https://commons.wikimedia.org/w/api.php", params={
        "action": "query", "generator": "search", "gsrsearch": f"{q} filetype:video", "gsrnamespace": 6, "gsrlimit": n, "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata", "format": "json"}).json()
    out = []
    for p in (d.get("query", {}).get("pages") or {}).values():
        ii = (p.get("imageinfo") or [{}])[0]
        md = ii.get("extmetadata") or {}
        lic = (md.get("LicenseShortName") or {}).get("value", "") or ("Public domain" if "PD" in (md.get("Categories") or {}).get("value", "") else "")
        if not (COMMONS_OK.match(lic.strip()) and license_ok(lic)):
            continue
        if int(ii.get("size") or 0) > 400 * 1024 * 1024:
            continue
        who = _clean((md.get("Artist") or {}).get("value"), 80) or "Wikimedia Commons"
        out.append({"provider": "commons", "id": str(p.get("pageid")), "title": _clean((md.get("ObjectName") or {}).get("value") or p.get("title"), 160),
                    "desc": _clean((md.get("ImageDescription") or {}).get("value")), "tags": str((md.get("Categories") or {}).get("value", "")).split("|"),
                    "duration": float(ii.get("duration") or 0), "w": int(ii.get("width") or 0), "h": int(ii.get("height") or 0), "url": ii.get("url"),
                    "page": ii.get("descriptionurl"), "license": lic, "credit": f"{who} / Wikimedia Commons ({lic})", "creator": who})
    return out


def pexels(q: str, n: int, orientation: str = "portrait") -> list[dict]:
    """orientation "landscape" (a 16:9 long video) picks the biggest file that is at most 1080p."""
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        return []
    d = _get("https://api.pexels.com/videos/search", params={"query": q, "per_page": n, "orientation": orientation, "size": "medium"},
             headers={"Authorization": key}).json()
    out = []
    for v in d.get("videos", []):
        files = [f for f in v.get("video_files", []) if f.get("file_type") == "video/mp4" and (f.get("height") or 0) >= 720]
        if orientation == "landscape":
            files = [f for f in files if (f.get("width") or 0) > (f.get("height") or 0) and min(f.get("width") or 0, f.get("height") or 0) <= 1080]
        if not files:
            continue
        if orientation == "landscape":
            f = max(files, key=lambda f: f.get("width") or 0)
        else:
            f = min(files, key=lambda f: abs((f.get("height") or 0) - 1920) if (f.get("height") or 0) > (f.get("width") or 0) else abs((f.get("width") or 0) - 1280))
        out.append({"provider": "pexels", "id": str(v["id"]), "title": f"Pexels video {v['id']} by {v.get('user', {}).get('name', '')}", "desc": _clean(v.get("url")),
                    "tags": re.findall(r"[a-z]+", str(v.get("url", "")).lower()), "duration": float(v.get("duration") or 0), "w": f.get("width") or 0,
                    "h": f.get("height") or 0, "url": f["link"], "page": v.get("url"), "license": "Pexels License (free for commercial use)",
                    "credit": f"Video: {v.get('user', {}).get('name', 'Pexels contributor')} / Pexels", "creator": v.get("user", {}).get("name", "Pexels")})
    return out


def pixabay(q: str, n: int, orientation: str | None = None) -> list[dict]:
    """orientation "landscape" / "portrait": only clips of that shape, in the biggest size that is at most 1080p."""
    key = os.environ.get("PIXABAY_API_KEY")
    if not key:
        return []
    d = _get("https://pixabay.com/api/videos/", params={"key": key, "q": q, "per_page": max(3, n), "safesearch": "true"}).json()
    out = []
    for v in d.get("hits", []):
        vv = v.get("videos") or {}
        if orientation:
            sizes = [x for x in (vv.get(k) or {} for k in ("large", "medium", "small")) if x.get("url") and 0 < min(x.get("width") or 0, x.get("height") or 0) <= 1080]
            sizes = [x for x in sizes if ((x.get("width") or 0) > (x.get("height") or 0)) == (orientation == "landscape")]
            f = max(sizes, key=lambda x: x.get("width") or 0) if sizes else {}
        else:
            f = vv.get("large") or vv.get("medium") or {}
        if not f.get("url"):
            continue
        out.append({"provider": "pixabay", "id": str(v["id"]), "title": f"Pixabay video {v['id']}", "desc": _clean(v.get("tags")),
                    "tags": [t.strip() for t in str(v.get("tags", "")).split(",")], "duration": float(v.get("duration") or 0), "w": f.get("width") or 0,
                    "h": f.get("height") or 0, "url": f["url"], "page": v.get("pageURL"), "license": "Pixabay Content License (free for commercial use)",
                    "credit": f"Video: {v.get('user', 'Pixabay contributor')} / Pixabay", "creator": v.get("user", "Pixabay")})
    return out


def pixabay_photos(q: str, n: int, orientation: str = "horizontal") -> list[dict]:
    """Pixabay photos (Pixabay Content License). The file is `largeImageURL` (at most 1280 wide): downloaded, never hotlinked."""
    key = os.environ.get("PIXABAY_API_KEY")
    if not key:
        return []
    d = _get("https://pixabay.com/api/", params={"key": key, "q": q, "image_type": "photo", "orientation": orientation, "per_page": max(3, n),
                                                 "safesearch": "true"}).json()
    out = []
    for h in d.get("hits", []):
        url = h.get("largeImageURL") or h.get("webformatURL")
        if not url:
            continue
        out.append({"provider": "pixabay", "kind": "photo", "id": str(h["id"]), "title": f"Pixabay photo {h['id']}", "desc": _clean(h.get("tags")),
                    "tags": [t.strip() for t in str(h.get("tags", "")).split(",")], "duration": 0.0, "w": h.get("imageWidth") or 0,
                    "h": h.get("imageHeight") or 0, "url": url, "page": h.get("pageURL"), "license": "Pixabay Content License (free for commercial use)",
                    "credit": f"Photo: {h.get('user', 'Pixabay contributor')} / Pixabay", "creator": h.get("user", "Pixabay")})
    return out


def pexels_photos(q: str, n: int, orientation: str = "landscape") -> list[dict]:
    """Pexels photos (Pexels License). The file is the `large2x` size (about 1880 wide)."""
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        return []
    d = _get("https://api.pexels.com/v1/search", params={"query": q, "per_page": n, "orientation": orientation},
             headers={"Authorization": key}).json()
    out = []
    for p in d.get("photos", []):
        src = p.get("src") or {}
        url = src.get("large2x") or src.get("large") or src.get("original")
        if not url:
            continue
        who = p.get("photographer") or "Pexels contributor"
        out.append({"provider": "pexels", "kind": "photo", "id": str(p["id"]), "title": _clean(p.get("alt"), 160) or f"Pexels photo {p['id']}",
                    "desc": _clean(p.get("alt") or p.get("url")), "tags": re.findall(r"[a-z]+", str(p.get("url", "")).lower()), "duration": 0.0,
                    "w": p.get("width") or 0, "h": p.get("height") or 0, "url": url, "page": p.get("url"),
                    "license": "Pexels License (free for commercial use)", "credit": f"Photo: {who} / Pexels", "creator": who})
    return out


PROVIDERS = {"nasa": nasa, "archive": archive, "commons": commons, "pexels": pexels, "pixabay": pixabay}
KEYS = {"pixabay": "PIXABAY_API_KEY", "pexels": "PEXELS_API_KEY"}


def keys() -> list[str]:
    """The stock sites this Mac has a key for (the keys themselves are only ever read from the environment, never stored or logged)."""
    return [name for name, var in KEYS.items() if os.environ.get(var)]


def broll(q: str, landscape: bool = True, n: int = 6, photos: bool = True) -> list[dict]:
    """B-roll for one shot of a long video from Pixabay and Pexels: videos (and photos) in the video's shape, at most 1080p, best first. Each
    result gets `kind` ("video" / "photo") and `score`. A site that fails (or has no key) is skipped, never fatal."""
    calls = [("pixabay", "video", lambda: pixabay(q, n, "landscape" if landscape else "portrait")),
             ("pexels", "video", lambda: pexels(q, n, "landscape" if landscape else "portrait"))]
    if photos:
        calls += [("pixabay", "photo", lambda: pixabay_photos(q, n, "horizontal" if landscape else "vertical")),
                  ("pexels", "photo", lambda: pexels_photos(q, n, "landscape" if landscape else "portrait"))]
    out: list[dict] = []
    import time as _t
    for name, kind, fn in calls:
        if name not in keys() or _COOL.get(name, 0) > _t.time():
            continue
        try:
            found = fn()
        except Exception as e:  # noqa: BLE001
            log.warning(f"stock {name} {kind} search {q!r} failed: {type(e).__name__}")       # (the message could carry the URL, and so the key)
            continue
        for rank, x in enumerate(found):
            if not license_ok(x.get("license")):
                continue
            w, h = int(x.get("w") or 0), int(x.get("h") or 0)
            if w and h and (w > h) != landscape:
                continue                                                # the wrong shape for this video
            x = {**x, "kind": x.get("kind") or "video"}
            x["score"] = broll_score(x, q, rank)
            out.append(x)
    return sorted(out, key=lambda x: -x["score"])


def broll_score(item: dict, query: str, rank: int = 0) -> float:
    """How well a stock result fits a B-roll shot: the site's own order (its search knows its pictures better than their titles say), the
    query's words in the title / tags, a usable length, and a video over a photo."""
    s = 0.5 * relevance(item, query) + 0.3 * max(0.0, 1.0 - rank / 6.0)
    if item.get("kind") == "video":
        d = float(item.get("duration") or 0)
        s += 0.2 if d >= 5 else (0.05 if d >= 3 or not d else -0.4)
    return round(s, 3)


def local_name(item: dict) -> str:
    """The file name a downloaded stock item gets in a shared folder: "<provider>_<id>.<ext>" (the same item, the same file, every time)."""
    ext = (Path(str(item.get("url") or "").split("?")[0]).suffix or (".jpg" if item.get("kind") == "photo" else ".mp4")).lower()
    safe = re.sub(r"[^\w.-]", "_", str(item["id"]))[:80]
    return f"{item['provider']}_{safe}{ext}"


# ------------------------------------------------------------------------------------------------ search + rank
def _tokens(s: str) -> set[str]:
    from .core.words import tokens
    return set(tokens(s))


def relevance(item: dict, query: str) -> float:
    """0..1: how many of the query's words the result's title / description / tags contain (title counts double)."""
    q = _tokens(query)
    if not q:
        return 0.0
    title, rest = _tokens(item.get("title", "")), _tokens(" ".join([item.get("desc", ""), " ".join(map(str, item.get("tags") or []))]))
    hit = sum(2 for w in q if w in title) + sum(1 for w in q if w in rest and w not in title)
    return min(1.0, hit / (2 * len(q)))


def score(item: dict, query: str, want_s: float = 5.0) -> float:
    """How well a result fits the shot we asked for: relevance first, then a usable length, portrait or HD, and a small bonus for the stock
    sites whose clips are cut for this (short, clean, vertical-friendly)."""
    s = relevance(item, query)
    d = float(item.get("duration") or 0)
    if d:
        s += 0.15 if want_s <= d <= 90 else (-0.2 if d < want_s else -0.05)
    if (item.get("h") or 0) > (item.get("w") or 0) > 0:
        s += 0.15
    if (item.get("h") or 0) >= 720 or (item.get("w") or 0) >= 1280:
        s += 0.1
    if item["provider"] in ("pexels", "pixabay"):
        s += 0.1
    return round(s, 3)


def search(cfg, query: str, n: int = 8, want_s: float = 5.0, providers: list[str] | None = None) -> list[dict]:
    """Best matches first, from every enabled provider. A provider that errors is skipped, never fatal."""
    scfg = cfg.get("stock", {}) or {}
    names = providers or scfg.get("providers") or list(PROVIDERS)
    per = int(scfg.get("per_provider", 8))
    out: list[dict] = []
    import time as _t
    for name in names:
        fn = PROVIDERS.get(name)
        if not fn or _COOL.get(name, 0) > _t.time():
            continue
        try:
            found = fn(query, per)
        except Exception as e:  # noqa: BLE001
            log.warning(f"stock {name} search {query!r} failed: {e}")
            continue
        out += [x for x in found if license_ok(x.get("license"))]
    for x in out:
        x["score"] = score(x, query, want_s)
    # a shot must really be about what was asked: at least `min_relevance` of the query's words (a news reel with the wrong picture is worse than none)
    out = [x for x in out if relevance(x, query) >= float(scfg.get("min_relevance", 0.5))]
    return sorted(out, key=lambda x: -x["score"])[:n]


# ------------------------------------------------------------------------------------------------ download
def cache_dir(cfg) -> Path:
    return Path(cfg.root) / (cfg.get("stock.dir") or "assets/stock")


def fetch(cfg, item: dict, into: Path | None = None) -> Path:
    """Download (once; cached by provider + id) and return the file. A `.stock.json` beside it records where it came from and its licence.
    into = a folder of its own (a long video's Stock folder): the file is named local_name(item) there."""
    if into is not None:
        d = Path(into)
        safe = Path(local_name(item)).stem
    else:
        d = cache_dir(cfg) / item["provider"]
        safe = re.sub(r"[^\w.-]", "_", str(item["id"]))[:80]
    d.mkdir(parents=True, exist_ok=True)
    for existing in d.glob(f"{safe}.*"):
        if existing.suffix not in (".json", ".part"):
            return existing
    url = item.get("url")
    if not url:
        url = _nasa_file(item) if item["provider"] == "nasa" else _archive_file(item) if item["provider"] == "archive" else None
    if not url:
        raise RuntimeError(f"no downloadable video file for {item['provider']}:{item['id']}")
    ext = Path(local_name(item)).suffix if into is not None else (Path(url.split("?")[0]).suffix or ".mp4").lower()
    dest, part = d / f"{safe}{ext}", d / f"{safe}{ext}.part"
    limit = float(cfg.get("stock.max_mb", 400)) * 1024 * 1024
    import time as _t
    if _COOL.get(item["provider"], 0) > _t.time():
        raise RuntimeError(f"{item['provider']} asked us to slow down; leaving it alone for a bit")
    for attempt in range(3):
        with requests.get(url, stream=True, timeout=60, headers=UA) as r:
            if r.status_code == 429:                                   # polite: wait as long as it says, then give that provider a rest
                wait = min(60.0, float(r.headers.get("Retry-After") or 15))
                if attempt == 2:
                    _COOL[item["provider"]] = _t.time() + 300
                    raise RuntimeError(f"{item['provider']}: too many requests")
                _t.sleep(wait)
                continue
            r.raise_for_status()
            got = 0
            with part.open("wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    got += len(chunk)
                    if got > limit:
                        part.unlink(missing_ok=True)
                        raise RuntimeError(f"{item['provider']}:{item['id']} is bigger than {limit / 1e6:.0f} MB")
                    fh.write(chunk)
            break
    if item["provider"] == "commons":
        _t.sleep(1.5)                                                   # Wikimedia is rate-limited for anonymous downloads
    part.replace(dest)
    meta = {k: item.get(k) for k in ("provider", "id", "kind", "title", "page", "license", "credit", "creator", "query") if item.get(k) is not None} | {"source_url": url}
    (d / f"{safe}.stock.json").write_text(json.dumps(meta, indent=1))
    return dest
