"""Real pictures of the things a video talks about: a country's FLAG, a company's LOGO, a person / place / thing's PHOTO. Found by name, no key needed.

    flag    flagcdn.com - flags of every country (and the UK nations, US states): public domain
    logo    Simple Icons (CC0 data: the marks of ~3000 brands, with their brand colours); else the company site's own icon; else its Wikipedia logo
    photo   Wikipedia / Wikimedia Commons: the article's lead picture, ONLY when its licence is public domain, CC0 or CC BY (credit kept)

Brand names and logos are trademarks of their owners; they are shown to identify the company being talked about (the way any commentary video does),
never as part of a claim. Photos with a licence that needs more than a credit are skipped. Everything is cached under assets/entities/.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import quote, unquote

import requests

from ..core.log import get_logger

log = get_logger("clipper.entities")
UA = {"User-Agent": "Clipfarm/1.0 (entity pictures for video graphics)"}
COUNTRY_ALIASES = {"usa": "us", "u.s.": "us", "u.s.a.": "us", "america": "us", "united states of america": "us", "the united states": "us", "uk": "gb",
                   "britain": "gb", "great britain": "gb", "the uk": "gb", "england": "gb-eng", "scotland": "gb-sct", "wales": "gb-wls",
                   "northern ireland": "gb-nir", "russia": "ru", "south korea": "kr", "north korea": "kp", "uae": "ae", "czechia": "cz",
                   "ivory coast": "ci", "holland": "nl", "the netherlands": "nl", "burma": "mm", "vatican": "va", "taiwan": "tw", "palestine": "ps"}


def cache_dir(cfg) -> Path:
    return Path(cfg.root) / (cfg.get("entities.dir") or "assets/entities")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")[:60]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", str(s).lower().replace("&", " and ")).strip()


def _get(url: str, **kw) -> requests.Response:
    r = requests.get(url, timeout=kw.pop("timeout", 25), headers={**UA, **kw.pop("headers", {})}, **kw)
    r.raise_for_status()
    return r


def _cached_json(cfg, name: str, url: str, max_age_days: float = 14.0):
    path = cache_dir(cfg) / name
    if path.exists() and time.time() - path.stat().st_mtime < max_age_days * 86400:
        try:
            return json.loads(path.read_text())
        except ValueError:
            pass
    data = _get(url).json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


# ------------------------------------------------------------------------------------------------ flags
def flag(cfg, name: str) -> dict | None:
    n = _norm(name)
    try:
        codes = _cached_json(cfg, "flag_codes.json", "https://flagcdn.com/en/codes.json")
    except Exception as e:  # noqa: BLE001
        log.warning(f"flag list unavailable: {e}")
        return None
    code = COUNTRY_ALIASES.get(n) or next((c for c, nm in codes.items() if _norm(nm) == n), None)
    if code is None and n.startswith("the "):
        code = next((c for c, nm in codes.items() if _norm(nm) == n[4:]), None)
    if code is None:
        return None
    out = cache_dir(cfg) / "flags" / f"{code}.png"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(_get(f"https://flagcdn.com/w1280/{code}.png").content)
    return {"file": str(out), "type": "flag", "name": codes.get(code) or name, "credit": "", "color": None}


# ------------------------------------------------------------------------------------------------ logos
def _raster_svg(svg: Path, out_png: Path, size: int = 512) -> bool:
    """A black-on-white picture of the SVG (macOS Quick Look: nothing to install)."""
    try:
        tmp = out_png.parent
        subprocess.run(["qlmanage", "-t", "-s", str(size), "-o", str(tmp), str(svg)], capture_output=True, timeout=60)
        made = tmp / (svg.name + ".png")
        if made.exists():
            made.replace(out_png)
            return True
    except (OSError, subprocess.SubprocessError) as e:
        log.debug(f"svg raster failed: {e}")
    return False


def _mark_png(svg_png: Path, out: Path) -> bool:
    """A black mark on white -> a white mark on transparent (the card tints it); the margin is cropped away."""
    from PIL import Image, ImageOps
    im = Image.open(svg_png).convert("L")
    mask = ImageOps.invert(im)
    bbox = mask.point(lambda v: 255 if v > 40 else 0).getbbox()
    if not bbox:
        return False
    mask = mask.crop(bbox)
    rgba = Image.new("RGBA", mask.size, (255, 255, 255, 0))
    rgba.putalpha(mask)
    rgba.save(out)
    return True


def brand(cfg, name: str, domain: str | None = None) -> dict | None:
    n = _norm(name)
    try:
        icons = _cached_json(cfg, "simple_icons.json", "https://cdn.jsdelivr.net/npm/simple-icons@latest/data/simple-icons.json")
        icons = icons.get("icons", icons) if isinstance(icons, dict) else icons
    except Exception as e:  # noqa: BLE001
        log.warning(f"brand list unavailable: {e}")
        icons = []
    hit = next((i for i in icons if _norm(i.get("title", "")) == n or i.get("slug") == _slug(name).replace("-", "")
                or any(_norm(a) == n for a in (i.get("aliases", {}).get("aka") or []))), None)
    folder = cache_dir(cfg) / "logos"
    folder.mkdir(parents=True, exist_ok=True)
    if hit:
        out = folder / f"{hit['slug']}.png"
        if not out.exists():
            svg = folder / f"{hit['slug']}.svg"
            svg.write_bytes(_get(f"https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/{hit['slug']}.svg").content)
            raw = folder / f"{hit['slug']}.raw.png"
            if not (_raster_svg(svg, raw) and _mark_png(raw, out)):
                out = None
            raw.unlink(missing_ok=True)
        if out and out.exists():
            return {"file": str(out), "type": "logo", "name": hit["title"], "credit": "", "color": "#" + str(hit.get("hex") or "FFFFFF"), "mark": True}
    guess = re.sub(r"[^a-z0-9]", "", str(name).lower())
    if not hit and guess:                                                                    # not in the list (brands can be withdrawn from it): try the icon file by name
        out = folder / f"{guess}.png"
        if not out.exists():
            try:
                svg = folder / f"{guess}.svg"
                svg.write_bytes(_get(f"https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/{guess}.svg").content)
                raw = folder / f"{guess}.raw.png"
                ok = _raster_svg(svg, raw) and _mark_png(raw, out)
                raw.unlink(missing_ok=True)
                if not ok:
                    out.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
        if out.exists():
            return {"file": str(out), "type": "logo", "name": name, "credit": "", "color": None, "mark": True}
    if domain and re.match(r"^[a-z0-9.-]+\.[a-z]{2,}$", domain.lower()):                  # the company's own site icon
        out = folder / f"fav_{_slug(domain)}.png"
        if not out.exists():
            data = _get(f"https://www.google.com/s2/favicons?domain={domain.lower()}&sz=256").content
            if len(data) > 700:                                                          # (the generic globe placeholder is tiny)
                out.write_bytes(data)
        if out.exists():
            return {"file": str(out), "type": "logo", "name": name, "credit": "", "color": None, "mark": False}
    got = commons_photo(cfg, name) if str(name) else None                                  # a free photo of the product itself beats a guessed logo
    return got or wiki_image(cfg, name, allow_nonfree=True, as_logo=True)


# ------------------------------------------------------------------------------------------------ photos
OK_LICENSE = re.compile(r"^(public domain|pd\b|cc0|cc[- ]by[- ][\d.]+)", re.I)


def _commons_license(title: str) -> tuple[str, str] | None:
    """(licence, artist) of a Commons file, or None."""
    d = _get("https://commons.wikimedia.org/w/api.php", params={"action": "query", "titles": title, "prop": "imageinfo", "iiprop": "extmetadata", "format": "json"}).json()
    page = next(iter((d.get("query", {}).get("pages") or {}).values()), {})
    md = (page.get("imageinfo") or [{}])[0].get("extmetadata") or {}
    lic = str((md.get("LicenseShortName") or {}).get("value") or "")
    who = re.sub(r"<[^>]+>", "", str((md.get("Artist") or {}).get("value") or "")).strip()
    return (lic, who) if lic else None


def commons_photo(cfg, name: str) -> dict | None:
    """A freely licensed photo of `name` from Wikimedia Commons: the best title match among public-domain / CC0 / CC BY pictures, wide enough to use."""
    d = _get("https://commons.wikimedia.org/w/api.php", params={
        "action": "query", "generator": "search", "gsrsearch": f"{name} filetype:bitmap", "gsrnamespace": 6, "gsrlimit": 20, "prop": "imageinfo",
        "iiprop": "url|size|extmetadata", "iiurlwidth": 1280, "format": "json"}).json()
    want = [w for w in re.findall(r"[a-z0-9]+", name.lower()) if len(w) > 1]
    best = None
    for pg in (d.get("query", {}).get("pages") or {}).values():
        ii = (pg.get("imageinfo") or [{}])[0]
        md = ii.get("extmetadata") or {}
        lic = str((md.get("LicenseShortName") or {}).get("value") or "").strip()
        if not OK_LICENSE.match(lic) or int(ii.get("width") or 0) < 800 or not ii.get("thumburl"):
            continue
        title = re.sub(r"[^a-z0-9 ]+", " ", pg.get("title", "").lower())
        hit = sum(1 for w in want if w in title.split())
        if hit < len(want) or re.search(r"\b(logo|map|diagram|flag|signature|stamp)\b", title):
            continue
        score = hit * 10 + min(2.0, int(ii.get("width") or 0) / 2000) - (0.1 * int(pg.get("index") or 20))
        if best is None or score > best[0]:
            who = re.sub(r"<[^>]+>", "", str((md.get("Artist") or {}).get("value") or "")).strip()[:80] or "Wikimedia Commons"
            best = (score, ii["thumburl"], f"{who} / Wikimedia Commons ({lic})")
    if best is None:
        return None
    out = cache_dir(cfg) / "photos" / f"{_slug(name)}.jpg"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(_get(best[1], timeout=60).content)
    return {"file": str(out), "type": "photo", "name": name, "credit": best[2], "color": None, "mark": False}


def wiki_image(cfg, name: str, allow_nonfree: bool = False, as_logo: bool = False) -> dict | None:
    """The lead picture of the Wikipedia article about `name`. Photos: only free licences. Logos (allow_nonfree): the article's logo as shown there."""
    try:
        d = _get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(name.replace(' ', '_'))}", headers={"Accept": "application/json"}).json()
    except Exception as e:  # noqa: BLE001
        log.debug(f"wikipedia has no page for {name!r}: {e}")
        return None
    if d.get("type") == "disambiguation":
        return None
    img = d.get("originalimage") or d.get("thumbnail") or {}
    src = str(img.get("source") or "")
    if not src:
        return None
    credit = ""
    m = re.search(r"/wikipedia/commons/(?:thumb/)?[0-9a-f]/[0-9a-f]{2}/([^/]+)", src)
    if m:
        lic = _commons_license("File:" + unquote(m.group(1)))
        if not lic or not OK_LICENSE.match(lic[0].strip()):
            return None
        credit = f"{lic[1] or 'Wikimedia Commons'} / Wikimedia Commons ({lic[0]})"
    elif not allow_nonfree:
        return None                                                                      # an article-only (non-free) picture: not for a photo
    ext = ".png" if src.lower().split("?")[0].endswith((".png", ".svg", ".svg.png")) else ".jpg"
    out = cache_dir(cfg) / ("logos" if as_logo else "photos") / f"{_slug(d.get('title') or name)}{ext}"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        thumb = (d.get("thumbnail") or {}).get("source") or src                          # the article's own thumbnail (1k wide at most): plenty, and quick
        out.write_bytes(_get(thumb, timeout=45).content)
    return {"file": str(out), "type": "logo" if as_logo else "photo", "name": d.get("title") or name, "credit": credit, "color": None, "mark": False}


# ------------------------------------------------------------------------------------------------ the one entry point
def resolve(cfg, item: dict) -> dict | None:
    """item = {"name", "type": country|company|product|person|place|thing, "domain"?} -> {"file", "type", "name", "credit", "color", "mark"} or None."""
    name = " ".join(str(item.get("name") or "").split())
    kind = str(item.get("type") or "thing").lower()
    if not name:
        return None
    try:
        if kind in ("country", "flag", "nation"):
            return flag(cfg, name) or commons_photo(cfg, name)
        if kind in ("company", "brand", "product", "organization", "app", "software", "service"):
            return brand(cfg, name, item.get("domain"))
        return commons_photo(cfg, name)                                                    # a person, a place, a thing: a freely licensed photo
    except Exception as e:  # noqa: BLE001 - a missing picture just means no graphic
        log.warning(f"no picture for {name!r} ({kind}): {e}")
        return None
