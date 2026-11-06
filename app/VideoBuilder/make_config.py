"""The settings shipped inside FirstCut (config/app.yaml): config/config.yaml without anything about this setup - no posting,
accounts, channels, machines or paths of this Mac - plus what the app needs. The packaging stops if anything personal is left.
    python make_config.py <config.yaml> <out app.yaml>
"""

import os
import re
import sys

import yaml

KEEP = ("paths", "llm", "video", "captions", "music", "quality", "clipper", "catalog", "whisper", "power", "voice")
PERSONAL = re.compile(r"\b100\.\d+\.\d+\.\d+\b|ts\.net|metricool|blog_?id|/Users/|/Volumes/"
                      + "".join("|" + p for p in os.environ.get("FIRSTCUT_PRIVATE_PATTERNS", "").split("\n") if p.strip()), re.I)


def main(src: str, out: str) -> None:
    d = yaml.safe_load(open(src)) or {}
    c = {k: d[k] for k in KEEP if k in d}
    cl = c.setdefault("clipper", {})
    for k in ("auto", "commentary", "newsreel", "originals", "feed", "remote_ollama", "library", "taste", "performance", "twitch"):
        cl.pop(k, None)
    cl["remote_ollama"] = {"base_url": None}
    cl["fcp_unpack_dir"] = "~/Movies/FirstCut"
    av = cl.setdefault("ai_videos", {})
    av.update({"dir": "~/Movies/FirstCut", "host": None, "local": True})
    av["chatgpt_plan"] = {**(av.get("chatgpt_plan") or {}), "app_name": "FirstCut"}
    av.setdefault("openai", {})["host"] = None
    for k in ("cookies_file", "push_host", "handoff"):
        cl.pop(k, None)
    llm = c.setdefault("llm", {})
    llm["ollama"] = {**(llm.get("ollama") or {}), "base_url": "http://127.0.0.1:11434"}
    llm.pop("openai", None) if isinstance(llm.get("openai"), dict) and llm["openai"].get("api_key") else None
    c.pop("voice", None)
    c["catalog"] = {**(d.get("catalog") or {}), "roots": [], "dir": "data/catalog"}
    c["clipper"]["app"] = "studio"
    text = "# FirstCut's built-in settings (made from config/config.yaml by app/VideoBuilder/make_config.py; the app writes overlay.yaml on top).\n" \
        + yaml.safe_dump(c, sort_keys=False, allow_unicode=True)
    left = [ln.strip() for ln in text.splitlines() if PERSONAL.search(ln) and not ln.lstrip().startswith("#")]
    if left:
        sys.exit("personal details would ship in app.yaml:\n  " + "\n  ".join(left[:20]))
    open(out, "w").write(text)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
