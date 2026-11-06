"""Post caption assembly (Instagram limits: 2,200 chars, 30 hashtags)."""

from __future__ import annotations

import re

IG_MAX_CHARS = 2200
IG_MAX_HASHTAGS = 30


def clean_hashtags(tags: list[str], limit: int = 8) -> list[str]:
    out, seen = [], set()
    for t in tags:
        t = "#" + re.sub(r"[^\w]", "", t.lstrip("#"))
        if len(t) > 1 and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out[: min(limit, IG_MAX_HASHTAGS)]


def build_caption(body: str, hashtags: list[str] | None = None, credits: list[str] | None = None,
                  label: str | None = None, max_hashtags: int = 8) -> str:
    """label (e.g. AI disclosure) first, then body, credits, hashtags."""
    blocks = []
    if label:
        blocks.append(label.strip())
    if body and body.strip():
        blocks.append(body.strip())
    creds = [c.strip() for c in (credits or []) if c and c.strip()]
    if creds:
        blocks.append("\n".join(creds))
    tags = clean_hashtags(hashtags or [], max_hashtags)
    caption = "\n\n".join(blocks)
    while tags:
        candidate = caption + "\n\n" + " ".join(tags)
        if len(candidate) <= IG_MAX_CHARS:
            return candidate
        tags.pop()
    if len(caption) > IG_MAX_CHARS:  # credits/labels must survive; trim the body
        overflow = len(caption) - IG_MAX_CHARS + 1
        body_trim = body.strip()[: max(0, len(body.strip()) - overflow - 1)] + "…"
        return build_caption(body_trim, [], credits, label)
    return caption
