"""Plain-word helpers shared by the footage search and the stock-footage finder."""

from __future__ import annotations

import re

STOP = set("""the a an and or of to in on at for from with by vs versus is are was were be this that it its as new live today score game match
news update updates latest watch how what who when why where""".split())


def tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", str(text).lower().replace("'", "")) if w not in STOP and (len(w) > 2 or w.isdigit())]
