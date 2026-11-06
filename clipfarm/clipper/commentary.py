"""A line of commentary in a real voice, dropped in where the clip pauses ("...and that's the part nobody expects").

The AI reads the clip's transcript with the natural pause points marked and writes 1-2 short, original lines - a reaction, a
bit of context, a dry joke, or a question - for the best spots. The clip's own audio is ducked for a moment while the voice
(core/voice.py, an open-source TTS) speaks, then comes back. It adds a human-sounding point of view on top of other people's
footage, so the post isn't just a re-upload.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.log import get_logger
from . import moments as M

log = get_logger("clipper.commentary")
SYSTEM = (
    "You add one very short voiceover line to a clip. Choose at most the requested number of the numbered PAUSE points (each one is a real "
    "silence in the clip's own sound) and write ONE plain, simple sentence for it: concise, to the point, matter-of-fact. No filler or "
    "hesitation words (um, uh, hmm, well, so, like, you know, wow), no exclamation marks, no jokes that need a big reaction, no emoji or "
    "hashtags. Good lines: one fact of context, or the point the viewer is already wondering about. The line MUST fit the pause: use no more "
    "words than the pause allows. Never repeat what was just said, never describe the video, never insult real people, never invent facts. "
    "If nothing fits well, return an empty list - no line is better than a forced one.")
BANNED = {"um", "uh", "uhm", "hmm", "mm", "er", "erm", "like", "wow", "well", "so", "basically", "literally", "honestly", "actually", "oh", "ah"}
SCHEMA = {
    "type": "object",
    "properties": {"lines": {"type": "array", "items": {"type": "object", "properties": {
        "pause": {"type": "integer"}, "text": {"type": "string"}}, "required": ["pause", "text"]}}},
    "required": ["lines"],
}


@dataclass
class Line:
    source_time: float      # where in the SOURCE video the pause is (the line is voiced right after it)
    text: str
    idx: int = -1           # the sentence it follows


def pauses(sents: list[M.Sentence], min_gap: float = 0.45) -> list[tuple[int, float, float]]:
    """(sentence idx, its end time, silence after it). The very last sentence counts (the clip's tail)."""
    out = []
    for k, s in enumerate(sents):
        gap = (sents[k + 1].start - s.end) if k + 1 < len(sents) else 1.5
        if gap >= min_gap and (s.end - sents[0].start) >= 2.5:
            out.append((s.idx, s.end, gap))
    return out


def max_words(room_s: float) -> int:
    """How many words a spoken line can have to finish (with a breath to spare) inside `room_s` seconds of silence."""
    return max(0, min(10, int((room_s - 0.4) / 0.42)))


def plan(llm, sents: list[M.Sentence], title: str, max_lines: int = 1, room: dict[int, float] | None = None) -> list[Line]:
    """`room` = {sentence idx: seconds of real silence after it in the FINISHED clip}: a line only goes where it fits entirely inside
    the silence, so it never talks over what the clip says (the main points are never covered)."""
    ps = [p for p in pauses(sents) if room is None or (p[0] in room and max_words(room[p[0]]) >= 3)]
    if not ps or not sents:
        return []
    said = "\n".join(f"  [{s.idx}] {s.text}" for s in sents)
    slots = "\n".join(f"  PAUSE {n}: after sentence [{i}] (silence {(room or {}).get(i, gap):.1f}s, at most {max_words((room or {}).get(i, gap))} words)"
                       for n, (i, _e, gap) in enumerate(ps))
    prompt = f"Clip title: {title}\nTranscript:\n{said}\n\nPause points:\n{slots}\n\nWrite up to {max_lines} line(s)."
    try:
        res = llm.complete_json("clip_commentary", SYSTEM, prompt, SCHEMA)
    except Exception as e:  # noqa: BLE001
        log.warning(f"commentary planning failed: {e}")
        return []
    out, used = [], set()
    for ln in res.data.get("lines") or []:
        try:
            n, text = int(ln["pause"]), " ".join(str(ln["text"]).split())
        except (KeyError, TypeError, ValueError):
            continue
        limit = max_words((room or {}).get(ps[n][0], 3.0)) if 0 <= n < len(ps) else 0
        toks = [t.strip(".,!?;:'\"").lower() for t in text.split()]
        if not (0 <= n < len(ps)) or n in used or not text or len(toks) > limit or "!" in text or any(t in BANNED for t in toks):
            continue
        used.add(n)
        out.append(Line(ps[n][1], text, ps[n][0]))
        if len(out) >= max_lines:
            break
    return sorted(out, key=lambda l: l.source_time)


PODCAST_RE = __import__("re").compile(r"podcast|interview|\bepisode\b|\bep\.? ?\d|conversation with|sits? down with|\bw/ ", __import__("re").I)


def is_podcast(source: dict | None) -> bool:
    """A podcast / interview clip never gets a voiced aside: two real people are talking and a third, synthetic voice only distracts."""
    s = source or {}
    return bool(PODCAST_RE.search(" ".join(str(s.get(k) or "") for k in ("title", "topic", "channel", "uploader"))))


MAX_WORDS_PER_S = 0.8        # more speech than this and the clip already has a voice: ours would only talk over it


def talk_density(words: list[dict], start: float, end: float) -> float:
    """Spoken words per second between `start` and `end` (source time)."""
    n = sum(1 for w in words if start - 0.05 <= w["start"] and w["end"] <= end + 0.05)
    return n / max(1.0, end - start)


def allowed(source: dict | None, density: float, limit: float = MAX_WORDS_PER_S) -> bool:
    """A voiced aside only goes on a clip with little talking in it (a sports play, a scene with no dialogue): never on a podcast / interview, never on a
    clip where people already talk most of the time. It never made those clips better, only more crowded."""
    return not is_podcast(source) and density < limit
