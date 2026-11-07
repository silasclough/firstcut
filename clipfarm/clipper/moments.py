"""Find the best standalone moments in a long transcript.

1. Words -> sentences (split on . ? ! or a pause).
2. The transcript is read in ~8-minute chunks; the LLM proposes self-contained clips in each
   (sentence index ranges + a hook title, score and post caption). Chunks keep prompts small
   enough for a local 14B model and let a 3-hour podcast work the same as a 10-minute video.
3. Every proposal is snapped to sentence boundaries and stretched/shrunk into the allowed length,
   overlapping proposals are merged (best score wins), and the top N are returned.
Offline fallback: sliding windows scored by a keyword/emotion heuristic plus loudness.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np

from . import clipgate

END = re.compile(r"[.!?…]['\"”’)]*$")
HOOKY = re.compile(r"\b(never|nobody|secret|truth|crazy|insane|wild|actually|biggest|worst|best|mistake|"
                   r"why|how|what if|i (?:was|got|had)|money|million|died|fired|caught|lied|shocked|"
                   r"you won't|here's the thing|the reason|turns out|i realized)\b", re.I)


@dataclass
class Sentence:
    idx: int
    start: float
    end: float
    text: str


@dataclass
class Moment:
    start: float
    end: float
    first: int                       # sentence index range (inclusive)
    last: int
    title: str = ""
    hook: str = ""
    caption: str = ""
    hashtags: list[str] = field(default_factory=list)
    score: float = 5.0
    reason: str = ""
    source: str = "llm"
    mood: str = ""                  # music mood for the "background music" option

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return asdict(self)


def sentences_from_words(words: list[dict], pause_s: float = 0.9, max_words: int = 45) -> list[Sentence]:
    out: list[Sentence] = []
    cur: list[dict] = []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt["start"] - w["end"]) if nxt else 99
        if END.search(w["text"]) or gap >= pause_s or len(cur) >= max_words or nxt is None:
            text = " ".join(x["text"] for x in cur).strip()
            out.append(Sentence(len(out), cur[0]["start"], cur[-1]["end"], text))
            cur = []
    return out


# Raw footage with several takes (the work Mac, 2026-11-06: a student reading the same scripted lines four times, with "Okay. Ready? Yes."
# and "That was a good one." between takes). The clips it made were a joke between takes and a blooper ("I think I hit a wire"), each grown
# to the 20 s minimum with crew talk. Such footage gets the best COMPLETE take of a line, nothing between takes, and no 20 s padding.
CREW = re.compile(r"^(?:(?:okay|ok|alright|all right|ready|yes|yeah|yep|go ahead|go|action|rolling|speed|cut|sorry|wait|hold on|hang on|"
                  r"(?:and )?go ahead|and go|perfect|nice|good|great|cool|again|one more(?: time)?|from the top|let'?s (?:do|go|try)(?: it| that| this)?(?: again)?|"
                  r"that was (?:a )?(?:good|great|nice|perfect)(?: one)?|we'?re (?:good|rolling)|take (?:one|two|three|four|five|\d+)|"
                  r"(?:are )?you ready|whenever you'?re ready|and go|in three,? two,? one|three,? two,? one)[\s.!?,-]*)+$", re.I)
RAW_NOTE = ("This is RAW FOOTAGE with several takes: the speaker repeats the same scripted lines, with crew talk between takes "
            "(\"Okay\", \"Ready?\", \"Go ahead\", \"That was a good one\", \"One more time\"). Each clip is the single best COMPLETE, clean take "
            "of a line or passage: start on its first word, end on its last. Never include crew talk or the chatter between takes, never "
            "a fluffed take or a restart, and never a blooper or mishap unless the notes ask for bloopers. Short is fine here: a clean "
            "10-second take beats a 20-second clip padded with other takes. The crew's approval right after a take (\"Perfect\", "
            "\"That was a good one\") marks the take to use; of several clean takes, prefer the last approved one.")


def crew_talk(text: str) -> bool:
    """A sentence that is only crew / slate talk ("Okay. Ready? Yes.", "That was a good one.")."""
    t = " ".join(str(text).split())
    return bool(t) and len(t.split()) <= 8 and bool(CREW.match(t))


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", str(text).lower()).split()[:6])


def raw_takes(sents: list[Sentence]) -> bool:
    """Footage of takes: crew talk between lines, and the same line started more than once."""
    crew = sum(1 for s in sents if crew_talk(s.text))
    starts: dict[str, int] = {}
    for s in sents:
        k = _norm(s.text)
        if len(k.split()) >= 4:
            starts[k] = starts.get(k, 0) + 1
    repeats = sum(n - 1 for n in starts.values() if n > 1)
    return (crew >= 3 and repeats >= 1) or repeats >= 3


def trim_crew(m: Moment, sents: list[Sentence]) -> Moment | None:
    """Crew talk cut off both ends of a clip (None when nothing else is left)."""
    first, last = m.first, m.last
    while first <= last and crew_talk(sents[first].text):
        first += 1
    while last >= first and crew_talk(sents[last].text):
        last -= 1
    if first > last:
        return None
    m.first, m.last = first, last
    m.start, m.end = sents[first].start, sents[last].end
    return m


def fit_length(m: Moment, sents: list[Sentence], min_len: float, max_len: float) -> Moment | None:
    """Grow (alternately forward/back) until >= min_len; only shrink past hard_cap(max_len), so a clip
    may run a little long to finish its thought rather than lose its payoff."""
    first, last = max(0, m.first), min(len(sents) - 1, m.last)
    if first > last:
        return None

    def dur(a, b):
        return sents[b].end - sents[a].start

    forward = True
    while dur(first, last) < min_len:
        if forward and last + 1 < len(sents):
            last += 1
        elif first > 0 and first > m.first - 2:        # grow backwards a little at most; hooks start clips
            first -= 1
        elif last + 1 < len(sents):
            last += 1
        else:
            break
        forward = not forward
    cap = hard_cap(max_len)                             # finishing the thought beats hitting max_len exactly
    while dur(first, last) > cap and last > first:
        last -= 1
    if dur(first, last) > cap:                          # one enormous "sentence": hard cut inside it
        m.start, m.end = sents[first].start, sents[first].start + max_len
    else:
        m.start, m.end = sents[first].start, sents[last].end
    m.first, m.last = first, last
    if m.duration < min_len * 0.6:
        return None
    return m


def hard_cap(max_len: float) -> float:
    if max_len <= 30:                                   # short-form: a thought may run a little over, not half as long again (long clips lose viewers)
        return max_len + 8.0
    return max(max_len + 20.0, max_len * 1.35)


CONTINUES = re.compile(r"^\W*(and|so|but|because|cause|'cause|which|then|that's why|that is why|or|plus|until|"
                       r"unless|except|like i said|anyway|hey|i mean|and then|so then)\b", re.I)


def complete_endings(moments: list[Moment], sents: list[Sentence], max_len: float, max_extra: int = 4) -> list[Moment]:
    """No-AI safety net: extend a clip while it ends on an unfinished sentence, or while the next sentence
    carries straight on ("And...", "So...", "But...", "Hey...") with barely a pause."""
    cap = hard_cap(max_len)
    for m in moments:
        if m.first < 0 or m.last < 0:
            continue
        for _ in range(max_extra):
            nxt = m.last + 1
            if nxt >= len(sents):
                break
            cur, nx = sents[m.last], sents[nxt]
            unfinished = not END.search(cur.text)
            carries_on = bool(CONTINUES.match(nx.text)) and nx.start - cur.end < 0.8
            if not (unfinished or carries_on) or nx.end - m.start > cap:
                break
            m.last, m.end = nxt, nx.end
    return moments


ENDING_SCHEMA = {
    "type": "object",
    "properties": {"clips": {"type": "array", "items": {"type": "object", "properties": {
        "clip": {"type": "integer"}, "last": {"type": "integer"}, "why": {"type": "string"}},
        "required": ["clip", "last", "why"]}}},
    "required": ["clips"],
}

ENDING_SYSTEM = """You check where short-form clips END. A clip must end where its thought fully lands: the \
punchline, the answer, the conclusion, the reaction - never on a setup or halfway through a story. Most bad clips \
stop one or two sentences too early. For each clip you see its last sentences and what is said right after. \
Return the index of the sentence the clip should end on: keep the current one if the thought is already complete, \
otherwise extend to where it completes. Never end on a sentence that is cut off mid-way."""


def llm_fix_endings(llm, moments: list[Moment], sents: list[Sentence], max_len: float, look_ahead: int = 8) -> list[Moment]:
    """One AI pass over all clips: move each end to where the thought completes (within hard_cap)."""
    todo = [(k, m) for k, m in enumerate(moments) if m.first >= 0 and m.last >= 0]
    if not todo:
        return moments
    cap = hard_cap(max_len)
    blocks = []
    for k, m in todo:
        lines = [f"  [{s.idx}] {s.text}" for s in sents[max(m.first, m.last - 1):m.last + 1]]
        after = [f"  [{s.idx}] {s.text}" for s in sents[m.last + 1:m.last + 1 + look_ahead]
                 if s.end - m.start <= cap]
        blocks.append(f"Clip {k} (\"{m.title}\"), currently ends at [{m.last}]:\n" + "\n".join(lines) +
                      ("\n  -- what is said next --\n" + "\n".join(after) if after else "\n  (nothing after fits)"))
    prompt = "\n\n".join(blocks) + "\n\nFor every clip, return clip number and the sentence index it should end on."

    def ok(c):
        k = int(c.get("clip", -1))
        if not (0 <= k < len(moments)):
            return False
        m = moments[k]
        last = int(c.get("last", -1))
        return m.last <= last < len(sents) and sents[last].end - m.start <= cap

    try:
        res = llm.complete_json("clip_endings", ENDING_SYSTEM, prompt, ENDING_SCHEMA)
    except Exception:  # noqa: BLE001 - the no-AI safety net still runs
        return moments
    for c in res.data.get("clips", []):
        try:
            if ok(c):
                m = moments[int(c["clip"])]
                m.last, m.end = int(c["last"]), sents[int(c["last"])].end
        except (TypeError, ValueError):
            continue
    return moments


START_SCHEMA = {
    "type": "object",
    "properties": {"clips": {"type": "array", "items": {"type": "object", "properties": {
        "clip": {"type": "integer"}, "start": {"type": "integer"}, "vague": {"type": "boolean"},
        "title": {"type": "string"}, "why": {"type": "string"}}, "required": ["clip", "start", "vague", "title", "why"]}}},
    "required": ["clips"],
}

START_SYSTEM = """You check how short-form clips START and whether they are worth posting. Viewers decide in the first \
two seconds, and on average they leave after about 7 seconds of a 15-30 second clip, so the first sentence has to hook.
For each clip you see the sentences just BEFORE it and its first sentences, numbered. Return:
- start: the index of the sentence the clip should start on. It must be a COMPLETE, striking statement that makes sense with zero \
context: a bold claim, a surprising fact, a question the clip answers, a punchline set-up, a concrete number. Never a continuation \
("and...", "so...", "but...", "which...", "because..."), never a fragment cut off mid-sentence, never a sentence that leans on \
something said earlier ("it", "that", "this one" with nothing to point at), never small talk ("hey guys", "so today"). Pick the \
best such sentence among the listed ones (the clip's own first sentences or the ones before it). Keep the current first sentence if it \
already hooks.
- vague: true when the clip has no concrete claim, number, story or payoff - general chat, vlog filler, 'time flew by', 'it was so good', \
a feeling with nothing behind it - something nobody would watch to the end. Be strict: when in doubt say true.
- title: the on-screen title. If the current title promises something the clip does not actually say, or misdescribes it, write a \
truthful curiosity title (max 55 characters, no clickbait lies, no emojis, no profanity) from what is really said. If it is already \
true and good, return it unchanged. Also rewrite it when it is vague: a bare question ("How do I do that?"), a label ("Streamer reacts to \
unexpected moment") or a greeting. A good title states the claim, the stake or the surprise with the concrete name, number or thing in it.
- why: max 10 words."""


def llm_fix_starts(llm, moments: list[Moment], sents: list[Sentence], min_len: float, look_back: int = 4, look_ahead: int = 3) -> list[Moment]:
    """One AI pass: move each clip's start onto a hooking, self-contained sentence, drop clips with nothing to say, make titles truthful."""
    todo = [(k, m) for k, m in enumerate(moments) if m.first >= 0 and m.last >= 0]
    if not todo:
        return moments
    blocks = []
    for k, m in todo:
        before = [f"  [{s.idx}] (before the clip) {s.text}" for s in sents[max(0, m.first - look_back):m.first]]
        inside = [f"  [{s.idx}] {s.text}" for s in sents[m.first:min(m.last, m.first + look_ahead) + 1]]
        blocks.append(f"Clip {k}, title \"{m.title}\", currently starts at [{m.first}]:\n" + "\n".join(before + inside))
    prompt = "\n\n".join(blocks) + "\n\nFor every clip return its number, the start sentence index, whether it is vague, and the (truthful) title."
    try:
        res = llm.complete_json("clip_starts", START_SYSTEM, prompt, START_SCHEMA)
    except Exception:  # noqa: BLE001 - the heuristic start check still runs
        return moments
    drop: set[int] = set()
    for c in res.data.get("clips", []):
        try:
            k = int(c["clip"])
            if not (0 <= k < len(moments)):
                continue
            m = moments[k]
            if c.get("vague"):
                drop.add(k)
                continue
            st = int(c.get("start", m.first))
            if max(0, m.first - look_back) <= st <= min(m.last, m.first + look_ahead) and st != m.first:
                if sents[m.last].end - sents[st].start >= min_len * 0.6:
                    m.first, m.start = st, sents[st].start
            t = re.sub(r"\s+", " ", str(c.get("title") or "")).strip()[:70]
            if t and len(t) >= 8:
                m.title = t
        except (TypeError, ValueError, KeyError):
            continue
    kept = [m for k, m in enumerate(moments) if k not in drop]
    return kept


RETITLE_SCHEMA = {"type": "object", "properties": {"titles": {"type": "array", "items": {"type": "object", "properties": {
    "clip": {"type": "integer"}, "title": {"type": "string"}}, "required": ["clip", "title"]}}}, "required": ["titles"]}
RETITLE_SYSTEM = """You rewrite the on-screen titles of short clips that got a vague title. Each clip shows its current title (the reason it was refused) and what is actually said.
Write ONE title per clip: max 55 characters, plain words, TRUE to what is said (never invent), no emojis, no profanity, no all-caps.
It must state the specific claim, the stake or the surprise and carry the concrete name, number or thing from the clip ("Kevin Nealon's terrifying Johnny Carson moment",
"What happened when the control rods were pulled out at Chernobyl", "How The Office almost got cancelled"). Never a bare question ("How do I do that?"), never a label
("Streamer reacts to unexpected moment"), a greeting, or a teaser that hides the subject ("Taste this before you even touch it", "Your brain decides before you know it")."""


def retitle_weak(llm, moments: list[Moment], sents: list[Sentence], max_clips: int = 12) -> list[Moment]:
    """One more try for the clips whose title clipgate.weak_title refuses: ask once for a title that states the claim with a name or number. The new title is kept
    only if it passes the same check; otherwise the old one stays (demote_weak_titles then lowers the score). Never raises."""
    todo = [(k, m, why) for k, m in enumerate(moments) if (why := clipgate.weak_title(m.title)) and m.first >= 0 and m.last >= m.first][:max_clips]
    if not todo:
        return moments
    blocks = []
    for k, m, why in todo:
        said = " ".join(s.text for s in sents[m.first:m.last + 1])[:700]
        blocks.append(f"Clip {k}: current title \"{m.title}\" (refused: {why})\nWhat is said: {said}")
    try:
        res = llm.complete_json("clip_title", RETITLE_SYSTEM, "\n\n".join(blocks), RETITLE_SCHEMA)
    except Exception:  # noqa: BLE001 - the old title and its lower score stay
        return moments
    for c in res.data.get("titles", []) or []:
        try:
            k = int(c["clip"])
            t = re.sub(r"\s+", " ", str(c.get("title") or "")).strip()[:70]
        except (TypeError, ValueError, KeyError):
            continue
        if 0 <= k < len(moments) and k in {x[0] for x in todo} and len(t) >= 8 and clipgate.weak_title(t) is None:
            moments[k].title = t
    return moments


def demote_weak_titles(moments: list[Moment]) -> list[Moment]:
    """A title that is a bare question, a label or a greeting sells nothing: those posts got 250-300 views while claims got 2,800+. The AI is told, and what still
    slips through loses 2.5 points here (the score floor then rejects it) and is refused again by clipgate."""
    for m in moments:
        why = clipgate.weak_title(m.title)
        if why:
            m.score = max(1.0, m.score - 2.5)
            m.reason = (m.reason + "; " if m.reason else "") + f"weak title ({why})"
    return moments


def fix_starts_heuristic(moments: list[Moment], sents: list[Sentence], min_len: float) -> list[Moment]:
    """No-AI safety net: a clip that opens on a continuation ("And...", "So...", "which...") or a fragment (lower case, under 4 words)
    starts a sentence or two later, as long as it stays long enough."""
    for m in moments:
        for _ in range(3):
            if m.first >= m.last or m.first < 0:
                break
            s0 = sents[m.first].text.strip()
            bad = bool(CONTINUES.match(s0)) or len(s0.split()) < 4 or (s0[:1].islower() and not s0.lower().startswith(("i ", "i'")))
            if not bad or sents[m.last].end - sents[m.first + 1].start < min_len * 0.7:
                break
            m.first += 1
            m.start = sents[m.first].start
    return moments


def dedupe(moments: list[Moment], max_overlap: float = 0.25) -> list[Moment]:
    kept: list[Moment] = []
    for m in sorted(moments, key=lambda x: -x.score):
        clash = False
        for k in kept:
            inter = max(0.0, min(m.end, k.end) - max(m.start, k.start))
            if inter / max(1e-6, min(m.duration, k.duration)) > max_overlap:
                clash = True
                break
        if not clash:
            kept.append(m)
    return kept


def chunks(sents: list[Sentence], chunk_s: float = 480, overlap_s: float = 45) -> list[list[Sentence]]:
    out, i = [], 0
    while i < len(sents):
        start = sents[i].start
        block = [s for s in sents[i:] if s.start < start + chunk_s]
        out.append(block)
        if block[-1].idx == sents[-1].idx:
            break
        nxt = next((j for j, s in enumerate(sents) if s.start >= start + chunk_s - overlap_s), None)
        i = max(i + 1, nxt if nxt is not None else len(sents))
    return out


SCHEMA = {
    "type": "object",
    "properties": {"clips": {"type": "array", "items": {"type": "object", "properties": {
        "first": {"type": "integer"}, "last": {"type": "integer"},
        "title": {"type": "string"}, "hook": {"type": "string"},
        "caption": {"type": "string"}, "hashtags": {"type": "array", "items": {"type": "string"}},
        "score": {"type": "number"}, "reason": {"type": "string"},
        "mood": {"type": "string", "enum": ["chill", "upbeat", "funny", "suspense", "emotional", "epic", "adventure"]}},
        "required": ["first", "last", "title", "hook", "caption", "hashtags", "score", "reason", "mood"]}}},
    "required": ["clips"],
}

SYSTEM = """You are a top short-form video editor who finds the moments in long videos that go viral as \
TikTok / Reels / YouTube Shorts clips.
Prefer clips that tell something that HAPPENED (a short story with a setup, tension and a payoff: an incident, a confrontation, a close call, a confession) over tips, recipes, how-tos and general chat.
A great clip: opens with a strong hook in its first sentence (a bold claim, a question, conflict, a surprising \
fact, a punchline set-up), is understandable with zero context, has a payoff (story resolves, joke lands, \
insight delivered), and keeps energy high. Avoid intros, outros, sponsor reads, housekeeping and moments that \
depend on something shown on screen.
For each clip return:
- first, last: sentence index range [first..last] from the transcript (inclusive). Start ON the hook. End where
  the thought fully lands (the punchline / answer / conclusion) - never on a setup; clips usually fail by
  stopping a sentence or two early. 15-25 seconds is the sweet spot: viewers of these Shorts leave after about 7 seconds, and clips over \
35 seconds do clearly worse. Never pad a clip to reach a length.
- title: on-screen hook title, max 55 characters, no clickbait lies, no emojis, and clean:
  no profanity or slurs in titles or captions (YouTube limits reach and ads for them), even if the speaker swears.
  State the claim, the stake or the surprise, with the concrete name, number or thing in it ("Cop finds suspect but is missing a second officer",
  "Why the comedy industry doesn't want real people"). Never a bare question or a label: no "How do I do that?", "How you doing?",
  "What just happened?", "Streamer reacts to unexpected moment", "A crazy story", "You won't believe this".
- hook: why someone keeps watching after 2 seconds (one short sentence).
- caption: one short post caption (max 20 words) in the speaker's spirit, no hashtags.
- hashtags: 3-5 relevant hashtags without '#'.
- score: 1-10 for viral potential as a standalone short. Use the whole range and RANK: within one answer no
  two clips get the same score, an average moment is a 5, and 9-10 are reserved for genuinely exceptional ones.
- reason: max 10 words.
- mood: the background-music mood that fits the clip's vibe (chill, upbeat, funny, suspense, emotional,
  epic or adventure).
Keep every field short - fewer, better clips beat many weak ones."""


def _fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def llm_moments(llm, sents: list[Sentence], per_chunk: int, min_len: float, max_len: float,
                context: str = "", progress=None) -> list[Moment]:
    blocks = chunks(sents)
    out: list[Moment] = []
    raw = raw_takes(sents)
    if raw:                                                          # takes, not a talk: see RAW_NOTE
        context = (context + "\n" if context else "") + RAW_NOTE
        min_len = min(min_len, 7.0)                                  # (the shortest clip anyone posts: clipgate's floor)
    for bi, block in enumerate(blocks):
        listing = "\n".join(f"[{s.idx}] ({_fmt(s.start)}) {s.text}" for s in block)
        prompt = (f"{context}\nTranscript part {bi + 1}/{len(blocks)} (sentence index, timestamp, text):\n{listing}\n\n"
                  f"Propose up to {per_chunk} clips from THIS part, each {int(min_len)}-{int(max_len)} seconds long "
                  f"(use the timestamps). Only propose genuinely strong moments; fewer is fine.")
        lo, hi = block[0].idx, block[-1].idx

        def in_range(c, lo=lo, hi=hi):
            try:
                return lo <= int(c["first"]) <= int(c["last"]) <= hi
            except (KeyError, TypeError, ValueError):
                return False

        def check(d):
            cs = d.get("clips", [])
            if cs and not any(in_range(c) for c in cs):          # only when nothing in the answer is usable
                raise ValueError(f"every proposed sentence range is outside {lo}-{hi}")

        try:
            from . import glossary
            res = llm.complete_json("clip_moments", SYSTEM + glossary.prompt_note(glossary.terms()), prompt, SCHEMA, validate=check)
        except Exception:  # noqa: BLE001 - one bad chunk shouldn't sink the job; heuristic covers gaps
            if progress:
                progress(bi + 1, len(blocks))
            continue
        for c in res.data.get("clips", []):
            if not in_range(c):
                continue                                         # drop just the bad proposal
            m = Moment(0, 0, int(c["first"]), int(c["last"]), title=str(c.get("title", ""))[:70].strip(),
                       hook=str(c.get("hook", "")), caption=str(c.get("caption", "")).strip(),
                       hashtags=[h.lstrip("#").strip() for h in c.get("hashtags", []) if str(h).strip()][:6],
                       score=float(c.get("score", 5)), reason=str(c.get("reason", "")), source=res.provider,
                       mood=str(c.get("mood", "")))
            if raw:
                m = trim_crew(m, sents)
                if m is None:
                    continue
            m = fit_length(m, sents, min_len, max_len)
            if m and raw:
                m = trim_crew(m, sents)                                  # (growing to the length must not bring the crew back in)
            if m:
                out.append(m)
        if progress:
            progress(bi + 1, len(blocks))
    return out


def loudness_per_second(samples: np.ndarray, sr: int) -> np.ndarray:
    n = len(samples) // sr
    if n == 0:
        return np.zeros(0)
    rms = np.sqrt((samples[: n * sr].reshape(n, sr) ** 2).mean(axis=1) + 1e-12)
    med = np.median(rms)
    mad = np.median(np.abs(rms - med)) or 1e-6
    return (rms - med) / (1.4826 * mad)


def heuristic_moments(sents: list[Sentence], loud_z: np.ndarray | None, min_len: float, max_len: float,
                      target: float = 40.0) -> list[Moment]:
    out = []
    for i, s in enumerate(sents):
        if not (HOOKY.search(s.text) or s.text.endswith("?")):
            continue
        m = fit_length(Moment(0, 0, i, i), sents, max(min_len, min(target, max_len)), max_len)
        if not m:
            continue
        text = " ".join(x.text for x in sents[m.first:m.last + 1])
        score = 3.0 + min(3.0, len(HOOKY.findall(text)) * 0.6) + min(1.5, text.count("?") * 0.5) + \
            min(1.0, text.count("!") * 0.5)
        if loud_z is not None and len(loud_z):
            seg = loud_z[int(m.start):max(int(m.start) + 1, int(m.end))]
            score += float(np.clip(np.mean(seg), -1, 2)) if len(seg) else 0
        first = s.text if len(s.text) <= 55 else s.text[:52].rsplit(" ", 1)[0] + "..."
        m.title, m.score, m.source, m.reason = first, round(score, 2), "heuristic", "hook words / questions / energy"
        m.caption = s.text
        out.append(m)
    return out


def pick(moments: list[Moment], n: int) -> list[Moment]:
    best = dedupe(moments)[:n]
    return sorted(best, key=lambda m: m.start)


def top_up(picked: list[Moment], extra: list[Moment], n: int, max_overlap: float = 0.25) -> list[Moment]:
    """Keep every LLM pick; fill the remaining slots with the best non-overlapping heuristic moments."""
    chosen = list(picked)
    for e in sorted(extra, key=lambda x: -x.score):
        if len(chosen) >= n:
            break
        if all(max(0.0, min(e.end, k.end) - max(e.start, k.start)) / max(1e-6, min(e.duration, k.duration))
               <= max_overlap for k in chosen):
            chosen.append(e)
    return sorted(chosen, key=lambda m: m.start)
