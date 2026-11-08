"""Who is talking? In an interview the editor must use the person being interviewed, not the person asking.

The AI reads the transcript and marks which sentences are the INTERVIEWER'S (questions, prompts, "mm-hmm", setting up the next topic).
Those words are then removed from what the editor works with, so nothing downstream - the story, the best moments, the captions - can
pick them. (The audio of a removed stretch is simply treated as silence and trimmed.)
"""

from __future__ import annotations

import re

from ..core.log import get_logger
from . import moments as M

log = get_logger("clipper.speakers")
CHUNK = 140
SYSTEM = (
    "You are reading the transcript of a recorded interview (there may be more than one guest, or only one speaker). The text has no "
    "speaker labels. Mark which numbered sentences were spoken by the INTERVIEWER / HOST: questions, prompts, 'tell me about...', "
    "reactions like 'mm-hmm' and 'right', setting up the next topic, and anything said to guide the conversation. The GUEST is the "
    "person answering and telling their own story, opinions and expertise. A sentence ending in '?' is usually the interviewer, but a "
    "guest can ask a rhetorical question inside an answer - judge by who is leading the conversation. Look at the whole flow: "
    "a question is almost always followed by an answer. Short prompts, reactions and requests from the interviewer count as the interviewer too ('say that again', 'what do you mean', 'right', 'mm-hm', laughing along).  Return the numbers of the interviewer's sentences only. If there is just one "
    "speaker (a monologue, no interviewer), return an empty list.")
SCHEMA = {"type": "object", "properties": {"host": {"type": "array", "items": {"type": "integer"}}, "has_interviewer": {"type": "boolean"}},
          "required": ["host", "has_interviewer"]}


def host_spans(ai, words: list[dict], progress=None) -> list[tuple[float, float]]:
    """-> [(start, end)] seconds of the interviewer's sentences."""
    return analyse(ai, words, progress)[0]


def analyse(ai, words: list[dict], progress=None) -> tuple[list[tuple[float, float]], bool]:
    """-> ([(start, end)] seconds of the interviewer's sentences, is there an interviewer at all). The second answer is the AI's own
    has_interviewer (any part of the transcript); when it never said, it is whether any sentence was marked."""
    flags: list[bool] = []
    sents = M.sentences_from_words(words, pause_s=0.5, max_words=22)      # short thoughts, so a question and its answer land in different ones
    spans: list[tuple[float, float]] = []
    by = {s.idx: s for s in sents}
    for n, i in enumerate(range(0, len(sents), CHUNK)):
        chunk = sents[i:i + CHUNK]
        if progress:
            progress(n * CHUNK / max(1, len(sents)))
        prev = sents[max(0, i - 6):i]
        lines = [f"  [{s.idx}] (context, already labelled) {s.text}" for s in prev] + [f"  [{s.idx}] {s.text}" for s in chunk]
        try:
            res = ai.complete_json("speaker_roles", SYSTEM, "Transcript:\n" + "\n".join(lines) +
                                   f"\n\nReturn the interviewer's sentences among [{chunk[0].idx}]..[{chunk[-1].idx}].", SCHEMA)
        except Exception as e:  # noqa: BLE001 - unlabelled is better than failing
            log.warning(f"labelling speakers failed for part {n + 1}: {e}")
            continue
        lo, hi = chunk[0].idx, chunk[-1].idx
        if "has_interviewer" in res.data:
            flags.append(bool(res.data.get("has_interviewer")))
        for k in res.data.get("host") or []:
            try:
                k = int(k)
            except (TypeError, ValueError):
                continue
            if lo <= k <= hi and k in by:
                spans.append((by[k].start, by[k].end))
    merged = merge(spans)
    return merged, (any(flags) if flags else bool(merged))


def merge(spans: list[tuple[float, float]], gap: float = 0.3) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def drop_spans(words: list[dict], spans: list[tuple[float, float]]) -> list[dict]:
    """The words that are NOT inside the interviewer's spans (a word counts as inside if its middle is)."""
    if not spans:
        return words
    out = []
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        if not any(a - 0.02 <= mid <= b + 0.02 for a, b in spans):
            out.append(w)
    return out


CHECK_SYSTEM = (
    "You are checking an edit of an interview. Below are sentences the editor chose to KEEP (marked >>) with the unchosen sentences around "
    "them for context. Some chosen sentences are really the INTERVIEWER speaking (a question, a prompt, a reaction such as 'right' or 'say "
    "that again', a follow-up) or the start of the interviewer's next question that was swept in at the edge. Return the numbers of the "
    "chosen sentences that are the interviewer's, or that begin a question to the guest. Only the guest's own answers may stay.")


def check_selected(ai, sents, selected: set[int], coach: bool = True) -> set[int]:
    """-> the sentence numbers among `selected` that are actually the interviewer (to be dropped). coach=False (no interviewer in this recording, or the
    style keeps them): the COACH words ("tell me about...") are not a reason, they are the speaker's own sentence."""
    by = {s.idx: s for s in sents}
    order = sorted(i for i in selected if i in by)
    bad: set[int] = set()
    for k in range(0, len(order), 60):
        chunk = order[k:k + 60]
        want = set(chunk)
        show: list[int] = []
        for i in chunk:
            for j in range(i - 3, i + 2):
                if j in by and j not in show:
                    show.append(j)
        show.sort()
        lines = [f"{'>>' if j in want else '  '} [{j}] {by[j].text}" for j in show]
        try:
            res = ai.complete_json("speaker_check", CHECK_SYSTEM, "Sentences:\n" + "\n".join(lines), SCHEMA_CHECK)
            bad |= {int(x) for x in res.data.get("interviewer") or [] if int(x) in want}
        except Exception as e:  # noqa: BLE001 - unchecked is better than failing
            log.warning(f"checking the chosen sentences failed: {e}")
    if coach:
        bad |= {i for i in order if COACH.search(by[i].text)}                # the interviewer coaching the guest, whatever the AI said
    return bad


# A prompt from the interviewer OPENS with one of these (after an "okay" / "so"). The same words inside a guest's answer ("my doctor said,
# tell me what's going on") are the guest telling their story and stay.
COACH = re.compile(r"^\W*(?:(?:so|okay|ok|and|now|alright|all right|um|uh|yeah|great|perfect)\b[\s,.!]*)*"
                   r"(you can say|go ahead|can you (tell|say|explain|share)|could you|tell me|where were you|what was happening in your|how did you)\b",
                   re.I)
SCHEMA_CHECK = {"type": "object", "properties": {"interviewer": {"type": "array", "items": {"type": "integer"}}}, "required": ["interviewer"]}
