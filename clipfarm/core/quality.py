"""Quality gates for generated scripts and rendered videos.

These are deliberately simple, explainable heuristics (plus the prompt itself telling the
model the rules). They catch the common failure modes; they are not a guarantee.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ENGLISH_STOPWORDS = set("""a about above after again against all am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further had has have having he her
here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on
once only or other our ours out over own same she should so some such than that the their theirs them then there these
they this those through to too under until up very was we were what when where which while who whom why will with would
you your yours yourself""".split())

# Common English first names. "<FirstName> <Capitalized>" not declared as a fictional
# character is treated as a possible real person's full name.
FIRST_NAMES = set("""James John Robert Michael William David Richard Joseph Thomas Charles Christopher Daniel Matthew
Anthony Mark Donald Steven Paul Andrew Joshua Kenneth Kevin Brian George Timothy Ronald Edward Jason Jeffrey Ryan Jacob
Gary Nicholas Eric Jonathan Stephen Larry Justin Scott Brandon Benjamin Samuel Gregory Alexander Frank Patrick Raymond
Jack Dennis Jerry Tyler Aaron Jose Adam Nathan Henry Douglas Zachary Peter Kyle Ethan Walter Noah Jeremy Christian Keith
Roger Terry Gerald Harold Sean Austin Carl Arthur Lawrence Dylan Jesse Jordan Bryan Billy Joe Bruce Gabriel Logan Albert
Willie Alan Juan Wayne Elijah Randy Roy Vincent Ralph Eugene Russell Bobby Mason Philip Louis Mary Patricia Jennifer Linda
Elizabeth Barbara Susan Jessica Sarah Karen Lisa Nancy Betty Margaret Sandra Ashley Kimberly Emily Donna Michelle Carol
Amanda Dorothy Melissa Deborah Stephanie Rebecca Sharon Laura Cynthia Kathleen Amy Angela Shirley Anna Brenda Pamela Emma
Nicole Helen Samantha Katherine Christine Debra Rachel Carolyn Janet Catherine Maria Heather Diane Ruth Julie Olivia Joyce
Virginia Victoria Kelly Lauren Christina Joan Evelyn Jolene Megan Andrea Cheryl Hannah Jacqueline Martha Gloria Teresa Ann
Sara Madison Frances Kathryn Janice Jean Abigail Alice Judy Sophia Grace Denise Amber Doris Marilyn Danielle Beverly Isabella
Theresa Diana Natalie Brittany Charlotte Marie Kayla Alexis Lori Elon Taylor Kim Kanye Jeff Bill Tim Tom Chris Mike Dave
Steve Bob Jim Tony Ben Sam Max Leo Liam Lucas Oliver Ava Mia Harper Ella Chloe""".split())

# Minecraft vocabulary and stock characters that are fine to name.
MINECRAFT_OK = set("""Steve Alex Herobrine Notch Enderman Endermen Ender Dragon Creeper Creepers Wither Warden Nether
End Overworld Villager Villagers Piglin Piglins Zombie Skeleton Spider Ghast Blaze Elytra Netherite Diamond Diamonds
Redstone Obsidian Beacon Totem Shulker Allay Sniffer Iron Golem Golem Pillager Illager Evoker Vindicator Ravager Strider
Hoglin Axolotl Bedrock Spawn Stronghold Mansion Woodland Ancient City Deep Dark Trial Chamber Breeze Bogged Creaking
Crafting Table Ender Pearl Pearl Eye Book Steve's Alex's""".split())
MINECRAFT_OK.discard("Notch")  # real person

LYRIC_MARKERS = [
    "lyrics", "the song goes", "sang the words", "singing the words", "as the song says", "the chorus",
    "famous quote", "once said", "as the saying goes", "the lyric", "verse two", "second verse",
]
FAMOUS_QUOTES = [
    "to be or not to be", "may the force be with you", "i'll be back", "here's looking at you",
    "houston, we have a problem", "life is like a box of chocolates", "you can't handle the truth",
    "i am your father", "winter is coming", "with great power comes great responsibility",
    "elementary, my dear watson", "i see dead people", "why so serious", "phone home",
    "ask not what your country", "one small step for man", "frankly, my dear", "you shall not pass",
    "i'm the king of the world", "show me the money", "there's no place like home",
]


@dataclass
class GateResult:
    ok: bool
    issues: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text)


def word_count(text: str) -> int:
    return len(words(text))


def first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])[\"')\]]*\s+", text.strip(), maxsplit=1)[0].strip()


def check_length(text: str, min_words: int, max_words: int) -> list[str]:
    n = word_count(text)
    target = f"aim for {min_words + 25}-{max_words - 15} words"
    if n < min_words:
        return [f"too short: {n} words (< {min_words}); {target}"]
    if n > max_words:
        return [f"too long: {n} words (> {max_words}); {target}"]
    return []


def detect_language(text: str) -> tuple[str, float]:
    try:
        from langdetect import DetectorFactory, detect_langs

        DetectorFactory.seed = 0
        best = detect_langs(text)[0]
        return best.lang, float(best.prob)
    except Exception:  # noqa: BLE001 - fall back to a stopword heuristic
        toks = [w.lower() for w in words(text)]
        if not toks:
            return "unknown", 0.0
        ratio = sum(t in ENGLISH_STOPWORDS for t in toks) / len(toks)
        ascii_ratio = sum(c.isascii() for c in text) / max(1, len(text))
        return ("en", min(1.0, ratio * 2.5)) if ratio > 0.2 and ascii_ratio > 0.97 else ("unknown", ratio)


def check_language(text: str, lang: str = "en", min_prob: float = 0.85) -> list[str]:
    got, prob = detect_language(text)
    if got != lang or prob < min_prob:
        return [f"language check failed: detected {got} ({prob:.2f}), want {lang}"]
    return []


def load_banned_names(path: Path | None) -> list[str]:
    if not path or not Path(path).exists():
        return []
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def check_named_people(text: str, allowed: list[str] | None = None, banned: list[str] | None = None) -> list[str]:
    issues = []
    allowed_tokens = {t for name in (allowed or []) for t in name.split()}
    for name in banned or []:
        if re.search(rf"(?<![\w@]){re.escape(name)}(?![\w])", text):
            issues.append(f"mentions a real person: '{name}'")
    for m in re.finditer(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+(?:-[A-Z][a-z]+)?)\b", text):
        first, last = m.group(1), m.group(2)
        if first in FIRST_NAMES and first not in allowed_tokens and last not in MINECRAFT_OK \
                and f"{first} {last}" not in (allowed or []):
            issues.append(f"possible real full name '{first} {last}' (not declared as a fictional character)")
    for m in re.finditer(r"\b(Mr|Mrs|Ms|Dr|President|Senator|King|Queen|Prince|Princess)\.?\s+([A-Z][a-z]+)", text):
        if m.group(2) not in allowed_tokens:
            issues.append(f"titled name '{m.group(0)}' not declared as a fictional character")
    return sorted(set(issues))


def check_quotes_and_lyrics(text: str) -> list[str]:
    issues = []
    low = text.lower()
    for marker in LYRIC_MARKERS:
        if marker in low:
            issues.append(f"lyric/quote marker '{marker}'")
    for q in FAMOUS_QUOTES:
        if q in low:
            issues.append(f"famous quote '{q}'")
    for m in re.finditer(r"[\"“]([^\"”]{1,600})[\"”]", text):
        if word_count(m.group(1)) > 25:
            issues.append(f"long quoted passage ({word_count(m.group(1))} words)")
    toks = [w.lower() for w in words(text)]
    grams = Counter(tuple(toks[i:i + 4]) for i in range(len(toks) - 3))
    rep = [g for g, c in grams.items() if c >= 3]
    if rep:
        issues.append(f"repetitive chorus-like phrase: '{' '.join(rep[0])}'")
    return issues


def check_story(story: dict, qcfg: dict, banned: list[str], check_people: bool = True) -> GateResult:
    """check_people=False for grounded true stories, which name documented public figures."""
    text = story.get("narration", "")
    issues = []
    issues += check_length(text, int(qcfg.get("min_words", 85)), int(qcfg.get("max_words", 165)))
    issues += check_language(text, qcfg.get("language", "en"))
    if check_people:
        issues += check_named_people(text + " " + story.get("caption", ""), story.get("characters", []), banned)
    issues += check_quotes_and_lyrics(text)
    hook = story.get("hook", "")
    max_hook = int(qcfg.get("max_hook_words", 14))
    if not hook or word_count(hook) > max_hook:
        issues.append(f"the first sentence (the hook) has {word_count(hook)} words; it must be {max_hook - 2} words "
                      "or fewer so the hook lands in the first ~2 seconds")
    elif not text.strip().lower().startswith(hook.strip().lower()[: max(10, len(hook) // 2)]):
        issues.append("narration must open with the hook")
    return GateResult(not issues, issues)


def check_duration(seconds: float, min_s: float, max_s: float) -> list[str]:
    if seconds < min_s:
        return [f"audio too short: {seconds:.1f}s (< {min_s}s)"]
    if seconds > max_s:
        return [f"audio too long: {seconds:.1f}s (> {max_s}s)"]
    return []
