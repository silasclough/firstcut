"""Keywords for footage that has only been LOOKED AT so far: the three words that best describe a file's pictures.

The moment a file is catalogued (its frames are embedded by the picture model) it is searchable, but nothing has described it yet (that
takes a vision model, later). So the page used to say "searchable now". This gives it words at once, from the pictures alone:

    * a fixed vocabulary of about 400 plain B-roll words (people, things they do, places, objects, light and camera);
    * each word embedded ONCE with the picture model's text side (the matrix is kept next to the catalogue: tagwords.npy + tagwords.json
      with a key that changes when the model or the vocabulary does), so tagging a file is one small matrix product on embeddings that
      are already on disk: no video is read again;
    * the top three words for a file, by the model's own calibrated sigmoid (the same one Catalog.chance() uses), with near-synonyms
      folded together, weak matches dropped, and, once the library is big enough, each word's library-wide average taken off so the
      words that are specific to this file beat the ones that fit everything ("outdoors");
    * when a description arrives later (a vision model, or the person's AI account), words from it are merged in by rarity, so the three
      keywords sharpen over time.

Without the picture model there are no picture words: the file's folder and file name give a few instead.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from pathlib import Path

from ..core.log import get_logger

log = get_logger("clipper.tagwords")

VERSION = 1
PROMPT = "a photo of {}"                  # (how SigLIP likes to be asked)
TOP = 3
MIN_CHANCE = 0.015                        # the model's own "does this show it" (SigLIP 2: ~0.1 is a clear match); below this a word is not offered
REL_CHANCE = 0.2                          # ...and not under this fraction of the file's best word
SYNONYM = 0.86                            # two words whose text vectors are this alike count as one idea
IDF_AFTER = 40                            # files tagged before the library-wide average is trusted
FLUSH_EVERY = 25                          # statistics written to disk every this many updates

# ----------------------------------------------------------------------------------------------------------- the vocabulary
PEOPLE = [
    "students", "a student", "friends laughing", "friends", "a crowd", "a group of people", "a small group", "two people talking", "a couple",
    "a family", "a child", "children", "kids playing", "teenagers", "young adults", "a teacher", "a professor", "a speaker", "a pastor",
    "a counselor", "a coach", "a team", "an athlete", "a musician", "a band", "a choir", "a performer", "a dancer", "a graduate",
    "graduates", "an audience", "a man", "a woman", "an older man", "an older woman", "a volunteer", "volunteers", "staff", "a worker",
    "a nurse", "a doctor", "a chef", "a leader", "a mentor", "a person alone", "a person smiling", "a person laughing", "a hug",
    "a handshake", "a high five", "a portrait", "faces", "hands", "feet", "a silhouette", "a shadow",
]
ACTIONS = [
    "walking", "running", "jumping", "dancing", "singing", "playing music", "playing guitar", "playing piano", "playing drums", "talking",
    "listening", "studying", "reading", "writing", "typing", "working on a laptop", "taking notes", "raising a hand", "clapping",
    "cheering", "celebrating", "laughing", "smiling", "hugging", "praying", "worshiping", "eating", "drinking coffee", "cooking",
    "serving food", "playing a game", "playing sports", "playing soccer", "playing basketball", "playing volleyball", "swimming",
    "climbing", "hiking", "biking", "paddling", "fishing", "a campfire", "roasting marshmallows", "building something", "painting",
    "drawing", "crafting", "performing on stage", "giving a speech", "presenting", "interviewing", "taking a photo", "filming",
    "waving", "pointing", "high fiving", "playing with a dog", "carrying boxes", "setting up chairs", "cleaning", "gardening",
    "driving", "riding", "waiting in line", "opening a door", "entering a building", "leaving", "arriving", "crossing a street",
    "sitting at a table", "standing in a circle", "gathering", "marching", "a parade", "a ceremony", "a graduation", "a wedding",
    "a worship service", "a game night", "a dinner", "a party", "a field trip", "a workshop", "a lecture", "a rehearsal", "a concert",
    "a tour", "move-in day", "an open house", "orientation",
]
PLACES = [
    "a classroom", "a lecture hall", "a library", "a campus", "a quad", "a courtyard", "a hallway", "a stairwell", "a cafeteria",
    "a dining hall", "a dorm room", "a dormitory", "a gym", "a basketball court", "a soccer field", "a sports field", "a stadium",
    "a track", "a pool", "a chapel", "a church", "a sanctuary", "an auditorium", "a stage", "a theater", "a studio", "a lab",
    "a science lab", "a nursing lab", "a computer lab", "an art room", "a music room", "a workshop", "a kitchen", "a bookstore",
    "an office", "a reception desk", "a conference room", "a lobby", "an entrance", "a front door", "a parking lot", "a parking garage",
    "a sidewalk", "a street", "a city street", "a crosswalk", "a bridge", "a park", "a playground", "a garden", "a lawn", "a forest",
    "a trail", "a lake", "a river", "a beach", "a field", "a barn", "a cabin", "a campground", "a tent", "a dock", "a mountain",
    "a hill", "a farm", "a town", "a neighborhood", "a house", "a living room", "a backyard", "a porch", "a rooftop", "a courthouse",
    "a hospital", "a classroom door", "an outdoor event", "an indoor event", "an exterior building", "a brick building", "a glass building",
    "a campus sign", "a statue", "a fountain", "a flag", "an empty room", "an empty hallway", "a crowded room",
]
THINGS = [
    "books", "a laptop", "a phone", "a camera", "a microphone", "a guitar", "a piano", "drums", "a whiteboard", "a chalkboard", "a screen",
    "a projector", "a notebook", "a backpack", "a desk", "a chair", "a table", "food", "pizza", "coffee", "a cake", "balloons", "banners",
    "signs", "posters", "a trophy", "a medal", "a diploma", "a cap and gown", "a uniform", "a jersey", "a ball", "a bicycle", "a bus",
    "a van", "a car", "a boat", "a canoe", "a kayak", "a tent", "a campfire", "a sleeping bag", "a flashlight", "a map", "a flag",
    "a banner", "a logo", "a sign with text", "a screen with text", "a name tag", "a bible", "a cross", "a candle", "flowers", "a tree",
    "leaves", "grass", "snow", "rain", "fog", "water", "a window", "a door", "stairs", "a railing", "a fence", "a wall", "a mural",
    "a stained glass window", "a pet", "a dog", "a cat", "a horse", "a bird", "a squirrel", "a tool", "a medical device", "a microscope",
    "a lab coat", "a stethoscope", "a mannequin", "a stack of papers", "a laptop screen", "a clock", "a calendar", "a gift", "a drink",
]
LOOK = [
    "sunset", "sunrise", "golden hour", "daytime", "night", "dusk", "an overcast day", "a sunny day", "indoors", "outdoors", "dim light",
    "bright light", "backlit", "silhouette against the sky", "stage lighting", "neon light", "candlelight", "string lights", "shallow depth of field",
    "a close-up", "an extreme close-up", "a wide shot", "an establishing shot", "a medium shot", "a low angle", "a high angle", "an aerial view",
    "a drone shot", "a handheld shot", "a tripod shot", "a gimbal shot", "a slow pan", "a tilt", "a push in", "a pull out", "a tracking shot",
    "a time lapse", "slow motion", "a blurry shot", "a shaky shot", "a still photo", "a black and white photo", "a screenshot",
    "a graphic", "text on screen", "a title card", "an empty frame", "a dark frame", "a bright frame", "a symmetrical frame",
    "a busy scene", "a quiet scene", "a calm scene", "an energetic scene", "a joyful moment", "a serious moment", "an emotional moment",
    "a candid moment", "a posed group photo", "a selfie", "a reaction shot", "a cutaway", "b-roll", "an interview setup", "a behind the scenes",
]
EVENTS = [
    "youth camp", "summer camp", "a retreat", "a conference", "graduation", "commencement", "a parade", "a festival", "a fair", "a fundraiser",
    "a ribbon cutting", "a groundbreaking", "a career fair", "a campus tour", "move-in", "homecoming", "a pep rally", "a chapel service",
    "a baptism", "a mission trip", "a service project", "a talent show", "a banquet", "a holiday event", "a christmas event", "a thanksgiving meal",
    "a sports game", "a tournament", "a race", "a practice", "a club meeting", "a student panel", "a campus visit", "an admissions event",
    "a recruiting event", "a webinar", "a podcast recording", "a press event", "a community event",
]


def vocabulary() -> list[str]:
    """The words, in a fixed order, each once, without the leading article ('a classroom' -> 'classroom' is what is shown and searched)."""
    seen, out = set(), []
    for group in (PEOPLE, ACTIONS, PLACES, THINGS, LOOK, EVENTS):
        for w in group:
            if w not in seen:
                seen.add(w)
                out.append(w)
    return out


def shown(word: str) -> str:
    """The word as the page shows and searches it: no 'a '/'an ' in front."""
    return re.sub(r"^(a|an)\s+", "", word.strip())


STOP = set("""a an the and or of in on at to for with without from by as is are was were be been being it its this that these those there here
into over under near next beside behind between across through while during very more most some any each both other another same such
person people man men woman women one two three four five six seven eight nine ten group wearing wears wear standing stands stand sitting
sits sit holding holds hold looking looks look showing shows show appears appear seems seem looks image picture photo photos video clip shot
scene scenes frame frames background foreground left right center middle top bottom side front back large small big little light dark
white black blue green red yellow brown gray grey orange pink purple color colored
""".split())
# file-name words that say nothing: camera prefixes, extensions, folder names everybody has
PATH_STOP = set("""img dsc dscn mvi mov mp4 m4v avi mxf mts jpg jpeg png heic cr3 arw raw clip clips video videos photo photos footage broll roll
video b-roll a-roll archive final export exports output camera card dcim private misc other new untitled folder copy test tests volumes
users projects project media drive backup edit edits proxy""".split())


# ------------------------------------------------------------------------------------------------------------- the tagger
def path_words(rel: str, k: int = TOP) -> list[str]:
    """The fallback with no picture model: words from the folders (nearest first) and the file name that actually say something."""
    parts = [p for p in str(rel or "").replace("\\", "/").split("/") if p]
    names = []
    for part in reversed(parts):
        stem = re.sub(r"\.[A-Za-z0-9]{2,4}$", "", part)
        names.append(stem)
    out: list[str] = []
    for nm in names[:1] + names[1:3]:                                    # the file name first (when it says anything), then the two folders above it
        for w in re.findall(r"[A-Za-z]+", nm):
            lw = w.lower()
            if len(lw) < 4 or lw in PATH_STOP or lw in STOP or lw in out or not re.search(r"[aeiouy]", lw):    # (no vowels: a camera's folder code)
                continue
            out.append(lw)
            if len(out) >= k:
                return out
    return out


def _pairs_ok(word: str, picked: list[str]) -> bool:
    """No two keywords that are the same idea in words ('walking' and 'people walking')."""
    w = shown(word)
    for p in picked:
        q = shown(p)
        if w == q or w in q or q in w:
            return False
    return True


class TagWords:
    """The vocabulary matrix for one picture model, and the picking. `embedder` is only used to embed the words (once) and for chance();
    it is never created here (so the model is never loaded or downloaded by tagging)."""

    def __init__(self, folder: Path, embedder, model_name: str = ""):
        self.dir = Path(folder)
        self.emb = embedder
        self.model_name = str(model_name or getattr(embedder, "name", "") or "")
        self.words = vocabulary()
        self._M = None
        self._lock = threading.RLock()                 # (flush() runs inside pick()/note_description()'s lock)
        self._stats = None                  # {"n": files tagged, "sum": [per-word sum of chance]}
        self._df = None                     # {"docs": descriptions seen, "df": {word: how many described files had it}}
        self._since = 0

    # -- the matrix --------------------------------------------------------------------------------------------------
    def key(self) -> str:
        h = hashlib.sha1(("|".join(self.words) + "|" + PROMPT).encode()).hexdigest()[:12]
        return f"{VERSION}:{self.model_name}:{h}"

    def matrix(self):
        """(words x dims) float32, from disk when its key matches, else embedded now (once) and kept. None when the words cannot be embedded."""
        import numpy as np
        with self._lock:
            if self._M is not None:
                return self._M
            npy, js = self.dir / "tagwords.npy", self.dir / "tagwords.json"
            try:
                if js.exists() and json.loads(js.read_text()).get("key") == self.key() and npy.exists():
                    M = np.load(npy)
                    if M.ndim == 2 and len(M) == len(self.words):
                        self._M = M.astype("float32")
                        return self._M
            except (OSError, ValueError):
                pass
            if self.emb is None:
                return None
            try:
                parts = [self.emb.text([PROMPT.format(shown(w)) for w in self.words[k:k + 64]]) for k in range(0, len(self.words), 64)]
                M = np.concatenate(parts).astype("float32")
            except Exception as e:  # noqa: BLE001 - no words, no keywords: the folder names are the fallback
                log.info(f"tag words not embedded: {e}")
                return None
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
                np.save(npy, M)
                js.write_text(json.dumps({"key": self.key(), "words": len(self.words)}))
            except OSError as e:
                log.info(f"tag words not kept: {e}")
            self._M = M
            return M

    # -- library-wide statistics (the cheap IDF) ---------------------------------------------------------------------
    def _load_stats(self):
        if self._stats is None:
            try:
                d = json.loads((self.dir / "tagstats.json").read_text())
                if d.get("key") == self.key():
                    self._stats = {"n": int(d["n"]), "sum": [float(x) for x in d["sum"]]}
                    self._df = {"docs": int(d.get("docs") or 0), "df": {str(k): int(v) for k, v in (d.get("df") or {}).items()}}
            except (OSError, ValueError, KeyError, TypeError):
                pass
            if self._stats is None:
                self._stats = {"n": 0, "sum": [0.0] * len(self.words)}
            if self._df is None:
                self._df = {"docs": 0, "df": {}}
        return self._stats

    def flush(self) -> None:
        with self._lock:
            s = self._load_stats()
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
                (self.dir / "tagstats.json").write_text(json.dumps({"key": self.key(), "n": s["n"], "sum": [round(x, 5) for x in s["sum"]],
                                                                    "docs": self._df["docs"], "df": self._df["df"]}))
                self._since = 0
            except OSError as e:
                log.info(f"tag statistics not kept: {e}")

    def _bump(self) -> None:
        self._since += 1
        if self._since >= FLUSH_EVERY:
            self.flush()

    # -- picking -----------------------------------------------------------------------------------------------------
    def chances(self, frames) -> "np.ndarray | None":  # noqa: F821
        """Per word, how likely the file's pictures show it: the best frame (60%) and the average frame (40%), through the model's sigmoid."""
        import numpy as np
        M = self.matrix()
        if M is None or frames is None or not len(frames):
            return None
        S = np.asarray(frames, dtype="float32") @ M.T                      # frames x words, cosine
        score = 0.6 * S.max(axis=0) + 0.4 * S.mean(axis=0)
        ch = getattr(self.emb, "chance", None)
        return np.array([float(ch(float(x))) if ch else float(x) for x in score], dtype="float32")

    def pick(self, frames, k: int = TOP, count: bool = True) -> list[str]:
        """The top words for a file's frame embeddings ([] when there is no model to say)."""
        import numpy as np
        ch = self.chances(frames)
        if ch is None:
            return []
        M = self._M
        with self._lock:
            st = self._load_stats()
            use_idf = st["n"] >= IDF_AFTER
            mean = np.array(st["sum"], dtype="float32") / max(1, st["n"])
            if count:
                st["n"] += 1
                st["sum"] = [a + float(b) for a, b in zip(st["sum"], ch)]
                self._bump()
        best = float(ch.max())
        floor = max(MIN_CHANCE, REL_CHANCE * best)
        adj = ch - mean if use_idf else ch
        order = np.argsort(-adj)
        picked_idx: list[int] = []
        for i in order:
            i = int(i)
            if ch[i] < floor or (use_idf and adj[i] <= 0):
                continue
            w = self.words[i]
            if not _pairs_ok(w, [self.words[j] for j in picked_idx]):
                continue
            if any(float(M[i] @ M[j]) >= SYNONYM for j in picked_idx):
                continue
            picked_idx.append(i)
            if len(picked_idx) >= k:
                break
        return [shown(self.words[i]) for i in picked_idx]

    # -- the description joins in ------------------------------------------------------------------------------------
    def note_description(self, text: str) -> None:
        """Count which words a description used (once per file), for the rarity below."""
        ws = set(self.desc_words(text))
        with self._lock:
            self._load_stats()
            self._df["docs"] += 1
            for w in ws:
                self._df["df"][w] = self._df["df"].get(w, 0) + 1
            self._bump()

    @staticmethod
    def desc_words(text: str) -> list[str]:
        out = []
        for w in re.findall(r"[a-z]+", str(text or "").lower()):
            if len(w) >= 4 and w not in STOP and w not in out:
                out.append(w.rstrip("s") if w.endswith("s") and len(w) > 5 else w)
        return out

    def merge(self, picture: list[str], descriptions: list[str], k: int = TOP) -> list[str]:
        """The top k of: the picture words (ranked: 0.9, 0.8, 0.7) and the words of the description, by rarity in this library's descriptions
        (a word that most descriptions use scores low; one that few do scores high enough to take the place of the weakest picture word)."""
        with self._lock:
            self._load_stats()
            docs, df = self._df["docs"], self._df["df"]
        cands: dict[str, float] = {}
        stem = lambda x: x[:-1] if x.endswith("s") and len(x) > 3 else x
        by_stem: dict[str, str] = {}
        for r, w in enumerate(picture or []):
            cands[w] = 0.9 - 0.1 * r
            by_stem[stem(shown(w))] = w
        text = " ".join(str(x) for x in descriptions or [])
        for pos, w in enumerate(self.desc_words(text)):
            if stem(w) in by_stem:
                cands[by_stem[stem(w)]] += 0.15                             # the pictures and the description agree
                continue
            rarity = math.log((docs + 2) / (df.get(w, 0) + 1)) if docs >= 20 else 0.0
            cands[w] = 0.55 + 0.4 * (rarity / (rarity + 2)) - 0.0005 * pos
        out: list[str] = []
        for w, _ in sorted(cands.items(), key=lambda x: -x[1]):
            if _pairs_ok(w, out):
                out.append(w)
            if len(out) >= k:
                break
        return out
