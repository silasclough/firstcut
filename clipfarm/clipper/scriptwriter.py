"""Scripts that sound like a YouTube video, not a lecture: plan the story, write it spoken, have a retention editor tear it apart, rewrite until it holds.

    write(ai, topic, srcs, minutes, menu, package=None, say=print) -> {"title", "beats": [{"section", "narration", "visual"}], "plan", "review"}

The package is the idea the owner approved ({"title", "thumb_text", "hook", "why"}): the video is the PAYOFF of that title and thumbnail, so the plan has to promise the same
thing and deliver it early. Facts still come only from the sources (the fact check in explainer.py runs after this); analogies and imagined scenes are allowed when they are
plainly framed as "imagine ...", never as a claim about the real world.
"""

from __future__ import annotations

import json
import re

from ..core.log import get_logger

log = get_logger("clipper.scriptwriter")

PLAN_SCHEMA = {"type": "object", "properties": {
    "viewer_question": {"type": "string"}, "hook_line": {"type": "string"}, "payoff": {"type": "string"}, "running_example": {"type": "string"},
    "chapters": {"type": "array", "items": {"type": "object", "properties": {
        "question": {"type": "string"}, "answer": {"type": "string"}, "scene": {"type": "string"}, "surprise": {"type": "string"}, "facts": {"type": "array", "items": {"type": "string"}}},
        "required": ["question", "answer", "scene", "facts"]}},
    "ending": {"type": "string"}}, "required": ["viewer_question", "hook_line", "payoff", "chapters", "ending"]}

REVIEW_SCHEMA = {"type": "object", "properties": {
    "score": {"type": "number"}, "takeaways": {"type": "array", "items": {"type": "string"}}, "lecture_lines": {"type": "array", "items": {"type": "integer"}},
    "problems": {"type": "array", "items": {"type": "object", "properties": {"beat": {"type": "integer"}, "problem": {"type": "string"}, "fix": {"type": "string"}}, "required": ["problem", "fix"]}},
    "verdict": {"type": "string"}}, "required": ["score", "problems", "verdict"]}

EXAMPLE = (
    "EXAMPLE OF THE VOICE (a different topic; copy the VOICE, rhythm and structure, never the content):\n"
    "[0] Your fridge doesn't make cold. It never has. Not once.\n"
    "[1] What it does instead is way stranger, and by the end you'll know where all that heat actually goes.\n"
    "[2] (section: Where The Cold Goes) So if it isn't making cold, what is it doing? Put your hand behind it right now. Feel that warmth? That's where the cold went.\n"
    "[3] Here's the trick: cold isn't a thing. It's just a shortage of heat. So the fridge's whole job is to steal heat from the inside and dump it round the back.\n"
    "[4] The problem? Heat only flows one way by itself: from hot to cold. Pushing it the other way takes a helper.\n"
    "[5] And that helper is a liquid with a strange talent. It boils at a temperature far below freezing.\n"
    "[6] Think of a pan on the stove. The pan stays hot, but the water leaves as steam, carrying a ridiculous amount of heat with it. Same idea, inside your fridge, with a liquid that boils on contact with your leftovers.\n"
    "[7] Fine. But now you've got a hot gas and nowhere to put the heat. So here's the clever bit. You squeeze it.\n"
    "[8] Squash a gas and it heats up, like the nozzle of a bike pump. Suddenly it's hotter than your kitchen, so the heat spills out the back. Which is exactly the warmth you just felt.\n"
    "[9] Let it cool and expand, and it's cold again, ready for another lap. All day. All night. Round and round.\n"
    "[10] Which means your fridge isn't a cooler. It's a heat pump, and your kitchen is the place the heat ends up. Next time you stand there with the door open, you're not cooling your house. You're heating it.\n"
)

VOICE = (
    "THE VOICE: a sharp, funny friend explaining something they find genuinely amazing, over a table, to someone smart who knows nothing about it. Never a lecturer, never a textbook, "
    "never a news reader. Talk TO the viewer ('you', 'your', 'let's'). Contractions always ('it's', 'doesn't'). Short, punchy sentences mixed with the odd longer one. Concrete beats abstract "
    "every time: a thing you can picture, a number you can feel, a small scene ('you type three words, hit enter...'). When a number is huge or tiny, make it tangible. Dry humour and "
    "understatement are good; a little awe is good; hype is not. Every technical word gets a plain-words explanation the first time it appears, or is replaced by one. "
    "TEACH SOMETHING REAL: a viewer should finish able to explain the thing to a friend. Every chapter teaches one concrete mechanism or fact with its actual specifics from the sources "
    "(a number, a name, a date, a step-by-step 'first this, then that'); never a vague gesture like 'it works cleverly' or 'there's a trick'. Say HOW and WHY, not just THAT. "
    "SOUND HUMAN, NOT LIKE A CHATBOT: do not string together short punchy fragments ('Nice for you. For a camera? That's the torch.'), do not open lines with 'Here's the...', 'Picture this', "
    "'Imagine' more than twice in the whole video, do not use the shape 'X? Y.' (a question answered at once) more than three times, no 'And that's the thing', no 'Good luck.' style quips as a "
    "habit. Vary sentence length; let a thought run for a full sentence when it needs to; prefer a plain specific statement over a clever-sounding one. "
    "BANNED: 'in this video', 'let's dive in', 'delve', 'fascinating', 'game-changer', 'revolutionary', 'it is worth noting', 'in conclusion', 'to summarise', 'as we have seen', "
    "'there are several', 'is a type of', and any sentence of the shape 'X is a Y that Z' used as a definition up front. No lists of facts in a row: every line must make the viewer "
    "want the NEXT line (a question the last line raised, a 'but', a consequence, a twist). Do not invent contrived metaphors (no 'stickers', 'tiles' or 'coat hooks' to stand in "
    "for the thing itself): use the real thing, or a plain everyday comparison people already know."
)

PLAN_SYSTEM = (
    "You are the head writer of a YouTube channel that explains how the world works (AI, technology, machines, physics, history, the stories behind the headlines). Plan ONE video "
    "that people watch to the end. You are given the TITLE and THUMBNAIL the owner approved, a hook line, and SOURCE TEXT (the ONLY place facts may come from).\n"
    "The title and thumbnail made a promise. The video must PAY IT OFF: the thumbnail's words must be explained, with the surprising fact it points at revealed no later than about 90 "
    "seconds in, and everything after that must keep paying off new questions.\n"
    "DO NOT outline the article. A textbook walks through every topic in the source in order; a video picks ONE journey and goes deep. Choose the 3 or 4 most surprising or useful "
    "things in the sources and LEAVE THE REST OUT (no history detours, no storage or size arithmetic, no vocabulary lessons, no caveats about what is not known unless that IS the "
    "twist). Prefer following one concrete thing through the whole video (the life of one message, one object, one event) so every chapter is the next step of a story with tension: "
    "a mystery, an investigation, a twist, a resolution. At most FIVE technical terms in the whole video, each introduced only when the viewer needs it and tied to a picture.\n"
    "The hook_line must work with ZERO context for a person who knows nothing: a vivid scene or a strange true statement in plain words, no jargon, no 'a model', no abstract noun "
    "pile (good: 'Right now, billions of times a day, a computer is guessing your next word.' bad: 'Examples can change a model's answer format.').\n"
    "Return: viewer_question (the one question the viewer has in their head when they click); hook_line (the very first thing said: at most 18 words, a concrete surprising true "
    "statement or a vivid scene from the sources, NOT a definition, NOT a question the video fails to answer); payoff (what exactly the title/thumbnail promise and where it is "
    "revealed); running_example (one everyday thing or scene the viewer can picture, reused through the video: e.g. autocomplete on a phone for a language model); chapters (4 to 6, "
    "each the NEXT question a curious viewer would ask, in order, each with: question, answer (one plain sentence), scene (a concrete picture, analogy or tiny imagined scenario that "
    "makes the answer click), surprise (the twist or number that makes it memorable, from the sources), facts (the specific source facts it will use, quoted or closely paraphrased)); "
    "ending (a last beat that lands a thought, not a summary: something that changes how the viewer sees the thing)."
)

DRAFT_SYSTEM = (
    "You write the narration of ONE YouTube explainer of about {minutes} minutes ({words} words, {beats} beats) for the plan you are given, using ONLY facts from the SOURCE TEXT. "
    "A beat is one spoken passage of 14-30 words. {voice}\n"
    "STRUCTURE (non-negotiable): beat 0 is the plan's hook_line (polish it, keep it under 18 words, no section title). Beat 1 raises the stakes or the mystery in one or two lines and "
    "promises what the viewer will know by the end ('By the end you'll know exactly why...'), WITHOUT saying 'in this video'. By roughly beat 8 the thumbnail's surprising fact has been "
    "explained. Each chapter starts by asking the viewer's next question out loud, then answers it through the chapter's scene. Plant at least two open loops ('I'll explain why in a "
    "minute') and close them. Around the middle, re-hook: 'But here's where it gets weird.' The last beat lands the plan's ending, and refers back to the opening.\n"
    "Use the SOURCE TEXT for every specific number, name, date and claim of record (leave out any specific you cannot find there). To explain HOW something works you may also use well-established general knowledge in plain words, as long as you are certain it is correct and uncontroversial; never invent a statistic. Imagined scenes and analogies are welcome (frame them with 'imagine', 'picture', 'it's like', ONCE, with no disclaimers such as 'illustrative' or 'not attributed to any model'); you may explain how \
things work in plain textbook-level words, but every specific number, name, date and claim of record must come from the sources. Numbers as people say them. No parentheses, "
    "symbols, abbreviations or emoji.\n"
    "{example}\nPICTURES: for every beat plan what the viewer SEES while it is spoken. The picture must SHOW the idea, not decorate it. Prefer, in this order: 'animation' = a custom animated "
    "diagram drawn for the line (give 'concept' = one sentence of exactly what is drawn and how it moves; use it for any mechanism, process, comparison or imagined scene; UP TO HALF "
    "of the beats (never more than 18) may be animations, each about a different thing, but describe only simple shapes, text and movement); 'flow' = a process drawn step by step ('title' and 3-5 "
    "'steps'); 'bars' = growing bars for numbers the sources give ('title','labels','values','unit'); 'timeline' = dated events the sources give ('title','events'); 'stat' = one big "
    "number ('value' in digits at most 8 characters, 'label'); 'slide' = a short card listing 2-4 things ('title','bullets'); and 'photo' ONLY when one of the listed IMAGES really "
    "shows what the line says (give its id in 'image'): at most a quarter of the beats, never two photos in a row, never a chart or logo unrelated to the line. Never the same kind "
    "three times in a row; at most 4 'stat' cards and 5 'slide' cards in the whole video (more is a slideshow: use an animation instead), and never the same number twice.\n"
    "Put a short section title (2-5 words) in 'section' on the first beat of each chapter; it is ONLY a label for the chapter list and must never appear in the narration."
)

REVIEW_SYSTEM = (
    "You are a ruthless YouTube retention editor. You get a narration script (numbered beats) for a video whose title and thumbnail are given. Judge it the way a 20-year-old "
    "scrolling a phone would: would they stop watching in the first 15 seconds? at the first boring stretch? Score 1-10 (10 = they watch to the end and share it; 6 = a decent school "
    "lecture; be hard). List 'lecture_lines': beat numbers that sound like a textbook, a Wikipedia summary, a definition or a list of facts. For every real problem give the beat number, "
    "what is wrong (flat hook, jargon not explained, no concrete picture, stakes missing, repeats an earlier point, the thumbnail's promise not paid off, an unexplained leap, filler) and a "
    "concrete FIX (say what the line should do, with example wording). Also list 'takeaways': the specific things a viewer could now explain to a friend (each one a concrete fact or mechanism, one short line). If there are fewer than 6 genuine takeaways, "
    "or the script sounds like a chatbot (staccato fragments, 'Here's the thing', quips, vague claims) score it 5 or lower and say exactly which lines to make concrete. Judge: the hook; whether a newcomer understands every line without looking anything up; whether the title and "
    "thumbnail promise is paid off early and then more is paid off; whether it is a textbook outline (every sub-topic in order) instead of one story with tension; whether it uses more than five technical terms; whether lines are hedged or disclaimed instead of confident; variety and rhythm; whether any stretch of 3 beats is only facts. verdict = two sentences."
)

REWRITE_SYSTEM = (
    "You rewrite a YouTube narration script after an editor's review. Keep the structure and the facts (ONLY the SOURCE TEXT's facts), but fix every problem listed and make EVERY line "
    "in 'lecture_lines' spoken, concrete and curious. {voice}\nReturn the full script again in the same format (same JSON), improving also the picture plan where the review says a picture "
    "does not show the idea. Beat 0 stays the hook and has no section title."
)


def _package_text(package: dict | None) -> str:
    if not package:
        return ""
    return (f"TITLE: {package.get('title', '')}\nTHUMBNAIL WORDS (a surprising fact the thumbnail points at): {package.get('thumb_text', '')}\n"
            f"OPENING LINE IDEA: {package.get('hook', '')}\nWHY IT WORKS: {package.get('why', '')}\n")


def plan(ai, topic: str, package: dict | None, src_text: str, feedback: str = "") -> dict:
    extra = (f"\n\nA PREVIOUS PLAN FAILED THIS EDITOR REVIEW (it became a textbook outline): {feedback}\nMake a DIFFERENT plan: a different through-line, fewer and deeper chapters, "
             "a more concrete hook.") if feedback else ""
    res = ai.complete_json("flow_script_plan", PLAN_SYSTEM, f"TOPIC: {topic}\n{_package_text(package)}\nSOURCE TEXT:\n{src_text}{extra}", PLAN_SCHEMA)
    return res.data


def draft(ai, schema: dict, topic: str, the_plan: dict, package: dict | None, src_text: str, menu: list[dict], minutes: float, words: int, beats: int) -> dict:
    system = DRAFT_SYSTEM.replace("{example}", EXAMPLE).replace("{minutes}", str(int(round(minutes)))).replace("{words}", str(words)).replace("{beats}", str(beats)).replace("{voice}", VOICE)
    prompt = (f"TOPIC: {topic}\n{_package_text(package)}\nPLAN:\n{json.dumps(the_plan, indent=1)}\n\nIMAGES (id: description):\n"
              + "\n".join(f"{m['id']}: {m['desc']}" for m in menu) + f"\n\nSOURCE TEXT:\n{src_text}")
    return ai.complete_json("flow_script_draft", system, prompt, schema).data


def review(ai, script: dict, package: dict | None) -> dict:
    numbered = "\n".join(f"[{i}] {b['narration']}" for i, b in enumerate(script["beats"]))
    res = ai.complete_json("flow_script_review", REVIEW_SYSTEM, f"{_package_text(package)}\nSCRIPT:\n{numbered}", REVIEW_SCHEMA)
    return res.data


def rewrite(ai, schema: dict, script: dict, rv: dict, package: dict | None, src_text: str, menu: list[dict]) -> dict:
    system = REWRITE_SYSTEM.replace("{voice}", VOICE + "\n" + EXAMPLE)
    prompt = (f"{_package_text(package)}\nREVIEW:\n{json.dumps(rv, indent=1)}\n\nCURRENT SCRIPT (JSON):\n{json.dumps({'title': script['title'], 'beats': script['beats']})}\n\n"
              "IMAGES (id: description):\n" + "\n".join(f"{m['id']}: {m['desc']}" for m in menu) + f"\n\nSOURCE TEXT:\n{src_text}")
    return ai.complete_json("flow_script_rewrite", system, prompt, schema).data


def write(ai, topic: str, srcs_text: str, minutes: float, menu: list[dict], schema: dict, package: dict | None = None, words_per_min: int = 150, rounds: int = 1, say=print) -> dict:
    words = int(minutes * words_per_min)
    beats = max(12, int(minutes * 60 / 8))
    say("script: planning the story")
    the_plan = plan(ai, topic, package, srcs_text)
    say(f"script: hook idea: {the_plan.get('hook_line', '')!r}; {len(the_plan.get('chapters', []))} chapters")
    script = draft(ai, schema, topic, the_plan, package, srcs_text, menu, minutes, words, beats)
    script["plan"] = the_plan
    history = []
    replanned = False
    for r in range(rounds + 1):
        script["beats"] = [b for b in script.get("beats", []) if len((b.get("narration") or "").split()) >= 4]
        rv = review(ai, script, package)
        history.append({"round": r, "score": rv.get("score"), "verdict": rv.get("verdict"), "lecture_lines": rv.get("lecture_lines", [])})
        say(f"script review {r}: {rv.get('score')} of 10 - {str(rv.get('verdict', ''))[:140]}")
        if float(rv.get("score", 0)) >= 8.0 or r == rounds:
            break
        if float(rv.get("score", 0)) < 6.0 and not replanned:                      # a weak first draft is a weak plan: plan again, do not polish it
            replanned = True
            say("script: the first draft scored low, planning a different story")
            the_plan = plan(ai, topic, package, srcs_text, feedback=str(rv.get("verdict", "")) + " " + "; ".join(p.get("problem", "") for p in rv.get("problems", [])[:4]))
            script = draft(ai, schema, topic, the_plan, package, srcs_text, menu, minutes, words, beats)
            script["plan"] = the_plan
            continue
        better = rewrite(ai, schema, script, rv, package, srcs_text, menu)
        if len(better.get("beats", [])) >= max(8, int(len(script["beats"]) * 0.7)):
            better["plan"] = the_plan
            script = better
    script["review"] = history
    return script
