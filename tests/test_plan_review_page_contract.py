"""What the page really sends to the plan-review endpoints (tester, WP16): JSON bodies as the browser builds them, not Python ints."""
import json
import re

import pytest

from clipfarm.clipper import planreview as PR
from clipfarm.clipper import studio_web as W
from tests.test_plan_review import HTML, runner  # noqa: F401  (the fixture)


def test_the_radio_value_the_page_sends_is_accepted_as_the_choice(runner):
    """studio.html puts the choice id in a radio's value and posts `choice: v.value`: a STRING ("0"/"1"), because JSON from a DOM value is text."""
    q = PR.review(runner, "p1")["questions"][0]
    wire = json.loads(json.dumps({"question": q["id"], "choice": q["choices"][1]["id"]}))      # exactly what JSON.stringify(...) of the page produces
    assert isinstance(wire["choice"], str)
    r = PR.answer(runner, "p1", wire["question"], wire["choice"])
    assert r["questions"] == []


def test_the_page_posts_the_radio_value_unconverted():
    src = HTML.read_text()
    m = re.search(r'act\("answer", \{question: qb\.dataset\.q, choice: ([^}]+)\}\)', src)
    assert m, "the answer call moved: update this test"
    assert m.group(1).strip() != "+v.value" and "Number" not in m.group(1) and "parseInt" not in m.group(1), "the page converts to a number: then the string test above is moot"
