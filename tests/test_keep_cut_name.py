"""A kept video ("Add B-roll to a video") gets a name from what is said in it, not its file name; the quality check's rows come back from the report."""
from types import SimpleNamespace as NS

from clipfarm.clipper import flow as FL, notes as NT


class AI:
    def __init__(self, title):
        self.title = title

    def complete_json(self, task, system, prompt, schema):
        assert task == "video_title"
        return NS(data={"title": self.title, "summary": "x"})


def sents(n=30):
    return [NS(text="Students talk about the new building and why it matters to them") for _ in range(n // 10 + 1)]


def test_a_title_comes_from_the_words():
    assert FL.name_it(AI('"Why Maple Hall Is Filling Up"'), sents()) == "Why Maple Hall Is Filling Up"


def test_too_few_words_keep_the_file_name():
    assert FL.name_it(AI("Whatever"), [NS(text="hello there")]) == ""


def test_a_failing_ai_keeps_the_file_name():
    class Bad:
        def complete_json(self, *a):
            raise RuntimeError("no")
    assert FL.name_it(Bad(), sents()) == ""


def test_quality_rows_survive_the_report_file():
    md = NT.quality_md([{"level": "warn", "start": 5.0, "end": 9.0, "what": "a quiet stretch", "do": "add music"}], {}, "T")
    rows = NT.rows_from_md(md)
    assert rows[0]["level"] == "Look" and rows[0]["what"] == "A quiet stretch" and rows[0]["do"] == "add music"
