import json

import pytest

from aaa_bench.outputs import read_run
from aaa_bench.parsing import cue_action, explicit_tag, parse_action
from aaa_bench.paths import MODEL_OUTPUTS


@pytest.mark.parametrize("text, action", [
    ("[ASK] Which dog do you mean?", "ask"),
    ("Answer: [ABSTAIN]. The object is too small.", "abstain"),
    ("Reasoning: there are two cups. [ASK] Which cup?", "ask"),
    ("[ANSWER] It is red.", "answer"),
    ("ABSTAIN: the image does not show it", "abstain"),
    ("The car is blue.", "answer"),
    ("Which one do you mean?", "ask"),
    ("I cannot determine the colour. There is not enough detail.", "abstain"),
])
def test_parse_action(text, action):
    assert parse_action(text) == action


def test_first_tag_wins():
    assert explicit_tag("[ASK] or maybe [ANSWER]") == "ask"
    assert parse_action("Answer: [ABSTAIN]") == "abstain"
    assert cue_action("Answer: [ABSTAIN]") == "answer"


def test_empty_output_is_not_an_answer():
    with pytest.raises(ValueError):
        parse_action("   ")


def test_parser_reproduces_saved_actions():
    files = sorted(MODEL_OUTPUTS.glob("*.jsonl"))
    assert len(files) == 21
    for path in files:
        for r in read_run(path).values():
            assert parse_action(r["raw_output"]) == r["action"], (path.name, r["item_id"])
