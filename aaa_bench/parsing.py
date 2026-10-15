"""Mapping a model's free-text response to an action.

The option and chain-of-thought prompts ask the model to mark its choice with [ANSWER], [ASK] or
[ABSTAIN]. When such a tag appears anywhere in the response, the first one decides. Otherwise the
action is read from cue phrases, in priority order: an unbracketed leading marker ("ASK:"), then
abstention cues, then clarification cues, and ANSWER when nothing else applies.

This is the parser behind every number in the paper.
"""

import re

TAG = re.compile(r"\[(ANSWER|ASK|ABSTAIN)\]", re.I)

ASK_PATTERNS = [
    r"(?i)\bcould you (please )?(clarify|specify|tell me|explain)",
    r"(?i)\bwhich (one|.*?) (do you|are you)",
    r"(?i)\bdo you mean",
    r"(?i)\bcan you (be more specific|clarify|provide)",
    r"(?i)\bwhat (exactly|specifically) (do you|are you)",
    r"(?i)\bare you (asking|referring|talking) about",
    r"(?i)\bI need (more|additional) (information|context|details)",
    r"(?i)\bplease (clarify|specify|elaborate)",
    r"(?i)\?\s*$",  # ends with a question mark (weak signal, used only on short responses)
]

ABSTAIN_PATTERNS = [
    r"(?i)\bI (cannot|can't|am unable to) (determine|tell|answer|see)",
    r"(?i)\b(not possible|impossible) to (determine|tell|know)",
    r"(?i)\b(insufficient|not enough) (information|context|detail|resolution)",
    r"(?i)\bI (don't|do not) have enough",
    r"(?i)\bI'm (not sure|uncertain|unable)",
    r"(?i)\bcannot be determined",
    r"(?i)\bI abstain",
    r"(?i)\bI decline to answer",
    r"(?i)\bthe image (doesn't|does not) (provide|show|contain) enough",
]


def explicit_tag(raw_output):
    """The first [ANSWER]/[ASK]/[ABSTAIN] tag in the response, lower-cased, or None."""
    m = TAG.search(raw_output or "")
    return m.group(1).lower() if m else None


def cue_action(raw_output):
    """The action implied by cue phrases alone, ignoring bracketed tags that are not at the start."""
    text = raw_output.strip()
    head = text.upper()
    if head.startswith("[ABSTAIN]") or head.startswith("ABSTAIN:"):
        return "abstain"
    if head.startswith("[ASK]") or head.startswith("ASK:"):
        return "ask"
    if head.startswith("[ANSWER]") or head.startswith("ANSWER:"):
        return "answer"

    abstain_score = sum(1 for p in ABSTAIN_PATTERNS if re.search(p, text))
    ask_score = sum(1 for p in ASK_PATTERNS if re.search(p, text))
    if abstain_score >= 2:
        return "abstain"
    strong_ask = sum(1 for p in ASK_PATTERNS[:-1] if re.search(p, text))
    if strong_ask >= 1:
        return "ask"
    if ask_score >= 1 and abstain_score == 0 and len(text) < 200 and text.count("?") >= 1:
        return "ask"
    return "answer"


def parse_action(raw_output):
    """Action ('answer', 'ask' or 'abstain') for a model response.

    An empty response raises ValueError: a model that returned nothing has not chosen an action, and
    scoring it as an answer would turn every failed request into a confident reply.
    """
    if not raw_output or not raw_output.strip():
        raise ValueError("empty model output: no action was produced")
    return explicit_tag(raw_output) or cue_action(raw_output)


def _strip_prefix(raw_output, prefixes):
    text = raw_output.strip()
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text


def answer_text(raw_output, action):
    return _strip_prefix(raw_output, ("[ANSWER]", "ANSWER:", "Answer:")) if action == "answer" else None


def clarifying_question(raw_output, action):
    return _strip_prefix(raw_output, ("[ASK]", "ASK:", "Ask:")) if action == "ask" else None


def abstain_reason(raw_output, action):
    return _strip_prefix(raw_output, ("[ABSTAIN]", "ABSTAIN:", "Abstain:")) if action == "abstain" else None
