"""The two-turn evaluation protocol.

Turn 1: the model sees the image and the question under one prompting strategy, and its response
is parsed into an action. If the action is ASK and the item has a gold clarification, the model is
given that clarification (turn 1.5) and answers again (turn 2). Every metric in the paper is computed
from the turn-1 action.
"""

from .data import image_file
from .parsing import abstain_reason, answer_text, clarifying_question, parse_action
from .prompts import turn1_prompt, turn2_prompt


def select_clarification(item, asked):
    """The item's clarification whose question shares the most words with the model's question."""
    options = item.get("clarifications") or []
    if not options:
        return None
    if asked is None:
        return options[0]
    words = set(asked.lower().split())
    best, best_overlap = options[0], 0
    for c in options:
        overlap = len(words & set(c["clarifying_question"].lower().split()))
        if overlap > best_overlap:
            best, best_overlap = c, overlap
    return best


def run_item(model, item, strategy, coco_dir):
    """Run one item through the protocol and return a response record."""
    image = str(image_file(item, coco_dir))
    raw = model.generate(image, turn1_prompt(strategy, item["question"]))
    action = parse_action(raw)
    asked = clarifying_question(raw, action)

    turn2_raw = None
    if action == "ask" and item.get("clarifications"):
        clar = select_clarification(item, asked)
        turn2_raw = model.generate(image, turn2_prompt(item["question"], asked or "Could you clarify?",
                                                       clar["clarification_response"]))

    return {
        "item_id": item["item_id"],
        "model_name": model.model_name,
        "prompting_strategy": strategy,
        "action": action,
        "answer_text": answer_text(raw, action),
        "clarifying_question_text": asked,
        "abstain_reason": abstain_reason(raw, action),
        "confidence": None,
        "raw_output": raw,
        "turn2_answer": turn2_raw.strip() if turn2_raw is not None else None,
        "turn2_raw_output": turn2_raw,
    }
