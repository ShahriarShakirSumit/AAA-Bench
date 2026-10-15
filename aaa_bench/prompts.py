"""The three prompting strategies of the paper and the turn-2 prompt.

direct  the question alone, measuring what a model does by default
option  the three actions are offered, with no reasoning scaffold (the paper's primary setting)
cot     the three actions are offered after step-by-step reasoning
"""

PROMPTS = {
    "direct": (
        "Answer the following question about the image.\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    "option": (
        "Answer the following question about the image. "
        "If the question is ambiguous or you cannot determine the answer "
        "from the image alone, you have three options:\n"
        "1. [ANSWER] Provide your best answer\n"
        "2. [ASK] Ask a clarifying question to resolve the ambiguity\n"
        "3. [ABSTAIN] Decline to answer with a brief justification\n\n"
        "Question: {question}\n\n"
        "Response:"
    ),
    "cot": (
        "Look at the image and consider the following question carefully.\n\n"
        "Question: {question}\n\n"
        "First, think step-by-step about whether this question can be "
        "answered unambiguously from the image alone. Consider:\n"
        "- Are there multiple possible referents?\n"
        "- Is any information unclear or missing?\n"
        "- Is the question's intent clear?\n\n"
        "Then choose one action:\n"
        "- [ANSWER] if you can answer confidently\n"
        "- [ASK] if a clarifying question would help\n"
        "- [ABSTAIN] if the question cannot be reliably answered\n\n"
        "Reasoning and Response:"
    ),
}

STRATEGIES = tuple(PROMPTS)


def turn1_prompt(strategy, question):
    if strategy not in PROMPTS:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
    return PROMPTS[strategy].format(question=question)


def turn2_prompt(question, clarifying_question, clarification_response):
    """Prompt for the second turn, after the model asked and received the item's clarification."""
    return (
        f"You previously asked: \"{clarifying_question}\"\n\n"
        f"The user responded: \"{clarification_response}\"\n\n"
        f"Now, given this clarification and the image, please answer "
        f"the original question:\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
