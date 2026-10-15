# Saved model responses

`<model>_<strategy>.jsonl` holds the responses of one model to the test split under one prompting strategy: seven
models (`llava-1.5-7b`, `llava-next-7b`, `qwen2-vl-7b`, `qwen2.5-vl-7b`, `internvl3-8b`, `phi-3.5-vision`,
`idefics3-8b`) times three strategies (`direct`, `option`, `cot`; the prompts are in `aaa_bench/prompts.py`). The
runs used greedy decoding with at most 512 new tokens, the checkpoints pinned in `aaa_bench/models/__init__.py` and
the library versions in `requirements-gpu.txt`.

Each file covers 1,448 of the 1,450 test items. The pair `AAAv2-004578` / `AAAv2-004579` entered the test split
after these runs, when the splits were made image-disjoint, and has no responses. Contrastive accuracy as computed
by `aaa_bench.score` counts this pair as unsolved.

Each line is one response:

| Field | Meaning |
|---|---|
| `item_id`, `model_name`, `prompting_strategy` | which item, model and prompt |
| `action` | the turn-1 action, `parse_action(raw_output)` from `aaa_bench/parsing.py`; every metric uses this field |
| `raw_output` | the model's full turn-1 reply |
| `answer_text`, `clarifying_question_text`, `abstain_reason` | `raw_output` without its leading marker, in the field that matches `action` |
| `turn2_raw_output`, `turn2_answer` | the reply after the model asked and was given the item's clarification |
| `confidence` | always null; no model reported a usable confidence |

Only items whose gold action is ASK carry a clarification, so a model that asks on any other item gets no second
turn. A second turn was run for 189 responses. Another 96 responses count as ASK only under the final parser, which
reads an explicit `[ASK]` tag wherever it appears; they were not given a clarification. No number in the paper uses
turn 2.

`reference/` holds the two image-only pipelines of the paper, which are not language models and are kept apart so
that a glob over this folder never mixes them into a model table: `detector-rules.jsonl` (YOLO11-L detections with
the benchmark's own rules) and `learned-head.jsonl` (a logistic regression over detection features, fitted on the
train split). `detector_features.json` caches the detection features of the train and test items, from which
`reproduce/analysis_detector.py` recomputes both pipelines.
