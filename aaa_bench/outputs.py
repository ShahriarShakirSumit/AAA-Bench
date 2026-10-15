"""Reading saved model responses (model_outputs/<model>_<strategy>.jsonl)."""

import json
from pathlib import Path

from .paths import MODEL_OUTPUTS

# Display names used in the paper's tables and figures.
MODEL_LABELS = {
    "qwen2.5-vl-7b": "Qwen2.5-VL", "qwen2-vl-7b": "Qwen2-VL", "internvl3-8b": "InternVL3",
    "llava-next-7b": "LLaVA-Next", "llava-1.5-7b": "LLaVA-1.5", "phi-3.5-vision": "Phi-3.5-V",
    "idefics3-8b": "Idefics3",
}
MIN_COVERAGE = 0.9   # a run is tabulated only if it covers at least 90% of the test items


def label(model):
    return MODEL_LABELS.get(model, model)


def read_run(path):
    """{item_id: record} for one response file."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("item_id"):
                    out[r["item_id"]] = r
    return out


def load_runs(strategy="option", gold=None, outputs_dir=MODEL_OUTPUTS, min_coverage=MIN_COVERAGE):
    """{model: {item_id: record}} for every saved run of one strategy.

    Only files directly inside `outputs_dir` are read; the reference pipelines in its reference/
    subfolder are not language models and are loaded separately. With `gold`, records are restricted
    to gold items and runs covering less than `min_coverage` of them are left out.
    """
    runs = {}
    suffix = f"_{strategy}.jsonl"
    for path in sorted(Path(outputs_dir).glob(f"*{suffix}")):
        model = path.name[: -len(suffix)]
        preds = read_run(path)
        if gold is not None:
            preds = {i: r for i, r in preds.items() if i in gold}
            if len(preds) < min_coverage * len(gold):
                continue
        runs[model] = preds
    return runs
