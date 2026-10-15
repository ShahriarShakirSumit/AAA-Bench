"""Reading AAA-Bench items.

Each line of data/{train,val,test}.jsonl is one item: a COCO image reference, a question and the
action the scene licenses (answer, ask or abstain). Items come in counterfactual pairs: the two
members of a pair share the same question but are about different images and have different gold
actions. See data/schema.json for every field.
"""

import json
from collections import defaultdict
from pathlib import Path

from .paths import DATA

SPLITS = ("train", "val", "test")
ACTIONS = ("answer", "ask", "abstain")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def load_split(split="test", data_dir=DATA):
    """Items of one split, in file order."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; choose from {SPLITS}")
    return read_jsonl(Path(data_dir) / f"{split}.jsonl")


def load_items(splits=SPLITS, data_dir=DATA):
    """Items of several splits (all three by default), concatenated in split order."""
    if isinstance(splits, str):
        splits = (splits,)
    return [r for s in splits for r in load_split(s, data_dir)]


def by_id(items):
    return {r["item_id"]: r for r in items}


def pairs_of(items):
    """The complete counterfactual pairs among `items`, each as a two-item list."""
    groups = defaultdict(list)
    for r in items:
        groups[r["pair_id"]].append(r)
    return [members for members in groups.values() if len(members) == 2]


def image_file(item, coco_dir):
    """Path of an item's image inside a COCO 2017 folder (the one holding train2017/ and val2017/)."""
    return Path(coco_dir) / item["image_path"]
