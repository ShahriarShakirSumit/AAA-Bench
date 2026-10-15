import hashlib
from collections import Counter

from aaa_bench.data import load_items, load_split, pairs_of
from aaa_bench.paths import DATA


def test_split_sizes():
    assert {s: len(load_split(s)) for s in ("train", "val", "test")} == {"train": 4994, "val": 956, "test": 1450}


def test_pairs_share_wording_and_differ_in_action():
    items = load_items()
    pairs = pairs_of(items)
    assert len(pairs) == 3700 and 2 * len(pairs) == len(items)
    same_image = []
    for a, b in pairs:
        assert a["question"] == b["question"]
        assert a["gold_action"] != b["gold_action"]
        assert a["split"] == b["split"]
        if a["image_id"] == b["image_id"]:
            same_image.append(a["pair_id"])
    # the one known pair whose members share an image (see DATASHEET.md, known issues)
    assert same_image == ["P004605"]


def test_splits_are_image_disjoint():
    images = {s: {r["image_id"] for r in load_split(s)} for s in ("train", "val", "test")}
    assert not images["train"] & images["val"]
    assert not images["train"] & images["test"]
    assert not images["val"] & images["test"]


def test_composition():
    items = load_items()
    assert Counter(r["gold_action"] for r in items) == {"answer": 3700, "ask": 2488, "abstain": 1212}
    assert len({r["visual_evidence"]["category"] for r in items}) == 59


def test_checksums():
    for line in open(DATA / "checksums.sha256"):
        digest, name = line.split()
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest, name
