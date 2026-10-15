"""Assign whole pairs to train / val / test (60 / 15 / 25%).

Pairs are shuffled with a fixed seed and assigned in order; a pair with an image already placed in a
split follows that split. A pair whose images already sit in two different splits cannot follow both,
so a few images can end up in two splits here; relabel_from_human_study.py resolves them. Writes
{train,val,test,full}.jsonl to the output folder.

    python tools/make_splits.py --items construction/items_raw.jsonl --out-dir construction/prestudy
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 202608
FRAC = {"train": 0.60, "val": 0.15, "test": 0.25}


def split_pairs(rows, seed=SEED):
    """{pair id: split} for the pairs in `rows`."""
    rng = random.Random(seed)
    by_pair = defaultdict(list)
    for r in rows:
        by_pair[r.get("pair_id") or r["item_id"]].append(r)
    pairs = list(by_pair)
    rng.shuffle(pairs)
    n_tr, n_va = int(len(pairs) * FRAC["train"]), int(len(pairs) * FRAC["val"])
    assigned, img_split = {}, {}
    for i, p in enumerate(pairs):
        sp = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
        imgs = {r["image_id"] for r in by_pair[p]}
        clash = {img_split[im] for im in imgs if im in img_split}
        if len(clash) == 1:
            sp = clash.pop()
        assigned[p] = sp
        for im in imgs:
            img_split.setdefault(im, sp)
    return assigned


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", default="construction/items_raw.jsonl")
    ap.add_argument("--out-dir", default="construction/prestudy")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.items)]
    split_of = split_pairs(rows, args.seed)
    by_pair = defaultdict(list)
    for r in rows:
        by_pair[r.get("pair_id") or r["item_id"]].append(r)
    out = defaultdict(list)
    for p, members in by_pair.items():
        for r in members:
            r["split"] = split_of[p]
            out[split_of[p]].append(r)

    d = Path(args.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    for sp in ("train", "val", "test"):
        with open(d / f"{sp}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in out[sp])
    with open(d / "full.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for sp in ("train", "val", "test") for r in out[sp])

    seen = defaultdict(set)
    for sp in ("train", "val", "test"):
        for r in out[sp]:
            seen[r["image_id"]].add(sp)
        print(f"  {sp:5s}: {len(out[sp]):6d} items | {dict(Counter(r['gold_action'] for r in out[sp]))}")
    print(f"  images in more than one split: {sum(1 for v in seen.values() if len(v) > 1)}")


if __name__ == "__main__":
    main()
