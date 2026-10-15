"""Apply the human study's verdict to the pre-study item pool.

The K = 3 study (human_study/) audited the derivation rules, and two did not survive:
  absent -> ANSWER   annotators abstain on 49 of 50 absent-object items, since a single frame cannot
                     show that an object is absent rather than out of view. These items become ABSTAIN,
                     which also turns present/absent pairs from answer/answer into answer/abstain.
  single_occluded    annotators answer 40 of 50 attribute questions about a small or cut-off object.
                     Relabelling against the annotators or keeping a rule they reject were both ruled
                     out, so every pair built on this rule is removed.
A few images still sit in two splits at this point. Pairs that share an image are grouped and each
group moves, whole, to the split most of its items were already in, so the splits become pair-level
and image-disjoint without redrawing the test split.

The script always starts from the pre-study pool, so running it twice cannot compound.

    python tools/relabel_from_human_study.py --prestudy construction/prestudy_items.jsonl --out-dir data_rebuilt
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

RETIRED = {"single_occluded"}
RELABEL = {"absent": "abstain"}


def corrected(r):
    a = RELABEL.get(r.get("decision_basis"))
    if a:
        r = dict(r)
        r["gold_action"] = a
        r["is_ambiguous"] = a != "answer"
        if "pair_role" in r:
            r["pair_role"] = a
    return r


def apply_verdict(rows):
    """(kept rows, {split: rows}) after the verdict and the image-disjointness repair."""
    pairs = defaultdict(list)
    for r in rows:
        pairs[r.get("pair_id")].append(r)
    kept = [corrected(r) for m in pairs.values() if not any(x.get("decision_basis") in RETIRED for x in m) for r in m]

    pair_of = defaultdict(list)
    for r in kept:
        pair_of[r["pair_id"]].append(r)
    parent = {p: p for p in pair_of}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    image_pair = {}
    for pid, members in pair_of.items():
        for r in members:
            prev = image_pair.setdefault(r["image_path"], pid)
            ra, rb = find(prev), find(pid)
            if ra != rb:
                parent[rb] = ra
    groups = defaultdict(list)
    for pid in pair_of:
        groups[find(pid)].append(pid)
    assigned = {}
    for pids in groups.values():
        split = Counter(r.get("split", "train") for p in pids for r in pair_of[p]).most_common(1)[0][0]
        for p in pids:
            assigned[p] = split
    by_split = defaultdict(list)
    for pid, members in pair_of.items():
        for r in members:
            r["split"] = assigned[pid]
            by_split[assigned[pid]].append(r)
    return kept, by_split


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prestudy", default="construction/prestudy_items.jsonl")
    ap.add_argument("--out-dir", default="data_rebuilt")
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.prestudy)]
    kept, by_split = apply_verdict(rows)

    d = Path(args.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "full.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in kept)
    for sp in ("train", "val", "test"):
        with open(d / f"{sp}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in by_split.get(sp, []))
    pairs = defaultdict(set)
    for r in kept:
        pairs[r["pair_id"]].add(r["gold_action"])
    contrastive = sum(len(v) > 1 for v in pairs.values())
    print(f"  kept {len(kept)} of {len(rows)} items, {len(pairs)} pairs, {contrastive / len(pairs):.1%} contrastive")
    print("  splits: " + ", ".join(f"{sp} {len(by_split.get(sp, []))}" for sp in ("train", "val", "test")))


if __name__ == "__main__":
    main()
