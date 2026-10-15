"""Spatial ASK items whose two largest instances lie on the same side of the image.

Definition 1 of the paper requires the interpretations of an ASK item to give different answers. For
a spatial question ("Is the cup on the left or the right?") the answer is the side of the image, so an
ASK item whose two largest instances are on the same side (left, centre or right, by box centre) has
two interpretations with the same answer. This counts such items, reports what the human majority
chose on them in the study, and recomputes contrastive accuracy without the test pairs that contain
one. Image widths come from the COCO annotations (reproduce/data/image_widths.json). Writes
results/same_side.json.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from aaa_bench.construction import side_of
from aaa_bench.data import by_id, load_items, load_split, pairs_of
from aaa_bench.outputs import label, load_runs
from aaa_bench.paths import RESULTS
from compute_agreement import load_study, majority


def same_side(item, widths):
    sides = {side_of(b, widths[item["image_id"]]) for b in item["visual_evidence"]["boxes"][:2]}
    return len(sides) == 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--widths", default=str(Path(__file__).resolve().parent / "data" / "image_widths.json"),
                    help="COCO image widths by image id")
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()
    widths = json.load(open(args.widths))

    ask = lambda r: r["ambiguity_subtype"] == "referential_spatial" and r["gold_action"] == "ask"  # noqa: E731
    items = [r for r in load_items() if ask(r)]
    n_same = sum(same_side(r, widths) for r in items)
    test = load_split("test")
    gold = by_id(test)
    pairs = pairs_of(test)
    flagged = [p for p in pairs if any(ask(x) and same_side(x, widths) for x in p)]
    keep = [p for p in pairs if p not in flagged]
    print(f"  spatial ASK items with both instances on one side: {n_same} of {len(items)} "
          f"({sum(same_side(r, widths) for r in items if r['split'] == 'test')} of "
          f"{sum(r['split'] == 'test' for r in items)} in the test split)")

    study_items, votes, _ = load_study()
    human = {"same side": Counter(), "different sides": Counter()}
    for i, r in study_items.items():
        if r["decision_basis"] == "multi_comparable" and r["ambiguity_subtype"] == "referential_spatial":
            human["same side" if same_side(r, widths) else "different sides"][majority(votes[i])] += 1
    print(f"  human majority on such study items: {dict(human['same side'])}; "
          f"with instances on different sides: {dict(human['different sides'])}")

    changes = {}
    for m, run in load_runs("option", gold).items():
        p = {i: r["action"] for i, r in run.items()}
        ca = lambda ps: sum(all(p.get(x["item_id"]) == x["gold_action"] for x in q) for q in ps) / len(ps)  # noqa: E731
        changes[m] = {"all_pairs": ca(pairs), "without_flagged": ca(keep)}
        print(f"    {label(m):11s} CA {ca(pairs):.3f} -> {ca(keep):.3f} without the {len(flagged)} pairs")
    summary = {"items_same_side": n_same, "items": len(items), "test_pairs_flagged": len(flagged),
               "test_pairs": len(pairs), "human_majority": {k: dict(v) for k, v in human.items()},
               "contrastive": changes,
               "max_change": max(abs(v["without_flagged"] - v["all_pairs"]) for v in changes.values()),
               "max_without_flagged": max(v["without_flagged"] for v in changes.values())}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "same_side.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  largest change {summary['max_change']:.3f}; wrote {Path(args.out) / 'same_side.json'}")


if __name__ == "__main__":
    main()
