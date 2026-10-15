"""Check that every released label follows from the COCO annotations.

For each item it recomputes, from instances_{train,val}2017.json, the referents of the queried
category, the decision basis, the gold action (with the human study's verdict), the stored evidence
boxes and areas, the question wording and the image path, and reports every mismatch.

    python tools/verify_labels.py --annotations /data/coco/annotations
    python tools/verify_labels.py --index construction/coco_index.json      # reuse a built index
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aaa_bench.construction import (STUDY_RELABEL, TEMPLATES, evidence, question,  # noqa: E402
                                    referential_rule, referents)
from aaa_bench.data import load_items  # noqa: E402


def check(item, index):
    """Names of the fields that do not follow from the annotations (empty if all do)."""
    bad = []
    im = index["images"][item["image_id"]]
    cid = str(item["coco_category_ids"][0])
    cat = index["categories"][cid]
    insts = im["cats"].get(cid, [])
    W, H = im["width"], im["height"]
    real, A = referents(insts, W, H)
    family, basis = item["ambiguity_subtype"], item["decision_basis"]

    if family.startswith("referential"):
        derived = referential_rule(real, A, W, H, "color" if family.endswith("color") else "spatial")
        used = real
    elif family == "counting":
        if basis == "distinct":
            derived = ("answer", "distinct") if item["image_id"] in index["pools"][cid]["many"] else (None, "")
            used = real
        else:
            derived = ("abstain", "crowd") if item["image_id"] in index["pools"][cid]["crowd"] else (None, "")
            used = real or insts
    else:
        derived = ("answer", "present") if insts else ("answer", "absent")
        used = (real or insts) if insts else []
    action, derived_basis = derived
    action = STUDY_RELABEL.get(derived_basis, action)
    if derived_basis != basis:
        bad.append("decision_basis")
    if action != item["gold_action"]:
        bad.append("gold_action")
    if evidence(cat, used) != item["visual_evidence"]:
        bad.append("visual_evidence")
    if item["question"] not in [question(family, t, cat) for t in range(len(TEMPLATES[family]))]:
        bad.append("question")
    if item["image_path"] != f"{im['split']}2017/{im['file_name']}":
        bad.append("image_path")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--annotations", help="folder with instances_{train,val}2017.json")
    src.add_argument("--index", help="an index written by build_coco_index.py")
    args = ap.parse_args()
    if args.index:
        index = json.load(open(args.index))
    else:
        from build_coco_index import build
        index = build(args.annotations)
    for pools in index["pools"].values():
        for k in pools:
            pools[k] = {str(i) for i in pools[k]}

    items = load_items()
    problems = Counter()
    examples = {}
    for item in items:
        for field in check(item, index):
            problems[field] += 1
            examples.setdefault(field, item["item_id"])
    print(f"checked {len(items)} items against the COCO annotations")
    if not problems:
        print("every gold action, decision basis, evidence box, question and image path follows from the annotations")
        return
    for field, n in problems.most_common():
        print(f"  {field}: {n} items (e.g. {examples[field]})")
    sys.exit(1)


if __name__ == "__main__":
    main()
