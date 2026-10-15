"""Build a pool of counterfactual item pairs from the COCO index.

For each of 59 categories it samples images, derives each candidate's action with the rules in
aaa_bench/construction.py, and pairs items that share a question but differ in action:
  referential (spatial and colour)  ASK (several comparable) with ANSWER (one dominant)
  colour                            ANSWER (one clear) with ABSTAIN (one occluded)
  spatial                           ANSWER (one clear) with ASK (several comparable)
  counting                          ANSWER (countable) with ABSTAIN (crowd)
  presence                          ANSWER (present) with an absent-object item
The labels are those of the construction rules before the human study (absent -> ANSWER, the
occlusion rule in force); relabel_from_human_study.py applies the study's verdict afterwards.

Answer fields are filled only where COCO fixes the answer (the side for spatial items, "yes" for a
present object). Rebuilding draws a fresh sample under the same rules. It is deterministic, but it does not reproduce
the published files item for item; tools/verify_labels.py checks the published labels themselves.

    python tools/build_dataset.py --index construction/coco_index.json --out construction/items_raw.jsonl
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aaa_bench.construction import evidence, question, referential_rule, referents, side_of  # noqa: E402

SEED = 20260805
CATEGORIES = [
    "airplane", "apple", "backpack", "banana", "bear", "bed", "bench", "bicycle", "bird", "boat", "book",
    "bottle", "bowl", "bus", "cake", "car", "cat", "cell phone", "chair", "clock", "couch", "cow", "cup",
    "dog", "donut", "elephant", "fork", "giraffe", "handbag", "horse", "keyboard", "kite", "knife",
    "laptop", "motorcycle", "mouse", "orange", "person", "pizza", "potted plant", "refrigerator", "remote",
    "scissors", "sheep", "sink", "skateboard", "spoon", "suitcase", "surfboard", "teddy bear",
    "tennis racket", "toothbrush", "train", "truck", "tv", "umbrella", "vase", "wine glass", "zebra",
]


def make_item(n, image_id, im, cat, cid, family, action, basis, real, template, answer=None):
    W = im["width"]
    item = dict(item_id=f"AAAv2-{n:06d}", split="unassigned", image_id=str(image_id), image_source="coco",
                image_path=f"{im['split']}2017/{im['file_name']}", question=question(family, template, cat),
                ambiguity_subtype=family, coco_category_ids=[int(cid)],
                num_objects_in_scene=sum(len(v) for v in im["cats"].values()),
                pair_role=action, decision_basis=basis, visual_evidence=evidence(cat, real))
    if action == "answer":
        # COCO fixes the answer only for spatial items (the side) and present objects ("yes")
        ans = str(answer) if family == "referential_spatial" or basis == "present" else None
        item.update(is_ambiguous=False, ambiguity_type=None, difficulty=1 if basis != "multi_dominant" else 2,
                    gold_action="answer", direct_answer=ans,
                    valid_interpretations=[{"interpretation": "the intended referent", "answer": ans,
                                            "probability": 1.0}], clarifications=[])
    elif action == "ask":
        sides = [side_of(x["bbox"], W) for x in real]
        item.update(is_ambiguous=True, ambiguity_type="referential", difficulty=3, gold_action="ask",
                    direct_answer=None,
                    valid_interpretations=[{"interpretation": f"the {sides[i]} {cat}", "answer": f"referent {i + 1}",
                                            "probability": round(1 / len(real), 3)} for i in range(len(real))],
                    clarifications=[{"clarifying_question":
                                     f"Which {cat} do you mean, the one on the {sides[0]} or the one on the {sides[-1]}?",
                                     "clarification_response": f"The {cat} on the {sides[0]}.",
                                     "disambiguated_answer": "referent 1", "interpretation_id": 0}])
    else:
        item.update(is_ambiguous=True, ambiguity_type="resolution", difficulty=3, gold_action="abstain",
                    direct_answer=None, valid_interpretations=[], clarifications=[])
    return item


def build(index, per_cat=30, seed=SEED):
    rng = random.Random(seed)
    cats, images, pools = index["categories"], index["images"], index["pools"]
    name2id = {v: k for k, v in cats.items()}
    cat_ids = [name2id[n] for n in CATEGORIES if n in name2id]
    items, pair = [], [0]

    def add(image_id, family, action, basis, real, cid, template, answer=None, pid=None):
        it = make_item(len(items), image_id, images[image_id], cats[cid], cid, family, action, basis, real,
                       template, answer)
        it["pair_id"] = pid
        items.append(it)

    def new_pair():
        pair[0] += 1
        return f"P{pair[0] - 1:06d}", rng.randrange(3)

    for family, kind in (("referential_spatial", "spatial"), ("referential_color", "color")):
        for cid in cat_ids:
            imgs = [str(i) for i in pools[cid]["one"] + pools[cid]["many"]]
            rng.shuffle(imgs)
            buckets = defaultdict(list)
            for iid in imgs[:600]:
                im = images[iid]
                real, A = referents(im["cats"][cid], im["width"], im["height"])
                act, basis = referential_rule(real, A, im["width"], im["height"], kind)
                if act is not None:
                    buckets[basis].append((iid, real))
            for (ia, ra), (ib, rb) in zip(buckets["multi_comparable"][:per_cat], buckets["multi_dominant"][:per_cat]):
                pid, t = new_pair()
                add(ia, family, "ask", "multi_comparable", ra, cid, t, pid=pid)
                dom = side_of(rb[0]["bbox"], images[ib]["width"]) if kind == "spatial" else None
                add(ib, family, "answer", "multi_dominant", rb, cid, t, answer=dom, pid=pid)
            if kind == "color":
                for (ia, ra), (ib, rb) in zip(buckets["single_clear"][:per_cat], buckets["single_occluded"][:per_cat]):
                    pid, t = new_pair()
                    add(ia, family, "answer", "single_clear", ra, cid, t, pid=pid)
                    add(ib, family, "abstain", "single_occluded", rb, cid, t, pid=pid)
            else:
                for (ia, ra), (ib, rb) in zip(buckets["single_clear"][:per_cat],
                                              buckets["multi_comparable"][per_cat:2 * per_cat]):
                    pid, t = new_pair()
                    add(ia, family, "answer", "single_clear", ra, cid, t,
                        answer=side_of(ra[0]["bbox"], images[ia]["width"]), pid=pid)
                    add(ib, family, "ask", "multi_comparable", rb, cid, t, pid=pid)

    for cid in cat_ids:
        distinct = [str(i) for i in pools[cid]["many"]]
        crowd = [str(i) for i in pools[cid]["crowd"]]
        rng.shuffle(distinct)
        rng.shuffle(crowd)
        for j in range(min(per_cat // 2, len(crowd), len(distinct))):
            pid, t = new_pair()
            im = images[distinct[j]]
            real, _ = referents(im["cats"][cid], im["width"], im["height"])
            add(distinct[j], "counting", "answer", "distinct", real, cid, t, answer=len(real), pid=pid)
            imc = images[crowd[j]]
            realc, _ = referents(imc["cats"][cid], imc["width"], imc["height"])
            add(crowd[j], "counting", "abstain", "crowd", realc or imc["cats"][cid], cid, t, pid=pid)

    all_imgs = list(images)
    for cid in cat_ids:
        present = [str(i) for i in pools[cid]["one"] + pools[cid]["many"]]
        rng.shuffle(present)
        for j in range(min(per_cat // 3, len(present))):
            pid, t = new_pair()
            im = images[present[j]]
            real, _ = referents(im["cats"][cid], im["width"], im["height"])
            add(present[j], "presence", "answer", "present", real or im["cats"][cid], cid, t, answer="yes", pid=pid)
            for _ in range(25):
                cand = rng.choice(all_imgs)
                if cid not in images[cand]["cats"]:
                    add(cand, "presence", "answer", "absent", [], cid, t, answer="no", pid=pid)
                    break
    rng.shuffle(items)
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default="construction/coco_index.json")
    ap.add_argument("--out", default="construction/items_raw.jsonl")
    ap.add_argument("--per-cat", type=int, default=30, help="target pairs per category and family")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    items = build(json.load(open(args.index)), args.per_cat, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"  {len(items)} items, {len({i['pair_id'] for i in items})} pairs -> {args.out}")
    print(f"  families: {dict(Counter(i['ambiguity_subtype'] for i in items))}")
    print(f"  decision bases: {dict(Counter(i['decision_basis'] for i in items))}")


if __name__ == "__main__":
    main()
