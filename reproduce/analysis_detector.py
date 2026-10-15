"""Two image-only pipelines that show the task is solvable without gold annotations.

Both read the detections of a COCO-pretrained detector (YOLO11-L, confidence 0.25) for the queried
category:
  detector + rules  the benchmark's own derivation rules (D = 3, K = 0.5, crowd >= 10) applied to the
                    detections instead of the annotations
  learned head      a multinomial logistic regression over detection features, fitted on the train
                    split and scored on the test split
The detections are cached in model_outputs/reference/detector_features.json, so this runs on a CPU
(the learned head needs scikit-learn; version 1.9.1 gives the paper's numbers, and older versions move
them in the third decimal). To recompute the detections, pass --weights yolo11l.pt and --coco-dir;
that needs ultralytics and a GPU.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from aaa_bench.data import load_split
from aaa_bench.paths import REFERENCE_OUTPUTS, RESULTS

D_RATIO, K_RATIO, CROWD_N = 3.0, 0.5, 10
FAMILIES = ["referential_color", "referential_spatial", "counting", "presence"]
ACTIONS = ["answer", "ask", "abstain"]


def featurise(family, areas, img_area):
    a0 = areas[0] if areas else 0.0
    a1 = areas[1] if len(areas) > 1 else 0.0
    return ([float(len(areas)), min(len(areas), 10) / 10.0, a0 / img_area, a1 / img_area,
             (a1 / a0) if a0 > 0 else 0.0, sum(areas) / img_area,
             float(len(areas) == 0), float(len(areas) == 1), float(len(areas) >= 10)]
            + [float(family == f) for f in FAMILIES])


def decide(family, f):
    """The derivation rules applied to detections (f is the feature vector of featurise)."""
    n, a0, a1 = int(f[0]), f[2], f[3]
    if family == "presence":
        return "answer" if n > 0 else "abstain"
    if family == "counting":
        return "abstain" if n >= CROWD_N else "answer"
    if n == 0:
        return "abstain"
    if n == 1:
        return "answer"
    if a0 >= D_RATIO * a1:
        return "answer"
    if a1 >= K_RATIO * a0:
        return "ask"
    return "answer"


def detect(weights, coco_dir, conf=0.25):
    """Detection features for the train and test items (GPU)."""
    from ultralytics import YOLO
    model = YOLO(weights)
    names = {v: k for k, v in model.names.items()}
    feats = {}
    for split in ("train", "test"):
        rows = load_split(split)
        by_img = defaultdict(list)
        for r in rows:
            by_img[r["image_path"]].append(r)
        out, paths = {}, sorted(by_img)
        for i in range(0, len(paths), 64):
            batch = paths[i:i + 64]
            for p, res in zip(batch, model.predict([str(Path(coco_dir) / p) for p in batch], conf=conf,
                                                   verbose=False, device=0)):
                h, w = res.orig_shape
                per_cat = defaultdict(list)
                if res.boxes is not None:
                    for c, (_, _, bw, bh) in zip(res.boxes.cls.tolist(), res.boxes.xywh.tolist()):
                        per_cat[int(c)].append(bw * bh)
                for item in by_img[p]:
                    cid = names.get(item["visual_evidence"]["category"])
                    areas = sorted(per_cat.get(cid, []), reverse=True)
                    out[item["item_id"]] = featurise(item["ambiguity_subtype"], areas, float(h * w))
        feats[split] = out
    return feats


def scores(test, pred):
    gold = {r["item_id"]: r for r in test}
    acc = sum(pred[i] == g["gold_action"] for i, g in gold.items()) / len(gold)
    groups = defaultdict(list)
    for r in test:
        groups[r["pair_id"]].append(r["item_id"])
    pairs = [v for v in groups.values() if len(v) == 2]
    ca = sum(all(pred[k] == gold[k]["gold_action"] for k in p) for p in pairs) / len(pairs)
    return acc, ca


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default=str(REFERENCE_OUTPUTS / "detector_features.json"))
    ap.add_argument("--weights", help="recompute the detections with these YOLO weights (GPU)")
    ap.add_argument("--coco-dir", help="COCO 2017 folder, needed with --weights")
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()
    feats = detect(args.weights, args.coco_dir) if args.weights else json.load(open(args.features))
    train, test = load_split("train"), load_split("test")

    rules = {r["item_id"]: decide(r["ambiguity_subtype"], feats["test"][r["item_id"]]) for r in test}
    acc, ca = scores(test, rules)
    by_family = defaultdict(list)
    for r in test:
        by_family[r["ambiguity_subtype"]].append(int(rules[r["item_id"]] == r["gold_action"]))
    per_family = {f: sum(v) / len(v) for f, v in sorted(by_family.items())}
    print(f"  detector + rules: action accuracy {acc:.3f}, contrastive accuracy {ca:.3f}")
    print("    by family: " + ", ".join(f"{f} {v:.2f}" for f, v in per_family.items()))
    summary = {"detector_rules": {"action_acc": acc, "contrastive": ca, "per_family": per_family}}

    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("  learned head skipped: install scikit-learn to fit it")
    else:
        X = np.array([feats["train"][r["item_id"]] for r in train])
        y = np.array([ACTIONS.index(r["gold_action"]) for r in train])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(X, y)
        head = dict(zip([r["item_id"] for r in test],
                        (ACTIONS[k] for k in clf.predict(np.array([feats["test"][r["item_id"]] for r in test])))))
        h_acc, h_ca = scores(test, head)
        agree = sum(head[i] == rules[i] for i in rules) / len(rules)
        print(f"  learned head:     action accuracy {h_acc:.3f}, contrastive accuracy {h_ca:.3f}")
        print(f"  the two pipelines agree on {agree:.1%} of test items")
        summary["learned_head"] = {"action_acc": h_acc, "contrastive": h_ca, "agreement_with_rules": agree}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "detector_pipelines.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  wrote {out / 'detector_pipelines.json'}")


if __name__ == "__main__":
    main()
