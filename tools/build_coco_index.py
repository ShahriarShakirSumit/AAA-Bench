"""Index the COCO 2017 instance annotations by image and category.

Reads instances_val2017.json and instances_train2017.json and writes one JSON file with, per image,
its size and the boxes, areas and crowd flags of every instance, and, per category, the images with
exactly one instance, with two or more, and with a crowd annotation. build_dataset.py and
verify_labels.py read this index.

    python tools/build_coco_index.py --annotations /data/coco/annotations --out construction/coco_index.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def build(annotation_dir):
    categories, images = {}, {}
    for split in ("val", "train"):
        path = Path(annotation_dir) / f"instances_{split}2017.json"
        print(f"  reading {path}")
        d = json.load(open(path))
        for c in d["categories"]:
            categories[c["id"]] = c["name"]
        for im in d["images"]:
            images[im["id"]] = {"file_name": im["file_name"], "width": im["width"], "height": im["height"],
                                "split": split, "cats": defaultdict(list)}
        for a in d["annotations"]:
            if a["image_id"] in images:
                images[a["image_id"]]["cats"][a["category_id"]].append(
                    {"bbox": [round(x, 1) for x in a["bbox"]], "area": round(a["area"], 1),
                     "iscrowd": int(a.get("iscrowd", 0))})
    pools = defaultdict(lambda: {"one": [], "many": [], "crowd": []})
    for iid, im in images.items():
        for cat, insts in im["cats"].items():
            n = sum(1 for x in insts if not x["iscrowd"])
            has_crowd = any(x["iscrowd"] for x in insts)
            if n == 1 and not has_crowd:
                pools[cat]["one"].append(iid)
            elif n >= 2:
                pools[cat]["many"].append(iid)
            if has_crowd:
                pools[cat]["crowd"].append(iid)
    for im in images.values():
        im["cats"] = {str(k): v for k, v in im["cats"].items()}
    return {"categories": {str(k): v for k, v in categories.items()},
            "pools": {str(k): v for k, v in pools.items()},
            "images": {str(k): v for k, v in images.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True, help="folder with instances_{train,val}2017.json")
    ap.add_argument("--out", default="construction/coco_index.json")
    args = ap.parse_args()
    index = build(args.annotations)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(index, open(args.out, "w"))
    print(f"  {len(index['images'])} images, {len(index['categories'])} categories -> {args.out}")


if __name__ == "__main__":
    main()
