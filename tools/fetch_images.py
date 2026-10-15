"""Get the COCO 2017 images that AAA-Bench items refer to.

The benchmark does not redistribute images. Each item names a file such as
train2017/000000384993.jpg, and this script puts those files under one folder (default: coco/)
with the same layout, either by linking them from a COCO 2017 copy you already have, or by
downloading them one at a time from images.cocodataset.org.

    python tools/fetch_images.py --split test                       # download the 1,424 test images
    python tools/fetch_images.py --split all --from-dir /data/coco  # link from a local COCO copy
    python tools/fetch_images.py --human-study                      # images for the annotation tool

Then pass --coco-dir coco to aaa_bench.evaluate. Files already present are skipped.
"""

import argparse
import os
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aaa_bench.data import SPLITS, load_split, read_jsonl  # noqa: E402
from aaa_bench.paths import HUMAN_STUDY  # noqa: E402

URL = "http://images.cocodataset.org/{}"


def wanted(split, human_study):
    if human_study:
        return sorted({r["image_path"] for r in read_jsonl(HUMAN_STUDY / "sample_items.jsonl")})
    splits = SPLITS if split == "all" else (split,)
    return sorted({r["image_path"] for s in splits for r in load_split(s)})


def fetch(rel, out_dir, from_dir, retries=3):
    dst = out_dir / rel
    if dst.exists() and dst.stat().st_size > 0:
        return "present"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if from_dir:
        src = Path(from_dir) / rel
        if not src.exists():
            return "missing"
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)
        return "linked"
    for attempt in range(retries):
        try:
            tmp = dst.with_suffix(".part")
            with urllib.request.urlopen(URL.format(rel), timeout=60) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            tmp.rename(dst)
            return "downloaded"
        except OSError:
            time.sleep(2 * (attempt + 1))
    return "failed"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=(*SPLITS, "all"))
    ap.add_argument("--human-study", action="store_true",
                    help="fetch the 399 images of the human study and copy them into human_study/kit/images/")
    ap.add_argument("--out", default="coco", help="destination folder (gets train2017/ and val2017/)")
    ap.add_argument("--from-dir", help="a local COCO 2017 folder holding train2017/ and val2017/")
    ap.add_argument("--limit", type=int, help="only the first N images (for a quick check)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    files = wanted(args.split, args.human_study)[: args.limit]
    out_dir = Path(args.out)
    counts = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for status in pool.map(lambda rel: fetch(rel, out_dir, args.from_dir), files):
            counts[status] = counts.get(status, 0) + 1
    print(f"{len(files)} images under {out_dir}/: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))

    if args.human_study:
        kit = HUMAN_STUDY / "kit" / "images"
        kit.mkdir(parents=True, exist_ok=True)
        for rel in files:
            src = out_dir / rel
            if src.exists():
                shutil.copy2(src, kit / Path(rel).name)
        print(f"copied the study images into {kit}/; open human_study/kit/annotate.html in a browser")
    if counts.get("failed") or counts.get("missing"):
        sys.exit(1)


if __name__ == "__main__":
    main()
