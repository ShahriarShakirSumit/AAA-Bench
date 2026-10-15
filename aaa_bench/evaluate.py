"""Run a model on AAA-Bench and save its responses.

    python -m aaa_bench.evaluate --model qwen2.5-vl-7b --strategy option --coco-dir /data/coco

Writes runs/<model>_<strategy>.jsonl, one response per line, in the format of model_outputs/.
An interrupted run resumes where it stopped. Score the file with `python -m aaa_bench.score`.
"""

import argparse
import json
import logging
import time
from pathlib import Path

from .data import load_split
from .models import MODELS, get_model
from .paths import DATA
from .prompts import STRATEGIES
from .protocol import run_item

log = logging.getLogger("aaa_bench")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--strategy", default="option", choices=STRATEGIES)
    ap.add_argument("--coco-dir", required=True, help="folder that holds train2017/ and val2017/")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--data-dir", default=str(DATA))
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--limit", type=int, help="evaluate only the first N items of the split")
    ap.add_argument("--revision", help="model commit to load (default: the one used in the paper)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    items = load_split(args.split, args.data_dir)
    if args.limit:
        items = items[: args.limit]
    missing = [it["image_path"] for it in items if not (Path(args.coco_dir) / it["image_path"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images not found under {args.coco_dir}, e.g. {missing[0]}. "
                         f"See tools/fetch_images.py.")

    out = Path(args.out_dir) / f"{args.model}_{args.strategy}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            done = {json.loads(line)["item_id"] for line in f if line.strip()}
        log.info("resuming: %d items already in %s", len(done), out)

    from . import compat
    compat.apply()
    model = get_model(args.model, revision=args.revision)
    model.load_model()
    log.info("loaded %s (%s @ %s)", args.model, model.hf_id, model.revision[:12])

    remaining = [it for it in items if it["item_id"] not in done]
    start = time.time()
    for n, item in enumerate(remaining, 1):
        try:
            record = run_item(model, item, args.strategy, args.coco_dir)
        except Exception:
            # An empty response or a failed generation is skipped and logged, never scored as an answer.
            log.exception("item %s failed", item["item_id"])
            continue
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
        if n % 50 == 0 or n == len(remaining):
            rate = n / max(time.time() - start, 1e-6)
            log.info("%d/%d items, %.2f items/s", len(done) + n, len(items), rate)
    log.info("responses in %s", out)


if __name__ == "__main__":
    main()
