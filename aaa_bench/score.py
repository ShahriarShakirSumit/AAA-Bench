"""Score response files against the gold actions.

    python -m aaa_bench.score model_outputs/qwen2.5-vl-7b_option.jsonl
    python -m aaa_bench.score runs/*.jsonl --json scores.json

A response file has one JSON object per line with at least "item_id" and "action"
("answer", "ask" or "abstain"). Files written by `python -m aaa_bench.evaluate` have this form.
"""

import argparse
import json
import sys
from pathlib import Path

from .data import by_id, load_split
from .metrics import summary
from .outputs import read_run
from .paths import DATA


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="response files (JSONL)")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--data-dir", default=str(DATA))
    ap.add_argument("--json", help="also write the scores to this file")
    args = ap.parse_args(argv)

    gold = by_id(load_split(args.split, args.data_dir))
    rows = {}
    for f in args.files:
        preds = read_run(f)
        bad = {r.get("action") for r in preds.values()} - {"answer", "ask", "abstain"}
        if bad:
            sys.exit(f"{f}: unknown action values {sorted(map(str, bad))}")
        rows[Path(f).stem] = summary(gold, preds)

    print(f"{'run':38s} {'scored':>7s} {'action':>7s} {'amb-F1':>7s} {'CA':>6s} {'percep':>7s} {'decis':>6s}")
    for name, s in rows.items():
        print(f"{name:38s} {s['n_scored']:7d} {s['action_acc']:7.3f} {s['amb_f1']:7.3f} {s['contrastive']:6.3f} "
              f"{s['perception']:7.3f} {s['decision']:6.3f}")
    print(f"\n{len(gold)} {args.split} items. CA = contrastive accuracy over counterfactual pairs; "
          f"a constant policy scores 0.")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
