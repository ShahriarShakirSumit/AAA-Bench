"""Merge the files returned by annotators into one responses sheet.

Each annotator using human_study/kit/annotate.html downloads a CSV with the columns
item_id, action, ambiguous. This combines several such files, in the order given, into a sheet in the
format of human_study/annotator_responses.csv, which reproduce/compute_agreement.py reads.

    python tools/merge_annotations.py labels_a.csv labels_b.csv labels_c.csv --out my_responses.csv
    python reproduce/compute_agreement.py --responses my_responses.csv
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aaa_bench.data import read_jsonl  # noqa: E402
from aaa_bench.paths import HUMAN_STUDY  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="CSV files downloaded from the annotation tool")
    ap.add_argument("--sample", default=str(HUMAN_STUDY / "sample_items.jsonl"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = read_jsonl(args.sample)
    picks = [{r["item_id"]: r for r in csv.DictReader(open(f, encoding="utf-8"))} for f in args.files]
    cols = ["item_id", "image", "question", "options"]
    for n in range(1, len(picks) + 1):
        cols += [f"annot{n}_ambiguous(yes/no)", f"annot{n}_action(answer/ask/abstain)"]
    complete = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for it in items:
            row = {"item_id": it["item_id"], "image": it["image_path"].split("/")[-1], "question": it["question"],
                   "options": "answer | ask | abstain"}
            for n, d in enumerate(picks, 1):
                act = (d.get(it["item_id"], {}).get("action") or "").strip()
                row[f"annot{n}_action(answer/ask/abstain)"] = act
                row[f"annot{n}_ambiguous(yes/no)"] = "no" if act == "answer" else ("yes" if act else "")
            complete += all(row[f"annot{n}_action(answer/ask/abstain)"] for n in range(1, len(picks) + 1))
            w.writerow(row)
    for path, d in zip(args.files, picks):
        print(f"  {path}: {sum(1 for r in d.values() if r.get('action'))} items labelled")
    print(f"  {complete} of {len(items)} items labelled by every annotator -> {args.out}")


if __name__ == "__main__":
    main()
