"""Human reference points for reading model scores, from the K=3 study.

  self-consistency  one annotator against the majority of the other two, on items where those two
                    agree
  action accuracy   the human majority against the label after the study's verdict
  contrastive       the human majority on study pairs whose two members were both annotated, scored
                    like a model (both members must match)
Each value carries a percentile bootstrap interval. Writes results/tables/tab_human.tex and
results/human_reference.json.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from aaa_bench.paths import HUMAN_STUDY, RESULTS
from compute_agreement import after_study, load_study, majority

B, SEED = 2000, 20260809


def boot(vals, rng):
    n = len(vals)
    if n == 0:
        return (0.0, 0.0)
    d = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return d[int(.025 * B)], d[int(.975 * B)]


def reference_points(items, votes):
    sc = []
    for v in votes.values():
        for i in range(3):
            others = Counter(v[:i] + v[i + 1:])
            if len(others) == 1:
                sc.append(int(v[i] == others.most_common(1)[0][0]))

    labelled = {i: after_study(items[i]) for i in votes}
    labelled = {i: a for i, a in labelled.items() if a is not None}
    maj = {i: majority(votes[i]) for i in labelled}
    acc = [int(maj[i] == labelled[i]) for i in labelled]

    by_pair = defaultdict(list)
    for i in labelled:
        by_pair[items[i]["pair_id"]].append(i)
    pairs = [p for p in by_pair.values() if len(p) == 2]
    ca = [int(all(maj[i] == labelled[i] for i in p)) for p in pairs]
    return [("Self-consistency (annotator vs.\\ other two)", sc),
            ("Action accuracy (majority vs.\\ gold)", acc),
            ("Contrastive accuracy (majority vs.\\ gold)", ca)]


def human_reference(items, votes):
    """[(row label, value, (lo, hi), n)] with the seeded bootstrap intervals of the paper."""
    rng = random.Random(SEED)
    out = []
    for label, vals in reference_points(items, votes):
        out.append((label, sum(vals) / len(vals), boot(vals, rng), len(vals)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses", default=str(HUMAN_STUDY / "annotator_responses.csv"))
    ap.add_argument("--sample", default=str(HUMAN_STUDY / "sample_items.jsonl"))
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()
    items, votes, _ = load_study(args.responses, args.sample)

    lines = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lcc}", r"\toprule",
             r"Human reference & Value $\pm$ CI & $n$ \\", r"\midrule"]
    summary = {}
    for label, v, (lo, hi), n in human_reference(items, votes):
        lines.append(f"{label} & {v:.3f}$\\pm${(hi - lo) / 2:.3f} & {n:,} \\\\".replace(",", "{,}"))
        summary[label.split(" (")[0].lower()] = dict(value=v, ci=[lo, hi], half_width=(hi - lo) / 2, n=n)
        print(f"  {label.replace(chr(92), ''):46s} {v:.3f} [{lo:.3f}, {hi:.3f}]  n={n}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "tables" / "tab_human.tex").write_text("\n".join(lines) + "\n")
    (out / "human_reference.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  wrote {out / 'tables' / 'tab_human.tex'}")


if __name__ == "__main__":
    main()
