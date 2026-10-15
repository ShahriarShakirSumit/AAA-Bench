"""Reference policies for contrastive accuracy (CA) on the test split.

A constant policy scores 0. A policy that reads only the question can still know which non-answer
action a family uses (referential pairs are answer/ask, counting and presence pairs answer/abstain),
so guessing the remaining answer-vs-other bit at random reaches 0.5 per item and 0.25 per pair.
This measures:
  constant policies (answer / ask / abstain), uniform random over the three actions,
  text-only random (knows the family's alternative, flips a fair coin),
  text-only prior (knows the family and the train-split answer rate),
  the best of the seven models, the detector-plus-rules pipeline, and the human majority.
Randomised rows are the mean over 400 draws with half the 95% range. Writes
results/tables/tab_baselines.tex.
"""

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from aaa_bench.data import load_split
from aaa_bench.outputs import load_runs, read_run
from aaa_bench.paths import HUMAN_STUDY, REFERENCE_OUTPUTS, RESULTS
from analysis_human_ceiling import human_reference
from compute_agreement import load_study

REPS, SEED = 400, 20260809


def contrastive(pairs, action_of):
    return sum(all(action_of(r) == r["gold_action"] for r in m) for m in pairs) / len(pairs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()
    rng = random.Random(SEED)
    test = load_split("test")
    groups = defaultdict(list)
    for r in test:
        groups[r["pair_id"]].append(r)
    pairs = [m for m in groups.values() if len(m) == 2]

    train = load_split("train")
    alt = {}
    for f in {r.get("ambiguity_subtype") for r in train}:
        c = Counter(r["gold_action"] for r in train if r.get("ambiguity_subtype") == f and r["gold_action"] != "answer")
        alt[f] = c.most_common(1)[0][0] if c else "ask"
    p_answer = sum(r["gold_action"] == "answer" for r in train) / len(train)
    print(f"  alternative action per family (train split): {alt}")
    print(f"  P(answer) in the train split: {p_answer:.3f}")

    rows = []
    for name, act in (("Always \\ans{}", "answer"), ("Always \\ask{}", "ask"), ("Always \\abs{}", "abstain")):
        rows.append((name, contrastive(pairs, lambda r, a=act: a), 0.0))

    def rand_mean(fn):
        vals = sorted(contrastive(pairs, fn) for _ in range(REPS))
        return sum(vals) / len(vals), (vals[int(.975 * REPS)] - vals[int(.025 * REPS)]) / 2

    rows.append(("Uniform random (3 actions)", *rand_mean(lambda r: rng.choice(["answer", "ask", "abstain"]))))
    rows.append(("\\textbf{Text-only random} (knows family)",
                 *rand_mean(lambda r: rng.choice(["answer", alt.get(r.get("ambiguity_subtype"), "ask")]))))
    rows.append(("Text-only prior (family $+$ marginal)",
                 *rand_mean(lambda r: "answer" if rng.random() < p_answer else alt.get(r.get("ambiguity_subtype"), "ask"))))

    best = None
    for model, run in load_runs("option").items():
        p = {i: r["action"] for i, r in run.items()}
        ok = [m for m in pairs if all(x["item_id"] in p for x in m)]
        if len(ok) < 0.9 * len(pairs):
            continue
        ca = contrastive(ok, lambda r: p[r["item_id"]])
        if best is None or ca > best[1]:
            best = (model, ca)
    rows.append((f"\\emph{{Best VLM}} ({best[0]})", best[1], 0.0))

    det = {i: r["action"] for i, r in read_run(REFERENCE_OUTPUTS / "detector-rules.jsonl").items()}
    rows.append(("\\textbf{Detector $+$ rules}", contrastive(pairs, lambda r: det[r["item_id"]]), 0.0))

    items, votes, _ = load_study(HUMAN_STUDY / "annotator_responses.csv", HUMAN_STUDY / "sample_items.jsonl")
    _, h, (lo, hi), n = human_reference(items, votes)[2]
    rows.append((f"\\emph{{Human majority}} ({n} pairs)", h, (hi - lo) / 2))

    lines = [r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lc}", r"\toprule",
             r"Policy & Contrastive acc. \\", r"\midrule"]
    for name, v, h in rows:
        lines.append(f"{name} & {v:.3f}" + (f"$\\pm${h:.3f}" if h else "") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = Path(args.out) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tab_baselines.tex").write_text("\n".join(lines) + "\n")
    for name, v, h in rows:
        print(f"    {name:44s} CA={v:.3f}" + (f" +-{h:.3f}" if h else ""))
    print(f"  wrote {out / 'tab_baselines.tex'}")


if __name__ == "__main__":
    main()
