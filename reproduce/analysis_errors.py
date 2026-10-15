"""Where the option-prompt errors are.

  (a) action accuracy per decision basis, for each model (results/tables/tab_family.tex and
      results/figures/errors.{pdf,png})
  (b) the confusion between gold and predicted actions, pooled over the seven models
      (results/figures/confusion.{pdf,png})
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from aaa_bench.data import by_id, load_split  # noqa: E402
from aaa_bench.outputs import label, load_runs  # noqa: E402
from aaa_bench.paths import RESULTS  # noqa: E402

BASIS_LABEL = {"single_clear": "one clear", "multi_dominant": "one dominant", "multi_comparable": "several comparable",
               "distinct": "countable", "crowd": "crowd", "present": "present", "absent": "absent"}
BASIS_HEAD = {"single_clear": "Clear", "multi_dominant": "Dom.", "multi_comparable": "Comp.",
              "distinct": "Cnt.", "crowd": "Crowd", "present": "Pres.", "absent": "Abs."}
ACTIONS = ["answer", "ask", "abstain"]
plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#4a4a4a", "axes.linewidth": .8})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()
    gold = by_id(load_split("test"))
    preds = {m: {i: r["action"] for i, r in run.items()} for m, run in load_runs("option", gold).items()}
    bases = [b for b in BASIS_LABEL if any(g.get("decision_basis") == b for g in gold.values())]
    models = sorted(preds, key=lambda m: -sum(preds[m].get(i) == g["gold_action"] for i, g in gold.items() if i in preds[m]))

    lines = [r"\setlength{\tabcolsep}{2.5pt}", r"\footnotesize", r"\begin{tabular}{l" + "r" * len(bases) + "}",
             r"\toprule", "Model & " + " & ".join(BASIS_HEAD[b] for b in bases) + r" \\", r"\midrule"]
    grid = np.zeros((len(models), len(bases)))
    for i, m in enumerate(models):
        cells = []
        for j, b in enumerate(bases):
            ids = [k for k, g in gold.items() if g.get("decision_basis") == b and k in preds[m]]
            a = sum(preds[m][k] == gold[k]["gold_action"] for k in ids) / len(ids) if ids else 0.0
            grid[i, j] = a
            cells.append(f"{a:.2f}")
        lines.append(f"{label(m)} & " + " & ".join(cells) + r" \\")
    counts = [str(sum(1 for g in gold.values() if g.get("decision_basis") == b)) for b in bases]
    lines += [r"\midrule", "Items & " + " & ".join(counts) + r" \\", r"\bottomrule", r"\end{tabular}"]
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables" / "tab_family.tex").write_text("\n".join(lines) + "\n")
    for j, b in enumerate(bases):
        print(f"    {BASIS_LABEL[b]:20s} mean accuracy over models = {grid[:, j].mean():.3f}")

    conf = np.zeros((3, 3))
    for m, p in preds.items():
        for k, g in gold.items():
            if k in p and p[k] in ACTIONS:
                conf[ACTIONS.index(g["gold_action"]), ACTIONS.index(p[k])] += 1
    conf = conf / conf.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(bases)))
    ax.set_xticklabels([BASIS_LABEL[b] for b in bases], rotation=32, ha="right", fontsize=7)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([label(m) for m in models], fontsize=7.2)
    for i in range(len(models)):
        for j in range(len(bases)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=6.2,
                    color="black" if 0.25 < grid[i, j] < 0.85 else "white")
    fig.colorbar(im, ax=ax, fraction=.03, pad=.02)
    fig.tight_layout(pad=.4)
    for e in ("pdf", "png"):
        fig.savefig(out / "figures" / f"errors.{e}", dpi=200)

    fig2, ax = plt.subplots(figsize=(3.35, 1.85))
    ax.imshow(conf, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(ACTIONS, fontsize=7.2)
    ax.set_yticks(range(3))
    ax.set_yticklabels(ACTIONS, fontsize=7.2)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{conf[i, j]:.2f}", ha="center", va="center", fontsize=7.4,
                    color="white" if conf[i, j] > .55 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    fig2.tight_layout(pad=.4)
    for e in ("pdf", "png"):
        fig2.savefig(out / "figures" / f"confusion.{e}", dpi=200)
    for i, a in enumerate(ACTIONS):
        print(f"    gold={a:8s} -> " + ", ".join(f"{ACTIONS[j]} {conf[i, j]:.2f}" for j in range(3)))
    (out / "errors.json").write_text(json.dumps(
        {"confusion_pooled": {g: {p: conf[i, j] for j, p in enumerate(ACTIONS)} for i, g in enumerate(ACTIONS)},
         "accuracy_by_basis": {m: {b: grid[i, j] for j, b in enumerate(bases)} for i, m in enumerate(models)}},
        indent=2) + "\n")
    print("  wrote tab_family.tex, errors.pdf and confusion.pdf")


if __name__ == "__main__":
    main()
