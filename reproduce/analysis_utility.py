"""Cost-robust utility of each policy on the test split.

The win-rate of a policy M is Pr_C[u(M; C) > u(always-answer; C)] over cost matrices C drawn
uniformly from the admissible region (c_correct = 1 > 0 > c_ask, c_abstain > c_wrong, with c_wrong
in [-8, -0.1]). It compares every policy against one named reference, so adding or removing a
competitor cannot change it. A second sweep varies the cost of asking relative to the cost of a
wrong answer and compares the best model with always-ask. Writes results/tables/tab_utility.tex and
results/figures/utility_sweep.{pdf,png}.
"""

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aaa_bench.data import by_id, load_split  # noqa: E402
from aaa_bench.metrics import utility  # noqa: E402
from aaa_bench.outputs import label, load_runs  # noqa: E402
from aaa_bench.paths import RESULTS  # noqa: E402

REF = "Always \\ans{}"
NDRAW, SEED = 4000, 20260809
BLUE, ORANGE, GREY, INK = "#0072B2", "#D55E00", "#9aa4b0", "#1a1a1a"


def admissible(rng):
    cw = rng.uniform(-8.0, -0.1)
    return (1.0, cw, rng.uniform(cw, 0.0), rng.uniform(cw, 0.0))


def utilities(gold, policies, grid):
    out = {p: [] for p in policies}
    for (rc, cw, ca, cab) in grid:
        for p, pr in policies.items():
            out[p].append(utility(gold, pr, rc, cw, ca, cab))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()
    rng = random.Random(SEED)
    gold = by_id(load_split("test"))
    policies = {label(m): run for m, run in load_runs("option", gold).items()}
    models = set(policies)
    for name, a in ((REF, "answer"), ("Always \\ask{}", "ask"), ("Always \\abs{}", "abstain")):
        policies[name] = {i: {"item_id": i, "action": a} for i in gold}
    policies["Oracle"] = {i: {"item_id": i, "action": g["gold_action"]} for i, g in gold.items()}

    grid = [admissible(rng) for _ in range(NDRAW)]
    us = utilities(gold, policies, grid)
    rows = sorted(((p, sum(a > b for a, b in zip(v, us[REF])) / len(v), sum(v) / len(v)) for p, v in us.items()),
                  key=lambda r: -r[2])
    lines = [r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lrr}", r"\toprule",
             r"Policy & Win-rate vs.\ \ans{}-all & Mean utility \\", r"\midrule"]
    sep = False
    for p, wr, mu in rows:
        if p in models and not sep:
            lines.append(r"\midrule")
            sep = True
        lab = p if p in models else rf"\textit{{{p}}}"
        lines.append(f"{lab} & {wr:.2f} & ${mu:+.2f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables" / "tab_utility.tex").write_text("\n".join(lines) + "\n")
    for p, wr, mu in rows:
        print(f"    {p.replace(chr(92), ''):18s} win-rate vs always-answer={wr:.2f}  mean utility={mu:+.3f}")

    best = max(models, key=lambda m: sum(us[m]) / len(us[m]))
    ratios = [r / 40 for r in range(1, 41)]
    mu_model, mu_ask = [], []
    for rr in ratios:
        g = []
        for _ in range(600):
            cw = rng.uniform(-8.0, -0.1)
            g.append((1.0, cw, rr * cw, rng.uniform(cw, 0.0)))
        u = utilities(gold, {best: policies[best], "ask": policies["Always \\ask{}"]}, g)
        mu_model.append(sum(u[best]) / len(u[best]))
        mu_ask.append(sum(u["ask"]) / len(u["ask"]))

    fig, ax = plt.subplots(figsize=(3.35, 2.0))
    ax.plot(ratios, mu_model, color=BLUE, lw=1.6, label=f"{best} (best model)")
    ax.plot(ratios, mu_ask, color=ORANGE, lw=1.6, ls="--", label="always ask")
    cross = next((r for r, a, b in zip(ratios, mu_model, mu_ask) if a > b), None)
    if cross:
        ax.axvline(cross, color=GREY, lw=1.0, ls=":")
        ax.text(cross + .02, ax.get_ylim()[0] + .1, f"crossover\n$\\approx{cross:.2f}$", fontsize=6.6, color=INK)
    ax.set_xlabel("$|c_{\\rm ask}| / |c_{\\rm wrong}|$")
    ax.set_ylabel("mean utility")
    ax.legend(frameon=False, fontsize=6.8, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("whether asking beats the model depends on\nthe cost of asking", fontsize=7.6, pad=4)
    fig.tight_layout(pad=.4)
    for e in ("pdf", "png"):
        fig.savefig(out / "figures" / f"utility_sweep.{e}", dpi=200)
    print(f"  wrote tab_utility.tex and utility_sweep.pdf (crossover at |c_ask|/|c_wrong| = {cross})")


if __name__ == "__main__":
    main()
