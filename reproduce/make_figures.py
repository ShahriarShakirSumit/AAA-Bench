"""The data figures of the paper, from the saved option-prompt and chain-of-thought runs.

  results.{pdf,png}     the seven models and, as the top two rows, the two image-only reference
                        pipelines of analysis_detector.py.
                        (a) contrastive accuracy with pair-bootstrap intervals, against the
                        text-only coin flip (0.251) and uniform random (0.111) of analysis_baselines.py;
                        (b) action accuracy against the 0.500 majority baseline; (c) contrastive
                        accuracy on answer/abstain pairs and on answer/ask pairs
  cot_effect.{pdf,png}  perception without and with chain-of-thought, per model
  theory.{pdf,png}      (a) the answer threshold tau of Theorem 1 against the wrong-answer penalty;
                        (b) how often each model answers, against the share of items the scene
                        licenses it to answer
"""

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aaa_bench.data import by_id, load_split  # noqa: E402
from aaa_bench.metrics import summary  # noqa: E402
from aaa_bench.outputs import MODEL_LABELS, label, load_runs, read_run  # noqa: E402
from aaa_bench.paths import REFERENCE_OUTPUTS, RESULTS  # noqa: E402

BLUE, ORANGE, GREEN, GREY, INK = "#0072B2", "#D55E00", "#009E73", "#9aa4b0", "#1a1a1a"
B, SEED = 2000, 20260809
TEXT_ONLY_COIN, UNIFORM_RANDOM = 0.251, 0.111
plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#4a4a4a", "axes.linewidth": 0.8,
                     "xtick.color": INK, "ytick.color": INK, "text.color": INK, "axes.labelcolor": INK,
                     "axes.spines.top": False, "axes.spines.right": False})


def ci(vals, rng):
    n = len(vals)
    d = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return d[int(.025 * B)], d[int(.975 * B)]


def fig_results(gold, preds, out):
    rng = random.Random(SEED)
    groups = defaultdict(list)
    for r in gold.values():
        groups[r["pair_id"]].append(r["item_id"])
    pairs = [v for v in groups.values() if len(v) == 2]
    hits = {}
    for m, p in preds.items():
        item = [int(p.get(i) == g["gold_action"]) for i, g in gold.items() if i in p]
        pair = [int(p.get(a) == gold[a]["gold_action"] and p.get(b) == gold[b]["gold_action"])
                for a, b in pairs if a in p and b in p]
        hits[m] = (item, pair)
    maj = max(Counter(r["gold_action"] for r in gold.values()).values()) / len(gold)
    order = sorted(hits, key=lambda m: -sum(hits[m][1]) / len(hits[m][1]))

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 1.48))
    ax = axes[0]
    for i, m in enumerate(order):
        v = sum(hits[m][1]) / len(hits[m][1])
        lo, hi = ci(hits[m][1], rng)
        ax.barh(i, v, color=(BLUE if v >= TEXT_ONLY_COIN else GREY), height=.62, zorder=3)
        ax.plot([lo, hi], [i, i], color=INK, lw=1.1, zorder=4)
    ax.axvline(TEXT_ONLY_COIN, color=ORANGE, lw=1.5, ls="--", zorder=5)
    ax.axvline(UNIFORM_RANDOM, color=GREY, lw=1.2, ls=":", zorder=5)
    ax.text(0.243, 1.9, "text-only\ncoin flip", fontsize=6.6, color=ORANGE, ha="right", va="center")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([label(m) for m in order], fontsize=7.2)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.80)
    ax.set_xlabel("contrastive accuracy")
    ax.set_title("(a) every VLM below the coin flip", fontsize=7.8, pad=4)

    ax = axes[1]
    for i, m in enumerate(order):
        v = sum(hits[m][0]) / len(hits[m][0])
        lo, hi = ci(hits[m][0], rng)
        ax.barh(i, v, color=GREEN if lo > maj else GREY, height=.62, zorder=3)
        ax.plot([lo, hi], [i, i], color=INK, lw=1.1, zorder=4)
    ax.axvline(maj, color=ORANGE, lw=1.4, ls="--", zorder=5)
    ax.text(0.97, 0.06, f"majority $={maj:.3f}$", transform=ax.transAxes, fontsize=6.8, color=ORANGE,
            ha="right", va="bottom")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlim(.40, .90)
    ax.set_xlabel("action accuracy")
    ax.set_title("(b) grey $=$ not above chance", fontsize=7.8, pad=4)

    ax = axes[2]
    kinds = defaultdict(list)
    for a, b in pairs:
        kinds[tuple(sorted({gold[a]["gold_action"], gold[b]["gold_action"]}))].append((a, b))

    def ca(p, kind):
        h = [int(p.get(a) == gold[a]["gold_action"] and p.get(b) == gold[b]["gold_action"])
             for a, b in kinds.get(kind, []) if a in p and b in p]
        return sum(h) / len(h) if h else 0.0

    ask = [ca(preds[m], ("answer", "ask")) for m in order]
    abstain = [ca(preds[m], ("abstain", "answer")) for m in order]
    idx = range(len(order))
    ax.barh([i - .19 for i in idx], abstain, height=.34, color=BLUE, label="answer/abstain", zorder=3)
    ax.barh([i + .19 for i in idx], ask, height=.34, color=ORANGE, label="answer/ask", zorder=3)
    ax.set_yticks(list(idx))
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlim(0, .90)
    ax.set_xlabel("contrastive accuracy")
    ax.legend(frameon=False, fontsize=6.6, loc="lower right")
    ax.set_title("(c) presence, not ambiguity", fontsize=7.8, pad=4)
    fig.tight_layout(pad=.45, w_pad=.9)
    for e in ("pdf", "png"):
        fig.savefig(out / f"results.{e}", dpi=200)


def fig_cot(gold, out):
    option = {m: summary(gold, p) for m, p in load_runs("option", gold).items()}
    cot = {m: summary(gold, p) for m, p in load_runs("cot", gold).items()}
    models = [m for m in MODEL_LABELS if m in option and m in cot]
    models.sort(key=lambda m: -option[m]["perception"])
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    for i, m in enumerate(models):
        a, b = option[m]["perception"], cot[m]["perception"]
        ax.annotate("", xy=(b, i), xytext=(a, i),
                    arrowprops=dict(arrowstyle="-|>", color=(ORANGE if b < a else GREEN), lw=1.5, shrinkA=0, shrinkB=0))
        ax.scatter([a], [i], s=18, color=GREY, zorder=3)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([label(m) for m in models], fontsize=7.2)
    ax.invert_yaxis()
    ax.set_xlabel("perception: no-CoT $\\rightarrow$ CoT")
    ax.set_xlim(-.02, 1.02)
    ax.set_title("chain-of-thought moves perception the wrong way", fontsize=8, pad=4)
    fig.tight_layout(pad=.4)
    for e in ("pdf", "png"):
        fig.savefig(out / f"cot_effect.{e}", dpi=200)


def fig_theory(gold, preds, out):
    licensed = sum(1 for r in gold.values() if r["gold_action"] == "answer") / len(gold)
    rates = {m: Counter(p.values())["answer"] / len(p) for m, p in preds.items()}
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.15))
    ax = axes[0]
    pen = [x / 20 for x in range(2, 201)]
    for c_ask, style in ((-0.1, "-"), (-0.5, "--"), (-1.0, ":")):
        tau = [(c_ask + 1.0 + w) / (1.0 + w) for w in pen]
        ax.plot(pen, tau, style, color=BLUE, lw=1.4, label=f"$c_{{\\rm ask}}={c_ask}$")
    ax.set_xscale("log")
    ax.set_xlabel("wrong-answer penalty $|c_{\\rm wrong}|$")
    ax.set_ylabel("answer threshold $\\tau$")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=6.8, loc="lower right")
    ax.set_title("(a) Theorem 1: answer only above $\\tau$", fontsize=8, pad=4)
    ax = axes[1]
    order = sorted(rates, key=lambda m: -rates[m])
    for i, m in enumerate(order):
        ax.barh(i, rates[m], color=ORANGE if rates[m] > licensed + .15 else GREY, height=.62, zorder=3)
        ax.text(rates[m] + .012, i, f"{rates[m]:.2f}", va="center", fontsize=6.8, color=INK)
    ax.axvline(licensed, color=GREEN, lw=1.5, ls="--", zorder=5)
    ax.text(licensed - .02, -0.62, f"licensed by the scene $={licensed:.2f}$", fontsize=6.8, color=GREEN,
            ha="right", va="center")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([label(m) for m in order], fontsize=7.2)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.14)
    ax.set_ylim(len(order) - 0.4, -1.05)
    ax.set_xlabel("fraction of items the model answers")
    ax.set_title("(b) models answer far past what the scene allows", fontsize=8, pad=4)
    fig.tight_layout(pad=.45, w_pad=1.1)
    for e in ("pdf", "png"):
        fig.savefig(out / f"theory.{e}", dpi=200)
    print("  answer rates: " + ", ".join(f"{label(m)} {rates[m]:.2f}" for m in order) + f" (licensed {licensed:.2f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()
    gold = by_id(load_split("test"))
    preds = {m: {i: r["action"] for i, r in run.items()} for m, run in load_runs("option", gold).items()}
    with_pipelines = {name: {i: r["action"] for i, r in read_run(REFERENCE_OUTPUTS / f"{name}.jsonl").items()}
                      for name in ("detector-rules", "learned-head")}
    with_pipelines.update(preds)
    out = Path(args.out) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_results(gold, dict(sorted(with_pipelines.items())), out)
    fig_cot(gold, out)
    fig_theory(gold, preds, out)
    print(f"  wrote results, cot_effect and theory figures in {out}")


if __name__ == "__main__":
    main()
