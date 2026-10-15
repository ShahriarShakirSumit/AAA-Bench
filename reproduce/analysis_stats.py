"""Significance and intervals for the option-prompt results.

  McNemar     exact two-sided test of each model against the always-answer policy, item by item
              (the majority baseline is a deterministic policy, not a Bernoulli(0.5) process)
  Holm        Holm-Bonferroni adjustment over the seven models
  bootstrap   percentile intervals that resample pairs, since the two members of a pair are
              anti-correlated by construction
  null        the contrastive accuracy a policy reaches by emitting the model's own action
              frequencies at random; a model's CA is read as excess over this
  best vs runner-up  paired bootstrap over pairs of the difference in CA between the two models
              with the highest CA
Writes results/tables/tab_stats_ci.tex and results/stats.json.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

from aaa_bench.data import by_id, load_split
from aaa_bench.outputs import label, load_runs
from aaa_bench.paths import RESULTS

B, SEED = 2000, 20260809


def mcnemar_exact(b, c):
    """Two-sided exact McNemar: binomial(b; b+c, 1/2) on the discordant items."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def pair_boot(pair_hits, rng, stat, b=B):
    n = len(pair_hits)
    if n == 0:
        return (0.0, 0.0)
    d = sorted(stat([pair_hits[rng.randrange(n)] for _ in range(n)]) for _ in range(b))
    return d[int(.025 * b)], d[int(.975 * b)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()
    rng = random.Random(SEED)
    gold = by_id(load_split("test"))
    groups = defaultdict(list)
    for r in gold.values():
        groups[r["pair_id"]].append(r["item_id"])
    pairs = [v for v in groups.values() if len(v) == 2]
    preds = {m: {i: r["action"] for i, r in run.items()} for m, run in load_runs("option", gold).items()}
    maj_rate = max(Counter(g["gold_action"] for g in gold.values()).values()) / len(gold)
    print(f"  test items={len(gold)} pairs={len(pairs)} majority baseline={maj_rate:.3f}")

    rows, raw_p = [], []
    for m, p in preds.items():
        ph = [(int(p[a] == gold[a]["gold_action"]), int(p[b_] == gold[b_]["gold_action"]))
              for a, b_ in pairs if a in p and b_ in p]
        acc = sum(ha + hb for ha, hb in ph) / (2 * len(ph))
        ca = sum(ha and hb for ha, hb in ph) / len(ph)
        acc_lo, acc_hi = pair_boot(ph, rng, lambda s: sum(a + b for a, b in s) / (2 * len(s)))
        ca_lo, ca_hi = pair_boot(ph, rng, lambda s: sum(a and b for a, b in s) / len(s))
        bb = cc = 0
        for i, g in gold.items():
            if i not in p:
                continue
            model_ok, base_ok = p[i] == g["gold_action"], g["gold_action"] == "answer"
            bb += int(model_ok and not base_ok)
            cc += int(base_ok and not model_ok)
        pv = mcnemar_exact(bb, cc)
        marg = Counter(p.values())
        tot = sum(marg.values())
        q = {a: marg.get(a, 0) / tot for a in ("answer", "ask", "abstain")}
        null = sum(q.get(gold[a]["gold_action"], 0) * q.get(gold[b_]["gold_action"], 0)
                   for a, b_ in pairs if a in p and b_ in p) / len(ph)
        rows.append([m, acc, acc_lo, acc_hi, pv, bb, cc, ca, ca_lo, ca_hi, null])
        raw_p.append(pv)
    for r, a in zip(rows, holm(raw_p)):
        r.append(a)
    rows.sort(key=lambda r: -(r[7] - r[10]))

    def fmt_p(x):
        return "$<$.001" if x < 0.001 else f"{x:.3f}".lstrip("0")

    lines = [r"\footnotesize", r"\setlength{\tabcolsep}{2.6pt}", r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r"Model & Act. & McNemar & Holm & CA & null \\", r"\midrule"]
    for m, acc, alo, ahi, pv, bb, cc, ca, clo, chi, null, ap_ in rows:
        lines.append(f"{label(m)} & {acc:.3f} & {fmt_p(pv)} & {fmt_p(ap_)} & {ca:.3f} & {null:.3f} \\\\")
    lines += [r"\midrule", rf"Majority & {maj_rate:.3f} & n/a & n/a & 0.000 & n/a \\", r"\bottomrule", r"\end{tabular}"]
    out = Path(args.out) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tab_stats_ci.tex").write_text("\n".join(lines) + "\n")
    for m, acc, alo, ahi, pv, bb, cc, ca, clo, chi, null, ap_ in rows:
        print(f"    {label(m):12s} acc={acc:.3f} [{alo:.3f}, {ahi:.3f}]  McNemar p={pv:.2e} (b={bb}, c={cc})  "
              f"Holm={ap_:.2e} | CA={ca:.3f} [{clo:.3f}, {chi:.3f}]  null={null:.3f}  excess={ca - null:+.3f}")
    ns = [label(r[0]) for r in rows if r[11] > 0.05]
    print(f"  not distinguishable from always-answer after Holm: {', '.join(ns) or 'none'}")

    by_ca = sorted(rows, key=lambda r: -r[7])
    best, second = by_ca[0][0], by_ca[1][0]
    diff = [(int(preds[best][a] == gold[a]["gold_action"] and preds[best][b_] == gold[b_]["gold_action"])
             - int(preds[second][a] == gold[a]["gold_action"] and preds[second][b_] == gold[b_]["gold_action"]))
            for a, b_ in pairs if all(x in preds[best] and x in preds[second] for x in (a, b_))]
    drng = random.Random(SEED)
    d = sorted(sum(diff[drng.randrange(len(diff))] for _ in diff) / len(diff) for _ in range(B))
    lo, hi = d[int(.025 * B)], d[int(.975 * B)]
    print(f"  {label(best)} vs {label(second)}: delta CA {sum(diff) / len(diff):+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]")
    summary = {m: {"action_acc": acc, "action_ci": [alo, ahi], "mcnemar_p": pv, "holm_p": ap_,
                   "ca": ca, "ca_ci": [clo, chi], "null": null}
               for m, acc, alo, ahi, pv, bb, cc, ca, clo, chi, null, ap_ in rows}
    summary["best_vs_runner_up"] = {"models": [best, second], "delta_ca": sum(diff) / len(diff), "ci": [lo, hi]}
    (Path(args.out) / "stats.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  wrote {out / 'tab_stats_ci.tex'} and stats.json")


if __name__ == "__main__":
    main()
