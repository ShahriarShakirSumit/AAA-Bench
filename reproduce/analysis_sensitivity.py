"""Sensitivity of the referential labels to the two area thresholds.

An instance is dominant if its area is at least D times the next largest, and two instances are
comparable if the second is at least K times the largest. The benchmark uses D = 3 and K = 0.5.
This re-derives the referential test labels over a grid of (D, K) and reports how many labels
change, what share of pairs stays contrastive, and the best model's contrastive accuracy under the
re-derived labels. Writes results/tables/tab_sensitivity.tex.
"""

import argparse
from collections import defaultdict
from pathlib import Path

from aaa_bench.data import load_split
from aaa_bench.outputs import load_runs
from aaa_bench.paths import RESULTS

D_GRID = [2.0, 2.5, 3.0, 4.0, 5.0]
K_GRID = [0.4, 0.5, 0.6, 0.7]


def relabel_referential(r, D, K):
    a = sorted((r.get("visual_evidence") or {}).get("areas") or [], reverse=True)
    if len(a) < 2:
        return "answer"
    if a[0] >= D * a[1]:
        return "answer"
    if a[1] >= K * a[0]:
        return "ask"
    return "answer"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()
    ref = [r for r in load_split("test") if str(r.get("ambiguity_subtype", "")).startswith("referential")]
    base = {r["item_id"]: r["gold_action"] for r in ref}
    preds = {m: {i: x["action"] for i, x in run.items()} for m, run in load_runs("option").items()}
    groups = defaultdict(list)
    for r in ref:
        groups[r["pair_id"]].append(r["item_id"])
    pairs = {k: v for k, v in groups.items() if len(v) == 2}
    print(f"  referential test items={len(ref)} pairs={len(pairs)}")

    rows = []
    for D in D_GRID:
        for K in K_GRID:
            new = {r["item_id"]: relabel_referential(r, D, K) for r in ref}
            changed = sum(1 for i in base if new[i] != base[i])
            best = 0.0
            for p in preds.values():
                hits = [int(p.get(a) == new[a] and p.get(b) == new[b])
                        for a, b in pairs.values() if a in p and b in p and new[a] != new[b]]
                if hits:
                    best = max(best, sum(hits) / len(hits))
            n_contrast = sum(1 for a, b in pairs.values() if new[a] != new[b])
            rows.append((D, K, changed / len(base), n_contrast / len(pairs), best))

    d_free = len({(K, round(ch, 6), round(cp, 6), round(b, 6)) for D, K, ch, cp, b in rows}) == len(K_GRID)
    by_k = {K: (ch, cp, b) for D, K, ch, cp, b in rows}
    lines = [r"\begin{tabular}{cccc}", r"\toprule", r"$K$ & Labels chg. & Contrast. & Best CA \\", r"\midrule"]
    for K in K_GRID:
        ch, cp, b = by_k[K]
        star = r"$^{\star}$" if K == 0.5 else ""
        lines.append(f"{K:.1f}{star} & {ch * 100:.1f}\\% & {cp * 100:.1f}\\% & {b:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = Path(args.out) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tab_sensitivity.tex").write_text("\n".join(lines) + "\n")
    print(f"  every D in [{D_GRID[0]}, {D_GRID[-1]}] gives the same labelling: {d_free}")
    for D, K, ch, cp, b in rows:
        mark = "  <- as built" if (D == 3.0 and K == 0.5) else ""
        print(f"    D={D:.1f} K={K:.1f}  changed={ch:6.1%}  contrastive={cp:6.1%}  best CA={b:.3f}{mark}")
    print(f"  wrote {out / 'tab_sensitivity.tex'}")


if __name__ == "__main__":
    main()
