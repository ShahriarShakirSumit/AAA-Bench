"""Score every saved run and write the composition, main-results and chain-of-thought tables.

  results/scores.json             headline metrics of all 21 runs (7 models x 3 prompts)
  results/tables/tab_stats.tex    items and pairs per question family
  results/tables/tab_results.tex  option prompt, ranked by contrastive accuracy
  results/tables/tab_cot.tex      option prompt against chain-of-thought
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from aaa_bench.data import by_id, load_items, load_split
from aaa_bench.metrics import summary
from aaa_bench.outputs import label, load_runs
from aaa_bench.paths import RESULTS

FAMILY_LABEL = {"referential_color": "Referential (attribute)", "referential_spatial": "Referential (spatial)",
                "counting": "Counting", "presence": "Presence"}


def table_stats(rows):
    fam = defaultdict(int)
    for r in rows:
        fam[r["ambiguity_subtype"]] += 1
    lines = [r"\begin{tabular}{lrr}", r"\toprule", r"Family & Items & Pairs \\", r"\midrule"]
    for f, n in sorted(fam.items(), key=lambda x: -x[1]):
        lines.append(f"{FAMILY_LABEL.get(f, f)} & {n:,} & {n // 2:,} \\\\".replace(",", "{,}"))
    lines += [r"\midrule", f"Total & {len(rows):,} & {len(rows) // 2:,} \\\\".replace(",", "{,}"),
              r"\bottomrule", r"\end{tabular}"]
    return lines


def table_results(runs):
    prim = sorted(((m, v) for (m, s), v in runs.items() if s == "option"), key=lambda x: -x[1]["contrastive"])
    best = max(v["contrastive"] for _, v in prim)
    lines = [r"\setlength{\tabcolsep}{2.7pt}", r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Model & Act. & F1 & \textbf{CA} & Perc. & Dec. \\", r"\midrule"]
    for m, v in prim:
        c = f"{v['contrastive']:.2f}"
        if v["contrastive"] == best:
            c = rf"\textbf{{{c}}}"
        lines.append(f"{label(m)} & {v['action_acc']:.2f} & {v['amb_f1']:.2f} & {c} & "
                     f"{v['perception']:.2f} & {v['decision']:.2f} \\\\")
    lines += [r"\midrule", r"Majority baseline & 0.50 & 0.00 & 0.00 & 0.00 & --- \\", r"\bottomrule", r"\end{tabular}"]
    return lines


def table_cot(runs):
    def cell(m, s, k):
        v = runs.get((m, s), {}).get(k)
        return f"{v:.2f}" if v is not None else "n/a"
    lines = [r"\footnotesize", r"\setlength{\tabcolsep}{3.2pt}", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"& \multicolumn{2}{c}{Contrastive} & \multicolumn{2}{c}{Perception} & \multicolumn{2}{c}{Action acc.} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Model & base & CoT & base & CoT & base & CoT \\", r"\midrule"]
    for m in sorted({m for m, _ in runs}):
        lines.append(f"{label(m)} & {cell(m, 'option', 'contrastive')} & {cell(m, 'cot', 'contrastive')}"
                     f" & {cell(m, 'option', 'perception')} & {cell(m, 'cot', 'perception')}"
                     f" & {cell(m, 'option', 'action_acc')} & {cell(m, 'cot', 'action_acc')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()
    gold = by_id(load_split("test"))
    runs = {}
    for strategy in ("option", "cot", "direct"):
        for m, preds in load_runs(strategy, gold).items():
            runs[(m, strategy)] = summary(gold, preds)

    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "scores.json").write_text(json.dumps({f"{m}|{s}": v for (m, s), v in runs.items()}, indent=2) + "\n")
    for name, lines in (("tab_stats", table_stats(load_items())), ("tab_results", table_results(runs)),
                        ("tab_cot", table_cot(runs))):
        (out / "tables" / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print(f"{'model':16s}{'prompt':8s}{'scored':>7s}{'action':>8s}{'amb-F1':>8s}{'CA':>7s}{'percep':>8s}{'decis':>7s}")
    for (m, s), v in sorted(runs.items(), key=lambda x: (x[0][1], -x[1]["contrastive"])):
        print(f"{label(m):16s}{s:8s}{v['n_scored']:7d}{v['action_acc']:8.3f}{v['amb_f1']:8.3f}{v['contrastive']:7.3f}"
              f"{v['perception']:8.3f}{v['decision']:7.3f}")
    counts = Counter(r["gold_action"] for r in load_items())
    print(f"\n  gold actions over all splits: {dict(counts)}")
    print(f"  wrote scores.json, tab_stats.tex, tab_results.tex, tab_cot.tex in {out}")


if __name__ == "__main__":
    main()
