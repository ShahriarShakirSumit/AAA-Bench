"""Agreement in the human study.

Three annotators independently chose an action for each of the 400 study items
(human_study/sample_items.jsonl), 50 per decision basis, drawn with the benchmark's construction
rules. This reports agreement between annotators (Fleiss' kappa, mean pairwise Cohen's kappa) and how
often the human majority matches the derived label, overall and per decision basis:
  as derived   the labels the rules gave the items when they were sampled
  after study  the same items under the study's verdict, which the benchmark adopts: absent objects
               become ABSTAIN, and the partial-occlusion rule is retired (its items drop out)
"After study" is partly fitted, since the verdict was drawn from this sample, so the agreement on
the six bases the verdict left untouched is reported as well. Writes results/agreement_report.txt.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from aaa_bench.data import read_jsonl
from aaa_bench.paths import HUMAN_STUDY, RESULTS

ACTIONS = ["answer", "ask", "abstain"]
RELABELLED = {"absent": "abstain"}
RETIRED = {"single_occluded"}
OCCLUSION_CUTOFFS = (5000, 10000, 20000, 40000)


def after_study(item):
    """The item's label under the study's verdict, or None if its rule was retired."""
    basis = item["decision_basis"]
    if basis in RETIRED:
        return None
    return RELABELLED.get(basis, item["gold_action"])


def fleiss_kappa(rows):
    """rows: one Counter over ACTIONS per item, each summing to the number of raters."""
    N = len(rows)
    n = sum(rows[0].values()) if rows else 0
    if N == 0 or n < 2:
        return None
    p_j = {a: sum(r[a] for r in rows) / (N * n) for a in ACTIONS}
    P_i = [(sum(r[a] ** 2 for a in ACTIONS) - n) / (n * (n - 1)) for r in rows]
    P_bar, P_e = sum(P_i) / N, sum(v ** 2 for v in p_j.values())
    return (P_bar - P_e) / (1 - P_e) if P_e < 1 else 1.0


def cohen_kappa(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    if not pairs:
        return None
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in ACTIONS)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def load_study(responses=HUMAN_STUDY / "annotator_responses.csv", sample=HUMAN_STUDY / "sample_items.jsonl"):
    """(items by id, {item_id: [votes]}, action column names)."""
    items = {r["item_id"]: r for r in read_jsonl(sample)}
    rows = list(csv.DictReader(open(responses, encoding="utf-8")))
    cols = [c for c in rows[0] if c.startswith("annot") and "action" in c]
    votes = {row["item_id"]: [row[c].strip().lower() for c in cols] for row in rows}
    missing = set(votes) - set(items)
    if missing:
        raise SystemExit(f"{len(missing)} responses refer to items not in {sample}")
    return items, votes, cols


def majority(votes):
    valid = [v for v in votes if v in ACTIONS]
    return Counter(valid).most_common(1)[0][0] if len(valid) >= 2 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses", default=str(HUMAN_STUDY / "annotator_responses.csv"))
    ap.add_argument("--sample", default=str(HUMAN_STUDY / "sample_items.jsonl"))
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()
    items, votes, cols = load_study(args.responses, args.sample)

    per_item = []
    per_rater = defaultdict(list)
    for v in votes.values():
        if sum(x in ACTIONS for x in v) < 2:
            continue
        cnt = Counter(x for x in v if x in ACTIONS)
        per_item.append(Counter({a: cnt.get(a, 0) for a in ACTIONS}))
        for c, x in zip(cols, v):
            per_rater[c].append(x if x in ACTIONS else "")
    raters = list(per_rater)
    cks = [cohen_kappa(per_rater[raters[i]], per_rater[raters[j]])
           for i in range(len(raters)) for j in range(i + 1, len(raters))]
    cks = [c for c in cks if c is not None]

    derived = defaultdict(lambda: [0, 0])
    study = defaultdict(lambda: [0, 0])
    for iid, v in votes.items():
        m = majority(v)
        if m is None:
            continue
        item = items[iid]
        b = item["decision_basis"]
        for key in (b, "OVERALL"):
            derived[key][0] += m == item["gold_action"]
            derived[key][1] += 1
        label = after_study(item)
        if label is not None:
            for key in (b, "OVERALL"):
                study[key][0] += m == label
                study[key][1] += 1

    lines = [f"study items with at least two valid ratings: {len(per_item)}",
             f"Fleiss' kappa (action)          = {fleiss_kappa(per_item):.3f}",
             f"mean pairwise Cohen's kappa     = {sum(cks) / len(cks):.3f}",
             "", "human majority vs the label, per decision basis:",
             f"  {'basis':18s} {'as derived':>16s} {'after study':>16s}"]
    for b in sorted(derived, key=lambda k: (k == "OVERALL", k)):
        ok, n = derived[b]
        s = f"{study[b][0] / study[b][1]:.3f} (n={study[b][1]})" if study[b][1] else "retired"
        lines.append(f"  {b:18s} {ok / n:9.3f} (n={n}) {s:>16s}")
    kept = [study[b] for b in study if b != "OVERALL" and b not in RELABELLED and b not in RETIRED]
    ok, n = sum(k for k, _ in kept), sum(m for _, m in kept)
    lines.append(f"  bases the study left unchanged: {ok / n:.3f} (n={n})")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "agreement_report.txt").write_text(text)
    absent = Counter(majority(v) for i, v in votes.items() if items[i]["decision_basis"] == "absent")
    occluded = Counter(majority(v) for i, v in votes.items() if items[i]["decision_basis"] == "single_occluded")
    # the retired occlusion rule restricted to ever smaller instances (largest instance area <= cut-off)
    sweep = {}
    for cut in OCCLUSION_CUTOFFS:
        maj = [majority(v) for i, v in votes.items() if items[i]["decision_basis"] == "single_occluded"
               and items[i]["visual_evidence"]["areas"][0] <= cut]
        sweep[cut] = {"n": len(maj), "answer": maj.count("answer") / len(maj),
                      "abstain": maj.count("abstain") / len(maj)}
    extra = ["", "retired occlusion rule, items whose instance covers at most the cut-off:"]
    extra += [f"  <= {c:6d} px: n={v['n']:2d}  answer {v['answer']:.2f}  abstain {v['abstain']:.2f}"
              for c, v in sweep.items()]
    print("\n".join(extra))
    (Path(args.out) / "agreement_report.txt").write_text(text + "\n".join(extra) + "\n")
    summary = {"fleiss_kappa": fleiss_kappa(per_item), "mean_cohen_kappa": sum(cks) / len(cks),
               "as_derived": {b: {"agreement": ok / n, "n": n} for b, (ok, n) in derived.items()},
               "after_study": {b: {"agreement": ok / n, "n": n} for b, (ok, n) in study.items() if n},
               "unchanged_bases": {"agreement": sum(k for k, _ in kept) / sum(m for _, m in kept),
                                   "n": sum(m for _, m in kept)},
               "absent_majority": dict(absent), "occluded_majority": dict(occluded),
               "occlusion_sweep": sweep,
               "majority_by_basis": {b: dict(Counter(majority(v) for i, v in votes.items()
                                                     if items[i]["decision_basis"] == b))
                                     for b in derived if b != "OVERALL"}}
    (Path(args.out) / "agreement.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
