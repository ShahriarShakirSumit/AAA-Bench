"""Check the paper's numbers against what the scripts in reproduce/ compute.

Run reproduce/run_all.sh first. This compares every generated table in results/tables/ with the
paper's version in reproduce/expected/, then checks the numbers the paper states in its text, each
rounded as printed. It exits with status 1 if anything differs.
"""

import json
import sys
from collections import Counter, defaultdict

from aaa_bench.data import load_items, read_jsonl
from aaa_bench.paths import CONSTRUCTION, RESULTS, ROOT

EXPECTED = ROOT / "reproduce" / "expected"
failures = 0


def report(ok, what, detail=""):
    global failures
    failures += not ok
    print(f"  [{'ok' if ok else 'DIFFERS'}] {what}{(': ' + detail) if detail else ''}")


def same(claim, value, digits):
    """True if `value` prints as `claim` at `digits` decimals."""
    return f"{value:.{digits}f}" == f"{claim:.{digits}f}"


def load(name):
    return json.load(open(RESULTS / name))


def tables():
    print("tables")
    for path in sorted(EXPECTED.glob("*.tex")):
        made = RESULTS / "tables" / path.name
        if not made.exists():
            report(False, path.name, "not generated")
            continue
        want = [l for l in path.read_text().splitlines() if not l.lstrip().startswith("%")]
        got = [l for l in made.read_text().splitlines() if not l.lstrip().startswith("%")]
        diff = [f"paper `{a}` vs computed `{b}`" for a, b in zip(want, got) if a != b]
        report(want == got, path.name, "; ".join(diff[:2]) if diff else "")


def numbers():
    print("numbers stated in the text")
    items = load_items()
    test = [r for r in items if r["split"] == "test"]
    pre = read_jsonl(CONSTRUCTION / "prestudy_items.jsonl")
    pre_pairs = defaultdict(set)
    for r in pre:
        pre_pairs[r["pair_id"]].add(r["gold_action"])
    checks = [
        ("items", 7400, len(items), 0),
        ("counterfactual pairs", 3700, len({r["pair_id"] for r in items}), 0),
        ("COCO categories", 59, len({r["visual_evidence"]["category"] for r in items}), 0),
        ("test items", 1450, len(test), 0),
        ("majority action share on the test split", 0.500,
         max(Counter(r["gold_action"] for r in test).values()) / len(test), 3),
        ("items before the human study", 10842, len(pre), 0),
        ("items removed with the occlusion rule", 3442, len(pre) - len(items), 0),
        ("contrastive pairs before the study (%)", 89.1,
         100 * sum(len(v) > 1 for v in pre_pairs.values()) / len(pre_pairs), 1),
        ("majority baseline before the study", 0.554,
         max(Counter(r["gold_action"] for r in pre).values()) / len(pre), 3),
    ]
    d = load("deconfound.json")
    checks += [
        ("unpaired: text-only", 0.983, d["unpaired"]["text_only"], 3),
        ("unpaired: majority", 0.440, d["unpaired"]["majority"], 3),
        ("AAA-Bench: text-only", 0.500, d["aaa_bench"]["text_only"], 3),
        ("AAA-Bench: count-only", 0.650, d["aaa_bench"]["count_only"], 3),
        ("AAA-Bench: count-only, count-matched pairs", 0.582, d["aaa_bench"]["count_only_matched"], 3),
        ("AAA-Bench: count-only, referential pairs not count-matched", 0.659, d["aaa_bench"]["count_only_unmatched"], 3),
        ("AAA-Bench: text + count", 0.620, d["aaa_bench"]["text_count"], 3),
    ]
    a = load("agreement.json")
    after, derived, maj = a["after_study"], a["as_derived"], a["majority_by_basis"]
    sweep = a["occlusion_sweep"]
    checks += [
        ("Fleiss' kappa", 0.833, a["fleiss_kappa"], 3),
        ("mean pairwise Cohen's kappa", 0.833, a["mean_cohen_kappa"], 3),
        ("agreement with the labels as derived", 0.527, derived["OVERALL"]["agreement"], 3),
        ("agreement after the study", 0.726, after["OVERALL"]["agreement"], 3),
        ("agreement after the study, n", 350, after["OVERALL"]["n"], 0),
        ("agreement on the bases the study left unchanged", 0.683, a["unchanged_bases"]["agreement"], 3),
        ("agreement on the bases the study left unchanged, n", 300, a["unchanged_bases"]["n"], 0),
        ("agreement on held-out items in the abstract (%)", 68, 100 * a["unchanged_bases"]["agreement"], 0),
        ("agreement, one clear instance", 0.98, after["single_clear"]["agreement"], 2),
        ("agreement, absent objects after the revision", 0.98, after["absent"]["agreement"], 2),
        ("agreement, presence", 0.86, after["present"]["agreement"], 2),
        ("agreement, countability", 0.68, after["distinct"]["agreement"], 2),
        ("agreement, several comparable instances", 0.66, after["multi_comparable"]["agreement"], 2),
        ("agreement, one dominant instance", 0.46, after["multi_dominant"]["agreement"], 2),
        ("agreement, crowds", 0.46, after["crowd"]["agreement"], 2),
        ("one dominant instance: majority asks", 27, maj["multi_dominant"].get("ask", 0), 0),
        ("crowds: majority answers", 27, maj["crowd"].get("answer", 0), 0),
        ("absent objects: majority abstains", 49, a["absent_majority"].get("abstain", 0), 0),
        ("occluded items: majority answers", 40, a["occluded_majority"].get("answer", 0), 0),
        ("occluded items: majority abstains", 5, a["occluded_majority"].get("abstain", 0), 0),
        ("occluded items: majority asks", 5, a["occluded_majority"].get("ask", 0), 0),
        ("occlusion rule agreement (%)", 10, 100 * derived["single_occluded"]["agreement"], 0),
        ("occlusion sweep, lowest answer share (%)", 75, 100 * min(v["answer"] for v in sweep.values()), 0),
        ("occlusion sweep, highest answer share (%)", 81, 100 * max(v["answer"] for v in sweep.values()), 0),
        ("occlusion sweep, lowest abstain share (%)", 9, 100 * min(v["abstain"] for v in sweep.values()), 0),
        ("occlusion sweep, highest abstain share (%)", 12, 100 * max(v["abstain"] for v in sweep.values()), 0),
    ]
    h = load("human_reference.json")
    checks += [
        ("human self-consistency", 0.950, h["self-consistency"]["value"], 3),
        ("human self-consistency, n", 1086, h["self-consistency"]["n"], 0),
        ("human contrastive accuracy", 0.560, h["contrastive accuracy"]["value"], 3),
        ("human contrastive accuracy, pairs", 50, h["contrastive accuracy"]["n"], 0),
    ]
    det = load("detector_pipelines.json")
    fam = det["detector_rules"]["per_family"]
    checks += [
        ("detector + rules: action accuracy", 0.861, det["detector_rules"]["action_acc"], 3),
        ("detector + rules: contrastive accuracy", 0.742, det["detector_rules"]["contrastive"], 3),
        ("detector + rules: counting", 0.80, fam["counting"], 2),
        ("detector + rules: presence", 0.93, fam["presence"], 2),
        ("detector + rules: referential (attribute)", 0.84, fam["referential_color"], 2),
        ("detector + rules: referential (spatial)", 0.87, fam["referential_spatial"], 2),
    ]
    if "learned_head" in det:
        checks += [
            ("learned head: action accuracy", 0.860, det["learned_head"]["action_acc"], 3),
            ("learned head: contrastive accuracy", 0.742, det["learned_head"]["contrastive"], 3),
            ("pipelines agree on test items (%)", 94.8, 100 * det["learned_head"]["agreement_with_rules"], 1),
        ]
    else:
        print("  (learned head not fitted: install scikit-learn to check its three numbers)")
    s = load("scores.json")
    option = {k: v for k, v in s.items() if k.endswith("|option")}
    st = load("stats.json")
    best = st["best_vs_runner_up"]
    p = load("parser.json")
    e = load("errors.json")
    ss = load("same_side.json")
    checks += [
        ("best contrastive accuracy (option prompt)", 0.167, max(v["contrastive"] for v in option.values()), 3),
        ("lowest contrastive accuracy (option prompt)", 0.006, min(v["contrastive"] for v in option.values()), 3),
        ("lowest action accuracy (option prompt)", 0.50, min(v["action_acc"] for v in option.values()), 2),
        ("highest action accuracy (option prompt)", 0.57, max(v["action_acc"] for v in option.values()), 2),
        ("points above the majority baseline, at most", 7, 100 * (max(v["action_acc"] for v in option.values()) - 0.5), 0),
        ("best model CA, lower 95% bound", 0.141, st["qwen2.5-vl-7b"]["ca_ci"][0], 3),
        ("best model CA, upper 95% bound", 0.195, st["qwen2.5-vl-7b"]["ca_ci"][1], 3),
        ("best vs runner-up, lower 95% bound", -0.017, best["ci"][0], 3),
        ("best vs runner-up, upper 95% bound", 0.028, best["ci"][1], 3),
        ("tagged option and chain-of-thought responses", 14000, p["tagged_option_cot"], 0),
        ("cue rules recover the explicit tag (%)", 97.3, 100 * p["cue_rules_recover_tag"], 1),
        ("gold-ASK items answered, pooled over models (%)", 94, 100 * e["confusion_pooled"]["ask"]["answer"], 0),
        ("gold-ASK items asked, pooled over models (%)", 3, 100 * e["confusion_pooled"]["ask"]["ask"], 0),
        ("spatial ASK items with both instances on one side", 418, ss["items_same_side"], 0),
        ("spatial ASK items", 1653, ss["items"], 0),
        ("test pairs with such an item", 77, ss["test_pairs_flagged"], 0),
        ("largest change in CA without them", 0.018, ss["max_change"], 3),
        ("same side: human majority asks", 4, ss["human_majority"]["same side"].get("ask", 0), 0),
        ("same side: human majority answers", 4, ss["human_majority"]["same side"].get("answer", 0), 0),
        ("different sides: human majority asks", 21, ss["human_majority"]["different sides"].get("ask", 0), 0),
        ("different sides: human majority answers", 6, ss["human_majority"]["different sides"].get("answer", 0), 0),
    ]
    for what, claim, value, digits in checks:
        report(same(claim, value, digits), what, "" if same(claim, value, digits) else f"paper {claim}, computed {value}")
    report(max(v["contrastive"] for v in s.values()) < 0.23, "contrastive accuracy stays below 0.23 under every prompt")
    report(ss["max_without_flagged"] < 0.25, "every model stays below 0.25 without those pairs")


def main():
    tables()
    numbers()
    print(f"\n{'all checks passed' if not failures else f'{failures} check(s) differ'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
