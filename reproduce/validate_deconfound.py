"""Shortcut attacks on the benchmark: can the gold action be predicted without the image?

Each attacker is trained on the train split and scored on the test split:
  text-only    multinomial Naive Bayes on the question's words
  count-only   the majority gold action per instance-count bin (counts capped at 6)
  text+count   Naive Bayes on the question's words plus one token for the count bin
The same attacks are run on the unpaired variant (reproduce/data/unpaired_variant.jsonl), which is
the construction without counterfactual pairing. Count-only accuracy is also reported on the
count-matched pairs (several comparable instances against one dominant instance, so both images show
two or more instances) and on the referential pairs that are not count-matched (one clear instance
against several comparable ones). Writes results/deconfound_report.txt and results/tables/tab_deconf.tex.
"""

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from aaa_bench.data import load_items, read_jsonl
from aaa_bench.paths import RESULTS, ROOT

TOK = re.compile(r"[a-z]+")
MATCHED = {"multi_comparable", "multi_dominant"}
UNMATCHED = {"multi_comparable", "single_clear"}


def tokens(s):
    return TOK.findall((s or "").lower())


def nb_train(rows, features):
    classes, prior, vocab = defaultdict(Counter), Counter(), set()
    for r in rows:
        y = r["gold_action"]
        prior[y] += 1
        for t in features(r):
            classes[y][t] += 1
            vocab.add(t)
    return dict(classes=classes, prior=prior, totals={y: sum(c.values()) for y, c in classes.items()},
                V=len(vocab), n=sum(prior.values()), labels=list(prior))


def nb_predict(m, toks):
    best, best_lp = None, -1e18
    for y in m["labels"]:
        lp = math.log(m["prior"][y] / m["n"])
        c, tot = m["classes"][y], m["totals"][y]
        for t in toks:
            lp += math.log((c[t] + 1) / (tot + m["V"]))
        if lp > best_lp:
            best, best_lp = y, lp
    return best


def count_of(r):
    c = (r.get("visual_evidence") or {}).get("instance_count")
    if c is None:
        c = r.get("num_objects_in_scene")
    return min(int(c or 0), 6)


def attack(rows):
    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]
    words = lambda r: tokens(r["question"])  # noqa: E731
    # the count enters as its own token, after tokenisation, so the word tokeniser cannot drop it
    words_count = lambda r: tokens(r["question"]) + [f"<count={count_of(r)}>"]  # noqa: E731

    m = nb_train(tr, words)
    text_acc = sum(nb_predict(m, words(r)) == r["gold_action"] for r in te) / len(te)
    maj_label, maj_n = Counter(r["gold_action"] for r in te).most_common(1)[0]

    by_count = defaultdict(Counter)
    for r in tr:
        by_count[count_of(r)][r["gold_action"]] += 1
    count_rule = {c: cc.most_common(1)[0][0] for c, cc in by_count.items()}
    fallback = Counter(r["gold_action"] for r in tr).most_common(1)[0][0]
    count_pred = lambda r: count_rule.get(count_of(r), fallback)  # noqa: E731
    count_acc = sum(count_pred(r) == r["gold_action"] for r in te) / len(te)

    mc = nb_train(tr, words_count)
    text_count_acc = sum(nb_predict(mc, words_count(r)) == r["gold_action"] for r in te) / len(te)

    out = dict(items=len(rows), test=len(te), majority_label=maj_label, majority=maj_n / len(te),
               text_only=text_acc, count_only=count_acc, text_count=text_count_acc)

    pairs = defaultdict(list)
    for r in te:
        if r.get("pair_id"):
            pairs[r["pair_id"]].append(r)
    for name, bases in (("count_only_matched", MATCHED), ("count_only_unmatched", UNMATCHED)):
        members = [x for v in pairs.values() if len(v) == 2 and {y.get("decision_basis") for y in v} == bases for x in v]
        if members:
            out[name] = sum(count_pred(r) == r["gold_action"] for r in members) / len(members)
    full = [v for v in pairs.values() if len(v) == 2 and len({x["gold_action"] for x in v}) == 2]
    if full:
        hits = sum(nb_predict(m, words(x)) == x["gold_action"] for v in full for x in v)
        out["paired_text_only"] = hits / (2 * len(full))

    fam_key = "ambiguity_subtype" if any(r.get("ambiguity_subtype") for r in te) else "ambiguity_type"
    by_fam = defaultdict(list)
    for r in te:
        by_fam[str(r.get(fam_key))].append(r)
    out["text_only_by_family"] = {f: (sum(nb_predict(m, words(r)) == r["gold_action"] for r in rs) / len(rs), len(rs))
                                  for f, rs in sorted(by_fam.items())}
    return out


def report(name, r):
    lines = [f"===== {name} =====",
             f"  items={r['items']}  test={r['test']}  majority ('{r['majority_label']}') = {r['majority']:.3f}",
             f"  TEXT-ONLY  acc = {r['text_only']:.3f}   (lift over majority {r['text_only'] - r['majority']:+.3f})",
             f"  COUNT-ONLY acc = {r['count_only']:.3f}   (lift over majority {r['count_only'] - r['majority']:+.3f})",
             f"  TEXT+COUNT acc = {r['text_count']:.3f}   (lift over majority {r['text_count'] - r['majority']:+.3f})"]
    if "count_only_matched" in r:
        lines.append(f"  COUNT-ONLY on count-matched pairs = {r['count_only_matched']:.3f}, "
                     f"on referential pairs that are not count-matched = {r['count_only_unmatched']:.3f}")
    if "paired_text_only" in r:
        lines.append(f"  text-only accuracy within contrastive pairs = {r['paired_text_only']:.3f}  (0.500 = chance)")
    lines.append("  text-only accuracy per family:")
    lines += [f"    {f:22s} {a:.3f}  (n={n})" for f, (a, n) in r["text_only_by_family"].items()]
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unpaired", default=str(ROOT / "reproduce" / "data" / "unpaired_variant.jsonl"))
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()

    unpaired = attack(read_jsonl(args.unpaired))
    bench = attack(load_items())
    lines = ["Shortcut attacks (no image). Lift over the majority baseline is leakage.", ""]
    lines += report("unpaired variant", unpaired) + [""] + report("AAA-Bench", bench)
    text = "\n".join(lines) + "\n"
    print(text, end="")

    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "deconfound_report.txt").write_text(text)
    (out / "deconfound.json").write_text(json.dumps({"unpaired": unpaired, "aaa_bench": bench}, indent=2) + "\n")
    f3 = lambda v: f"{v:.3f}"  # noqa: E731
    table = [r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lcc}", r"\toprule",
             r"Attacker & Unpaired & \bench{} \\", r"\midrule",
             rf"Majority baseline & {f3(unpaired['majority'])} & {f3(bench['majority'])} \\",
             rf"Text-only (question) & \textbf{{{f3(unpaired['text_only'])}}} & \textbf{{{f3(bench['text_only'])}}} \\",
             rf"Count-only (instances) & {f3(unpaired['count_only'])} & {f3(bench['count_only'])} \\",
             rf"\quad count-matched pairs & --- & {f3(bench['count_only_matched'])} \\",
             rf"Text $+$ count & {f3(unpaired['text_count'])} & {f3(bench['text_count'])} \\",
             rf"Paired, same wording & --- & \textbf{{{f3(bench['paired_text_only'])}}} \\",
             r"\bottomrule", r"\end{tabular}"]
    (out / "tables" / "tab_deconf.tex").write_text("\n".join(table) + "\n")
    print(f"\nwrote {out / 'deconfound_report.txt'} and {out / 'tables' / 'tab_deconf.tex'}")


if __name__ == "__main__":
    main()
