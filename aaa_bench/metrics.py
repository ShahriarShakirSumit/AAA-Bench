"""Metrics for AAA-Bench.

All metrics take `gold` ({item_id: item}) and `preds` ({item_id: record with an "action" field}).
Items without a prediction are skipped, except in contrastive accuracy, where a pair counts as solved
only when both members are predicted correctly.

action_accuracy       share of items whose predicted action equals the gold action
ambiguity_detection   ASK or ABSTAIN read as "flags the item"; precision, recall and F1
contrastive_accuracy  share of counterfactual pairs with both members correct (the headline metric;
                      any constant policy scores 0)
oracle_decomposition  perception (recall of ambiguity), decision (ASK vs ABSTAIN once flagged) and
                      answer fidelity (answering the unambiguous items)
utility               mean decision utility under one cost matrix
ece                   expected calibration error, for predictions that carry a confidence
"""

from collections import Counter, defaultdict

AMBIGUOUS = {"ask", "abstain"}


def action_accuracy(gold, preds):
    by_gold = defaultdict(lambda: [0, 0])
    by_family = defaultdict(lambda: [0, 0])
    total = correct = 0
    for iid, g in gold.items():
        p = preds.get(iid)
        if not p:
            continue
        c = int(p["action"] == g["gold_action"])
        total += 1
        correct += c
        by_gold[g["gold_action"]][0] += c
        by_gold[g["gold_action"]][1] += 1
        by_family[g["ambiguity_subtype"]][0] += c
        by_family[g["ambiguity_subtype"]][1] += 1
    return dict(accuracy=correct / max(total, 1), n=total,
                per_gold={k: v[0] / v[1] for k, v in by_gold.items()},
                per_family={k: v[0] / v[1] for k, v in by_family.items()})


def ambiguity_detection(gold, preds):
    tp = fp = fn = tn = 0
    for iid, g in gold.items():
        p = preds.get(iid)
        if not p:
            continue
        gold_amb = g["gold_action"] in AMBIGUOUS
        pred_amb = p["action"] in AMBIGUOUS
        tp += gold_amb and pred_amb
        fp += (not gold_amb) and pred_amb
        fn += gold_amb and (not pred_amb)
        tn += (not gold_amb) and (not pred_amb)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return dict(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


def contrastive_accuracy(gold, preds):
    by_pair = defaultdict(list)
    for g in gold.values():
        if g.get("pair_id"):
            by_pair[g["pair_id"]].append(g)
    solved = n = 0
    for members in by_pair.values():
        if len(members) != 2 or len({m["gold_action"] for m in members}) < 2:
            continue
        n += 1
        if all(preds.get(m["item_id"], {}).get("action") == m["gold_action"] for m in members):
            solved += 1
    return dict(contrastive_accuracy=solved / max(n, 1), n_pairs=n)


def oracle_decomposition(gold, preds):
    det_ok = det_n = dec_ok = dec_n = ans_ok = ans_n = 0
    for iid, g in gold.items():
        p = preds.get(iid)
        if not p:
            continue
        ga = g["gold_action"]
        if ga in AMBIGUOUS:
            det_n += 1
            if p["action"] in AMBIGUOUS:
                det_ok += 1
                dec_n += 1
                dec_ok += int(p["action"] == ga)
        else:
            ans_n += 1
            ans_ok += int(p["action"] == "answer")
    return dict(perception_recall=det_ok / max(det_n, 1),
                decision_acc_given_detect=dec_ok / max(dec_n, 1),
                answer_fidelity=ans_ok / max(ans_n, 1))


def utility(gold, preds, r_correct, c_wrong, c_ask, c_abstain):
    """Mean decision utility under one cost matrix.

    The right action earns r_correct plus the intrinsic cost of asking or abstaining. A wrong action
    pays that action's cost, and answering when the gold action is ASK or ABSTAIN pays c_wrong.
    """
    u = 0.0
    n = 0
    for iid, g in gold.items():
        p = preds.get(iid)
        if not p:
            continue
        n += 1
        a, ga = p["action"], g["gold_action"]
        if a == ga:
            u += r_correct + (c_ask if a == "ask" else (c_abstain if a == "abstain" else 0.0))
        elif a == "answer":
            u += c_wrong
        elif a == "ask":
            u += c_ask
        else:
            u += c_abstain
    return u / max(n, 1)


def ece(gold, preds, bins=10):
    """Expected calibration error of the action, over predictions that report a confidence."""
    buckets = [[] for _ in range(bins)]
    for iid, g in gold.items():
        p = preds.get(iid)
        if not p or p.get("confidence") is None:
            continue
        conf = min(max(float(p["confidence"]), 0.0), 1.0)
        buckets[min(int(conf * bins), bins - 1)].append((conf, int(p["action"] == g["gold_action"])))
    n = sum(len(b) for b in buckets)
    if n == 0:
        return dict(ece=None, n=0)
    e = 0.0
    for b in buckets:
        if b:
            acc = sum(c for _, c in b) / len(b)
            conf = sum(cf for cf, _ in b) / len(b)
            e += (len(b) / n) * abs(acc - conf)
    return dict(ece=e, n=n)


def evaluate(gold, preds):
    """All metrics for one run."""
    return dict(action=action_accuracy(gold, preds),
                ambiguity_detection=ambiguity_detection(gold, preds),
                contrastive=contrastive_accuracy(gold, preds),
                ece=ece(gold, preds),
                oracle=oracle_decomposition(gold, preds))


def summary(gold, preds):
    """The headline numbers of one run as a flat dict."""
    r = evaluate(gold, preds)
    return dict(n_scored=r["action"]["n"],
                action_acc=r["action"]["accuracy"],
                amb_f1=r["ambiguity_detection"]["f1"],
                contrastive=r["contrastive"]["contrastive_accuracy"],
                perception=r["oracle"]["perception_recall"],
                decision=r["oracle"]["decision_acc_given_detect"],
                answer_fidelity=r["oracle"]["answer_fidelity"],
                action_counts=dict(Counter(p["action"] for i, p in preds.items() if i in gold)))
