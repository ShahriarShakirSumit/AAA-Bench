from aaa_bench.data import by_id, load_split
from aaa_bench.metrics import contrastive_accuracy, evaluate, utility


def _gold():
    return by_id(load_split("test"))


def test_constant_policies_score_zero_contrastive():
    gold = _gold()
    for a in ("answer", "ask", "abstain"):
        preds = {i: {"action": a} for i in gold}
        assert contrastive_accuracy(gold, preds)["contrastive_accuracy"] == 0.0


def test_oracle_is_perfect():
    gold = _gold()
    preds = {i: {"action": g["gold_action"]} for i, g in gold.items()}
    r = evaluate(gold, preds)
    assert r["action"]["accuracy"] == 1.0
    assert r["contrastive"]["contrastive_accuracy"] == 1.0
    assert r["contrastive"]["n_pairs"] == 725
    assert r["oracle"]["perception_recall"] == 1.0
    assert r["ambiguity_detection"]["f1"] == 1.0


def test_always_answer_is_the_majority_baseline():
    gold = _gold()
    preds = {i: {"action": "answer"} for i in gold}
    assert evaluate(gold, preds)["action"]["accuracy"] == 0.5


def test_missing_prediction_breaks_a_pair():
    gold = _gold()
    preds = {i: {"action": g["gold_action"]} for i, g in gold.items()}
    preds.pop(next(iter(gold)))
    assert contrastive_accuracy(gold, preds)["contrastive_accuracy"] == 724 / 725


def test_utility_rewards_the_right_action():
    gold = {"a": {"gold_action": "ask"}, "b": {"gold_action": "answer"}}
    right = {"a": {"action": "ask"}, "b": {"action": "answer"}}
    wrong = {"a": {"action": "answer"}, "b": {"action": "answer"}}
    assert utility(gold, right, 1.0, -4.0, -0.5, -1.0) == (0.5 + 1.0) / 2
    assert utility(gold, wrong, 1.0, -4.0, -0.5, -1.0) == (-4.0 + 1.0) / 2
