from aaa_bench.construction import plural, question, referential_rule, referents


def inst(area, box=(100, 100, 50, 50), crowd=0):
    return {"area": area, "bbox": list(box), "iscrowd": crowd}


W, H = 640, 480
A = W * H


def test_small_and_crowd_instances_are_not_referents():
    real, _ = referents([inst(0.01 * A), inst(0.2 * A, crowd=1), inst(0.1 * A)], W, H)
    assert [x["area"] for x in real] == [0.1 * A]


def test_referential_rules():
    one, _ = referents([inst(0.2 * A)], W, H)
    assert referential_rule(one, A, W, H, "spatial") == ("answer", "single_clear")
    dominant, _ = referents([inst(0.3 * A), inst(0.05 * A)], W, H)
    assert referential_rule(dominant, A, W, H, "spatial") == ("answer", "multi_dominant")
    comparable, _ = referents([inst(0.2 * A), inst(0.15 * A)], W, H)
    assert referential_rule(comparable, A, W, H, "color") == ("ask", "multi_comparable")
    between, _ = referents([inst(0.2 * A), inst(0.08 * A)], W, H)
    assert referential_rule(between, A, W, H, "color")[0] is None


def test_occlusion_rule_applies_to_colour_only():
    cut, _ = referents([inst(0.2 * A, box=(0, 100, 200, 200))], W, H)
    assert referential_rule(cut, A, W, H, "color") == ("abstain", "single_occluded")
    assert referential_rule(cut, A, W, H, "spatial") == ("answer", "single_clear")


def test_question_templates():
    assert question("counting", 0, "bus") == "How many buses are in the image?"
    assert question("presence", 0, "apple") == "Is there an apple in the image?"
    assert plural("knife") == "knifes" and plural("toothbrush") == "toothbrushes"
