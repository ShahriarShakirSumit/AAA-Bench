"""The rules that derive an item's gold action from COCO instance annotations.

For a question about category c in an image, the instances of c that count as referents are the
non-crowd ones covering at least MIN_FRAC of the image, largest first. Then:

  referential families (spatial, colour)
    no referent                                  -> not used
    one referent, clearly visible                -> ANSWER (single_clear)
    one referent, small or cut off by the frame  -> ABSTAIN (single_occluded, colour only; this rule
                                                    was retired after the human study)
    largest >= DOMINANT x second largest         -> ANSWER (multi_dominant)
    second largest >= COMPARABLE x largest       -> ASK (multi_comparable)
    anything in between                          -> not used
  counting
    two or more non-crowd instances              -> ANSWER (distinct)
    a crowd annotation for c                     -> ABSTAIN (crowd)
  presence
    c present                                    -> ANSWER (present)
    c absent                                     -> ABSTAIN (absent; the rule before the human study
                                                    was ANSWER)
"""

MIN_FRAC = 0.015
SMALL_FRAC = 0.03
DOMINANT = 3.0
COMPARABLE = 0.5

SPATIAL_T = ["Is the {c} on the left or the right side of the image?",
             "Which side of the image is the {c} on, the left or the right?",
             "On which side of the picture is the {c}?"]
COLOR_T = ["What color is the {c}?", "What is the colour of the {c} in the image?",
           "Describe the color of the {c}."]
COUNT_T = ["How many {cp} are in the image?", "Count the {cp} in the picture.",
           "How many {cp} can you see in the image?"]
PRES_T = ["Is there {art} {c} in the image?", "Does the image contain {art} {c}?",
          "Can you see {art} {c} in the picture?"]
TEMPLATES = {"referential_spatial": SPATIAL_T, "referential_color": COLOR_T, "counting": COUNT_T,
             "presence": PRES_T}

# Changes the human study made to the derived labels (see relabel_from_human_study.py).
STUDY_RELABEL = {"absent": "abstain"}
STUDY_RETIRED = {"single_occluded"}


def article(w):
    return "an" if w[0].lower() in "aeiou" else "a"


def plural(w):
    if w.endswith(("s", "x", "ch", "sh")):
        return w + "es"
    if w.endswith("y") and w[-2] not in "aeiou":
        return w[:-1] + "ies"
    return w + "s"


def question(family, template, category):
    return TEMPLATES[family][template].format(c=category, cp=plural(category), art=article(category))


def touches_edge(box, W, H, m=3):
    x, y, w, h = box
    return x <= m or y <= m or x + w >= W - m or y + h >= H - m


def side_of(box, W):
    x, _, w, _ = box
    cx = (x + w / 2) / max(W, 1)
    return "left" if cx < 0.44 else ("right" if cx > 0.56 else "center")


def referents(instances, W, H):
    """(non-crowd instances covering at least MIN_FRAC of the image, largest first; image area)."""
    A = W * H
    real = sorted([x for x in instances if not x["iscrowd"] and x["area"] >= MIN_FRAC * A], key=lambda x: -x["area"])
    return real, A


def referential_rule(real, A, W, H, kind):
    """(action, decision basis) for a referring question, or (None, reason) if the item is not used."""
    n = len(real)
    if n == 0:
        return None, "no prominent instance"
    if n == 1:
        occluded = real[0]["area"] < SMALL_FRAC * A or touches_edge(real[0]["bbox"], W, H)
        if kind == "color" and occluded:
            return "abstain", "single_occluded"
        return "answer", "single_clear"
    a1, a2 = real[0]["area"], real[1]["area"]
    if a1 >= DOMINANT * a2:
        return "answer", "multi_dominant"
    if a2 >= COMPARABLE * a1:
        return "ask", "multi_comparable"
    return None, "borderline"


def evidence(category, instances):
    return {"category": category, "instance_count": len(instances),
            "areas": [round(x["area"]) for x in instances[:4]], "boxes": [x["bbox"] for x in instances[:4]]}
