"""Figure 1 of the paper: one worked counterfactual pair per family (spatial, counting, presence).

Each row is one question asked about two images whose gold actions differ. For each family the pair
with the shortest question in a category not used yet is shown. The spatial row uses a count-matched
pair (one dominant instance against several comparable ones), and its ASK image must have its two
largest instances clearly on opposite sides (box centres below 0.35 and above 0.65 of the image
width), so the two readings of the question give different answers. The counting row shows pair
P004558: two teddy bears against a large pile of them that COCO annotates as a crowd. A crowd image is
labelled without an instance count, because the count stored for it does not include the objects
inside the crowd region. Needs the COCO images (tools/fetch_images.py) and Pillow.

    python reproduce/make_examples.py --coco-dir coco
"""

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from aaa_bench.data import load_items, pairs_of  # noqa: E402
from aaa_bench.paths import RESULTS  # noqa: E402

ASPECT = 1.55
ACT = {"answer": ("#009E73", "ANSWER"), "ask": ("#0072B2", "ASK"), "abstain": ("#D55E00", "ABSTAIN")}
BASIS_TEXT = {"single_clear": "one clear instance", "multi_dominant": "several, one dominant",
              "multi_comparable": "several, comparable", "distinct": "countable instances",
              "crowd": "uncountable crowd", "present": "category present", "absent": "category absent"}
WANT = [("referential_spatial", ("answer", "ask")), ("counting", ("answer", "abstain")),
        ("presence", ("answer", "abstain"))]
COUNTING_PAIR = "P004558"


def crop(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if w / h > ASPECT:
        nw = int(h * ASPECT)
        return im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    nh = int(w / ASPECT)
    return im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))


def clearly_two_sides(item, coco_dir):
    width = Image.open(Path(coco_dir) / item["image_path"]).size[0]
    centres = [(b[0] + b[2] / 2) / width for b in item["visual_evidence"]["boxes"][:2]]
    return min(centres) < 0.35 and max(centres) > 0.65


def pick_rows(coco_dir):
    rows, used = [], set()
    for family, order in WANT:
        cands = []
        for pair in pairs_of(load_items()):
            if pair[0]["ambiguity_subtype"] != family or {x["gold_action"] for x in pair} != set(order):
                continue
            if not all((Path(coco_dir) / x["image_path"]).exists() for x in pair):
                continue
            if family == "counting" and pair[0]["pair_id"] != COUNTING_PAIR:
                continue
            if family == "referential_spatial":
                ask = next(x for x in pair if x["gold_action"] == "ask")
                answer = next(x for x in pair if x["gold_action"] == "answer")
                if answer["decision_basis"] != "multi_dominant" or not clearly_two_sides(ask, coco_dir):
                    continue
            cands.append((pair[0]["visual_evidence"]["category"], pair))
        fresh = [c for c in cands if c[0] not in used] or cands
        cat, pair = min(fresh, key=lambda c: len(c[1][0]["question"]))
        used.add(cat)
        rows.append(sorted(pair, key=lambda x: order.index(x["gold_action"])))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco-dir", required=True)
    ap.add_argument("--out", default=str(RESULTS / "figures"))
    args = ap.parse_args()
    rows = pick_rows(args.coco_dir)
    n = len(rows)
    W = 3.35
    L, R, WS = 0.006, 0.994, 0.016
    panel_w = (W * (R - L) - WS * W) / 2
    panel_h = panel_w / ASPECT
    cap_h, top_pad, bot_pad = 0.30, 0.19, 0.155
    H = n * (panel_h + cap_h) + top_pad + bot_pad
    fig, axes = plt.subplots(n, 2, figsize=(W, H))
    fig.subplots_adjust(left=L, right=R, top=1 - top_pad / H, bottom=bot_pad / H,
                        wspace=WS * W / panel_w, hspace=cap_h / panel_h)
    for pair, row_axes in zip(rows, axes):
        for ax, m in zip(row_axes, pair):
            ax.imshow(crop(Path(args.coco_dir) / m["image_path"]))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#cccccc")
                s.set_linewidth(0.5)
            colour, label = ACT[m["gold_action"]]
            ax.text(0.045, 0.94, label, transform=ax.transAxes, color="white", fontsize=5.6, fontweight="bold",
                    va="top", ha="left", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.30", facecolor=colour, edgecolor="none"))
            ev = m["visual_evidence"]
            count = "" if m["decision_basis"] == "crowd" else f" ×{ev['instance_count']}"
            ax.set_xlabel(f"{ev['category']}{count}: {BASIS_TEXT[m['decision_basis']]}",
                          fontsize=5.4, labelpad=1.6)
        bl, br = row_axes[0].get_position(), row_axes[1].get_position()
        fig.text((bl.x0 + br.x1) / 2, bl.y1 + 0.006,
                 "\n".join(textwrap.wrap(f"“{pair[0]['question'].strip()}”", 50)),
                 ha="center", va="bottom", fontsize=6.4, style="italic")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"examples.{ext}", dpi=200)
    print("  " + "; ".join(f"{p[0]['visual_evidence']['category']}: {p[0]['item_id']} / {p[1]['item_id']}" for p in rows))
    print(f"  wrote {out / 'examples.pdf'}")


if __name__ == "__main__":
    main()
