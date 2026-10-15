"""How the action parser reads the saved responses.

Under the option and chain-of-thought prompts many responses carry an explicit tag ([ANSWER],
[ASK] or [ABSTAIN]); the parser reads the first tag directly. For those responses the tag is the
model's own statement of its action, so it can be used to check the cue rules that decide every
untagged response: this reports how often the cue rules alone recover the tag, and where they
disagree. Writes results/parser_report.txt.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from aaa_bench.outputs import load_runs
from aaa_bench.parsing import cue_action, explicit_tag, parse_action
from aaa_bench.paths import RESULTS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()
    lines = []
    tagged_all = agree_all = 0
    kinds = Counter()
    for strategy in ("direct", "option", "cot"):
        n = tagged = agree = reproduced = 0
        for run in load_runs(strategy).values():
            for r in run.values():
                n += 1
                reproduced += parse_action(r["raw_output"]) == r["action"]
                tag = explicit_tag(r["raw_output"])
                if tag is None:
                    continue
                tagged += 1
                cue = cue_action(r["raw_output"])
                agree += cue == tag
                if cue != tag:
                    head = r["raw_output"].strip().upper()
                    lead = head.startswith("[ANSWER]") or head.startswith("ANSWER:")
                    kinds["a leading answer marker precedes the tag" if lead else "the tag is not at the start"] += 1
        tagged_all += tagged if strategy != "direct" else 0
        agree_all += agree if strategy != "direct" else 0
        share = f"{agree / tagged:.4f}" if tagged else "n/a"
        lines.append(f"{strategy:7s} responses={n:6d}  with a tag={tagged:6d} ({tagged / n:.3f})  "
                     f"cue rules recover the tag: {share}  parser reproduces saved action: {reproduced}/{n}")
    lines.append(f"option + cot: {tagged_all} tagged responses, cue rules alone recover the tag on "
                 f"{agree_all / tagged_all:.4f}")
    lines.append("where the cue rules miss the tag:")
    lines += [f"  {k}: {v}" for k, v in kinds.most_common()]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "parser_report.txt").write_text(text)
    (Path(args.out) / "parser.json").write_text(json.dumps(
        {"tagged_option_cot": tagged_all, "cue_rules_recover_tag": agree_all / tagged_all,
         "misses": dict(kinds)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
