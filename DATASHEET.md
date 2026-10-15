# Datasheet for AAA-Bench

Following *Datasheets for Datasets* (Gebru et al., 2021).

## Motivation

**Why was it created?** To measure whether vision-language models decide correctly whether to answer a question
about an image, ask for clarification, or abstain, on a benchmark where that decision cannot be read off the
question wording. In earlier template-built ambiguity benchmarks the label leaks through the text; on an unpaired
version of our own task, a bag-of-words classifier that never sees the image reaches 98.3% action accuracy.

**Who created it?** Shahriar Shakir Sumit and Murat Tahtali (UNSW Canberra) and Seyedali Mirjalili (Torrens
University Australia).

## Composition

**What is an item?** A COCO 2017 image, a question about one object category in it, and the action the scene
licenses: answer, ask or abstain. Items come in counterfactual pairs whose two members share exactly the same
question, are about different images, and have different gold actions.

**How many?** 7,400 items in 3,700 pairs, over 6,927 distinct images (6,646 from COCO train2017, 281 from
val2017) and 59 object categories (26 to 230 items each).

| Family | Decision basis | Gold action | Items |
|---|---|---|---|
| Referential (spatial) | one clear instance | answer | 795 |
| | one instance dominates by area | answer | 858 |
| | several comparable instances | ask | 1,653 |
| Referential (attribute) | one instance dominates by area | answer | 835 |
| | several comparable instances | ask | 835 |
| Counting | separable instances | answer | 622 |
| | a crowd annotation | abstain | 622 |
| Presence | object present | answer | 590 |
| | object absent | abstain | 590 |

Totals: answer 3,700, ask 2,488, abstain 1,212. There are 615 distinct questions, from three wordings per family.

**Splits.** Train 4,994 / validation 956 / test 1,450 items (2,497 / 478 / 725 pairs). Pairs are never split, and
no image appears in two splits. The paper evaluates zero-shot on the test split.

**Fields.** `data/schema.json` describes every field. Besides the label, each item stores its decision basis and
the evidence behind it: the number, areas and boxes of the instances that count as referents.

**Images.** Not included. Items refer to COCO 2017 files by path, and `tools/fetch_images.py` fetches them.

**Known issues.**

* `direct_answer` holds an answer only where COCO fixes one: the side of the referent for spatial answer items
  and "yes" for present objects. It is null for every other item, including colour questions, since COCO has no
  colour labels. No metric uses it.
* In 418 of the 1,653 spatial ASK items (77 of the 317 in the test split) the two largest instances lie on the
  same side of the image, so both interpretations give the same answer to a left-or-right question. Excluding the
  77 test pairs changes contrastive accuracy by at most 0.018 (`reproduce/analysis_same_side.py`).
* One test pair, `AAAv2-009210` / `AAAv2-009211` (pair `P004605`), uses the same image twice: COCO marks the
  airplanes in it both as twelve separate instances and as a crowd, so the rules labelled one copy countable and
  the other a crowd. No policy that sees only the image and question can solve this pair, which caps contrastive
  accuracy on the test split at 724/725.
* For ASK items, `valid_interpretations` names each referent by its side of the image, and two referents can share
  a side ("the right donut" twice).
* Object count carries some signal. An attacker that predicts the action from the instance count alone reaches
  0.650 on the test split and 0.582 on the count-matched pairs (several comparable instances against one
  dominant instance), against a 0.500 majority baseline.

## Collection process

**How were the labels made?** They were derived, not hand-labelled. For the queried category, the instances that
count as referents are the non-crowd ones covering at least 1.5% of the image. One referent means answer; for
several, the largest at least three times the area of the second means answer, and the second at least half the
area of the largest means ask; a crowd annotation means abstain, and so does an absent object. Items in between
these cut-offs were not used. `aaa_bench/construction.py` states the rules, and `tools/verify_labels.py` checks that
every released label follows from the COCO annotations.

**Was there a human check?** Yes. Three annotators chose an action for 400 items drawn with the same rules, 50 per
decision basis, without seeing the labels. They agreed with each other at Fleiss' kappa 0.833. Their majority
matched the derived labels on 52.7% of items, and the disagreement sat in two rules, which were changed: absent
objects had been labelled answer ("no") and the annotators abstained on 49 of 50, so they became abstain; colour
questions about one small or cut-off object had been labelled abstain and the annotators answered 40 of 50, so
that rule was retired and the 3,442 items of its pairs were removed. Under the revised rules the majority matches
72.6% of the 350 remaining study items, and 68.3% of the 300 items whose rules the study did not touch. Two
boundaries are the least settled. When one instance dominates by area, the annotators asked on 27 of 50 items
where the rules say answer, and with several comparable instances they answered on 17 of 50 where the rules say
ask. For crowds they answered (that is, counted) on 27 of 50 items where the rules say abstain.

**Who were the annotators?** Members of the research team, working independently. No personal data was
collected.

## Preprocessing

All pairs built on the retired rule were removed, and absent-object items were relabelled, which made every
remaining pair contrastive (before the study, the 590 present/absent pairs were answer/answer). Images that sat in
two splits were resolved by moving whole groups of pairs that share an image to one split. `construction/prestudy_items.jsonl` is the pool before these steps, and
`tools/relabel_from_human_study.py` turns it into the released files exactly.

## Uses

**Intended.** Evaluating and developing decision policies for vision-language models under visual ambiguity:
when to answer, when to ask, when to abstain. Report contrastive accuracy alongside action accuracy, since the
majority baseline already reaches 0.500 action accuracy, and report the families separately.

**Not intended.** An accuracy leaderboard for answer text (the benchmark scores actions), or a model of ambiguity
in general: it covers object-instance ambiguity in COCO scenes and nothing about intent, social context or time.
Train and validation are for fitting decision heads; do not tune on the test split.

## Distribution

GitHub (code, data, saved model responses, human study) and Hugging Face (data and human study). The annotations
are under CC BY 4.0, the code under MIT. The COCO images stay under the COCO terms of use and their original
Flickr licences.

## Maintenance

The authors maintain the dataset; please report problems through GitHub issues. Corrections will be released as
new versions with a changelog, and `data/checksums.sha256` identifies the files of each version.
