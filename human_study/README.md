# The human study

Three annotators each chose an action (answer, ask or abstain) for 400 items, working independently from the image
and the question, without seeing our labels and without discussing items until all three had finished. The
annotators are members of the research team, so the study is best read as an internal audit of our labelling
rules, one that went on to overturn two of them, rather than as validation by outside raters.

## The items

`sample_items.jsonl` holds the 400 items in the order the tool shows them, with ids `HS-001` to `HS-400`. They were
drawn from the test split of a build made with the benchmark's construction rules, 50 for each of the eight
decision bases that existed at the time (the seven of the final benchmark plus `single_occluded`). Each item
carries the label the rules gave it then, in the same format as `data/`. These are not benchmark items: the
benchmark was rebuilt from a fresh sample under the same rules, so match study items to their labels through this
file and never through benchmark ids. `pair_id` (`HSP-...`) links the study items that formed a counterfactual pair
in that build; 54 pairs have both members in the sample.

## The responses

`annotator_responses.csv` has one row per item: its id, image file, question, and for each of the three annotators
the chosen action (`annotN_action(answer/ask/abstain)`). The `annotN_ambiguous(yes/no)` columns were filled in by
the tool from the action (yes for ask or abstain) and add no separate judgement.

`python reproduce/compute_agreement.py` reports agreement between the annotators and with the labels, per decision
basis, and `python reproduce/analysis_human_ceiling.py` the human reference points used in the paper.

## The instructions

The annotators received the tool in `kit/` with these instructions (wording as given, layout adapted):

> Look at the image + the question, then imagine an AI assistant got that question about that image. Pick ONE:
>
> * ANSWER (key 1): there's one clear answer; the assistant should just answer.
> * ASK (key 2): it's ambiguous: two or more equally likely things could be meant, so the assistant should ask
>   "which one?" first.
> * ABSTAIN (key 3): it can't be answered from this image (thing too tiny / cut off / an uncountable crowd), so the
>   assistant should decline rather than guess.
>
> Rules of thumb:
>
> * Exactly one obvious thing matches, OR one clearly dominates the frame: ANSWER
> * Two+ equally prominent things could be "the X": ASK
> * The relevant thing is tiny, cut off, or an uncountable crowd: ABSTAIN

## Running the tool

`kit/annotate.html` is a single page that runs in any modern browser, with nothing uploaded anywhere. It needs the
images next to it:

```bash
python tools/fetch_images.py --human-study      # puts the 399 study images into human_study/kit/images/
```

Open `kit/annotate.html`, type a name (it keys the saved progress), and label with the keys 1, 2 and 3. Progress
is kept in the browser, so the page can be closed and reopened. When every item is labelled, the download button
saves `aaa_labels_<name>.csv`. Combine the files of several annotators and score them with

```bash
python tools/merge_annotations.py aaa_labels_a.csv aaa_labels_b.csv aaa_labels_c.csv --out my_responses.csv
python reproduce/compute_agreement.py --responses my_responses.csv
```

To annotate other items, replace `kit/items.js` with your own list in the same format and write a matching
`sample_items.jsonl`.
