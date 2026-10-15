# The pre-study item pool

`prestudy_items.jsonl` has the 10,842 items (5,421 pairs) that the construction rules produced before the human
study, with their splits and their labels as derived then: absent objects were ANSWER, and colour questions about a
small or cut-off object were ABSTAIN (`single_occluded`). This pool is not the benchmark. It is the input of
`tools/relabel_from_human_study.py`, which applies the study's verdict and gives back `data/` exactly:

```bash
python tools/relabel_from_human_study.py --prestudy construction/prestudy_items.jsonl --out-dir data_rebuilt
```

The construction pipeline in `tools/` writes its intermediate files to this folder as well (`coco_index.json`,
`items_raw.jsonl`, `prestudy/`).
