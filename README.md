# AAA-Bench: Counterfactual, Verifier-Grounded Evaluation of Vision-Language Models as Decision-Makers Under Visual Ambiguity

Shahriar Shakir Sumit<sup>1</sup>, Murat Tahtali<sup>1</sup>, Seyedali Mirjalili<sup>2</sup>

<sup>1</sup>UNSW Canberra, Australia &nbsp; <sup>2</sup>Torrens University Australia

IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2027

Dataset: [huggingface.co/datasets/Shahriar10/AAA-Bench](https://huggingface.co/datasets/Shahriar10/AAA-Bench)

## Overview

AAA-Bench evaluates whether a vision-language model (VLM) makes the right decision when an image does not settle
a question: answer it, ask a clarifying question, or abstain. Each item is one member of a counterfactual pair.
The two members of a pair have exactly the same question but different COCO images, and their correct actions
differ. As a result, the question text carries no information about the correct action: a text-only classifier
reaches the 50% majority baseline on AAA-Bench, against 98.3% on an unpaired construction of the same task. Gold
actions are derived from COCO instance annotations by stated rules, and a three-annotator study audits these rules.

This repository contains the benchmark, the evaluation code, the responses of the seven models evaluated in the
paper, and the scripts that reproduce every table, figure and number in the paper.

## Dataset

AAA-Bench contains 7,400 items in 3,700 counterfactual pairs over 59 COCO object categories.

| Family | Example question | Items | Actions in a pair |
|---|---|---|---|
| Referential (spatial) | "Is the kite on the left or the right side of the image?" | 3,306 | answer / ask |
| Referential (attribute) | "What color is the couch?" | 1,670 | answer / ask |
| Counting | "How many buses are in the image?" | 1,244 | answer / abstain |
| Presence | "Is there an apple in the image?" | 1,180 | answer / abstain |

**Splits.** 4,994 training, 956 validation and 1,450 test items. The splits are made by pair and by image, so no
image appears in two splits and no pair is divided. The paper evaluates zero-shot on the test split.

**Labels.** Gold actions follow from the COCO instance annotations. For a referential question, only the
non-crowd instances of the queried category that cover at least 1.5% of the image are considered. One such
instance, or several with one at least three times larger than the next, gives ANSWER. Two or more instances of
comparable size, where the second is at least half the area of the largest, give ASK. A counting question gives ANSWER
when the instances are annotated separately and ABSTAIN when COCO marks them as a crowd. A presence question gives
ANSWER when the category is present and ABSTAIN when it is absent. Each item stores the boxes and areas behind its
label, and `tools/verify_labels.py` recomputes every label from the COCO annotations.

**Human study.** Three annotators labelled 400 items built with the same rules, with inter-annotator agreement of
Fleiss' kappa 0.833. Their majority agrees with the final labels on 72.6% of the items, and on 68.3% of the items
whose rules the study did not change. The study changed two rules: absent objects are labelled ABSTAIN (the
majority abstained on 49 of 50), and the rule that labelled small or truncated objects ABSTAIN was removed (the
majority answered 40 of 50). The items, the annotations and the annotation tool are in `human_study/`.

**Format.** `data/train.jsonl`, `data/val.jsonl` and `data/test.jsonl` hold one item per line; `data/schema.json`
describes every field, and `croissant.json` describes the dataset in the Croissant format.

```python
from aaa_bench import load_split, pairs_of

test = load_split("test")        # 1,450 items
pairs = pairs_of(test)           # 725 pairs, each [item, partner]

# or, from Hugging Face
from datasets import load_dataset
test = load_dataset("Shahriar10/AAA-Bench", split="test")
```

**Images.** The COCO images are not redistributed. `tools/fetch_images.py` downloads the COCO 2017 images that an
item list needs, or links them from a local copy.

## Evaluation

**Protocol.** In turn 1 the model receives the image and the question under one of three prompts: `direct` (the
question only), `option` (the three actions are offered; the main setting) or `cot` (the actions are offered after
step-by-step reasoning). `aaa_bench.parsing.parse_action` maps the response to an action. A model that asks
receives the item's clarification and answers again in turn 2. All metrics use the turn-1 action.

| Metric | Definition |
|---|---|
| Contrastive accuracy (CA) | Fraction of pairs whose two items both receive the correct action. The primary metric: any policy that depends only on the question scores 0, and a random text-only policy at most 0.25. |
| Action accuracy | Fraction of items with the correct action. Always answering scores 0.500. |
| Ambiguity F1 | F1 score for flagging an item as ambiguous (ASK or ABSTAIN). |
| Perception, decision | Recall of the items whose correct action is ASK or ABSTAIN, and accuracy of the choice between ASK and ABSTAIN on the items a model flags. |
| Cost-robust utility | Fraction of random cost matrices under which a model has higher utility than always answering. |

## Results

Test split, `option` prompt.

| Model | Action acc. | Ambiguity F1 | CA | Perception | Decision |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B | 0.55 | 0.42 | **0.17** | 0.30 | 0.67 |
| Phi-3.5-Vision | 0.54 | 0.36 | 0.16 | 0.24 | 0.77 |
| LLaVA-NeXT-7B | 0.57 | 0.31 | 0.15 | 0.19 | 0.92 |
| InternVL3-8B | 0.56 | 0.27 | 0.14 | 0.16 | 0.93 |
| Qwen2-VL-7B | 0.50 | 0.29 | 0.08 | 0.19 | 0.64 |
| LLaVA-1.5-7B | 0.51 | 0.07 | 0.02 | 0.04 | 1.00 |
| Idefics3-8B | 0.50 | 0.01 | 0.01 | 0.01 | 0.80 |
| Majority baseline | 0.50 | 0.00 | 0.00 | 0.00 | - |

For reference, a random policy that reads only the question reaches a CA of 0.251. A COCO-pretrained detector
(YOLO11-L) that applies the labelling rules to its own detections reaches 0.742, and the human majority reaches
0.560 on the 50 study pairs. Results for the other prompts are produced by `reproduce/run_all.sh`.

## Repository structure

```
aaa_bench/       data loading, action parser, two-turn protocol, metrics, and wrappers for the seven models,
                 pinned to the evaluated checkpoints
data/            the benchmark (train, val, test), its schema, checksums and licence
human_study/     the 400 study items, the three annotators' labels and the browser annotation tool
model_outputs/   responses of the seven models to the test split under the three prompts; the two detector
                 pipelines and their cached detections are in reference/
construction/    the item pool before the human study, from which the benchmark is derived
reproduce/       scripts that reproduce the tables, figures and numbers of the paper on a CPU
tools/           image download, label verification and the construction pipeline
tests/           unit tests
```

## Installation

```bash
git clone https://github.com/ShahriarShakirSumit/AAA-Bench.git
cd AAA-Bench
python -m venv .venv && source .venv/bin/activate     # Python 3.11
pip install -e . -r requirements.txt
pytest -q tests
```

The `aaa_bench` package depends only on the Python standard library. `requirements.txt` pins the versions of
numpy, matplotlib, scikit-learn and pytest used for the reproduction scripts and the tests. With other versions
of scikit-learn, the learned-head results may differ in the third decimal; no other result depends on the
versions.

## Reproducing the paper (CPU, about one minute)

No GPU, model weights or images are needed, since all model responses are stored in `model_outputs/`.

```bash
bash reproduce/run_all.sh
```

The tables (in LaTeX), figures and reports are written to `results/`. The final step,
`reproduce/audit_claims.py`, compares each table with the version in the paper (`reproduce/expected/`) and checks
the numbers stated in the text of the paper. Each script can also be run on its own; its header describes what it
computes.

The detector pipelines use cached detections from `model_outputs/reference/detector_features.json`. Recomputing
the detections requires the images, the `ultralytics` package and the YOLO11-L weights:
`python reproduce/analysis_detector.py --weights yolo11l.pt --coco-dir coco`.

Figure 1 shows example images and therefore requires them:

```bash
python tools/fetch_images.py --split all          # or --from-dir /data/coco to link a local COCO copy
python reproduce/make_examples.py --coco-dir coco
```

## Evaluating a model (GPU)

**1. Images.**

```bash
python tools/fetch_images.py --split test                        # downloads the 1,424 test images into coco/
python tools/fetch_images.py --split test --from-dir /data/coco   # or links them from a COCO 2017 copy
```

**2. Inference.** All models were run with the library versions in `requirements-gpu.txt` on one NVIDIA V100
GPU with 32 GB of memory. The wrappers load the exact checkpoint revisions that were evaluated.

```bash
pip install -r requirements-gpu.txt
python -m aaa_bench.evaluate --model qwen2.5-vl-7b --strategy option --coco-dir coco
```

Available models: `llava-1.5-7b`, `llava-next-7b`, `qwen2-vl-7b`, `qwen2.5-vl-7b`, `internvl3-8b`,
`phi-3.5-vision` and `idefics3-8b`. Available prompts: `direct`, `option` and `cot`. Responses are written to
`runs/<model>_<strategy>.jsonl` in the format of `model_outputs/`, and an interrupted run resumes where it stopped.
Use `--limit 20` for a quick test.

**3. Scoring.**

```bash
python -m aaa_bench.score runs/qwen2.5-vl-7b_option.jsonl
python -m aaa_bench.score model_outputs/*_option.jsonl        # the seven models of the paper
```

**Other models.** Any object with a `model_name` attribute and a `generate(image_path, prompt)` method that
returns text can be evaluated with the same protocol:

```python
from aaa_bench import load_split
from aaa_bench.protocol import run_item

class MyModel:
    model_name = "my-model"

    def generate(self, image_path, prompt):
        ...   # return the model's response as a string

responses = [run_item(MyModel(), item, "option", "coco") for item in load_split("test")]
```

Save the responses as JSON lines and score them as above. Responses obtained elsewhere should be mapped to actions
with `aaa_bench.parsing.parse_action`, the parser used for all results in the paper, so that scores remain
comparable.

## Verifying and rebuilding the labels

All labels follow from the COCO 2017 instance annotations
([annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)). The
following command recomputes the gold action, decision basis, evidence boxes, question and image path of all
7,400 items and reports any mismatch (about 3 GB of memory):

```bash
python tools/verify_labels.py --annotations /data/coco/annotations
```

The construction consists of four steps. The first two draw a new sample under the same rules, so they produce a
new benchmark of the same design rather than the released files:

```bash
python tools/build_coco_index.py --annotations /data/coco/annotations --out construction/coco_index.json
python tools/build_dataset.py --index construction/coco_index.json --out construction/items_raw.jsonl
python tools/make_splits.py --items construction/items_raw.jsonl --out-dir construction/prestudy
python tools/relabel_from_human_study.py --prestudy construction/prestudy/full.jsonl --out-dir data_rebuilt
```

The last step applies the outcome of the human study. Applied to the released pre-study pool, it reproduces
`data/` exactly:

```bash
python tools/relabel_from_human_study.py --prestudy construction/prestudy_items.jsonl --out-dir data_rebuilt
cmp data_rebuilt/test.jsonl data/test.jsonl
```

## Human study

`human_study/README.md` describes the study and the instructions given to the annotators. To repeat the study,
download the images with `python tools/fetch_images.py --human-study` and open `human_study/kit/annotate.html` in
a browser; the tool runs locally and saves each annotator's labels as a CSV file. `tools/merge_annotations.py`
combines the files and `reproduce/compute_agreement.py --responses <file>` computes the agreement.

## Limitations

* AAA-Bench covers object-instance ambiguity in COCO scenes. Relational, social and temporal ambiguity are out of
  scope.
* Only the action is scored. The `direct_answer` field is filled only where COCO determines the answer (the side
  of the image for spatial items and "yes" for present objects).
* In 418 of the 1,653 spatial ASK items (77 of the 725 test pairs), both instances lie on the same side of the
  image, so the two interpretations give the same answer. Excluding these pairs changes contrastive accuracy by at
  most 0.018.
* One test pair uses the same image twice, because COCO annotates the objects in it both as instances and as a
  crowd. No policy can solve this pair, so the maximum test CA is 724/725.
* Agreement between the human majority and the labels is lowest for one dominant instance and for crowds (0.46
  each). The annotators were members of the research team.

`DATASHEET.md` gives further details.

## Licence

The code is released under the MIT licence (`LICENSE`). The annotations, the human-study files and the model
responses are released under CC BY 4.0 (`data/LICENSE.md`). The COCO images remain subject to the COCO terms of
use.

## Citation

If you use AAA-Bench, please cite:

```bibtex
@inproceedings{sumit2027aaabench,
  title     = {{AAA-Bench}: Counterfactual, Verifier-Grounded Evaluation of Vision-Language Models as
               Decision-Makers Under Visual Ambiguity},
  author    = {Sumit, Shahriar Shakir and Tahtali, Murat and Mirjalili, Seyedali},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## Contact

For questions or problems, please open a GitHub issue or contact Shahriar Shakir Sumit
(m.sumit@unsw.edu.au or shahriar9121@gmail.com).

## Acknowledgements

Experiments were performed on the computational facility of the National Computational Infrastructure (NCI)
through the UNSW HPC Scheme (DOI: 10.26190/PMN5-7J50).
