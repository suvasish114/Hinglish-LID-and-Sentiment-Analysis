# Language Identification and Sentiment Analysis from Hindi English Code-Mixed Texts

This project is under active development for partial fullfillment of grade requirement for course code CS613-NLP at Indian Institute of Technology Gandhinagar.

### Enviornment setup

This project require specified python libraries. Make sure to use python enviornment for the experiments.

```sh
python3 -m venv venv # create environment (one time)
source venv/bin/activate # activate your enviornment (linux)
pip install -r requirements.txt # install all required packages (one time)
```

If you install any new packages, make sure to add those in `requirements.txt` by using

```sh
pip freeze > requirements.txt
```
> This project has been tested on CUDA12.4

### Download dataset

For all the experiment the datasets, models and hyperparameters remain same. Please use the data generation script to download and process the SentiMix dataset. Alongside, use `eval.py` to generate the result. Use below command to download and process the dataset.

```sh
python3 get_sentimix.py # one time 
```

This will download all the required files from the remote server and process it to `csv` format and store them in your current folder's `dataset/SentiMix/`. For evaluation use 

```sh
python3 eval.py
```

For experiment specific details, use branches `exp0`, `exp1`, ...

---

## Experiment 2 - Switch Point + Sentiment (branch `exp2`)

**Research question.** If the model is given the language *switch point* information, does the sentiment score improve?

```
Tokens:  Movie  bahut  acchi  thi  but  ending  was  terrible
LID:       EN     HI     HI    HI   EN     EN     EN     EN
Switch:     0      1      0     0    1      0      0      0
```

`switch_i = 1 if LID_i != LID_{i-1} else 0` (`switch_0 = 0`). Tokens tagged `o` (punctuation, mentions,
emoji) are transparent by default, i.e. `hin o eng` is a single switch; use `--naive_switch` to change that.

**Model.** Any HuggingFace encoder (default `xlm-roberta-base`) followed by *switch-biased attention pooling*

    a_i = softmax( W_h h_i + W_s e_i^sw + b ),     s = sum_i a_i h_i,     logits = W_c s

where `e_i^sw` is a small learned embedding (16-d) of the switch feature. Variants compared (`--pooling`):

| `--pooling` | switch feature `e_i^sw`                                    |
|-------------|-------------------------------------------------------------|
| `none`      | *baseline*: plain attention pooling, no switch information   |
| `binary`    | embedding of the binary switch flag {0, 1}                   |
| `distance`  | embedding of bucketed distance to nearest switch {0,1,2,3+}  |
| `cls`/`mean`| extra reference points (no attention pooling)                |

Switch points come either from the gold SentiMix LID tags (`--switch_source gold`) or from a small
token-level LID tagger (`lib/lid_tagger.py`, EN/HI/O) that is trained once and then **frozen**
(`--switch_source predicted`); the sentiment model never back-propagates into the tagger.

### Layout

```
lib/data.py        build/load train, dev, test CSVs (re-uses the shared get_sentimix.py, unchanged)
lib/switch.py      switch points, distance buckets, word -> sub-word alignment
lib/lid_tagger.py  token-level LID tagger (fine-tuned encoder), writes {split}_pred_tags.csv
lib/model.py       SwitchAwareSentimentModel (encoder + switch-biased attention pooling)
lib/train.py       training loop; writes results/exp2/<run>.csv + ground.csv for evaluate_all.py
main.py            CLI gateway (all hyper-parameters)
run.sh             full reproduction: tagger + 2 x 3 sentiment runs + shared evaluation
```

### Run

```sh
cp .env.example .env            # DATA_HOME=dataset, HG_DATACARD=RTT1/SentiMix
python3 lib/data.py             # download + build train/dev/test (test labels are joined from test_labels_hinglish.txt)
python3 lib/lid_tagger.py       # 1. LID tagger  -> checkpoints/lid_xlm-roberta-base/
python3 main.py --pooling none                        # 2a. baseline
python3 main.py --pooling binary                      # 2b. binary switch embedding (gold LID)
python3 main.py --pooling distance                    # 2c. distance-to-switch embedding (gold LID)
python3 main.py --pooling binary --switch_source predicted --pred_tags_dir checkpoints/lid_xlm-roberta-base
cd results/exp2 && python3 ../../evaluate_all.py --ground ground.csv --pred-glob "*.csv"   # 3. shared eval
```

or simply `bash run.sh [MODEL_NAME]`. Smoke test on CPU/GPU with
`python3 main.py --max_train_samples 400 --max_eval_samples 200 --epochs 1`.
Large decoder models (QLoRA): `python3 main.py --model_name Qwen/Qwen2.5-7B-Instruct --lora --load_in_4bit --batch_size 8 --lr 1e-4 --epochs 2`.
Other encoders: `--model_name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --lr 5e-5`.

Each run writes `results/exp2/<run>_metrics.json` with dev/test accuracy and macro P/R/F1
(same formula as `evaluate_all.py`) plus a breakdown on sentences **with** vs **without** a switch.

### Results (SentiMix test, 2 944 roman tweets, macro-F1 from the shared `evaluate_all.py`)

Frozen LID tagger (`lib/lid_tagger.py`, xlm-roberta-base, 2 epochs): test token-acc 0.912 / macro-F1 0.922.

Three encoders, same data, same training loop, same pooling variants. Small encoders: 3 epochs, batch 32,
full fine-tuning (lr 2e-5 for XLM-R, 5e-5 for MiniLM). Qwen2.5-7B-Instruct: LoRA r=8 on all linear layers,
4-bit NF4 base, batch 8, lr 1e-4, 2 epochs, ~40 min per run on one RTX A5000.
Cells with +- are mean +- std over seeds {42, 1, 2} (3 seeds) or {42, 1} where marked (2 seeds); the rest are seed 42 only.

| encoder | params | baseline (none) | binary, gold | distance, gold | binary, predicted | distance, predicted | distance + mlp, gold |
|---|---|---|---|---|---|---|---|
| MiniLM-L12 (paraphrase-multilingual) | 118M | 0.695 +- 0.006 | 0.693 +- 0.000 | 0.698 +- 0.006 | 0.689 | 0.704 | 0.700 |
| xlm-roberta-base | 278M | 0.715 +- 0.004 | 0.715 +- 0.003 | 0.712 +- 0.004 | 0.705 | 0.711 | 0.719 |
| Qwen2.5-7B-Instruct (QLoRA) | 7.6B | 0.729 +- 0.000 (2 seeds) | **0.735** | 0.730 | 0.724 | 0.734 | 0.732 +- 0.011 (2 seeds) |

Full per-run table incl. dev F1 / accuracy / P / R: `python3 lib/summarize.py` -> `results/exp2/summary.csv`.

Test macro-F1 by number of switch points per tweet (`lib/analyze.py`; every test tweet has >= 1 switch):

| encoder / pooling | 1-2 switches (n=916) | 3-5 (n=1640) | 6-9 (n=381) |
|---|---|---|---|
| xlm-roberta-base, none (3 seeds) | 0.734 | 0.708 | 0.699 |
| xlm-roberta-base, binary (3 seeds) | 0.732 | 0.708 | 0.705 |
| xlm-roberta-base, distance (3 seeds) | 0.731 | 0.703 | 0.703 |
| Qwen2.5-7B, none | 0.729 | 0.728 | 0.730 |
| Qwen2.5-7B, binary | 0.714 | 0.740 | **0.754** |
| Qwen2.5-7B, distance | 0.718 | 0.731 | 0.749 |
| Qwen2.5-7B, distance + mlp | 0.743 | 0.735 | 0.745 |

**Take-aways.**
1. Encoder size dominates: baselines go 0.695 -> 0.715 -> 0.729 from MiniLM to XLM-R to Qwen-7B, a bigger spread than any pooling variant produces within one encoder.
2. On the two small encoders, switch information does **not** move overall test F1 beyond seed noise (+-0.4-0.6 F1).
3. On Qwen2.5-7B every seed-42 switch-aware variant is at or above the baseline (0.730-0.740 vs 0.729), but a second seed of `distance + mlp` scored 0.724, so its 2-seed mean is 0.732 +- 0.011 vs a very stable baseline 0.729 +- 0.000. The Qwen gains are therefore also within seed noise; the single best run (0.740) was a favourable seed.
4. Where any gain lives: on switch-heavy tweets (6-9 switches) Qwen `binary` and `distance` beat the baseline by +2.4 / +1.9 F1 (seed 42), while on 1-2-switch tweets the linear variants are *below* the baseline (-1.1 to -1.5). The switch bias helps where there are many switches to exploit and slightly hurts where there are few; the two effects roughly cancel in the overall score. This bucket pattern is the most promising lead and needs more seeds to confirm.
5. Switch points from the frozen tagger (91% token accuracy) instead of gold LID cost 0-1 F1; the `distance` variant is the more robust of the two to tagger noise.
6. Dev F1 (~0.63-0.66) is consistently ~7 points below test F1 for every model; the SentiMix dev split is harder than test. Dev is only used to pick the best epoch.

Next steps: 3+ seeds for every Qwen variant (the deciding evidence is still missing), fuse `e_i^sw` into `h_i` (not only into the attention score),
PESTO-style switch-relative positional encoding, and the contrastive variants from the experiment notes.

Notes: the `emt` (emoticon) tag of SentiMix is folded into `o`; a handful of rows with a non-numeric
id emitted by the shared parser (a token literally spelled `meta`) and one exact duplicate id per
training split are dropped in `lib/data.py`. The test split (2 944 roman rows) is unaffected.

### Authors

- Parth Dangi - [parthgdangi](https://github.com/parthgdangi)
- Nishant Sharma - [rockbnishant](https://github.com/Rockbnishant)
- Durgesh Mishra - [durg3sh10](https://github.com/durg3sh10)
- Rahul Kumawat - [rahulkumawat835](https://github.com/rahulkumawat835)
- Lavish Jangid - [lavish-j](https://github.com/lavish-j)
- Harshiddhi Pathak - [horikita-99](https://github.com/horikita-99)
- Anuj Tiwari - [anujjtiwari](https://github.com/anujjtiwari) 
- Harsh Krishnadev Dubey [Hrshhh](https://github.com/Hrshhh)
- Suvasish Das - [suvasish114](https://github.com/suvasish114)


### Contributors

<a href="https://github.com/suvasish114/Hinglish-LID-and-Sentiment-Analysis/graphs/contributors"><img src="https://contrib.rocks/image?repo=suvasish114/Hinglish-LID-and-Sentiment-Analysis"/></a> 
