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

## Experiment 0 - SentiMix Hi-En baselines (branch `exp0`)

**Goal.** Re-create the baseline of the SentiMix paper (Patwa et al., 2020, *SemEval-2020 Task 9: Overview of
Sentiment Analysis of Code-Mixed Tweets*, Section 5) on our shared data pipeline, run the same recipe with the
models / hyper-parameters fixed for all our experiments, and provide the paper's evaluation
(precision, recall, F1 per class, weighted F1, micro-F1) for any `id,predictions` file.

### The paper's baseline (Section 5)

| | |
|---|---|
| model | `bert-base-multilingual-cased` (M-BERT), fine-tuned separately per language pair |
| input | tweet text, maximum sequence length 56 for Hinglish |
| training | 3 epochs, AdamW, learning rate 2e-5, all weights trained |
| metric | **weighted F1** (per-class F1 averaged with support weights); precision / recall per class |
| Hinglish test | Pos 72.8 / 68.8 / 70.7, Neu 56.2 / 60.2 / 58.1, Neg 69.1 / 67.4 / 68.3 (P / R / F1), **weighted F1 65.4** (rank 45 of 62); best system (KK2018, XLM-R) 75.0 |

### Layout

```
baseline.py             CLI. --paper = the Section-5 recipe above; otherwise the project recipe
eval_sentimix.py        paper-style evaluation of id,predictions CSVs: per-class P/R/F1, weighted / macro / micro F1
lib/sentimix_data.py    loads the CSVs written by the shared get_sentimix.py; sub-word <-> word alignment
lib/sentiment_model.py  any HF encoder / decoder + pooling + linear head; freezes all but the last k blocks
lib/trainer.py          training loop; writes results/exp0/<run>.csv + ground.csv in the format of evaluate_all.py
lib/metrics.py          metric definitions (scikit-learn)
run.sh                  every run below + both evaluations
notebooks/rnn.py        earlier RNN / LSTM baseline (Colab export), unchanged
```

### Run

```sh
cp .env.example .env            # DATA_HOME=dataset, HG_DATACARD=RTT1/SentiMix
python3 get_sentimix.py         # shared script, one time
python3 baseline.py --paper                                   # 1. paper baseline (mBERT, full fine-tuning, max len 56)
python3 baseline.py --model_name bert-base-multilingual-cased # 2. project recipe: last 2 blocks trainable, seed 42, batch 32
python3 baseline.py --model_name xlm-roberta-base
python3 baseline.py --model_name Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --device_map auto \
        --max_memory "0:7GiB,1:20GiB" --gradient_checkpointing --eval_batch_size 32   # 2 x 24 GB GPUs
python3 eval_sentimix.py --ground results/exp0/ground.csv --pred-glob "results/exp0/*.csv"   # paper-style table
python3 evaluate_all.py  --ground results/exp0/ground.csv --pred-glob "results/exp0/*.csv" --out results/exp0/evaluation_results.csv
```

or simply `bash run.sh`. Smoke test on any GPU/CPU: `python3 baseline.py --max_train_samples 400 --max_eval_samples 200 --epochs 1`.
Every run also writes `results/exp0/<run>_metrics.json` (per-epoch dev scores, final dev / test metrics, arguments).

### Data actually used

The shared `get_sentimix.py` keeps roman-script tweets only and lower-cases the tokens: **13,912 / 2,978 / 2,944**
train / dev / test tweets (paper: 14,000 / 3,000 / 3,000). Test labels come from `test_labels_hinglish.txt`.
Note for the maintainers of `main`: the last step of `get_sentimix.py` (deleting the raw files and renaming the
tags to `h/e/o`) looks up `dataset/sentimix` in lower case and therefore crashes on Linux after the CSVs have
been written; `lib/sentimix_data.py` accepts both tag schemes, so nothing else is affected.

### Results (SentiMix Hi-En test, 2,944 tweets, seed 42; scores in %)

| run | model | trained blocks | trainable | seeds | Pos F1 | Neu F1 | Neg F1 | weighted F1 | macro F1 | acc (micro F1) | dev wF1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mBERT_paper-baseline_full | bert-base-multilingual-cased | all | 177.9M | 1,2,42 | 75.3 +- 0.4 | 62.7 +- 1.0 | 71.5 +- 0.9 | 69.5 +- 0.4 | 69.8 +- 0.4 | 69.3 +- 0.4 | 61.8 +- 0.3 |
| Qwen2.5-7B-Instruct_last2 | Qwen2.5-7B-Instruct | last 2 | 466.1M | 42 | 73.9 | 60.1 | 70.8 | 67.9 | 68.3 | 67.9 | 62.2 |
| xlm-roberta-base_last2 | xlm-roberta-base | last 2 | 14.2M | 1,2,42 | 75.5 +- 0.2 | 59.0 +- 0.5 | 67.1 +- 0.6 | 66.9 +- 0.1 | 67.2 +- 0.1 | 66.9 +- 0.2 | 61.0 +- 0.1 |
| bert-base-multilingual-cased_last2 | bert-base-multilingual-cased | last 2 | 14.2M | 42 | 70.8 | 56.8 | 55.5 | 61.0 | 61.0 | 60.7 | 56.0 |

`+-` = mean +- std over seeds {42, 1, 2}; single numbers are seed 42. "weighted F1" is the paper's metric,
"macro F1" the one of the shared `evaluate_all.py`, "acc" = micro F1. Per-class precision / recall for every run:
`results/exp0/evaluation_paper_style.csv` (`python3 eval_sentimix.py`).

**Findings**

- **Paper baseline reproduced, slightly above the paper.** The Section-5 recipe gives **69.5 +- 0.4** weighted F1
  on our test split against **65.4** in the paper (Table 3, rank 45). The per-class picture is the same
  (neutral is the hardest class: 62.7 vs 58.1 F1; positive 75.3 vs 70.7; negative 71.5 vs 68.3). Likely sources of
  the +4: best-dev-epoch checkpoint selection, the roman-script-only test subset, batch size 32 and lower-cased
  input (see deviations below). Run-to-run noise of this recipe is about +-0.4.
- **Project recipe (only the last 2 blocks trained).** `Qwen2.5-7B-Instruct` 67.9 > `xlm-roberta-base`
  66.9 +- 0.1 > `bert-base-multilingual-cased` 61.0. Freezing all but two blocks costs mBERT 8.5 points
  (69.5 -> 61.0); XLM-R with 14 M trainable parameters is within 1 point of Qwen-7B with 466 M trainable
  parameters. All three beat the paper's baseline, all are far from the best shared-task system (75.0, XLM-R with
  full fine-tuning and adversarial training).
- The dev split is consistently 6-8 points harder than test for every model (dev weighted F1 ~61-62), so dev scores
  are only useful for relative comparisons / checkpoint selection.
- `ai4bharat/IndicBERT-v3-1B` could not be run (gated checkpoint, see notes).

### Deviations from the paper and other notes

- The paper does not state the batch size (we use 32, the project setting), the learning-rate schedule
  (linear decay with 10 % warm-up here) or how the final checkpoint was chosen (here: the epoch with the best
  dev weighted F1, then evaluated once on test).
- Our text is lower-cased by the shared pipeline while M-BERT is a cased model, and our test set is the
  2,944 roman-script tweets of the official 3,000.
- "Last 2 layers": only the last two transformer blocks, the final norm and the classifier are trained;
  embeddings and all other blocks stay frozen.
- `Qwen/Qwen2.5-7B-Instruct` is loaded with `AutoModel` (no LM head), bf16 frozen weights, fp32 copies of the two
  trained blocks, last-token pooling, sharded over two 24 GB GPUs with `--device_map auto`. Temperature does not
  apply (classification head, no generation).
- `ai4bharat/IndicBERT-v3-1B` is a gated checkpoint. The code path exists
  (`--trust_remote_code --dtype bfloat16 --pooling mean`) but the model could not be downloaded on our machine
  without an HF login that has accepted its terms, so it is missing from the table.

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
