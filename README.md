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

## Experiment 1 - LID + Sentiment (branch `exp1`)

**Research question.** Do word-level language-identification (LID) tags improve sentence-level sentiment
classification of Hinglish (Hindi-English code-mixed) tweets?

**Approach.** Fusion of the token representation with an embedding of its LID tag, passed through the sentiment
model. Each sub-word inherits the tag of its word (`h` Hindi, `e` English, `o` other; special tokens and padding get
a fixed zero vector). The tag is embedded (16-d, learned) and fused into the hidden states entering transformer
block `--lid_inject_layer` (0 = the input embeddings):

    concat :  h_i <- W_f [ h_i ; e_i^lid ] + b_f      W_f initialised to [ I | 0 ],  b_f = 0
    add    :  h_i <- h_i + W_f e_i^lid                W_f initialised to 0

With these initialisations the fused model is *exactly the baseline at step 0* (verified: identical logits), so any
difference after training is attributable to the LID signal. The LID model is **frozen**: it only supplies discrete
tags that are pre-computed for every split, so no gradient can reach it - the objective is better sentiment given
LID, not better LID. Sentiment models, data and hyper-parameters are those of Experiment 0 (last 2 blocks trained,
seed 42, batch 32, 3 epochs, lr 2e-5).

### Layout

```
lid_tagger.py           task 1: fine-tune a token-classification LID tagger (xlm-roberta-base, mBERT, ...) or build the
                        train-set lexicon baseline; writes results/exp1/lid/<tagger>/{train,dev,test}_pred_tags.csv
lid_eval.py             token accuracy, per-tag P/R/F1, macro-F1 of any id,pred_tags CSV (also for Microsoft LID / LLM outputs)
sentiment_lid.py        task 2: sentiment with --lid_fusion none|concat|add, --lid_source gold|predicted, --lid_inject_layer k
eval_sentimix.py        paper-style evaluation (per-class P/R/F1, weighted / macro / micro F1), same as branch exp0
lib/                    data loader + alignment, SentimentClassifier (freezing + LID fusion), training loop, metrics
run_exp1.sh             taggers -> best tagger -> baseline vs. fusion runs -> both evaluations

main.py, lib/inference.py, lib/finetune_model.py, lib/microsoft_lid.py, prompts/, run.sh, tune.sh, microsoft_lid.sh
                        earlier exp1 work: zero/one-shot LLM prompting and QLoRA fine-tuning for LID on COMI-Lingua,
                        and the Microsoft LID tool - unchanged
```

### Run

```sh
cp .env.example .env && python3 get_sentimix.py                  # shared data pipeline (one time)
python3 lid_tagger.py --model_name lexicon                       # 1. LID taggers ...
python3 lid_tagger.py --model_name xlm-roberta-base --save_model
python3 lid_eval.py --split test --pred-glob "results/exp1/lid/*/test_pred_tags.csv"
python3 sentiment_lid.py --model_name xlm-roberta-base --lid_fusion none                              # 2. baseline
python3 sentiment_lid.py --model_name xlm-roberta-base --lid_fusion concat --lid_source gold          #    + gold LID
python3 sentiment_lid.py --model_name xlm-roberta-base --lid_fusion concat --lid_source predicted \
        --pred_tags_dir results/exp1/lid/xlm-roberta-base_full                                        #    + frozen-tagger LID
python3 eval_sentimix.py --ground results/exp1/ground.csv --pred-glob "results/exp1/*.csv"            # 3. evaluation
python3 evaluate_all.py  --ground results/exp1/ground.csv --pred-glob "results/exp1/*.csv" --out results/exp1/evaluation_results.csv
```

or `bash run_exp1.sh [MODEL]`. For `Qwen/Qwen2.5-7B-Instruct` add
`--dtype bfloat16 --device_map auto --max_memory "0:7GiB,1:20GiB" --gradient_checkpointing --eval_batch_size 32`
(two 24 GB GPUs). Smoke test: `python3 sentiment_lid.py --max_train_samples 400 --max_eval_samples 200 --epochs 1`.

### Task 1 - LID taggers (SentiMix word-level tags; scores in %)

| tagger | dev acc | dev macro-F1 | test acc | test F1 h | test F1 e | test F1 o | test macro-F1 |
|---|---|---|---|---|---|---|---|
| xlm-roberta-base_full | 92.0 | 93.1 | 91.7 | 91.4 | 87.7 | 99.3 | 92.8 |
| bert-base-multilingual-cased_full | 92.1 | 93.1 | 89.5 | 91.3 | 84.9 | 93.6 | 89.9 |
| xlm-roberta-base_last2 | 89.1 | 90.5 | 89.0 | 88.5 | 83.9 | 98.8 | 90.4 |
| lexicon | 85.0 | 86.8 | 81.3 | 81.2 | 76.0 | 91.0 | 82.7 |

### Task 2 - sentiment with LID (SentiMix Hi-En test, 2,944 tweets, seed 42; scores in %)

| run | model | trained blocks | trainable | LID fusion | LID tags | seeds | Pos F1 | Neu F1 | Neg F1 | weighted F1 | macro F1 | acc (micro F1) | dev wF1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct_last2_lid-concat@0_gold | Qwen2.5-7B-Instruct | last 2 | 479.0M | concat @ block 0 | gold | 42 | 72.7 | 61.9 | 71.5 | 68.3 | 68.7 | 68.1 | 62.5 |
| xlm-roberta-base_last2_lid-concat@0_predicted | xlm-roberta-base | last 2 | 14.8M | concat @ block 0 | predicted | 1,2,42 | 76.1 +- 0.3 | 60.7 +- 0.3 | 68.0 +- 1.2 | 68.0 +- 0.3 | 68.3 +- 0.4 | 67.9 +- 0.4 | 61.7 +- 0.2 |
| Qwen2.5-7B-Instruct_last2 | Qwen2.5-7B-Instruct | last 2 | 466.1M | none (baseline) | - | 42 | 73.9 | 60.1 | 70.8 | 67.9 | 68.3 | 67.9 | 62.2 |
| xlm-roberta-base_last2_lid-concat@0_gold | xlm-roberta-base | last 2 | 14.8M | concat @ block 0 | gold | 1,2,42 | 76.2 +- 0.3 | 60.7 +- 0.3 | 67.8 +- 1.3 | 67.9 +- 0.3 | 68.3 +- 0.4 | 67.9 +- 0.4 | 61.8 +- 0.1 |
| Qwen2.5-7B-Instruct_last2_lid-concat@0_predicted | Qwen2.5-7B-Instruct | last 2 | 479.0M | concat @ block 0 | predicted | 42 | 71.0 | 62.0 | 71.5 | 67.8 | 68.2 | 67.6 | 62.2 |
| xlm-roberta-base_last2_lid-concat@10_gold | xlm-roberta-base | last 2 | 14.8M | concat @ block 10 | gold | 1,2,42 | 75.9 +- 0.1 | 60.2 +- 1.5 | 67.8 +- 0.8 | 67.7 +- 0.5 | 68.0 +- 0.4 | 67.6 +- 0.4 | 61.9 +- 0.1 |
| xlm-roberta-base_last2_lid-concat@10_predicted | xlm-roberta-base | last 2 | 14.8M | concat @ block 10 | predicted | 1,2,42 | 75.9 +- 0.2 | 60.0 +- 1.3 | 67.9 +- 0.9 | 67.6 +- 0.3 | 67.9 +- 0.3 | 67.6 +- 0.3 | 61.9 +- 0.1 |
| xlm-roberta-base_last2_lid-add@0_predicted | xlm-roberta-base | last 2 | 14.2M | add @ block 0 | predicted | 42 | 75.2 | 62.3 | 63.7 | 66.9 | 67.1 | 66.8 | 61.2 |
| xlm-roberta-base_last2 | xlm-roberta-base | last 2 | 14.2M | none (baseline) | - | 1,2,42 | 75.5 +- 0.2 | 59.0 +- 0.5 | 67.1 +- 0.6 | 66.9 +- 0.1 | 67.2 +- 0.1 | 66.9 +- 0.2 | 61.0 +- 0.1 |
| xlm-roberta-base_last2_lid-add@0_gold | xlm-roberta-base | last 2 | 14.2M | add @ block 0 | gold | 42 | 75.2 | 61.9 | 62.9 | 66.5 | 66.6 | 66.4 | 61.6 |
| Qwen2.5-7B-Instruct_last2_lid-concat@26_gold | Qwen2.5-7B-Instruct | last 2 | 479.0M | concat @ block 26 | gold | 42 | 70.3 | 66.1 | 62.6 | 66.4 | 66.3 | 66.4 | 61.4 |

`+-` = mean +- std over seeds {42, 1, 2}; single numbers are seed 42. The `none` rows are the Experiment-0 baselines
(identical code and configuration; the Qwen baseline is the Experiment-0 run). "weighted F1" is the SentiMix
metric, "macro F1" the one of the shared `evaluate_all.py`. Full per-class P/R: `results/exp1/evaluation_paper_style.csv`.

Weighted F1 per Code-Mixing-Index bucket of the test tweets (`python3 analyze_exp1.py`, seed 42; every test tweet is
code-mixed, no tweet has CMI 0): the 695 tweets with CMI <= 20, the 1,655 with 20 < CMI <= 40 and the 594 with CMI > 40.

| run (seed 42) | all | CMI <= 20 | 20 < CMI <= 40 | CMI > 40 |
|---|---|---|---|---|
| xlm-roberta-base baseline | 66.7 | 68.5 | 66.4 | 65.5 |
| xlm-roberta-base + concat @ block 0, gold | 67.6 | 67.9 | 67.7 | 66.6 |
| xlm-roberta-base + concat @ block 0, predicted | 67.5 | 68.7 | 67.3 | 66.6 |
| xlm-roberta-base + concat @ block 10, gold | 68.1 | 69.3 | 67.9 | 67.1 |
| Qwen2.5-7B-Instruct baseline | 67.9 | 67.4 | 67.2 | 69.9 |
| Qwen2.5-7B-Instruct + concat @ block 0, gold | 68.3 | 66.3 | 68.5 | 70.2 |
| Qwen2.5-7B-Instruct + concat @ block 0, predicted | 67.8 | 65.4 | 68.1 | 69.9 |

**Findings**

- **LID tagger (task 1).** Fine-tuning `xlm-roberta-base` end-to-end on the SentiMix word tags gives the best
  tagger: 91.7 % token accuracy / 92.8 macro-F1 on test (dev 93.1). mBERT ties on dev but generalises worse to
  test (89.9, weak on `o`), the project recipe (last 2 blocks) loses 2.4 points, and a train-set lexicon reaches
  82.7. The XLM-R tagger is the frozen LID model used for all `predicted` rows below; `lid_eval.py` scores any other
  tagger's `id,pred_tags` CSV (Microsoft LID tool, LLM prompting) in the same table.
- **LID tags help the small encoder a little, consistently.** With `xlm-roberta-base` (last 2 blocks trained),
  concatenating the LID embedding to the input embeddings raises weighted F1 from **66.9 +- 0.1** to
  **67.9 +- 0.3 (gold)** / **68.0 +- 0.3 (predicted)** - about +1 point, three times the seed std, with the
  gain spread over all code-mixing buckets. Fusing at the first trainable block instead (block 10) gives
  +0.8 / +0.7, and the additive variant gives nothing (66.5 / 66.9, seed 42): the LID signal is only useful when
  the frozen lower blocks can see it and the fusion has a full projection.
- **Predicted tags are as good as gold.** A frozen tagger at 91.7 % token accuracy loses nothing against the gold
  annotation (which is itself automatic, Bhat et al. 2014), so the cascaded LID -> sentiment pipeline is practical.
- **No measurable gain for Qwen2.5-7B-Instruct (one seed).** 67.9 -> 68.3 with gold tags at the input, 67.8
  with predicted tags, 66.4 when fused only at block 26. These differences are inside the +-1 run-to-run spread we
  observed for 7B models in Experiment 2, so the answer for the large decoder is "not with this fusion and budget".
  Fusion shifts errors rather than removing them (neutral F1 60.1 -> 61.9/62.0, positive F1 73.9 -> 72.7/71.0), and
  helps only on the more mixed tweets (CMI > 20) while hurting the least mixed ones.
- **Answer to the research question.** Word-level LID tags give a small but reproducible improvement (~ +1
  weighted F1) for a partially fine-tuned XLM-R when fused into the input embeddings, and none that we can
  distinguish from noise for Qwen2.5-7B. The additive Qwen variants (`add @ 0`) were not run for time reasons.
- **Next.** 3 seeds for every Qwen variant; `--lid_dim` / `--head_lr` sweeps; fusion with full fine-tuning; the
  Microsoft LID tool and the LLM taggers of the earlier exp1 pipeline as `--pred_tags_dir` sources; combine with the
  switch-point features of Experiment 2; `ai4bharat/IndicBERT-v3-1B` once the gated checkpoint is accessible.

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
