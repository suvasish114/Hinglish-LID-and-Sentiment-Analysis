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

**Research question.** If the sentiment model is given the language *switch-point* information, does the score improve?

```
Tokens:  Movie  bahut  acchi  thi  but  ending  was  terrible
LID:       e      h      h     h    e     e       e     e
Switch:    0      1      0     0    1     0       0     0
```

`switch_i = 1 if LID_i != LID_{i-1} else 0`, `switch_0 = 0` by convention. Tokens tagged `o` (punctuation, mentions,
emoji, numbers) are transparent: `h o e` is one switch. SentiMix test tweets have 1 to 16 switch points each (no
monolingual tweet).

### Setup shared by all five approaches

| | |
|---|---|
| data | shared `get_sentimix.py`, roman-script tweets, lower-cased: 13,912 / 2,978 / 2,944 train / dev / test; gold word-level LID tags {h, e, o} |
| models | `FacebookAI/xlm-roberta-base` (12 blocks, 278 M) and `Qwen/Qwen2.5-7B-Instruct` (28 blocks, 7.1 B; loaded with `AutoModel`, i.e. without the LM head, bf16 weights sharded over two 24 GB GPUs) |
| what is trained | only the **last 2 transformer blocks** (+ final norm) of the encoder and the heads (pooling, classifier); everything else is frozen. XLM-R: 14.2 M trainable parameters, Qwen: 466 M |
| optimisation | **seed 42** (XLM-R additionally seeds 1 and 2 to measure noise), **batch size 32**, 3 epochs, AdamW lr 2e-5, weight decay 0.01, linear schedule with 10 % warm-up, max length 128, bf16 autocast, checkpoint = epoch with the best dev weighted F1 |
| temperature | **T = 0** does not apply - there is no generation; a linear classifier reads a pooled state. (The contrastive τ below is a different quantity.) |
| features | encoder states `h_i` per sub-word; a word's switch feature is copied to all of its sub-words; special tokens are excluded from the pooling |
| metrics | weighted F1 (official SentiMix metric), macro F1 (shared `evaluate_all.py`), accuracy; mean +- std over seeds where available |

### The five approaches (one flag each: `--approach`)

**1. No switch - baseline** (`no_switch`). Attention pooling without any switch information:
`a_i = softmax_i( w^T h_i + b )`, sentiment vector `s = sum_i a_i h_i`, `logits = W_c dropout(s) + b_c`.

**2. Binary switch embedding** (`binary`). Switch-biased attention pooling, `a_i = softmax_i( W_h h_i + W_s e(sw_i) + b )`
with `e` a learned 16-d embedding of the binary flag `sw_i in {0, 1}`; `W_s` is zero-initialised so the model equals
approach 1 at step 0. Switch points come from the gold LID tags (`--switch_source gold`) or from the frozen LID tagger of
Task 1 (`--switch_source lid`).

**3. Distance-to-switch embedding** (`distance`). Same pooling with `e(d_i)`, `d_i = min(distance in words to the nearest
switch, 3) in {0, 1, 2, 3+}` - a smoother signal than the sparse binary flag. Same two switch sources.

**4. Contrastive learning** (`contrastive`) - *Multilingual Representation Distillation with Contrastive Learning*
(Tan, Heffernan, Schwenk, Koehn, EACL 2023), model LASER3-CO. What the paper does: a student encoder θ_s and a frozen
teacher θ_t; for a parallel pair (x, y) the student embedding `q = θ_s(x)` must match the teacher embedding of the
translation `k+ = θ_t(y)` against a queue of N = 4096 teacher embeddings of earlier batches (negatives), with InfoNCE
`L = -log exp(q.k+/τ) / sum_i exp(q.k_i/τ)`, τ = 0.05; LASER3-CO-Filter additionally drops "extremely hard" negatives with
`cos(k+, k_i) >= σ` (σ = 0.9). Our adaptation to SentiMix, which has no parallel sentences: the pair (x, y) becomes
(code-mixed tweet, one of its **language views**), where the Hindi view is the sequence of the words of the tweet's Hindi
segments (the maximal runs delimited by the switch points) and likewise for English; one view is drawn at random per step.
The teacher is the pre-trained encoder before fine-tuning, mean-pooled; its embeddings are pre-computed once (frozen by
construction) and mean-centred, because raw mean-pooled states have cosine > 0.9 between any two tweets. `q` is the
sentiment vector `s` of approach-1 pooling. Total loss `L = L_CE + λ L_contrastive`, λ chosen on the dev set among
{0.1, 0.01}; the σ-filter is available with `--contrastive_filter 0.9`. The approach uses the switch points only to build
the views - the pooling itself is the approach-1 baseline.

**5. Predicted switch points** (`beyond_detection`) - *Beyond Detection: Predicting Code-Switch Points in Multilingual
Conversations* (Xie, Zhang, Koshal, Sushmita, WiML @ NeurIPS 2025). The paper predicts upcoming switch points token by
token with two paradigms: (1) window-based models - BERT embeddings of the preceding tokens fed to a recurrent network,
with fixed and flexible context windows; (2) a transformer token classifier built on mBERT / XLM-RoBERTa; it reports ROC-AUC
(best RNN 0.91, mBERT 0.98 on Chinese-English ASCEND). On SentiMix we train both paradigms on the gold switch points
(`lib/switch_predictor.py`): (1) frozen `bert-base-multilingual-cased` word embeddings -> LSTM over the last 5 words or over
the whole prefix (flexible window) -> `P(switch at the next word)`, causal; (2) `xlm-roberta-base` token classifier (last 2
blocks trained) -> `P(switch at this word)`. We report AUC (overall and per direction h->e / e->h, as the paper does per
direction), P / R / F1 of the switch class, and select the predictor with the best dev F1 of its per-word switch flags.
Its *predicted* switch points then replace the LID-derived ones in the approach-2/3 pooling
(`--approach beyond_detection --pooling binary|distance`). Only the paradigms and the metric come from the paper (its full
text is not openly accessible); window size, LSTM size, class weighting and the decision threshold are our choices.

**Task 1 - LID tagger** (`lib/lid_tagger.py`, source of the `lid` switch points). `xlm-roberta-base` fine-tuned as a token
classifier on the SentiMix word tags (test token accuracy 91.7 %, macro-F1 92.8, see Experiment 1), then frozen: it only
writes `{split}_pred_tags.csv`, so no gradient ever reaches it.

### Layout

```
lib/data.py              SentiMix loader (output of the shared get_sentimix.py), sub-word <-> word alignment
lib/switch.py            switch points, directions, distance buckets, language views, upcoming-switch labels
lib/lid_tagger.py        Task 1: LID tagger -> results/exp2/lid/<tagger>/{train,dev,test}_pred_tags.csv
lib/switch_predictor.py  approach 5: window BERT+RNN / transformer switch predictors -> results/exp2/switch/<predictor>/{split}_pred_switch.csv
lib/model.py             SwitchAttentionPooling (approaches 1-3), ContrastiveDistillation (approach 4), SentimentModel
lib/train.py             training loop -> results/exp2/<run>.csv + ground.csv (format of evaluate_all.py) + <run>_metrics.json
lib/report.py            README tables: configurations (mean +- std over seeds), weighted F1 by #switches, switch-point quality
main.py                  CLI gateway: --approach no_switch|binary|distance|contrastive|beyond_detection
run.sh                   whole pipeline for one model (tagger, predictors, 8 sentiment runs, evaluation)
```

Shared scripts (`get_sentimix.py`, `evaluate_all.py`, `requirements.txt`) are untouched; the branch is synced with `main`.

### Run

```sh
cp .env.example .env && python3 get_sentimix.py                                   # shared data pipeline (one time)
python3 lib/lid_tagger.py --model_name xlm-roberta-base --save_model              # Task 1 (frozen tagger)
python3 lib/switch_predictor.py --paradigm window --window 5                      # approach 5, predictors
python3 lib/switch_predictor.py --paradigm window --window 0                      #   (flexible window)
python3 lib/switch_predictor.py --paradigm transformer                            #   (XLM-R token classifier)
python3 lib/report.py --switch_quality                                            # which switch-point source is best?
python3 main.py --approach no_switch                                              # 1
python3 main.py --approach binary   --switch_source gold                          # 2  (also: --switch_source lid --switch_dir results/exp2/lid/xlm-roberta-base_full)
python3 main.py --approach distance --switch_source gold                          # 3
python3 main.py --approach contrastive --contrastive_weight 0.1                   # 4
python3 main.py --approach beyond_detection --pooling binary --switch_dir results/exp2/switch/<best predictor>   # 5
python3 lib/report.py; python3 lib/report.py --by_switches                        # tables
python3 evaluate_all.py --ground results/exp2/ground.csv --pred-glob "results/exp2/*.csv" --out results/exp2/evaluation_results.csv
```

or `bash run.sh [MODEL]`. For Qwen: `EXTRA='--dtype bfloat16 --device_map auto --max_memory 0:7GiB,1:20GiB --gradient_checkpointing --eval_batch_size 32' bash run.sh Qwen/Qwen2.5-7B-Instruct`.
Smoke test: `python3 main.py --approach binary --max_train_samples 400 --max_eval_samples 200 --epochs 1`.

### Results

#### Switch-point sources (SentiMix test, per-word switch flags against the gold switches)

| switch-point source | precision | recall | F1 (switch class) | predicted switches / gold switches |
|---|---|---|---|---|
| lid tagger: xlm-roberta-base_full | 60.0 | 51.4 | 55.4 | 8725 / 10182 |
| switch predictor: transformer_xlm-roberta-base_last2 | 31.2 | 67.6 | 42.7 | 22046 / 10182 |
| switch predictor: window_lstm_w5 | 32.0 | 64.3 | 42.7 | 20457 / 10182 |
| switch predictor: window_lstm_wflex | 33.5 | 58.1 | 42.5 | 17627 / 10182 |

#### Sentiment (SentiMix test, 2,944 tweets; `+-` = mean +- std over seeds 42 / 1 / 2, single values = seed 42)

| model | approach | pooling | switch source | seeds | weighted F1 | macro F1 | acc | dev wF1 | min/run |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 1 no switch (attention pooling) | none | - | 42 | 70.0 | 70.3 | 69.8 | 63.2 | 18 |
| Qwen2.5-7B-Instruct | 2 binary switch embedding | binary | gold | 42 | 69.0 | 69.3 | 68.7 | 63.1 | 18 |
| Qwen2.5-7B-Instruct | 2 binary switch embedding | binary | lid (xlm-roberta-base_full) | 42 | 69.3 | 69.7 | 69.1 | 63.6 | 15 |
| Qwen2.5-7B-Instruct | 3 distance-to-switch embedding | distance | gold | 42 | 68.5 | 68.8 | 68.2 | 63.6 | 15 |
| Qwen2.5-7B-Instruct | 3 distance-to-switch embedding | distance | lid (xlm-roberta-base_full) | 42 | 69.0 | 69.3 | 68.7 | 63.6 | 15 |
| Qwen2.5-7B-Instruct | 4 contrastive distillation (LASER3-CO) | none | views of gold switches, lambda=0.01 | 42 | 69.9 | 70.2 | 69.6 | 63.8 | 15 |
| Qwen2.5-7B-Instruct | 5 Beyond-Detection predicted switches | binary | predictor (window_lstm_wflex) | 42 | 69.6 | 69.9 | 69.3 | 63.8 | 15 |
| Qwen2.5-7B-Instruct | 5 Beyond-Detection predicted switches | distance | predictor (window_lstm_wflex) | 42 | 69.1 | 69.4 | 68.9 | 63.9 | 15 |
| xlm-roberta-base | 1 no switch (attention pooling) | none | - | 1,2,42 | 68.3 +- 0.1 | 68.6 +- 0.1 | 68.2 +- 0.1 | 62.1 +- 0.1 | 1 |
| xlm-roberta-base | 2 binary switch embedding | binary | gold | 1,2,42 | 68.3 +- 0.1 | 68.5 +- 0.1 | 68.1 +- 0.1 | 61.8 +- 0.2 | 1 |
| xlm-roberta-base | 2 binary switch embedding | binary | lid (xlm-roberta-base_full) | 1,2,42 | 68.3 +- 0.0 | 68.6 +- 0.0 | 68.2 +- 0.0 | 61.8 +- 0.1 | 1 |
| xlm-roberta-base | 3 distance-to-switch embedding | distance | gold | 1,2,42 | 68.3 +- 0.2 | 68.6 +- 0.2 | 68.2 +- 0.2 | 62.1 +- 0.3 | 1 |
| xlm-roberta-base | 3 distance-to-switch embedding | distance | lid (xlm-roberta-base_full) | 1,2,42 | 68.3 +- 0.3 | 68.6 +- 0.2 | 68.3 +- 0.2 | 62.1 +- 0.4 | 1 |
| xlm-roberta-base | 4 contrastive distillation (LASER3-CO) | none | views of gold switches, lambda=0.01 | 1,2,42 | 68.4 +- 0.2 | 68.6 +- 0.3 | 68.2 +- 0.3 | 62.2 +- 0.1 | 1 |
| xlm-roberta-base | 4 contrastive distillation (LASER3-CO) | none | views of gold switches, lambda=0.01, sigma=0.9 | 42 | 68.5 | 68.7 | 68.3 | 62.2 | 1 |
| xlm-roberta-base | 4 contrastive distillation (LASER3-CO) | none | views of gold switches, lambda=0.1 | 42 | 65.1 | 65.2 | 65.0 | 60.4 | 1 |
| xlm-roberta-base | 5 Beyond-Detection predicted switches | binary | predictor (transformer_xlm-roberta-base_last2) | 42 | 68.3 | 68.6 | 68.2 | 62.0 | 1 |
| xlm-roberta-base | 5 Beyond-Detection predicted switches | binary | predictor (window_lstm_w5) | 42 | 68.3 | 68.6 | 68.2 | 62.1 | 1 |
| xlm-roberta-base | 5 Beyond-Detection predicted switches | binary | predictor (window_lstm_wflex) | 1,2,42 | 68.3 +- 0.0 | 68.6 +- 0.0 | 68.2 +- 0.0 | 61.8 +- 0.2 | 1 |
| xlm-roberta-base | 5 Beyond-Detection predicted switches | distance | predictor (window_lstm_wflex) | 1,2,42 | 68.3 +- 0.3 | 68.5 +- 0.3 | 68.2 +- 0.3 | 62.1 +- 0.4 | 1 |

#### Weighted F1 by number of switch points in the tweet (seed 42)

| model | approach | pooling | switch source | all | 1-2 switches (n=903) | 3-5 switches (n=1646) | 6+ switches (n=395) |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 1 no switch (attention pooling) | none | - | 70.0 | 71.2 | 69.5 | 69.6 |
| Qwen2.5-7B-Instruct | 2 binary switch embedding | binary | gold | 69.0 | 69.2 | 69.2 | 67.7 |
| Qwen2.5-7B-Instruct | 2 binary switch embedding | binary | lid (xlm-roberta-base_full) | 69.3 | 70.3 | 68.4 | 71.4 |
| Qwen2.5-7B-Instruct | 3 distance-to-switch embedding | distance | gold | 68.5 | 68.3 | 68.9 | 67.2 |
| Qwen2.5-7B-Instruct | 3 distance-to-switch embedding | distance | lid (xlm-roberta-base_full) | 69.0 | 70.0 | 68.6 | 67.7 |
| Qwen2.5-7B-Instruct | 4 contrastive distillation (LASER3-CO) | none | views of gold switches, lambda=0.01 | 69.9 | 71.9 | 69.0 | 68.8 |
| Qwen2.5-7B-Instruct | 5 Beyond-Detection predicted switches | binary | predictor (window_lstm_wflex) | 69.6 | 70.6 | 71.6 | 68.0 |
| Qwen2.5-7B-Instruct | 5 Beyond-Detection predicted switches | distance | predictor (window_lstm_wflex) | 69.1 | 69.1 | 71.4 | 67.6 |
| xlm-roberta-base | 1 no switch (attention pooling) | none | - | 68.2 | 69.3 | 67.2 | 70.3 |
| xlm-roberta-base | 2 binary switch embedding | binary | gold | 68.3 | 69.7 | 67.8 | 67.4 |
| xlm-roberta-base | 2 binary switch embedding | binary | lid (xlm-roberta-base_full) | 68.3 | 69.7 | 66.4 | 72.0 |
| xlm-roberta-base | 3 distance-to-switch embedding | distance | gold | 68.5 | 69.7 | 67.5 | 69.7 |
| xlm-roberta-base | 3 distance-to-switch embedding | distance | lid (xlm-roberta-base_full) | 68.6 | 70.3 | 66.4 | 72.1 |
| xlm-roberta-base | 4 contrastive distillation (LASER3-CO) | none | views of gold switches, lambda=0.01 | 68.4 | 69.1 | 67.7 | 69.6 |
| xlm-roberta-base | 4 contrastive distillation (LASER3-CO) | none | views of gold switches, lambda=0.01, sigma=0.9 | 68.5 | 69.2 | 67.8 | 69.9 |
| xlm-roberta-base | 4 contrastive distillation (LASER3-CO) | none | views of gold switches, lambda=0.1 | 65.1 | 64.4 | 64.9 | 67.9 |
| xlm-roberta-base | 5 Beyond-Detection predicted switches | binary | predictor (transformer_xlm-roberta-base_last2) | 68.3 | 62.7 | 67.6 | 69.0 |
| xlm-roberta-base | 5 Beyond-Detection predicted switches | binary | predictor (window_lstm_w5) | 68.3 | 65.5 | 68.2 | 68.7 |
| xlm-roberta-base | 5 Beyond-Detection predicted switches | binary | predictor (window_lstm_wflex) | 68.3 | 64.7 | 70.5 | 67.6 |
| xlm-roberta-base | 5 Beyond-Detection predicted switches | distance | predictor (window_lstm_wflex) | 68.6 | 66.8 | 71.0 | 67.5 |

XLM-R weighted F1 by switch bucket, averaged over the 3 seeds (the table above is seed 42 only):

| XLM-R configuration | 1-2 switches | 3-5 switches | 6+ switches | all |
|---|---|---|---|---|
| 1 no switch | 69.2 | 67.6 | 69.3 | 68.3 |
| 2 binary, gold / LID tagger | 69.6 / 69.8 | 67.5 / 66.5 | 68.3 / 70.5 | 68.3 / 68.3 |
| 3 distance, gold / LID tagger | 69.4 / 70.0 | 67.5 / 66.3 | 69.0 / 71.3 | 68.3 / 68.3 |
| 4 contrastive, λ = 0.01 | 68.9 | 68.0 | 68.8 | 68.4 |
| 5 predicted switches, binary / distance | 66.1 / 66.2 | 70.2 / 70.6 | 67.6 / 67.2 | 68.3 / 68.3 |

**Findings**

- **The baseline is the thing to beat, and nothing beats it.** Plain attention pooling (approach 1) gives
  **68.3 +- 0.1** (XLM-R) and **70.0** (Qwen). It is itself +1.4 / +2.1 over the CLS / last-token baselines of
  Experiment 0 with the same recipe - the pooling matters more than any switch information added to it. For Qwen it is
  also the best number of the whole experiment.
- **Approaches 2 and 3 (switch-biased pooling).** XLM-R: 68.3 for every variant - gold or LID-tagger switches, binary
  or distance - identical to the baseline within the +-0.1-0.3 seed spread. Qwen: 68.5-69.3, i.e. 0.7-1.5 *below* the
  baseline at one seed (the run-to-run spread of these 7B runs is about +-1). The switch source makes no difference
  (gold ~ LID tagger ~ predicted).
- **Approach 4 (contrastive distillation).** λ = 0.01: 68.4 +- 0.2 (XLM-R) and 69.9 (Qwen) - the baseline; the
  LASER3-CO σ-filter changes nothing (68.5). λ = 0.1 lets the contrastive term dominate the 14 M trainable parameters and
  costs 3 points (65.1). The InfoNCE loss hardly moves with two trainable blocks (XLM-R 5.7 -> 5.6, Qwen 12.3 -> 9.4
  over 3 epochs, against log 4097 = 8.3 for chance): the student cannot travel far enough in the frozen pre-trained
  space to align a code-mixed tweet with its monolingual views, so the objective works as a mild regulariser at best.
- **Approach 5 (predicted switch points).** The Beyond-Detection style predictors reach AUC 0.81-0.83 but a switch-class
  F1 of only 42.5-42.7 (precision ~ 33 %), far below switches derived from the LID tagger (F1 55.4): on romanised
  Hinglish both languages share one script and the next word's language is hard to anticipate (the paper's 0.91 / 0.98
  AUC were on Chinese-English dialogue with distinct scripts). The flexible-window LSTM is marginally the best predictor
  on dev. Despite the noisier switch points the sentiment scores equal those with gold switches (68.3 XLM-R; 69.6 / 69.1
  Qwen) - again a sign that the pooling is not using the feature.
- **Switch information redistributes errors instead of removing them.** Averaged over seeds, LID-derived switches give
  XLM-R +1.2 / +2.0 on tweets with 6+ switch points (binary / distance) but -1.1 / -1.3 on the 3-5 bucket; predicted
  switches give +2.6 / +3.0 on the 3-5 bucket and -3.1 on the 1-2 bucket. The buckets cancel in the overall score. Qwen
  shows the same pattern at seed 42 (binary + LID switches: +1.8 on 6+ switches, -1.1 on 3-5).
- **Answer to the research question.** Under the project recipe (last 2 blocks, seed 42, batch 32) the language
  switch-point information does **not** improve SentiMix sentiment for either model: all five approaches end within noise
  of (XLM-R) or below (Qwen) the attention-pooling baseline. The LID-tag fusion of Experiment 1 (+1.0 for XLM-R) remains the
  only language signal with a measurable overall effect so far.
- **Next.** 3 seeds for Qwen; inject the switch embedding into the hidden states (as the LID tags in Experiment 1)
  instead of only into the attention logits; more trainable blocks for the contrastive objective; PESTO-style
  switch-relative positional encodings; the "negative token attention" item of the scope list.

*Previous iteration of this branch (PR #7, first version): the same pooling variants with full fine-tuning (XLM-R) and
QLoRA (Qwen) instead of the project recipe; those numbers are superseded by the tables above.*

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
