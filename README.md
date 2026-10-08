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

<!-- RESULTS_EXP2_SWITCH -->

#### Sentiment (SentiMix test, 2,944 tweets; `+-` = mean +- std over seeds 42 / 1 / 2, single values = seed 42)

<!-- RESULTS_EXP2_SENT -->

#### Weighted F1 by number of switch points in the tweet (seed 42)

<!-- RESULTS_EXP2_BUCKETS -->

<!-- FINDINGS_EXP2 -->

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
