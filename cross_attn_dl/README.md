# Joint Contrastive Learning of Language Identification and Sentiment on Hindi-English Code-Mixed Tweets

Course assignment (CS613-NLP, IIT Gandhinagar), part of the group project
[Hinglish LID and Sentiment Analysis](https://github.com/suvasish114/Hinglish-LID-and-Sentiment-Analysis). The previous experiment (Experiment 2, branch `exp2` of the group
repository, approach 4) used a contrastive
loss on **one** task only - the sentiment vector of a tweet was pulled towards a frozen teacher's embedding of its
monolingual view - and the LID tags only served to build the views. This assignment trains **both tasks jointly**,
word-level language identification (LID) and tweet-level sentiment, in one model, with a supervised contrastive loss
on **each** task, and fuses the two through a cross-attention pooling.

### Environment and data

```sh
pip install -r requirements.txt
cp .env.example .env
# dataset/SentiMix/{train,dev,test}.csv must exist here. They are written by get_sentimix.py of the group repository:
# run it there once, then copy or link its dataset/ folder into this one
ln -s /path/to/Hinglish-LID-and-Sentiment-Analysis/dataset dataset     # inside the group repository: ln -s ../dataset dataset
```

Data: SentiMix Hi-En (SemEval-2020 Task 9) from the shared `get_sentimix.py`, roman script, lower-cased:
13,912 / 2,978 / 2,944 train / dev / test tweets, gold word-level LID tags {h, e, o} and a 3-way sentiment label.
The shared `get_sentimix.py` and `evaluate_all.py` of the group repository are not modified; this repository has its own
`evaluate.py` because
only precision, recall and F1 are reported here.

### Model (`lib/model.py`)

```
                      ┌──────────────── LID head ────────────────┐
  tweet ──► encoder ──► h_i ──► lid_logits_i = W_l h_i ──► CE_lid │ SupCon_lid (tokens, label = language)
   (last 2 blocks       │                │ softmax p_i              │
    trained)            ▼                ▼                          │
                   k_i = v_i = h_i + W_e p_i   ◄── language-aware token states
                        │
   learned query q ──► multi-head cross-attention ──► s ──► W_c s ──► CE_sent │ SupCon_sent (tweets, label = sentiment)
```

| component | definition |
|---|---|
| encoder | pre-trained transformer; only the last 2 blocks + final norm are trained (project recipe) |
| LID head | `lid_logits_i = W_l h_i` on every sub-word; each sub-word carries the tag of its word; a word is scored at its first sub-word |
| cross-attention pooling | a learned query attends over `k_i = v_i = LayerNorm(h_i + W_e softmax(lid_logits_i))`; `W_e` is zero-initialised, so at step 0 it is a plain multi-head attention pooling; the sentiment loss back-propagates into the LID logits |
| sentiment head | `logits = W_c dropout(s)` |
| SupCon (Khosla et al., 2020) | `L = -1/|A| Σ_a 1/|P(a)| Σ_{p∈P(a)} log exp(z_a·z_p/τ) / Σ_{n≠a} exp(z_a·z_n/τ)`, `z = normalise(g(x))`, `g` a 2-layer MLP to 128 d, τ = 0.1 |
| SupCon on LID | anchors = the word sub-words of the batch (capped at 2048 random tokens), positives = tokens with the same language tag |
| SupCon on sentiment | anchors = the tweets of the batch, positives = tweets with the same sentiment label |
| total loss | `L = CE_sent + α CE_lid + λ_lid SupCon_lid + λ_sent SupCon_sent`, α = 1, λ_lid = λ_sent = 0.1 |

### Modes (`--mode`)

| mode | LID head | pooling | contrastive | purpose |
|---|---|---|---|---|
| `sentiment` | - | attention pooling | - | single-task baseline (= approach 1 of Experiment 2) |
| `joint` | yes | cross-attention | - | multi-task, no contrastive term |
| `joint_contrastive` | yes | cross-attention | LID + sentiment | **proposed model** |
| `contrastive_lid` | yes | cross-attention | LID only | ablation |
| `contrastive_sent` | yes | cross-attention | sentiment only | ablation |

### Recipe (shared by all runs)

seed 42, batch size 32, 3 epochs, AdamW lr 2e-5, weight decay 0.01, linear schedule with 10 % warm-up, max length 128,
bf16 autocast, dropout 0.1, checkpoint = epoch with the best dev **sentiment macro F1**.
Models: `FacebookAI/xlm-roberta-base`, `Qwen/Qwen2.5-7B-Instruct` (loaded with `AutoModel`, bf16, on one
24 GB GPU, last 2 of 28 blocks trained), `ai4bharat/IndicBERT-v3-1B` (gated repo, needs a Hugging Face token).

### Metrics

Macro-averaged **precision, recall and F1** (plus per-class values), for sentiment (3 classes, test tweets) and for
LID (3 tags, test words). Nothing else is reported.

### Run

```sh
bash run.sh xlm-roberta-base                                                                   # 5 modes + tables
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  EXTRA='--dtype bfloat16 --gradient_checkpointing --eval_batch_size 32' bash run.sh Qwen/Qwen2.5-7B-Instruct   # fits one 24 GB GPU
# (sharding over both GPUs with --device_map auto --max_memory 0:7GiB,1:20GiB also works when GPU 0 is free; on 2026-11-07,
#  with another user's job loading GPU 0, layers placed on GPU 0 returned NaN, so the single-GPU command is used)
bash run.sh ai4bharat/IndicBERT-v3-1B                                                          # after `huggingface-cli login`

python3 main.py --mode joint_contrastive --model_name xlm-roberta-base                         # one run
python3 evaluate.py --split test --per_class                                                   # results/evaluation_results_test.{csv,md}
python3 main.py --mode joint_contrastive --max_train_samples 320 --max_eval_samples 160 --epochs 1   # smoke test

# basic baseline: zero-/one-shot prompting, no training (results/<model>_prompt_<setting>*.{csv,json})
CUDA_VISIBLE_DEVICES=1 python3 prompt_baseline.py --model_name Qwen/Qwen2.5-7B-Instruct --batch_size 1
CUDA_VISIBLE_DEVICES=1 ~/venvs/gemma4/bin/python prompt_baseline.py --model_name google/gemma-4-E4B-it --batch_size 1
#   (one-time: python3 -m venv --system-site-packages ~/venvs/gemma4 && ~/venvs/gemma4/bin/pip install "transformers>=5.5,<5.7")
```

### Checkpoints and demo

```sh
# 1. train and store the best-dev-epoch weights as checkpoints/<run>.pt
python3 main.py --mode joint_contrastive --model_name xlm-roberta-base --save_model --results_dir checkpoints
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 main.py --mode joint_contrastive \
  --model_name Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --gradient_checkpointing --eval_batch_size 32 --save_model --results_dir checkpoints

# 2. inference from the .pt file (the tokenizer and the frozen encoder come from the pre-trained checkpoint)
python3 joint_contrastive.py "yaar movie bahut acchi thi but ending was terrible"
python3 joint_contrastive.py --verify                        # re-scores the test split, compares with the scores stored in the .pt
JCL_CHECKPOINT=checkpoints/Qwen2.5-7B-Instruct_joint_contrastive_seed42.pt python3 joint_contrastive.py --verify --batch_size 32   # needs a 24 GB GPU

# 3. local UI at http://127.0.0.1:7860
pip install torch transformers sentencepiece gradio     # demo laptop; on this server use ~/venvs/demo/bin/python (see below)
python3 gradio.py                                       # files needed on the laptop: gradio.py, joint_contrastive.py, checkpoints/<run>.pt
```

A `.pt` file holds the weights that were trained (last two encoder blocks + all heads) at the best dev epoch, the name of
the pre-trained encoder, the label lists and the dev / test scores of that run; the frozen encoder weights are not
stored because they equal the pre-trained checkpoint. XLM-R: 68 MB, runs on a laptop CPU; Qwen2.5-7B: 2.1 GB, needs a
24 GB GPU. Both retrains reproduced the scores of the results table exactly, and `--verify` reproduces the stored test
scores from the `.pt` file (Qwen with `--batch_size 32`, its evaluation batch size; another batch size shifts the
bf16 rounding of a few tweets, 0.14 F1 at 64). `joint_contrastive.py` is
self-contained (it does not import `lib/`) and exposes the two functions used for merging the group demo:

```python
from joint_contrastive import get_lid, get_sentiment
get_lid(["yaar", "movie", "bahut", "acchi", "thi", "but", "ending", "was", "terrible"])
# -> ["h", "e", "h", "h", "h", "e", "e", "e", "e"]        one tag per word, in {h, e, o}
get_sentiment("kya mast gaana hai yaar , i love it")
# -> "positive"        negative | neutral | positive
```

The UI has a "Model" box that lists every `.pt` file in `checkpoints/` (the runs of the results table): XLM-R is loaded
at start, the Qwen2.5-7B checkpoint is loaded when it is selected (needs a GPU, about 16 GB; on a machine without one
the UI shows an error and XLM-R keeps working). Loaded checkpoints stay in memory, so switching back is instant, and
the line under the box shows which checkpoint is active with the test scores of that run. On this server:
`CUDA_VISIBLE_DEVICES=1 GRADIO_SERVER_PORT=7870 ~/venvs/demo/bin/python gradio.py`.

Limits to keep in mind for a live demo: the model was trained on SentiMix tweets (81 % are replies with @mentions, median
28 words, negatives are mostly angry or abusive replies). On short, politely worded complaints typed by hand it often
answers neutral; tweet-style input works best. Inference on CPU (float32) and on GPU (bf16) agrees on all sentiment
labels of 600 test tweets and on all but about 0.1 % of the word tags (borderline words).

Notes: (1) the UI file is called `gradio.py` as agreed in the group, which shadows the Gradio library for every other
file in the same folder; the file works around this itself, but renaming it (e.g. `app.py`) is the safer choice.
(2) On this server Gradio lives in a separate environment (`python3 -m venv --system-site-packages ~/venvs/demo &&
~/venvs/demo/bin/pip install "gradio>=5,<6"`), because the current Gradio 6 would upgrade `huggingface_hub` and break
the transformers 4.51 of the base environment. Checked with Gradio 5.50 + transformers 4.51 and Gradio 6.32 + transformers 5.23.

Outputs per run in `results/`: `<run>.csv` (sentiment predictions), `<run>_lid.csv` (word tags), `<run>_metrics.json`,
`ground.csv`.

### Layout

```
lib/data.py      SentiMix loader, sub-word <-> word alignment, JointDataset (LID labels per sub-word + sentiment)
lib/model.py     JointModel: encoder, LID head, cross-attention pooling, SupCon losses, ablation switches
lib/train.py     training loop, macro P/R/F1 of both tasks, result files
main.py          CLI gateway (--mode, loss weights, project hyper-parameters)
evaluate.py      collects results/*_metrics.json into the results table
run.sh           all modes for one model (resumable: finished runs are skipped)
joint_contrastive.py  inference from checkpoints/*.pt: get_lid(words), get_sentiment(sentence); self-contained
gradio.py        local demo UI (http://127.0.0.1:7860) on top of those two functions
checkpoints/     <run>.pt written by main.py --save_model, with the predictions and metrics of that run
prompt_baseline.py  basic baseline: zero-/one-shot sentiment prompting of a chat model (Qwen2.5-7B-Instruct, Gemma 4 E4B)
prompts/         the zero-shot and one-shot sentiment prompt templates (same text as the group repo)
report_sheet.py  rows for the group's result sheet (with accuracy), recomputed from the saved predictions
queue_qwen.sh    helper that waits for free GPU memory before launching the Qwen pipeline
```

### Results (SentiMix test, seed 42, macro precision / recall / F1 x100; `results/evaluation_results_test.md`)

| model | mode | contrastive | sentiment P | sentiment R | sentiment F1 | LID P | LID R | LID F1 | min/run |
|---|---|---|---|---|---|---|---|---|---|
| xlm-roberta-base | sentiment (baseline) | - | 68.75 | 68.34 | **68.52** | - | - | - | 1.2 |
| xlm-roberta-base | joint | - | 68.36 | 67.95 | 68.13 | 89.19 | 89.10 | 89.13 | 1.4 |
| xlm-roberta-base | joint_contrastive | LID + sentiment | 68.44 | 67.82 | 68.08 | 89.15 | 89.08 | 89.10 | 2.0 |
| xlm-roberta-base | contrastive_lid | LID | 68.31 | 67.68 | 67.94 | 89.19 | 89.12 | **89.14** | 1.8 |
| xlm-roberta-base | contrastive_sent | sentiment | 68.52 | 68.13 | 68.29 | 89.16 | 89.06 | 89.09 | 1.6 |
| Qwen2.5-7B-Instruct | sentiment (baseline) | - | 71.41 | 69.78 | **70.34** | - | - | - | 15.4 |
| Qwen2.5-7B-Instruct | joint | - | 72.78 | 67.20 | 68.32 | 88.63 | 88.56 | 88.59 | 15.7 |
| Qwen2.5-7B-Instruct | joint_contrastive | LID + sentiment | 73.47 | 66.80 | 67.97 | 88.65 | 88.63 | 88.64 | 16.0 |
| Qwen2.5-7B-Instruct | contrastive_lid | LID | 73.03 | 67.15 | 68.30 | 88.65 | 88.65 | **88.65** | 15.9 |
| Qwen2.5-7B-Instruct | contrastive_sent | sentiment | 73.35 | 66.94 | 68.12 | 88.62 | 88.51 | 88.56 | 15.8 |
| IndicBERT-v3-1B | *(not run yet: needs a Hugging Face token with access to the gated repo)* | | | | | | | | |
| Qwen2.5-7B-Instruct | zero-shot prompting (no training) | - | 63.60 | 65.56 | 62.32 | - | - | - | 6.8 (inference) |
| Qwen2.5-7B-Instruct | one-shot prompting (no training) | - | 65.49 | 62.96 | 62.69 | - | - | - | 5.2 (inference) |
| gemma-4-E4B-it | zero-shot prompting (no training) | - | 70.13 | 68.82 | 62.90 | - | - | - | 5.2 (inference) |
| gemma-4-E4B-it | one-shot prompting (no training) | - | 68.63 | 67.62 | 64.59 | - | - | - | 5.4 (inference) |

The last four rows are the basic prompting baseline (`prompt_baseline.py`, same approach as the group's exp1 prompting
scripts): the test tweet is detokenised into a sentence, placed in the zero-shot or one-shot prompt
(`prompts/*_sentiment_prompt.txt`), the chat model answers with greedy decoding (at most 10 new tokens), and the first of
positive / negative / neutral in the answer is the prediction (neutral if none; 0 of 2,944 answers needed this).
No training data is used. Gemma 4 E4B runs in a separate virtual environment with transformers 5.6
(`~/venvs/gemma4`), since the base environment's transformers 4.51 does not know the model.

Per-class F1, joint_contrastive: XLM-R sentiment negative 68.3 / neutral 61.1 / positive 74.9, LID h 86.8 / e 82.1 / o 98.4;
Qwen sentiment negative 65.6 / neutral 65.8 / positive 72.5, LID h 86.3 / e 80.7 / o 98.9.
Qwen ran on a single 24 GB GPU (bf16, gradient checkpointing, batch 32, ~1.8 it/s); XLM-R on one GPU in fp32 + bf16 autocast.

**Observations (one seed each; the seed spread of this recipe was +-0.1-0.3 for XLM-R in Experiment 2, about +-1 for Qwen)**

- **Joint training works for LID.** With the same two trainable blocks the joint model tags words at macro F1 89.1
  (XLM-R) and 88.6 (Qwen) while also predicting sentiment, i.e. one encoder serves both tasks. The frozen single-task
  XLM-R tagger of Experiment 1 reached 92.8 with all blocks trained.
- **Joint training does not help sentiment.** XLM-R: every joint variant is within 0.2-0.6 of the single-task baseline
  (68.5). Qwen: the joint variants are 2.0-2.4 *below* the baseline (68.0-68.3 vs 70.3), with higher precision but
  lower recall; their best dev epoch is epoch 1 (dev F1 64.2 -> 62.9 -> 63.6), whereas the baseline peaks at epoch 3
  (57.5 -> 61.6 -> 63.5) - the extra LID gradient makes the two trainable blocks fit faster and then drift. The
  negative class loses most (72.2 -> 65.6-67.0).
- **The contrastive terms change nothing.** Relative to plain joint training, adding SupCon on LID, on sentiment or on
  both moves sentiment macro F1 by at most +-0.4 and LID macro F1 by at most +-0.06, for both models. The SupCon losses
  barely move during training (XLM-R, epoch means: LID tokens 6.92 -> 6.71 -> 6.70, tweets 3.40 -> 3.36 -> 3.35;
  Qwen: 6.87 -> 6.75 -> 6.71 and 3.38 -> 3.30 -> 3.15) while the cross-entropies fall normally. With only the last two
  blocks trainable the contrastive objectives act as a mild regulariser at most - the same picture as the LASER3-CO
  distillation of Experiment 2.
- **Prompting without training is clearly weaker than fine-tuning.** Zero-/one-shot prompting reaches macro F1
  62.3-62.7 (Qwen) and 62.9-64.6 (Gemma 4 E4B) against 70.3 for the fine-tuned Qwen baseline; the one-shot example
  helps both models a little (+0.4 / +1.7). The gap is almost entirely the neutral class (F1 37-53 when prompted vs
  64 fine-tuned): the prompted models commit to positive or negative on tweets the annotators labelled neutral.
- **Answer.** Under the project recipe, joint contrastive learning of LID and sentiment gives a model that does both
  tasks, but the sentiment score is not better than the single-task attention-pooling baseline (XLM-R: equal within
  noise; Qwen: 2 points lower at one seed). More trainable blocks, a larger contrastive weight schedule, or a
  per-task learning rate would be the next things to try.
