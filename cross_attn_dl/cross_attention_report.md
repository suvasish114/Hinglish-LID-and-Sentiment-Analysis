# Cross-Attention Pooling for Joint LID and Sentiment on Hindi-English Tweets

Experiment report for `cross_attn_dl/` (CS613-NLP, IIT Gandhinagar). Written 2026-11-15.
All scores are macro-averaged and multiplied by 100. Every trained run uses seed 42.

## 1. Summary

Cross-attention pooling over language-aware token states did not improve sentiment over plain attention pooling.
On the SentiMix Hi-En test set, XLM-R scored 68.08 macro F1 against a 68.52 single-task baseline, and Qwen2.5-7B
scored 67.97 against 70.34.

| Test macro F1 | Baseline: sentiment only, attention pooling | Joint, cross-attention | Joint, cross-attention, contrastive (proposed) |
|---|---|---|---|
| Sentiment, XLM-R | **68.52** | 68.13 | 68.08 |
| Sentiment, Qwen2.5-7B | **70.34** | 68.32 | 67.97 |
| LID, XLM-R | not trained | 89.13 | 89.10 |
| LID, Qwen2.5-7B | not trained | 88.59 | 88.64 |

- **One model does both tasks.** With two trainable encoder blocks the joint model tags words at 89.1 (XLM-R) and
  88.6 (Qwen2.5-7B) macro F1 while also predicting sentiment.
- **The contrastive terms have no measurable effect.** Against plain joint training they move sentiment F1 by at
  most 0.35 and LID F1 by at most 0.06.
- **The trained cross-attention does not use language identity.** A probe of the two saved checkpoints shows that
  removing the LID signal from the pooling changes 2 of 2,944 test predictions for XLM-R and 1 for Qwen2.5-7B. The
  injected LID vector is 0.04 % and 0.01 % of the size of the token state it is added to (section 6).
- **So the idea itself is still untested.** The fusion weights start at zero, and the shared learning rate of 2e-5
  over 1,305 steps cannot move them far. The joint runs measured multi-task training with a different pooling
  module, not LID-informed pooling.
- **Fine-tuning beats prompting.** Zero-shot and one-shot prompting reach 62.3 to 64.6 macro F1 with no training.
- **Caveats.** Every number is one seed. No run separates the pooling module from the added LID task. The Qwen
  joint models were selected at epoch 1, the Qwen baseline at epoch 3.

## 2. Question and background

The experiment asks whether feeding word-level language identity into the sentiment representation helps sentiment
on Hindi-English code-mixed tweets.

The previous experiment (`hlsa-exp2`, approach 4) applied a contrastive loss to sentiment only. LID tags were used
there just to build monolingual views of each tweet. This experiment changes three things at once:

1. Both tasks are trained in one model: word-level language identification (LID) and tweet-level sentiment.
2. Each task gets its own supervised contrastive loss.
3. The two tasks are joined through a cross-attention pooling that reads the LID prediction of every token.

## 3. Model

Code: `cross_attn_dl/lib/model.py` (`JointModel`, `CrossAttentionPooling`, `AttentionPooling`, `supcon_loss`).

### 3.1 Architecture

```
 tweet (lower-cased words)
      |  tokenizer: sub-words, max length 128
      v
 +------------------------------+
 | pre-trained encoder          |   last 2 blocks + final norm trained, everything else frozen
 +------------------------------+
      |  h_i : one state per sub-word
      |
      +----> LID head:  lid_logits_i = W_l dropout(h_i) ----> CE_lid        (+ SupCon_lid on g_l(h_i))
      |                        |
      |                        | softmax
      |                        v
      |                   p_i : posterior over {h, e, o}
      v                        |
 k_i = v_i = LayerNorm(h_i + W_e p_i)      <-- language-aware token states, W_e starts at zero
      |
 learned query q ----> 8-head cross-attention ----> s ----> W_c dropout(s) ----> CE_sent   (+ SupCon_sent on g_s(s))
```

### 3.2 Cross-attention pooling

| Part | Definition in the code |
|---|---|
| Query | One learned vector `q` of the hidden size, initialised as `0.02 * randn`. The same query is used for every tweet. |
| LID injection | `W_e`: a bias-free linear map from the 3 LID probabilities to the hidden size, **initialised to zero**. |
| Keys and values | `k_i = v_i = LayerNorm(h_i + W_e p_i)`, where `p_i = softmax(lid_logits_i)`. |
| Attention | `nn.MultiheadAttention`, 8 heads, dropout 0.1. Output `s` is the pooled tweet vector. |
| Mask | Only sub-words that belong to a word are attended. Special tokens and padding are masked out. |
| Gradient path | `p_i` is not detached, so the sentiment loss can reach the LID head through `W_e`. |
| Attention weights | Returned as `out["attention"]` (averaged over heads). The training runs never saved or analysed them. |

At step 0, `W_e = 0`, so the module is a multi-head attention pooling with no language information. Language
information enters only as far as `W_e` grows during training.

### 3.3 The pooling it is compared with

The baseline uses plain attention pooling: `a_i = softmax(w^T h_i + b)` and `s = sum_i a_i h_i`. Its pooled vector is
a weighted average of raw encoder states. The cross-attention output instead passes through a LayerNorm and two
freshly initialised linear maps inside the attention module. The baseline and the joint models therefore differ in
two ways: the LID task and the pooling module.

### 3.4 Losses

```
L = CE_sent + alpha * CE_lid + lambda_lid * SupCon_lid + lambda_sent * SupCon_sent
alpha = 1,  lambda_lid = lambda_sent = 0.1,  temperature tau = 0.1
```

| Term | Anchors | Positives |
|---|---|---|
| `CE_lid` | every sub-word of a word, labelled with the word's tag | not applicable |
| `SupCon_lid` | word sub-words of the batch, capped at 2,048 random tokens | sub-words with the same language tag |
| `SupCon_sent` | the 32 tweets of the batch | tweets with the same sentiment label |

SupCon is the supervised contrastive loss of Khosla et al. (2020). Each contrastive term has its own projection
head, a 2-layer MLP to 128 dimensions, and is used during training only.

### 3.5 Model sizes

| Model | Mode | Total parameters | Trainable | Trainable share |
|---|---|---|---|---|
| xlm-roberta-base (12 blocks, blocks 10 and 11 trained) | sentiment | 278 M | 14.2 M | 5.10 % |
| | joint | 280 M | 16.5 M | 5.90 % |
| | contrastive_lid, contrastive_sent | 281 M | 17.2 M | 6.13 % |
| | joint_contrastive | 282 M | 17.9 M | 6.36 % |
| Qwen2.5-7B-Instruct (28 blocks, blocks 26 and 27 trained) | sentiment | 7,071 M | 466.1 M | 6.59 % |
| | joint | 7,122 M | 517.6 M | 7.27 % |
| | contrastive_lid, contrastive_sent | 7,135 M | 530.9 M | 7.44 % |
| | joint_contrastive | 7,149 M | 544.2 M | 7.61 % |

The cross-attention module adds about 2.4 M parameters for XLM-R and about 51 M for Qwen2.5-7B. The baseline's
attention pooling has 769 and 3,585.

## 4. Experimental setup

### 4.1 Data

SentiMix Hi-En (SemEval-2020 Task 9), roman script, lower-cased, produced by the shared `get_sentimix.py`.

| Split | Tweets | Words | Negative / neutral / positive | Hindi / English / other words |
|---|---|---|---|---|
| train | 13,912 | 363,170 | 29.4 % / 37.6 % / 33.1 % | 46.5 % / 33.4 % / 20.1 % |
| dev | 2,978 | 78,069 | 29.7 % / 37.6 % / 32.7 % | 47.1 % / 32.7 % / 20.2 % |
| test | 2,944 | 76,916 | 30.4 % / 36.9 % / 32.7 % | 46.2 % / 33.7 % / 20.1 % |

Tweets have a median of 28 words. Every word carries a gold LID tag: `h` Hindi, `e` English, `o` other.

### 4.2 Configurations

| Mode (`--mode`) | LID head | Pooling | Contrastive | Role |
|---|---|---|---|---|
| `sentiment` | no | attention | none | single-task baseline |
| `joint` | yes | cross-attention | none | multi-task, no contrastive term |
| `joint_contrastive` | yes | cross-attention | LID and sentiment | proposed model |
| `contrastive_lid` | yes | cross-attention | LID only | ablation |
| `contrastive_sent` | yes | cross-attention | sentiment only | ablation |
| prompting, zero-shot and one-shot | no | not applicable | none | no-training baseline |

Trained models: `FacebookAI/xlm-roberta-base` and `Qwen/Qwen2.5-7B-Instruct`. Prompted models:
`Qwen/Qwen2.5-7B-Instruct` and `google/gemma-4-E4B-it`. `ai4bharat/IndicBERT-v3-1B` was planned but not run,
because the repository is gated and needs a Hugging Face token.

### 4.3 Training recipe (identical for all trained runs)

| Setting | Value |
|---|---|
| Seed | 42 |
| Batch size, epochs, steps | 32, 3, 1,305 |
| Optimiser | AdamW, learning rate 2e-5 for the encoder blocks **and** all new heads, weight decay 0.01 |
| Schedule | linear, 10 % warm-up |
| Other | max length 128, dropout 0.1, gradient clipping at 1.0, bf16 autocast |
| Checkpoint | the epoch with the best dev **sentiment** macro F1 |
| Qwen2.5-7B specifics | bf16 weights, gradient checkpointing, evaluation batch 32, one 24 GB GPU |

### 4.4 Evaluation

Macro precision, recall and F1, plus per-class values. Sentiment is scored on the 2,944 test tweets. LID is scored
on the 76,916 test words, taking the tag predicted at the first sub-word of each word.

The prompting baseline (`prompt_baseline.py`) puts the detokenised tweet into the prompt in `prompts/`, decodes
greedily for at most 10 new tokens, and takes the first of positive, negative or neutral in the answer. None of the
2,944 answers per run failed to parse.

### 4.5 Run log

| Date | What ran |
|---|---|
| 2026-11-06 | XLM-R, all five modes (1.2 to 2.0 minutes each) |
| 2026-11-07 | Qwen2.5-7B, all five modes (15.4 to 16.0 minutes each) |
| 2026-11-11 | Prompting baselines for Qwen2.5-7B and Gemma 4 E4B (5.2 to 6.8 minutes of inference each) |
| 2026-11-15 | `joint_contrastive` retrained with `--save_model` for both models; test scores identical to the first runs |
| 2026-11-15 | Demo UI (`gradio.py`), and the checkpoint probe of section 6 |

The README records that Qwen2.5-7B was first meant to be split across two GPUs. Layers placed on a GPU shared with
another job returned NaN on 2026-11-07, so all reported Qwen runs use one GPU.

## 5. Results

### 5.1 Test scores

| Model | Mode | Sentiment P | Sentiment R | Sentiment F1 | F1 vs baseline | LID P | LID R | LID F1 |
|---|---|---|---|---|---|---|---|---|
| xlm-roberta-base | sentiment (baseline) | 68.75 | 68.34 | **68.52** | | | | |
| xlm-roberta-base | joint | 68.36 | 67.95 | 68.13 | -0.39 | 89.19 | 89.10 | 89.13 |
| xlm-roberta-base | joint_contrastive | 68.44 | 67.82 | 68.08 | -0.44 | 89.15 | 89.08 | 89.10 |
| xlm-roberta-base | contrastive_lid | 68.31 | 67.68 | 67.94 | -0.58 | 89.19 | 89.12 | **89.14** |
| xlm-roberta-base | contrastive_sent | 68.52 | 68.13 | 68.29 | -0.23 | 89.16 | 89.06 | 89.09 |
| Qwen2.5-7B-Instruct | sentiment (baseline) | 71.41 | 69.78 | **70.34** | | | | |
| Qwen2.5-7B-Instruct | joint | 72.78 | 67.20 | 68.32 | -2.02 | 88.63 | 88.56 | 88.59 |
| Qwen2.5-7B-Instruct | joint_contrastive | 73.47 | 66.80 | 67.97 | -2.37 | 88.65 | 88.63 | 88.64 |
| Qwen2.5-7B-Instruct | contrastive_lid | 73.03 | 67.15 | 68.30 | -2.04 | 88.65 | 88.65 | **88.65** |
| Qwen2.5-7B-Instruct | contrastive_sent | 73.35 | 66.94 | 68.12 | -2.22 | 88.62 | 88.51 | 88.56 |
| Qwen2.5-7B-Instruct | zero-shot prompting | 63.60 | 65.56 | 62.32 | | | | |
| Qwen2.5-7B-Instruct | one-shot prompting | 65.49 | 62.96 | 62.69 | | | | |
| gemma-4-E4B-it | zero-shot prompting | 70.13 | 68.82 | 62.90 | | | | |
| gemma-4-E4B-it | one-shot prompting | 68.63 | 67.62 | 64.59 | | | | |

Source: `cross_attn_dl/results/evaluation_results_test.md` and the `*_metrics.json` files.

### 5.2 Per-class F1 on test

| Model | Mode | Negative | Neutral | Positive | LID h | LID e | LID o |
|---|---|---|---|---|---|---|---|
| xlm-roberta-base | sentiment | 68.71 | 61.57 | 75.29 | | | |
| xlm-roberta-base | joint | 68.64 | 60.90 | 74.84 | 86.87 | 82.13 | 98.39 |
| xlm-roberta-base | joint_contrastive | 68.29 | 61.06 | 74.89 | 86.84 | 82.05 | 98.41 |
| xlm-roberta-base | contrastive_lid | 68.06 | 60.94 | 74.83 | 86.89 | 82.10 | 98.43 |
| xlm-roberta-base | contrastive_sent | 68.93 | 61.11 | 74.84 | 86.83 | 82.08 | 98.36 |
| Qwen2.5-7B-Instruct | sentiment | 72.20 | 64.40 | 74.42 | | | |
| Qwen2.5-7B-Instruct | joint | 66.97 | 65.60 | 72.39 | 86.28 | 80.62 | 98.87 |
| Qwen2.5-7B-Instruct | joint_contrastive | 65.59 | 65.80 | 72.52 | 86.31 | 80.73 | 98.88 |
| Qwen2.5-7B-Instruct | contrastive_lid | 66.62 | 65.69 | 72.58 | 86.29 | 80.76 | 98.90 |
| Qwen2.5-7B-Instruct | contrastive_sent | 66.05 | 65.80 | 72.50 | 86.27 | 80.60 | 98.80 |
| Qwen2.5-7B-Instruct | zero-shot prompting | 67.56 | 42.87 | 76.51 | | | |
| Qwen2.5-7B-Instruct | one-shot prompting | 68.10 | 52.90 | 67.08 | | | |
| gemma-4-E4B-it | zero-shot prompting | 72.35 | 37.09 | 79.26 | | | |
| gemma-4-E4B-it | one-shot prompting | 70.45 | 46.66 | 76.67 | | | |

The Qwen joint models lose most on the negative class. Precision and recall show why: they label many more tweets
neutral.

| Qwen2.5-7B run | Negative P / R | Neutral P / R | Positive P / R |
|---|---|---|---|
| sentiment (baseline) | 74.52 / 70.02 | 59.90 / 69.61 | 79.81 / 69.71 |
| joint | 79.75 / 57.72 | 55.81 / 79.56 | 82.78 / 64.32 |
| joint_contrastive | 81.91 / 54.70 | 55.17 / 81.49 | 83.31 / 64.21 |
| contrastive_lid | 80.51 / 56.82 | 55.66 / 80.11 | 82.93 / 64.52 |
| contrastive_sent | 81.11 / 55.70 | 55.34 / 81.12 | 83.60 / 64.00 |

For XLM-R the per-class precision and recall of the joint models stay within 1.7 points of the baseline.

### 5.3 Dev scores and checkpoint selection

Dev sentiment macro F1 after each epoch. The selected epoch is the one that was scored on test.

| Model | Mode | Epoch 1 | Epoch 2 | Epoch 3 | Selected | Dev LID F1 at epochs 1 / 2 / 3 |
|---|---|---|---|---|---|---|
| xlm-roberta-base | sentiment | 60.17 | 61.09 | 62.35 | 3 | |
| xlm-roberta-base | joint | 60.63 | 60.67 | 62.07 | 3 | 88.51 / 89.03 / 89.17 |
| xlm-roberta-base | joint_contrastive | 60.45 | 60.72 | 61.87 | 3 | 88.55 / 89.11 / 89.22 |
| xlm-roberta-base | contrastive_lid | 60.62 | 60.37 | 61.87 | 3 | 88.57 / 89.14 / 89.26 |
| xlm-roberta-base | contrastive_sent | 60.59 | 60.80 | 62.07 | 3 | 88.46 / 89.01 / 89.14 |
| Qwen2.5-7B-Instruct | sentiment | 57.50 | 61.62 | 63.53 | 3 | |
| Qwen2.5-7B-Instruct | joint | 64.22 | 62.94 | 63.65 | **1** | 88.51 / 88.75 / 88.84 |
| Qwen2.5-7B-Instruct | joint_contrastive | 63.76 | 62.94 | 63.51 | **1** | 88.58 / 88.79 / 88.85 |
| Qwen2.5-7B-Instruct | contrastive_lid | 64.09 | 63.09 | 63.33 | **1** | 88.59 / 88.76 / 88.84 |
| Qwen2.5-7B-Instruct | contrastive_sent | 63.87 | 63.05 | 63.83 | **1** | 88.44 / 88.73 / 88.82 |

Three things follow from this table:

- On dev, the Qwen joint models are level with or above the baseline (63.8 to 64.2 against 63.5). The 2-point
  deficit appears on test only.
- All four Qwen joint models were selected at epoch 1, by margins of 0.04 to 0.76 over epoch 3. The baseline was
  selected at epoch 3. Test scores exist only for the selected epoch, so the effect of the joint objective cannot be
  separated from the effect of stopping two epochs earlier.
- Selecting on sentiment costs the Qwen models about 0.3 dev LID F1, since LID keeps improving until epoch 3.

### 5.4 Training losses

Mean training loss per epoch (epochs 1 / 2 / 3).

| Model | Run | CE sentiment | CE LID | SupCon LID | SupCon sentiment |
|---|---|---|---|---|---|
| xlm-roberta-base | sentiment | 0.955 / 0.843 / 0.824 | | | |
| xlm-roberta-base | joint_contrastive | 0.959 / 0.852 / 0.825 | 0.495 / 0.346 / 0.336 | 6.923 / 6.711 / 6.699 | 3.401 / 3.363 / 3.354 |
| Qwen2.5-7B-Instruct | sentiment | 1.086 / 0.756 / 0.560 | | | |
| Qwen2.5-7B-Instruct | joint_contrastive | 0.887 / 0.746 / 0.544 | 0.523 / 0.299 / 0.256 | 6.868 / 6.746 / 6.709 | 3.375 / 3.297 / 3.152 |

The cross-entropy terms fall normally. For the contrastive terms, two reference levels help (both are estimates
from the batch composition, not measurements):

- **Sentiment SupCon.** With 32 tweets, a model that does not separate the classes at all scores `ln(31) = 3.43`.
  Perfectly clustered classes would score about 2.3. XLM-R ends at 3.35 and Qwen at 3.15, so the pooled tweet
  vectors stayed close to unclustered.
- **LID SupCon.** With about 1,450 to 1,500 word sub-words per batch the unseparated level is about 7.3, and perfect
  clustering about 6.2. Both models end near 6.7, roughly half-way. The LID cross-entropy pushes the same token
  states towards language separation, so this term adds little of its own.

In its first epoch the Qwen joint model has a lower sentiment loss than the baseline (0.887 against 1.086) and its
best dev score. The runs cannot tell whether the LID gradient or the different pooling module causes this faster
start.

### 5.5 Prompting baseline

Prompting reaches 62.32 to 64.59 macro F1, which is 5.8 to 8.0 below the fine-tuned Qwen baseline. The one-shot
example adds 0.37 for Qwen2.5-7B and 1.69 for Gemma 4 E4B. The gap is almost entirely the neutral class: neutral F1
is 37.1 to 52.9 when prompted against 64.4 fine-tuned. The prompted models commit to positive or negative on tweets
that the annotators labelled neutral.

## 6. What the trained cross-attention does

The training runs returned the attention weights but did not examine them. For this report, the two saved
`joint_contrastive` checkpoints were probed on the test split on 2026-11-15. Nothing was retrained. The probe
re-runs inference, reproduces the stored test scores exactly (68.08 and 67.97), and then changes only what the
pooling receives as LID input.

### 6.1 The LID signal is negligible at inference

| Measurement on the 2,944 test tweets | XLM-R | Qwen2.5-7B |
|---|---|---|
| Sentiment macro F1, model as trained | 68.08 | 67.97 |
| Sentiment macro F1, LID injection removed (`W_e p_i = 0`) | 68.15 | 67.97 |
| Sentiment macro F1, uniform LID posterior (1/3 each) | 68.15 | 67.97 |
| Sentiment macro F1, gold LID tags as one-hot posterior | 68.11 | 67.97 |
| Predictions changed by removing the injection | 2 | 1 |
| Predictions changed by a uniform posterior | 2 | 0 |
| Predictions changed by gold LID tags | 1 | 0 |
| Mean norm of the token state `h_i` | 21.47 | 103.16 |
| Mean norm of the injected vector `W_e p_i` | 0.0077 | 0.0149 |
| Injection as a share of the token state | 0.036 % | 0.014 % |
| Largest absolute weight in `W_e` | 0.0014 | 0.0018 |

Perfect LID tags change at most one prediction. The sentiment output is, in practice, independent of language
identity.

### 6.2 The attention is close to an average over the tweet

| Attention statistic (mean over test tweets, weights averaged over the 8 heads) | XLM-R | Qwen2.5-7B |
|---|---|---|
| Word sub-words per tweet | 45.0 | 47.4 |
| Effective number of sub-words attended (exponential of the entropy) | 44.8 | 38.2 |
| Weight on the most-attended sub-word | 2.8 % | 8.4 % |
| Attention on Hindi sub-words / their share of sub-words | 46.9 % / 44.2 % | 44.1 % / 47.1 % |
| Attention on English sub-words / their share | 35.7 % / 36.2 % | 31.3 % / 35.0 % |
| Attention on other sub-words / their share | 17.4 % / 19.6 % | 24.6 % / 17.9 % |
| Centre of the attention along the tweet (0 = start, 1 = end) | 0.495 | 0.515 |
| Norm of the learned query `q` after training | 0.60 | 1.26 |

For XLM-R the pooling is an almost exact mean over the word sub-words. Qwen2.5-7B is somewhat more selective and
puts 1.4 times the proportional weight on the "other" class, which holds punctuation, emoji and similar tokens.
Neither model attends by language.

### 6.3 Why the fusion path stayed inactive

This follows from the initialisation and the learning rate, not from the data.

- **`W_e` starts at zero and cannot travel far.** The learning rates of all 1,305 steps add up to about 0.013. Adam
  moves a weight by roughly one learning rate per step at most, so no weight of `W_e` could end further than about
  0.013 from zero. The measured maximum is 0.0014 to 0.0018. Even at that ceiling the injected vector would stay
  below 2 % of the token state.
- **The query stays small.** `q` is initialised with an expected norm of 0.55 (XLM-R) and 1.20 (Qwen) and ends at
  0.60 and 1.26. The keys are LayerNorm outputs with a much larger norm, so a small query gives nearly equal
  attention scores to all tokens. This explanation is consistent with the measurements but was not tested
  separately.
- **The new modules share the encoder's learning rate.** `--head_lr` exists but was left unset, so the query, `W_e`
  and the attention matrices all train at 2e-5.

Only the `joint_contrastive` checkpoints were saved, so only they were probed. The other three joint modes use the
same initialisation and recipe, and the same behaviour is expected but not verified.

## 7. Findings

1. **Joint training gives a working tagger at little sentiment cost for XLM-R.** LID reaches 89.1 macro F1 and
   sentiment stays within 0.2 to 0.6 of the baseline. The dedicated XLM-R tagger of `hlsa-exp1`, with all blocks
   trained, reached 92.8.
2. **Sentiment did not improve.** XLM-R is 0.2 to 0.6 below the baseline, a difference of the same order as the
   seed spread. Qwen2.5-7B is 2.0 to 2.4 lower on test, with negative recall falling from 70.0 to between 54.7 and
   57.7 as predictions shift to neutral.
3. **The contrastive terms change nothing measurable.** They move sentiment F1 by at most 0.35 and LID F1 by at
   most 0.06 relative to plain joint training. The tweet-level term stays near its unclustered level.
4. **The cross-attention fusion was effectively switched off.** The LID injection is under 0.04 % of the token
   state, and removing it changes one or two test predictions. The question "does language identity help sentiment
   pooling" is therefore not answered by these runs, in either direction.
5. **Fine-tuning two blocks beats prompting by 5.8 to 8.0 macro F1,** mostly through the neutral class.

## 8. Limitations

- **One seed.** The README cites a seed spread of 0.1 to 0.3 for XLM-R and about 1 for Qwen under this recipe in
  `hlsa-exp2`. The differences between the joint variants are inside that range. The XLM-R gap to the baseline
  (0.2 to 0.6) is of the same order, so one seed cannot confirm it.
- **Two changes at once.** The baseline has no LID task and plain attention pooling; every joint run has both the
  LID task and cross-attention pooling. No run has the LID task with plain attention pooling.
- **Checkpoint selection differs for Qwen.** Joint models were tested at epoch 1 and the baseline at epoch 3
  (section 5.3). The 2-point test gap may partly be a selection effect.
- **The fusion never became active** (section 6), so the results do not test LID-informed pooling.
- **Head-averaged attention.** The probe reads attention weights averaged over the 8 heads. Single heads may be
  sharper than their average.
- **Only two checkpoints were saved,** both `joint_contrastive`. The baseline's attention pooling could not be
  probed for comparison.
- **Missing model.** IndicBERT-v3-1B was not run.

## 9. Recommended next experiments

| Experiment | What it settles | How | Cost |
|---|---|---|---|
| Joint training with plain attention pooling | Whether the LID task or the pooling module causes the difference from the baseline | `python3 main.py --mode joint --fusion attn --run_name xlm-roberta-base_joint_attn_seed42` | about 1.4 min (XLM-R), 16 min (Qwen) |
| Higher learning rate for the new modules | Whether the fusion helps once `W_e` and `q` can move | add `--head_lr 1e-4` and `--head_lr 1e-3` to the joint modes, with a new `--run_name`; then repeat the probe | 2 to 16 min per run |
| Seeds 1 and 2 | Whether the Qwen 2-point gap is real | `--seed 1`, `--seed 2` for `sentiment`, `joint`, `joint_contrastive` | 6 runs per model |
| Test score of the last epoch for Qwen | Whether the gap is a selection effect | `lib/train.py` keeps only the best-dev weights; also score epoch 3 | one retrain, 16 min |
| A fusion that cannot start at zero | Whether LID-informed pooling works at all | non-zero initialisation or a learned scale for `W_e`, or one query per language | code change in `lib/model.py` |
| Log attention statistics per run | Makes section 6 available for every run | `out["attention"]` is already returned by the model | small change in `lib/train.py` |
| More trainable blocks, other contrastive weights | The README's own follow-ups | `--unfreeze_last 4`, `--con_weight_lid`, `--con_weight_sent` | as above |

Pass `--run_name` whenever a mode is re-run with changed settings. The default name is `<model>_<mode>_seed<seed>`,
so a re-run without it overwrites the existing result files.

## 10. Reproduction

Run from `cross_attn_dl/`.

```sh
# all five modes and the result tables
bash run.sh xlm-roberta-base
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  EXTRA='--dtype bfloat16 --gradient_checkpointing --eval_batch_size 32' bash run.sh Qwen/Qwen2.5-7B-Instruct

# one run, and the tables
python3 main.py --mode joint_contrastive --model_name xlm-roberta-base
python3 evaluate.py --split test --per_class
python3 evaluate.py --split dev

# prompting baseline
CUDA_VISIBLE_DEVICES=1 python3 prompt_baseline.py --model_name Qwen/Qwen2.5-7B-Instruct --batch_size 1

# checkpoints and inference
python3 main.py --mode joint_contrastive --model_name xlm-roberta-base --save_model --results_dir checkpoints
python3 joint_contrastive.py --verify
```

## Appendix A. Files

| Path in `cross_attn_dl/` | Content |
|---|---|
| `lib/model.py` | `JointModel`, cross-attention pooling, attention pooling, SupCon loss |
| `lib/data.py` | SentiMix loader, sub-word to word alignment, LID labels per sub-word |
| `lib/train.py` | training loop, metrics, result files, checkpoint writer |
| `main.py` | command-line entry point, modes and hyper-parameters |
| `run.sh` | all five modes for one model, resumable |
| `evaluate.py` | collects `*_metrics.json` into the result tables |
| `prompt_baseline.py`, `prompts/` | zero-shot and one-shot prompting |
| `joint_contrastive.py`, `gradio.py` | inference from a checkpoint, demo UI |
| `results/` | predictions, word tags and `*_metrics.json` per run, result tables |
| `checkpoints/` | the two saved `joint_contrastive` models |
| `logs/` | one log per run |

## Appendix B. How the probe of section 6 was run

The probe loads a checkpoint through `joint_contrastive.load_model`, encodes the test tweets exactly as
`joint_contrastive.py --verify` does, and calls the pooling four times per batch with different LID inputs: the
model's own posterior, zeros, a uniform posterior, and the one-hot gold tags. Attention statistics come from the
weights that `nn.MultiheadAttention` returns for the unmodified model. Evaluation batch sizes match training (64 for
XLM-R, 32 for Qwen2.5-7B). The probe script is not part of the repository.
