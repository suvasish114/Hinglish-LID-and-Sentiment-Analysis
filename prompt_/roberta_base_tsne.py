# # imports
# import torch
# from pathlib import Path
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModel
# from tsne import plot_tsne, get_words_by_tag, get_embeddings

# # config
# model_card = "FacebookAI/xlm-roberta-base"

# # helper functions
# def get_model(model_path):
#     model_dir = Path(model_path)
#     model, tokenizer = None, None
#     if model_dir.is_dir():
#         tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
#         model = AutoModel.from_pretrained("xlm-roberta-base")

#         tokenizer.save_pretrained(model_dir)
#         model.save_pretrained(model_dir)
#     else:
#         MODEL_DIR = "models/xlm-roberta-base"
#         tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
#         model = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True)
#         model.eval()
#     return model, tokenizer

# # @torch.inference_mode()
# # def get_word_embedding(model, tokenizer, word: str) -> torch.Tensor:
# #     if not word.strip():
# #         raise ValueError("word must not be empty")
# #     inputs = tokenizer(word, return_tensors="pt", return_special_tokens_mask=True)
# #     special_mask = inputs.pop("special_tokens_mask")
# #     device = next(model.parameters()).device
# #     inputs = {key: value.to(device) for key, value in inputs.items()}
# #     hidden_states = model(**inputs).last_hidden_state[0]
# #     word_mask = ((special_mask[0] == 0) & (inputs["attention_mask"][0].cpu() == 1)).to(device)
# #     if not word_mask.any():
# #         raise ValueError("word produced no usable tokens")
# #     return hidden_states[word_mask].mean(dim=0).cpu()

# # drivng code
# if __name__ == "__main__":
#     model_dir = "models/xlm-roberta-base"
#     model, tokenizer = get_model(model_dir)
#     csv_path="/Users/papai/Documents/CS613/dataset/SentiMix/train.csv"
#     hi_samples = get_words_by_tag(csv_path, 'h', total_words=2000)
#     en_samples = get_words_by_tag(csv_path, 'e', total_words=2000)
#     # hi_embeddings = [get_word_embedding(model, tokenizer, word) for word in tqdm(hi_samples, desc="collecting hin emb:")]
#     # en_embeddings = [get_word_embedding(model, tokenizer, word) for word in tqdm(en_samples, desc="collecting eng emb:")]
#     # plot_tsne(hi_embeddings, en_embeddings, "hindi-roman", "english-roman", f"roberta_tsne_hin_eng_train")




# imports
import ast
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import (precision_recall_fscore_support, accuracy_score,
                             confusion_matrix)
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForMaskedLM

# config
MODEL_CARD = "FacebookAI/xlm-roberta-base"
MODEL_DIR = "models/xlm-roberta-base-mlm"

TEST_CSV = "dataset/SentiMix/test.csv"
ZERO_SHOT_PROMPT = "prompts/zero_shot_lid_prompt.txt"   
ONE_SHOT_PROMPT = "prompts/one_shot_lid_prompt.txt"
N_SAMPLES = 10                              # None = all of test.csv
MAX_SENTENCE_TOKENS = 100                   
LABELS = ["positive", "negative", "neutral"]
SAVE_EVERY = 250                            # partial save every N sentences

PREDICTIONS_CSV = "predictions_xlmr.csv"
METRICS_CSV = "metrics_xlmr.csv"


# data
def fix_mojibake(token: str) -> str:
    # Repair tokens like 'â€¦' or 'ðÿ˜…' (UTF-8 bytes that were read as cp1252).
    # The 'ÿ' case is an emoji whose 'Ÿ' got lower-cased by the dataset's
    # preprocessing, so it is restored before decoding. Tokens that cannot be
    # repaired are returned unchanged.
    
    if token.isascii():
        return token
    try:
        return token.replace("ÿ", "Ÿ").encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return token


def detokenize(tokens: list) -> str:
    # Join the token list back into a normal-looking sentence.
    # Re-attaches @mentions, #hashtags, URLs split into 'https // t . co / x',
    # underscores inside handles, and punctuation, then repairs broken emoji.
    
    toks = [fix_mojibake(str(t)) for t in tokens]
    out, i, n = [], 0, len(toks)
    attach_next = False                 # glue the next token to the previous one
    no_space_before = {".", ",", "!", "?", ";", ":", ")", "]", "%", "…"}
    glue_after = {"@", "#", "(", "["}

    while i < n:
        tok = toks[i]

        # URL: https // t . co / abc  ->  https://t.co/abc
        if tok in ("http", "https") and i + 1 < n and toks[i + 1] == "//":
            url, i, expect_word = tok + "://", i + 2, True
            while i < n:
                if toks[i] in {".", "/", "-"}:
                    url += toks[i]
                    expect_word = True
                elif expect_word:
                    url += toks[i]
                    expect_word = False
                else:
                    break
                i += 1
            out.append((url, attach_next))
            attach_next = False
            continue

        glue = attach_next or tok in no_space_before or tok == "_"
        if out and out[-1][0].endswith("_"):
            glue = True                 # keep @user_name together
        out.append((tok, glue))
        attach_next = tok in glue_after or tok == "_"
        i += 1

    text = ""
    for piece, glue in out:
        text += piece if (glue or not text) else " " + piece
    return text.strip()


def load_test_samples(csv_path: str, n: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path).head(n).copy()
    df["text"] = df["sentence"].apply(lambda s: detokenize(ast.literal_eval(s)))
    df["sentiment"] = df["sentiment"].str.strip().str.lower()
    return df


# model
def has_weights(folder: Path) -> bool:
    return any(folder.glob("*.safetensors")) or any(folder.glob("*.bin"))


def find_local_model(model_path: str):
    # Look for the model folder next to where you run from, then next to this
    # script. config.json alone is not enough: an interrupted download leaves it
    # behind without the weights, and that must count as "missing".
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        candidate = base / model_path
        if (candidate / "config.json").exists() and has_weights(candidate):
            return candidate
    return None


def download_model(model_path: str) -> Path:
    # Copy the repo's files as published (same method as the IndicBERT file).
    # The repo also holds TensorFlow / Flax / ONNX copies of the same weights,
    # which PyTorch never reads, so they are skipped. The .bin duplicate is
    # skipped too unless the repo has no .safetensors file.
    skip = ["*.h5", "*.msgpack", "*.onnx", "onnx/*", "*.ot"]
    snapshot_download(MODEL_CARD, local_dir=model_path,
                      ignore_patterns=skip + ["*.bin"])
    if not any(Path(model_path).glob("*.safetensors")):
        snapshot_download(MODEL_CARD, local_dir=model_path, ignore_patterns=skip)
    return Path(model_path)


def get_model(model_path: str = MODEL_DIR):
    # Load xlm-roberta-base WITH its masked-LM head (download once, then reuse).

    local = find_local_model(model_path)
    if local is None:
        print("[!] no model found at '{}' (looked in {} and {})".format(
            model_path, Path.cwd(), Path(__file__).resolve().parent))
        print("[+] downloading {} into '{}' ...".format(MODEL_CARD, model_path))
        local = download_model(model_path)
        if not has_weights(local):
            raise RuntimeError("download finished but {} has no weights".format(local))
    print("[+] loading model from", local)

    tokenizer = AutoTokenizer.from_pretrained(str(local), local_files_only=True)
    model, info = AutoModelForMaskedLM.from_pretrained(
        str(local), local_files_only=True, output_loading_info=True)
    if "lm_head.dense.weight" in info["missing_keys"]:
        raise RuntimeError("The LM head was not in the checkpoint, so it would be "
                           "random (was this folder saved with AutoModel?). "
                           "Delete {} and run again.".format(local))
    model.eval()
    return model, tokenizer


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def label_token_ids(tokenizer) -> list:
    # One vocabulary id per label: the first real sub-word of ' positive' etc.
    ids = []
    for label in LABELS:
        pieces = [t for t in tokenizer.encode(" " + label, add_special_tokens=False)
                  if tokenizer.convert_ids_to_tokens(t) != "▁"]
        ids.append(pieces[0])
    print("label tokens:", dict(zip(LABELS, tokenizer.convert_ids_to_tokens(ids))))
    if len(set(ids)) != len(ids):
        raise ValueError("two labels share their first sub-word; use another label word")
    return ids


#  prompting
def build_prompt(template: str, sentence: str, tokenizer) -> str:
    # Fill {text}, then turn the final 'Output:' into a slot for the model.
    # An encoder cannot write an answer, so we cut the prompt at the last
    # 'Output:' (dropping the trailing instruction in the zero-shot file) and put
    # <mask> where the label goes.
    
    ids = tokenizer.encode(sentence, add_special_tokens=False)[:MAX_SENTENCE_TOKENS]
    sentence = tokenizer.decode(ids)
    rendered = template.replace("{text}", sentence)
    head = rendered.rsplit("Output:", 1)[0]
    return head + "Output: " + tokenizer.mask_token


@torch.inference_mode()
def predict_label(model, tokenizer, prompt: str, label_ids: list, device) -> str:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    mask_pos = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero()
    if len(mask_pos) == 0:
        raise ValueError("<mask> was truncated away; shorten the prompt")
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc).logits[0, mask_pos[-1].item()]
    return LABELS[int(torch.argmax(logits[label_ids]))]


# scoring
def score(y_true: list, y_pred: list) -> dict:
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    result = {"accuracy": accuracy_score(y_true, y_pred), "per_class": {}}
    for i, label in enumerate(LABELS):
        result["per_class"][label] = (p[i], r[i], f[i], int(support[i]))
    for avg in ("macro", "weighted"):
        pa, ra, fa, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=LABELS, average=avg, zero_division=0)
        result[avg] = (pa, ra, fa)
    return result


def print_report(name: str, res: dict, y_true: list, y_pred: list) -> None:
    print("\n=== {} ===".format(name))
    print("{:<10}{:>10}{:>10}{:>10}{:>10}".format("class", "P", "R", "F1", "support"))
    for label, (p, r, f, s) in res["per_class"].items():
        print("{:<10}{:>10.3f}{:>10.3f}{:>10.3f}{:>10d}".format(label, p, r, f, s))
    for avg in ("macro", "weighted"):
        p, r, f = res[avg]
        print("{:<10}{:>10.3f}{:>10.3f}{:>10.3f}".format(avg, p, r, f))
    print("accuracy  {:.3f}".format(res["accuracy"]))
    print("confusion matrix (rows = true, cols = predicted, order {}):".format(LABELS))
    print(confusion_matrix(y_true, y_pred, labels=LABELS))


def metrics_rows(setting: str, res: dict) -> list:
    total = sum(v[3] for v in res["per_class"].values())
    rows = [dict(setting=setting, cls=label, precision=p, recall=r, f1=f, support=s)
            for label, (p, r, f, s) in res["per_class"].items()]
    for avg in ("macro", "weighted"):
        p, r, f = res[avg]
        rows.append(dict(setting=setting, cls=avg, precision=p, recall=r, f1=f,
                         support=total))
    rows.append(dict(setting=setting, cls="accuracy", precision=res["accuracy"],
                     recall=res["accuracy"], f1=res["accuracy"], support=total))
    return rows


def save_results(df: pd.DataFrame) -> None:
    # predictions: one row per sentence, both settings side by side
    df[["id", "text", "sentiment", "zero_shot", "one_shot"]].rename(
        columns={"text": "sentence", "sentiment": "actual_sentiment"}).to_csv(
        PREDICTIONS_CSV, index=False)
    print("\nwrote", PREDICTIONS_CSV)

    # metrics: P, R, F1 per class + macro / weighted / accuracy, per setting
    y_true = df["sentiment"].tolist()
    rows = (metrics_rows("zero_shot", score(y_true, df["zero_shot"].tolist()))
            + metrics_rows("one_shot", score(y_true, df["one_shot"].tolist())))
    pd.DataFrame(rows).rename(columns={"cls": "class"}).to_csv(METRICS_CSV, index=False)
    print("wrote", METRICS_CSV)


def run_setting(name, template, df, predict_fn) -> list:
    preds = []
    for i, sentence in enumerate(tqdm(df["text"], desc=name)):
        preds.append(predict_fn(template, sentence))
        if (i + 1) % SAVE_EVERY == 0:                    # partial save, just in case
            part = df.iloc[:len(preds)][["id", "text", "sentiment"]].copy()
            part[name] = preds
            part.to_csv("partial_xlmr_{}.csv".format(name), index=False)
    print_report(name, score(df["sentiment"].tolist(), preds),
                 df["sentiment"].tolist(), preds)
    return preds


# driving code
if __name__ == "__main__":
    df = load_test_samples(TEST_CSV, N_SAMPLES)
    print("sentences sent to the model:")
    for i, (text, gold) in enumerate(zip(df["text"], df["sentiment"])):
        print("{:>2}. [{}] {}".format(i + 1, gold, text))

    model, tokenizer = get_model(MODEL_DIR)
    device = pick_device()
    model.to(device)
    label_ids = label_token_ids(tokenizer)

    def predict(template, sentence):
        prompt = build_prompt(template, sentence, tokenizer)
        return predict_label(model, tokenizer, prompt, label_ids, device)

    zero_template = Path(ZERO_SHOT_PROMPT).read_text(encoding="utf-8")
    one_template = Path(ONE_SHOT_PROMPT).read_text(encoding="utf-8")

    df["zero_shot"] = run_setting("zero_shot", zero_template, df, predict)
    df["one_shot"] = run_setting("one_shot", one_template, df, predict)

    save_results(df)