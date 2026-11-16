"""Local demo UI: type a Hinglish sentence, get the word-level language tags and the sentiment.

    python3 gradio.py                      # then open http://127.0.0.1:7860
    GRADIO_SERVER_PORT=7861 python3 gradio.py

The UI only calls the two functions get_lid(list_of_words) and get_sentiment(sentence); here they come from
joint_contrastive.py. The "Model" box lists the checkpoints of the training runs found in checkpoints/*.pt
(xlm-roberta-base runs on a CPU, Qwen2.5-7B needs a GPU); the selected one is loaded on first use and kept in memory.
To show another experiment, change the import.
"""

import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
# This file is itself called gradio.py, so a plain `import gradio` would find this file instead of the Gradio library.
# Its folder is taken off the import path while the library is imported and put back afterwards.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]
if __name__ != "__main__":                 # another file in this folder ran `import gradio` and got this file:
    sys.modules.pop(__name__, None)        # hand it the real library instead
import gradio as gr  # noqa: E402

sys.path.insert(0, HERE)
from joint_contrastive import get_lid, get_sentiment, load_model, model_info  # noqa: E402

CHECKPOINT_DIR = os.path.join(HERE, "checkpoints")
DEFAULT_CHECKPOINT = "xlm-roberta-base_joint_contrastive_seed42.pt"
TAG_NAMES = {"h": "Hindi", "e": "English", "o": "Other"}
TAG_COLORS = {"Hindi": "#f59e0b", "English": "#3b82f6", "Other": "#9ca3af"}
EXAMPLES = [
    "happy birthday bhai , bhagwan aapko hamesha khush rakhe",
    "kya mast gaana hai yaar , i love it",
    "tum logon ko sharam nahi aati kya , itna jhoot bolte ho",
    "worst government ever , sab chor hain",
    "kal office jaana hai meeting ke liye",
    "aaj shaam ko 5 baje class hai room 101 mein",
]
_LOCK = threading.Lock()                   # one request at a time: a request may switch the active model


def checkpoints() -> list:
    """The .pt files of the training runs, the default (XLM-R) first."""
    files = sorted(f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")) if os.path.isdir(CHECKPOINT_DIR) else []
    return sorted(files, key=lambda f: f != DEFAULT_CHECKPOINT)


def use_checkpoint(name: str) -> str:
    """Make the chosen checkpoint the active model and describe it (call with _LOCK held)."""
    try:
        load_model(os.path.join(CHECKPOINT_DIR, name))
    except Exception as err:               # e.g. the 7B checkpoint on a machine without a GPU
        raise gr.Error(str(err))
    info = model_info()
    return (f"**Loaded:** `{info['checkpoint']}`  |  pre-trained model `{info['model_name']}`  |  device `{info['device']}`  |  "
            f"test macro F1 of this run: sentiment {100 * info['test_sentiment_macro_f1']:.1f}, "
            f"LID {100 * info['test_lid_macro_f1']:.1f}")


def select_model(name: str) -> str:
    with _LOCK:
        return use_checkpoint(name)


def analyse(sentence: str, name: str):
    """sentence, checkpoint -> (sentiment, [(word, language), ...], what is loaded)."""
    words = (sentence or "").split()
    with _LOCK:
        status = use_checkpoint(name)      # instant when this checkpoint is already loaded
        if not words:
            return "", [], status
        tags = get_lid(words)              # one tag per word: h / e / o
        sentiment = get_sentiment(sentence)    # negative / neutral / positive
    highlighted = []
    for word, tag in zip(words, tags):
        highlighted += [(word, TAG_NAMES.get(tag, tag)), (" ", None)]
    return sentiment, highlighted[:-1], status


def build_demo(models: list, status: str):
    with gr.Blocks(title="Hinglish LID + Sentiment") as demo:
        gr.Markdown("# Hinglish language identification and sentiment\n"
                    "Joint contrastive model: one encoder predicts the language of every word "
                    "(Hindi / English / Other) and the sentiment of the sentence.")
        model = gr.Dropdown(choices=models, value=models[0], label="Model  -  checkpoint of a training run (checkpoints/*.pt)")
        loaded = gr.Markdown(status)
        sentence = gr.Textbox(label="Sentence", placeholder="Type a Hindi-English code-mixed sentence", lines=2)
        button = gr.Button("Analyse", variant="primary")
        sentiment = gr.Label(label="Sentiment  -  get_sentiment(sentence)")
        lid = gr.HighlightedText(label="Language of each word  -  get_lid(words)", color_map=TAG_COLORS,
                                 show_legend=True, combine_adjacent=False)
        gr.Examples(examples=[[e] for e in EXAMPLES], inputs=sentence)
        model.change(select_model, inputs=model, outputs=loaded, api_name="select_model")
        gr.on(triggers=[button.click, sentence.submit], fn=analyse, inputs=[sentence, model],
              outputs=[sentiment, lid, loaded], api_name="analyse")
    return demo


if __name__ == "__main__":
    models = checkpoints()
    if not models:
        sys.exit(f"no .pt file in {CHECKPOINT_DIR} - create one with `python3 main.py --mode joint_contrastive --save_model`")
    print(f"[+] checkpoints: {models}\n[+] loading {models[0]} ...", flush=True)
    status = select_model(models[0])       # load the first checkpoint before the page opens
    build_demo(models, status).launch(server_name="127.0.0.1")
