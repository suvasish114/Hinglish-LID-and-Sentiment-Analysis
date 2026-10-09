import os, ast 
from tqdm import tqdm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from model import load_model, get_embeddings

def plot_tsne(emb1, emb2, emb1_label, emb2_label, filename):
    blue = np.asarray(emb1, dtype=np.float32)
    red = np.asarray(emb2, dtype=np.float32)
    combined = np.concatenate([blue, red], axis=0)
    points = TSNE(n_components=2, perplexity=min(30, len(combined) - 1), init="random", learning_rate="auto", random_state=42,).fit_transform(combined)
    split = len(blue)
    plt.figure(figsize=(5, 5))
    plt.scatter(points[:split, 0], points[:split, 1], color="blue", label=emb1_label, alpha=0.6, s=15)
    plt.scatter(points[split:, 0], points[split:, 1], color="red", label=emb2_label, alpha=0.6, s=15)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.legend()
    plt.tight_layout()
    os.makedirs("plot", exist_ok=True)
    plt.savefig(f"plot/{filename}.png", dpi=300, bbox_inches="tight")
    print(f"plot saved at: plot/{filename}.png")

def get_words_by_tag(csv_path, tag, total_words=1000):
    df = pd.read_csv(csv_path)
    collected = []
    for row_index, row in tqdm(df.iterrows(), desc=f"collecting samples: {tag}"):
        words = ast.literal_eval(row["sentence"])
        tags = ast.literal_eval(row["tags"])
        if len(words) != len(tags):
            raise ValueError(f"Row {row_index}: word and tag counts differ.")
        for word, word_tag in zip(words, tags):
            if word_tag == tag:
                collected.append(word)
                if len(collected) == total_words:
                    return collected
    return collected

def main(csv_path, set_name):
    model = load_model()
    hi_samples = get_words_by_tag(csv_path, 'h', total_words=2000)
    en_samples = get_words_by_tag(csv_path, 'e', total_words=2000)
    # ot_samples = get_words_by_tag(csv_path, 'o', total_words=2000)
    hi_embeddings = [get_embeddings(model, word)["embedding"] for word in tqdm(hi_samples, desc="collecting hin emb:")]
    en_embeddings = [get_embeddings(model, word)["embedding"] for word in tqdm(en_samples, desc="collecting eng emb:")]
    plot_tsne(hi_embeddings, en_embeddings, "hindi-roman", "english-roman", f"tsne_hin_eng_{set_name}")
    # ot_embeddings = [get_embeddings(model, word)["embedding"] for word in tqdm(ot_samples, desc="collecting oth emb:")]
    # plot_tsne(hi_embeddings, ot_samples, "hindi-roman", "other", "tsne_hin_oth")

if __name__ == "__main__":
    csv_path="/Users/papai/Documents/CS613/dataset/SentiMix/dev.csv"
    main(csv_path, "dev")
    csv_path="/Users/papai/Documents/CS613/dataset/SentiMix/train.csv"
    main(csv_path, "train")