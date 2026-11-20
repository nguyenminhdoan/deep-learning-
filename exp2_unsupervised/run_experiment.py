"""Experiment 2: does unsupervised Word2Vec pre-training help the classifier?

Run from the project root:
    python -m exp2_unsupervised.run_experiment           # full experiment
    python -m exp2_unsupervised.run_experiment --quick   # 1 seed, smoke test

Steps
  1. Train Word2Vec (skip-gram and CBOW) on tweet text only, no labels.
  2. Inspect the vectors: nearest neighbours and a t-SNE map.
  3. Linear probe: logistic regression on averaged word vectors.
  4. Transfer learning: the same BiLSTM with random / frozen Word2Vec /
     fine-tuned Word2Vec embeddings, repeated over several seeds.
  5. Label-efficiency: repeat with only 10 / 25 / 50 % of the training labels.

All output goes to results/exp2_unsupervised/.
"""
import argparse
import json
import os
import time

# Quieten TensorFlow / joblib start-up noise (must be set before they load).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import config
from common.data import DisasterTweetDataset, Vocabulary
from common.evaluation import (METRICS, apply_plot_style, compute_metrics,
                               plot_confusion_matrix, plot_training_curves)
from .classifier import BiLSTMClassifier
from .word2vec_trainer import Word2VecTrainer

OUT_DIR = config.RESULTS_DIR / "exp2_unsupervised"

# Words whose neighbours we inspect to sanity-check the learned vectors.
PROBE_WORDS = ["fire", "earthquake", "flood", "storm", "killed", "bomb",
               "evacuation", "police", "love", "lol", "<url>"]

# name -> (which Word2Vec model or None, are the embeddings trainable?)
VARIANTS = {
    "Random init (baseline)": (None, True),
    "W2V skip-gram, frozen": ("skip-gram", False),
    "W2V skip-gram, fine-tuned": ("skip-gram", True),
    "W2V CBOW, frozen": ("CBOW", False),
    "W2V CBOW, fine-tuned": ("CBOW", True),
}
VARIANT_COLORS = dict(zip(VARIANTS, config.SERIES_COLORS))
FRACTION_VARIANTS = ["Random init (baseline)", "W2V skip-gram, frozen",
                     "W2V skip-gram, fine-tuned"]


class UnsupervisedExperiment:
    """Runs every step of Experiment 2 and saves tables and figures."""

    def __init__(self, seeds: list[int], fractions: list[float]):
        self.seeds = seeds
        self.fractions = fractions
        self.w2v_cfg = config.Word2VecConfig()
        self.clf_cfg = config.ClassifierConfig()

        self.data = DisasterTweetDataset()
        self.train = self.data.subset("train")
        self.val = self.data.subset("val")
        self.test = self.data.subset("test")

        # One vocabulary for every variant, built from the unlabelled corpus,
        # so the baseline and the Word2Vec models see identical inputs.
        self.corpus = self.data.unsupervised_corpus()
        self.vocab = Vocabulary(self.corpus, min_count=self.w2v_cfg.min_count)
        encode = lambda frame: self.vocab.encode(frame["tokens"], self.clf_cfg.max_len)
        self.x = {"train": encode(self.train), "val": encode(self.val),
                  "test": encode(self.test)}
        self.y = {"train": self.train["target"].to_numpy(),
                  "val": self.val["target"].to_numpy(),
                  "test": self.test["target"].to_numpy()}

        self.w2v: dict[str, Word2VecTrainer] = {}
        self.matrices: dict[str, np.ndarray] = {}

    # ------------------------------------------- 1. unsupervised training ----
    def train_word2vec(self) -> None:
        """Train skip-gram and CBOW on the unlabelled corpus and save them."""
        print(f"Word2Vec corpus: {len(self.corpus)} tweets, "
              f"vocabulary {len(self.vocab)} tokens")
        for skip_gram in (True, False):
            trainer = Word2VecTrainer(self.w2v_cfg, skip_gram).fit(self.corpus)
            self.w2v[trainer.name] = trainer
            self.matrices[trainer.name] = trainer.embedding_matrix(self.vocab)
            file_name = f"word2vec_{trainer.name.lower().replace('-', '')}.kv"
            trainer.vectors.save(str(OUT_DIR / file_name))

    # --------------------------------------------- 2. inspect the vectors ----
    def inspect_embeddings(self) -> None:
        """Nearest-neighbour tables and t-SNE maps of the word vectors."""
        report = "\n\n".join(t.nearest_neighbours(PROBE_WORDS) for t in self.w2v.values())
        print(report)
        (OUT_DIR / "nearest_neighbours.txt").write_text(report, encoding="utf-8")

        # Share of each word's training tweets that are disasters (colour only).
        seen, positive = {}, {}
        for tokens, label in zip(self.train["tokens"], self.y["train"]):
            for token in set(tokens):
                seen[token] = seen.get(token, 0) + 1
                positive[token] = positive.get(token, 0) + label
        rate = {w: positive[w] / n for w, n in seen.items() if n >= 5}
        for trainer in self.w2v.values():
            trainer.plot_tsne(rate, OUT_DIR / f"tsne_{trainer.name.lower()}.png",
                              label_words=PROBE_WORDS + ["wildfire", "suicide", "new",
                                                         "hiroshima", "god", "video"])

    # ------------------------------------------------- 3. linear probing ----
    def linear_probe(self) -> pd.DataFrame:
        """Logistic regression on mean word vectors: features with no fine-tuning."""
        rows = []
        for name, trainer in self.w2v.items():
            probe = LogisticRegression(max_iter=1000)
            probe.fit(trainer.tweet_vectors(self.train["tokens"]), self.y["train"])
            prob = probe.predict_proba(trainer.tweet_vectors(self.test["tokens"]))[:, 1]
            rows.append({"model": f"Mean W2V {name} + logistic regression",
                         **compute_metrics(self.y["test"], prob)})
        table = pd.DataFrame(rows)
        table.to_csv(OUT_DIR / "linear_probe.csv", index=False)
        print("\nLinear probe on frozen mean-pooled vectors (test split):")
        print(table.round(4).to_string(index=False))
        return table

    # ---------------------------------------------- 4. transfer learning ----
    def _run_once(self, variant: str, seed: int, train_idx=None):
        """Train one BiLSTM; returns (test metrics, test probabilities, history)."""
        source, trainable = VARIANTS[variant]
        matrix = self.matrices[source] if source else None
        clf = BiLSTMClassifier(len(self.vocab), self.w2v_cfg.vector_size,
                               self.clf_cfg, matrix, trainable, seed)
        x_train, y_train = self.x["train"], self.y["train"]
        if train_idx is not None:            # label-efficiency runs use a subset
            x_train, y_train = x_train[train_idx], y_train[train_idx]
        start = time.time()
        history = clf.fit(x_train, y_train, self.x["val"], self.y["val"])
        prob = clf.predict_proba(self.x["test"])
        metrics = compute_metrics(self.y["test"], prob)
        metrics["val_accuracy"] = max(history["val_accuracy"])
        metrics["epochs"] = len(history["loss"])
        metrics["train_seconds"] = round(time.time() - start, 1)
        return metrics, prob, history

    def compare_embeddings(self) -> pd.DataFrame:
        """Train every variant on every seed and save tables and figures."""
        rows, histories = [], {}
        for variant in VARIANTS:
            for seed in self.seeds:
                metrics, prob, history = self._run_once(variant, seed)
                rows.append({"variant": variant, "seed": seed, **metrics})
                print(f"  {variant:<28} seed {seed}  "
                      f"acc {metrics['accuracy']:.4f}  f1 {metrics['f1']:.4f}  "
                      f"({metrics['epochs']} epochs)")
                if seed == self.seeds[0]:    # figures use the first seed
                    histories[variant] = history
                    slug = variant.lower().replace(" ", "_").replace(",", "") \
                                  .replace("(", "").replace(")", "")
                    plot_confusion_matrix(self.y["test"], prob, variant,
                                          OUT_DIR / f"confusion_{slug}.png")
        runs = pd.DataFrame(rows)
        runs.to_csv(OUT_DIR / "all_runs.csv", index=False)

        plot_training_curves(histories, "BiLSTM training curves by embedding type",
                             OUT_DIR / "training_curves.png")
        summary = self._summarise(runs)
        self._plot_f1_comparison(summary)
        return runs

    def _summarise(self, runs: pd.DataFrame) -> pd.DataFrame:
        """Mean and standard deviation over seeds, saved as CSV and Markdown."""
        grouped = runs.groupby("variant", sort=False)[METRICS + ["epochs"]]
        summary = grouped.mean().join(grouped.std(ddof=0)[METRICS].add_suffix("_std"))
        summary.to_csv(OUT_DIR / "summary.csv")

        lines = ["| Model | Accuracy | Precision | Recall | F1 |", "|---|---|---|---|---|"]
        for variant, row in summary.iterrows():
            cells = " | ".join(f"{row[m]:.3f} ± {row[m + '_std']:.3f}" for m in METRICS)
            lines.append(f"| BiLSTM, {variant} | {cells} |")
        table = "\n".join(lines)
        (OUT_DIR / "summary.md").write_text(
            f"Test split, mean ± std over {len(self.seeds)} seeds\n\n{table}\n",
            encoding="utf-8")
        print(f"\nTest split, mean ± std over {len(self.seeds)} seeds:\n{table}")
        return summary

    def _plot_f1_comparison(self, summary: pd.DataFrame) -> None:
        """Dot plot: mean test F1 per variant, whiskers = std over seeds.

        Dots (not bars) are used because the axis is zoomed in on the range
        where the models differ and does not start at zero.
        """
        names = list(summary.index)[::-1]
        f1 = summary.loc[names, "f1"]
        std = summary.loc[names, "f1_std"]
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        for i, name in enumerate(names):
            ax.errorbar(f1[name], i, xerr=std[name], fmt="o", markersize=9,
                        color=VARIANT_COLORS[name], ecolor="#52514e",
                        elinewidth=1.2, capsize=4)
            ax.text(f1[name], i + 0.22, f"{f1[name]:.3f}", ha="center",
                    fontsize=9, color="#0b0b0b")
        ax.set_yticks(range(len(names)), names)
        ax.set_ylim(-0.6, len(names) - 0.3)
        ax.set_xlim(f1.min() - std.max() - 0.02, f1.max() + std.max() + 0.02)
        ax.set_xlabel(f"Test F1 (mean of {len(self.seeds)} seeds, whiskers = std)")
        ax.grid(axis="y", visible=False)
        ax.set_title("Effect of the embedding initialisation on test F1")
        fig.savefig(OUT_DIR / "f1_comparison.png")
        plt.close(fig)

    # ----------------------------------------------- 5. label efficiency ----
    def label_efficiency(self, full_runs: pd.DataFrame) -> None:
        """Repeat the comparison with only a fraction of the training labels.

        Word2Vec keeps using all unlabelled text; only the number of *labelled*
        tweets shrinks. This is where pre-training is expected to matter most.
        """
        rows = []
        n_train = len(self.y["train"])
        for fraction in self.fractions:
            for seed in self.seeds:
                # Same labelled subset for every variant at a given seed.
                idx = np.random.default_rng(seed).choice(
                    n_train, int(n_train * fraction), replace=False)
                for variant in FRACTION_VARIANTS:
                    metrics, _, _ = self._run_once(variant, seed, idx)
                    rows.append({"fraction": fraction, "variant": variant,
                                 "seed": seed, **metrics})
                    print(f"  {fraction:>4.0%} labels  {variant:<28} seed {seed}  "
                          f"f1 {metrics['f1']:.4f}")
        # The 100 % point comes from the main comparison (no need to retrain).
        full = full_runs[full_runs["variant"].isin(FRACTION_VARIANTS)].assign(fraction=1.0)
        runs = pd.concat([pd.DataFrame(rows), full], ignore_index=True)
        runs.to_csv(OUT_DIR / "label_efficiency_runs.csv", index=False)

        stats = runs.groupby(["variant", "fraction"])["f1"].agg(["mean", "std"]).fillna(0)
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for variant in FRACTION_VARIANTS:
            part = stats.loc[variant]
            n_labels = part.index * n_train
            ax.plot(n_labels, part["mean"], marker="o", markersize=6,
                    color=VARIANT_COLORS[variant], label=variant)
            ax.fill_between(n_labels, part["mean"] - part["std"],
                            part["mean"] + part["std"],
                            color=VARIANT_COLORS[variant], alpha=0.12, linewidth=0)
        ax.set_xscale("log")
        ticks = [int(f * n_train) for f in sorted(set(runs["fraction"]))]
        ax.set_xticks(ticks, [f"{t}\n({t / n_train:.0%})" for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("Labelled training tweets")
        ax.set_ylabel("Test F1")
        ax.legend(loc="lower right")
        ax.set_title("Label efficiency: test F1 vs. amount of labelled data "
                     "(band = std over seeds)")
        fig.savefig(OUT_DIR / "label_efficiency.png")
        plt.close(fig)
        print("\nLabel efficiency, mean test F1:")
        print(stats["mean"].unstack("fraction").round(4).to_string())

    # ----------------------------------------------------------- driver ----
    def run(self) -> None:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")   # hide retracing warnings (many models)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        apply_plot_style()
        print("== 1. Training Word2Vec (unsupervised) ==")
        self.train_word2vec()
        print("\n== 2. Inspecting the embeddings ==")
        self.inspect_embeddings()
        print("\n== 3. Linear probe ==")
        self.linear_probe()
        print("\n== 4. Transfer learning into the BiLSTM ==")
        runs = self.compare_embeddings()
        if self.fractions:
            print("\n== 5. Label efficiency ==")
            self.label_efficiency(runs)

        # Record the settings so the results can be reproduced.
        settings = {"word2vec": vars(self.w2v_cfg), "classifier": vars(self.clf_cfg),
                    "seeds": self.seeds, "vocab_size": len(self.vocab),
                    "corpus_tweets": len(self.corpus),
                    "split_sizes": {k: int(len(v)) for k, v in self.y.items()}}
        (OUT_DIR / "settings.json").write_text(json.dumps(settings, indent=2))
        print(f"\nAll results saved to {OUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true",
                        help="one seed and no label-efficiency study (smoke test)")
    parser.add_argument("--seeds", type=int, default=3, help="number of seeds")
    args = parser.parse_args()
    if args.quick:
        experiment = UnsupervisedExperiment(seeds=[0], fractions=[])
    else:
        experiment = UnsupervisedExperiment(seeds=list(range(args.seeds)),
                                            fractions=[0.10, 0.25, 0.50])
    experiment.run()


if __name__ == "__main__":
    main()
