"""Exploratory data analysis of the 'NLP with Disaster Tweets' dataset.

Run:  python eda.py
Writes figures and a text summary to results/eda/.
"""
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from common import config
from common.data import DisasterTweetDataset, Vocabulary
from common.evaluation import apply_plot_style

OUT_DIR = config.RESULTS_DIR / "eda"
CLASS_COLORS = [config.COLOR_NEGATIVE, config.COLOR_POSITIVE]


class TweetExplorer:
    """Produces every EDA figure and statistic for the dataset."""

    def __init__(self, dataset: DisasterTweetDataset):
        self.data = dataset
        self.df = dataset.labelled.copy()
        self.df["n_tokens"] = self.df["tokens"].map(len)
        self.df["n_chars"] = self.df["text"].str.len()
        self.lines: list[str] = []   # text summary, written to summary.txt

    def log(self, text: str = "") -> None:
        """Print a line and keep it for the summary file."""
        print(text)
        self.lines.append(text)

    # ------------------------------------------------------- statistics ----
    def overview(self) -> None:
        """Sizes, missing values, class balance, duplicates, length stats."""
        df, unl = self.df, self.data.unlabelled
        self.log(f"Labelled tweets (train.csv): {len(df)}")
        self.log(f"Unlabelled tweets (test.csv): {len(unl)}")
        self.log("\nMissing values (labelled):")
        for col in ["keyword", "location", "text"]:
            self.log(f"  {col:<9} {df[col].isna().mean():6.1%}")
        self.log("\nClass balance:")
        for cls, name in enumerate(config.CLASS_NAMES):
            n = (df["target"] == cls).sum()
            self.log(f"  {name:<13} {n:5d}  ({n / len(df):.1%})")
        self.log("\nShared split (stratified, seed 42):")
        for name, part in df.groupby("split"):
            self.log(f"  {name:<6} {len(part):5d}  disaster rate {part['target'].mean():.1%}")

        # Duplicated tweets, and duplicates whose copies disagree on the label.
        groups = df.groupby("text")["target"].agg(["count", "nunique"])
        self.log(f"\nDuplicate rows (same text): {df.duplicated('text').sum()}")
        self.log(f"Texts labelled inconsistently (both 0 and 1): "
                 f"{(groups['nunique'] > 1).sum()}")

        self.log("\nTweet length in tokens (after cleaning):")
        pct = np.percentile(df["n_tokens"], [50, 90, 95, 99, 100])
        self.log("  median {:.0f} | p90 {:.0f} | p95 {:.0f} | p99 {:.0f} | max {:.0f}"
                 .format(*pct))
        covered = (df["n_tokens"] <= 32).mean()
        self.log(f"  tweets fully kept with max_len=32: {covered:.1%}")

    def vocabulary_stats(self) -> None:
        """Vocabulary size, rare words, and coverage of the held-out test split."""
        corpus = self.data.unsupervised_corpus()
        vocab_all = Vocabulary(corpus, min_count=1)
        vocab_2 = Vocabulary(corpus, min_count=2)
        singletons = sum(1 for c in vocab_all.counts.values() if c == 1)
        self.log(f"\nWord2Vec corpus: {len(corpus)} tweets, "
                 f"{sum(map(len, corpus))} tokens")
        self.log(f"Unique tokens: {len(vocab_all) - 2} "
                 f"({singletons} appear only once)")
        self.log(f"Vocabulary with min_count=2: {len(vocab_2) - 2}")
        test_tokens = [t for toks in self.data.subset("test")["tokens"] for t in toks]
        in_vocab = np.mean([t in vocab_2.stoi for t in test_tokens])
        self.log(f"Test-split tokens covered by that vocabulary: {in_vocab:.1%}")

    # ---------------------------------------------------------- figures ----
    def plot_class_balance(self) -> None:
        counts = self.df["target"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        bars = ax.bar(config.CLASS_NAMES, counts.values, color=CLASS_COLORS, width=0.55)
        ax.bar_label(bars, [f"{v}\n({v / len(self.df):.0%})" for v in counts.values],
                     padding=3, color="#0b0b0b")
        ax.set_ylim(0, counts.max() * 1.2)
        ax.set_ylabel("Tweets")
        ax.grid(axis="x", visible=False)
        ax.set_title("Class balance of the labelled tweets")
        fig.savefig(OUT_DIR / "01_class_balance.png")
        plt.close(fig)

    def plot_length_distributions(self) -> None:
        """Token and character length per class, as overlaid step histograms."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        specs = [("n_tokens", "Tokens per tweet", np.arange(0, 46, 2)),
                 ("n_chars", "Characters per tweet", np.arange(0, 165, 8))]
        for ax, (col, label, bins) in zip(axes, specs):
            for cls, name in enumerate(config.CLASS_NAMES):
                values = self.df.loc[self.df["target"] == cls, col]
                ax.hist(values, bins=bins, density=True, histtype="step",
                        linewidth=2, color=CLASS_COLORS[cls],
                        label=f"{name} (mean {values.mean():.1f})")
            ax.set_xlabel(label)
            ax.set_ylabel("Share of tweets (density)")
            ax.set_title(label)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "02_length_distributions.png")
        plt.close(fig)

    def plot_keyword_disaster_rate(self, top: int = 12) -> None:
        """Keywords most and least associated with real disasters."""
        stats = (self.df.dropna(subset=["keyword"])
                 .assign(keyword=lambda d: d["keyword"].str.replace("%20", " "))
                 .groupby("keyword")["target"].agg(["mean", "count"])
                 .sort_values("mean"))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True)
        panels = [(stats.tail(top), config.COLOR_POSITIVE, "Most disaster-related keywords"),
                  (stats.head(top).iloc[::-1], config.COLOR_NEGATIVE,
                   "Least disaster-related keywords")]
        for ax, (part, color, title) in zip(axes, panels):
            part = part.sort_values("mean")
            bars = ax.barh(part.index, part["mean"], color=color, height=0.6)
            ax.bar_label(bars, [f"{v:.0%}" for v in part["mean"]], padding=3,
                         fontsize=8, color="#0b0b0b")
            ax.set_xlim(0, 1.12)
            ax.set_xlabel("Share of tweets that are real disasters")
            ax.grid(axis="y", visible=False)
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "03_keyword_disaster_rate.png")
        plt.close(fig)

    def plot_top_words(self, top: int = 15) -> None:
        """Most frequent content words in each class (stop words removed)."""
        skip = set(ENGLISH_STOP_WORDS) | {"<url>", "<user>", "<number>", "!", "?",
                                          "i'm", "it's", "don't", "just", "like"}
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
        for cls, ax in enumerate(axes):
            tokens = self.df.loc[self.df["target"] == cls, "tokens"]
            counts = Counter(t for toks in tokens for t in toks
                             if t not in skip and len(t) > 2)
            words, freqs = zip(*counts.most_common(top)[::-1])
            ax.barh(words, freqs, color=CLASS_COLORS[cls], height=0.6)
            ax.set_xlabel("Occurrences")
            ax.grid(axis="y", visible=False)
            ax.set_title(f"Top words: {config.CLASS_NAMES[cls]}")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "04_top_words.png")
        plt.close(fig)

    def plot_tweet_features(self) -> None:
        """How often Twitter-specific elements appear in each class."""
        text = self.df["text"]
        features = {
            "Contains a link": text.str.contains(r"https?://"),
            "Contains a number": text.str.contains(r"\d"),
            "Contains a #hashtag": text.str.contains("#"),
            "Contains a @mention": text.str.contains("@"),
            "Contains ! or ?": text.str.contains(r"[!?]"),
            "Has a keyword": self.df["keyword"].notna(),
            "Has a location": self.df["location"].notna(),
        }
        names = list(features)[::-1]
        y = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for cls, offset in [(0, 0.19), (1, -0.19)]:
            mask = self.df["target"] == cls
            rates = [features[n][mask].mean() for n in names]
            bars = ax.barh(y + offset, rates, height=0.34, color=CLASS_COLORS[cls],
                           label=config.CLASS_NAMES[cls])
            ax.bar_label(bars, [f"{r:.0%}" for r in rates], padding=3, fontsize=8,
                         color="#0b0b0b")
        ax.set_yticks(y, names)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel("Share of tweets in the class")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right")
        ax.set_title("Twitter-specific features by class")
        fig.savefig(OUT_DIR / "05_tweet_features.png")
        plt.close(fig)

    def show_examples(self, n: int = 3) -> None:
        """Raw text next to its cleaned tokens, for a few tweets of each class."""
        self.log("\nExamples (raw -> tokens):")
        for cls, name in enumerate(config.CLASS_NAMES):
            sample = self.df[self.df["target"] == cls].sample(n, random_state=config.SEED)
            for _, row in sample.iterrows():
                self.log(f"  [{name}] {row['text']!r}\n      -> {' '.join(row['tokens'])}")

    def run(self) -> None:
        """Run the full analysis and write everything to results/eda/."""
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        apply_plot_style()
        self.overview()
        self.vocabulary_stats()
        self.show_examples()
        self.plot_class_balance()
        self.plot_length_distributions()
        self.plot_keyword_disaster_rate()
        self.plot_top_words()
        self.plot_tweet_features()
        (OUT_DIR / "summary.txt").write_text("\n".join(self.lines), encoding="utf-8")
        print(f"\nEDA figures and summary saved to {OUT_DIR}")


if __name__ == "__main__":
    TweetExplorer(DisasterTweetDataset()).run()
