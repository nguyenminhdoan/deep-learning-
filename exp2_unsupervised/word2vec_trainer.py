"""Unsupervised feature extraction: Word2Vec trained on unlabelled tweet text."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE

from common import config
from common.data import Vocabulary


class Word2VecTrainer:
    """Trains Word2Vec and exposes the learned vectors to the classifier.

    Word2Vec is unsupervised: it only sees token sequences and learns to
    predict a word from its neighbours (CBOW) or the neighbours from a word
    (skip-gram). The disaster labels are never used.
    """

    def __init__(self, cfg: config.Word2VecConfig, skip_gram: bool = True):
        self.cfg = cfg
        self.skip_gram = skip_gram
        self.name = "skip-gram" if skip_gram else "CBOW"
        self.model: Word2Vec | None = None

    def fit(self, corpus: list[list[str]]) -> "Word2VecTrainer":
        """Train on a list of tokenised tweets."""
        self.model = Word2Vec(
            sentences=corpus,
            vector_size=self.cfg.vector_size,
            window=self.cfg.window,
            min_count=self.cfg.min_count,
            sg=int(self.skip_gram),
            negative=self.cfg.negative,
            epochs=self.cfg.epochs,
            seed=self.cfg.seed,
            workers=1,          # a single worker keeps training reproducible
        )
        return self

    @property
    def vectors(self):
        """The trained gensim KeyedVectors."""
        return self.model.wv

    def embedding_matrix(self, vocab: Vocabulary) -> np.ndarray:
        """Build the (vocab_size, dim) matrix that initialises the Keras Embedding.

        Row i holds the Word2Vec vector of vocab word i. Row 0 (padding) stays
        zero; <unk> and any word Word2Vec did not keep get small random values.
        """
        rng = np.random.default_rng(self.cfg.seed)
        matrix = rng.normal(0, 0.05, (len(vocab), self.cfg.vector_size)).astype("float32")
        matrix[0] = 0.0
        for word, idx in vocab.stoi.items():
            if word in self.vectors:
                matrix[idx] = self.vectors[word]
        return matrix

    def tweet_vectors(self, token_lists) -> np.ndarray:
        """Represent each tweet by the mean of its word vectors.

        Used for the 'pure feature extraction' test: a linear classifier on
        these fixed features shows how much the unsupervised model captured.
        """
        dim = self.cfg.vector_size
        out = np.zeros((len(token_lists), dim), dtype="float32")
        for row, tokens in enumerate(token_lists):
            known = [self.vectors[t] for t in tokens if t in self.vectors]
            if known:
                out[row] = np.mean(known, axis=0)
        return out

    # ------------------------------------------------ qualitative checks ----
    def nearest_neighbours(self, words: list[str], top: int = 8) -> str:
        """Text table of the most similar words for each query word."""
        lines = [f"Nearest neighbours ({self.name})"]
        for word in words:
            if word not in self.vectors:
                lines.append(f"  {word:<12} (not in vocabulary)")
                continue
            similar = ", ".join(f"{w} ({s:.2f})"
                                for w, s in self.vectors.most_similar(word, topn=top))
            lines.append(f"  {word:<12} {similar}")
        return "\n".join(lines)

    def plot_tsne(self, word_disaster_rate: dict, path: Path, n_words: int = 400,
                  label_words: tuple = ()) -> None:
        """2-D t-SNE map of the most frequent word vectors.

        Each word is coloured by the share of its training tweets that are
        real disasters. The labels are used only to colour this picture, never
        to train Word2Vec, so visible clusters show structure found without
        supervision.
        """
        placeholders = {"<url>", "<user>", "<number>", "!", "?"}
        words = [w for w in self.vectors.index_to_key
                 if w in word_disaster_rate and w not in placeholders
                 and len(w) > 2][:n_words]
        coords = TSNE(n_components=2, perplexity=30, init="pca",
                      random_state=self.cfg.seed).fit_transform(self.vectors[words])
        rates = [word_disaster_rate[w] for w in words]

        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        # Diverging scale: blue = non-disaster words, orange = disaster words.
        cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
            "class", [config.COLOR_NEGATIVE, "#d9d8d2", config.COLOR_POSITIVE])
        points = ax.scatter(coords[:, 0], coords[:, 1], c=rates, cmap=cmap,
                            vmin=0, vmax=1, s=22, edgecolors="#fcfcfb", linewidths=0.4)
        for word in label_words:
            if word in words:
                x, y = coords[words.index(word)]
                ax.annotate(word, (x, y), xytext=(4, 4), textcoords="offset points",
                            fontsize=8, color="#0b0b0b")
        bar = fig.colorbar(points, ax=ax, shrink=0.6, pad=0.02)
        bar.set_label("Share of the word's tweets that are disasters")
        bar.outline.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(f"t-SNE of Word2Vec ({self.name}) vectors: "
                     f"{len(words)} most frequent words")
        fig.savefig(path)
        plt.close(fig)
