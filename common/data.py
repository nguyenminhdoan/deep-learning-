"""Data pipeline: tweet cleaning, the shared train/val/test split, vocabulary."""
import html
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


class TweetCleaner:
    """Normalises raw tweets and splits them into tokens.

    Twitter-specific noise (links, user handles, numbers) is replaced with
    placeholder tokens rather than deleted, because *the presence* of a link
    or a number is itself informative for disaster detection.
    """

    URL = re.compile(r"https?://\S+|www\.\S+")
    MENTION = re.compile(r"@\w+")
    NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")
    NON_ASCII = re.compile(r"[^\x00-\x7f]+")
    REPEATED = re.compile(r"(.)\1{2,}")           # "soooo" -> "soo"
    TOKEN = re.compile(r"<\w+>|[a-z]+(?:'[a-z]+)?|[!?]")

    def clean(self, text: str) -> str:
        """Return a lower-cased, normalised version of one tweet."""
        text = html.unescape(str(text))            # "&amp;" -> "&"
        text = self.URL.sub(" <url> ", text)
        text = self.MENTION.sub(" <user> ", text)
        text = self.NUMBER.sub(" <number> ", text)
        text = self.NON_ASCII.sub(" ", text)       # mojibake / emoji bytes
        text = text.replace("#", " ")              # keep the hashtag word itself
        text = self.REPEATED.sub(r"\1\1", text.lower())
        return re.sub(r"\s+", " ", text).strip()

    def tokenize(self, text: str) -> list[str]:
        """Clean one tweet and return its list of tokens."""
        return self.TOKEN.findall(self.clean(text))


class DisasterTweetDataset:
    """Loads the Kaggle files and exposes the shared train/val/test split.

    Kaggle's test.csv has no labels, so the labelled train.csv is split
    (stratified, fixed seed) into train / validation / test. The assignment is
    written to data/splits.csv so every team member evaluates on the same
    held-out tweets.
    """

    def __init__(self, cleaner: TweetCleaner | None = None):
        self.cleaner = cleaner or TweetCleaner()
        self.labelled = pd.read_csv(config.TRAIN_CSV)
        self.unlabelled = pd.read_csv(config.KAGGLE_TEST_CSV)
        # Tokenise once; every experiment reuses these token lists.
        for frame in (self.labelled, self.unlabelled):
            frame["tokens"] = frame["text"].map(self.cleaner.tokenize)
        self.labelled["split"] = self._load_or_create_split()

    def _load_or_create_split(self) -> pd.Series:
        """Reuse data/splits.csv if present, otherwise create and save it."""
        if config.SPLIT_CSV.exists():
            saved = pd.read_csv(config.SPLIT_CSV).set_index("id")["split"]
            return self.labelled["id"].map(saved)

        ids, y = self.labelled["id"], self.labelled["target"]
        held_out = config.VAL_SIZE + config.TEST_SIZE
        train_ids, rest_ids, _, rest_y = train_test_split(
            ids, y, test_size=held_out, stratify=y, random_state=config.SEED)
        val_ids, test_ids = train_test_split(
            rest_ids, test_size=config.TEST_SIZE / held_out,
            stratify=rest_y, random_state=config.SEED)

        split = pd.Series("train", index=self.labelled.index)
        split[ids.isin(val_ids)] = "val"
        split[ids.isin(test_ids)] = "test"
        pd.DataFrame({"id": ids, "split": split}).to_csv(config.SPLIT_CSV, index=False)
        return split

    def subset(self, name: str) -> pd.DataFrame:
        """Return the rows of one split: 'train', 'val' or 'test'."""
        return self.labelled[self.labelled["split"] == name].reset_index(drop=True)

    def unsupervised_corpus(self) -> list[list[str]]:
        """Token lists used to train Word2Vec. No labels are involved.

        The corpus is the training split plus Kaggle's unlabelled test.csv.
        The validation and test splits are deliberately left out, so the
        held-out tweets stay completely unseen until evaluation.
        """
        train_tokens = self.subset("train")["tokens"].tolist()
        return train_tokens + self.unlabelled["tokens"].tolist()


class Vocabulary:
    """Maps tokens to integer ids and tweets to fixed-length id sequences."""

    PAD, UNK = "<pad>", "<unk>"   # id 0 = padding, id 1 = out-of-vocabulary

    def __init__(self, corpus: list[list[str]], min_count: int = 1):
        counts = Counter(token for tokens in corpus for token in tokens)
        words = [w for w, c in counts.most_common() if c >= min_count]
        self.itos = [self.PAD, self.UNK] + words
        self.stoi = {word: i for i, word in enumerate(self.itos)}
        self.counts = counts

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, token_lists, max_len: int) -> np.ndarray:
        """Convert token lists to a (n_tweets, max_len) array, post-padded with 0."""
        out = np.zeros((len(token_lists), max_len), dtype="int32")
        for row, tokens in enumerate(token_lists):
            ids = [self.stoi.get(t, 1) for t in tokens[:max_len]]
            out[row, :len(ids)] = ids
        return out
