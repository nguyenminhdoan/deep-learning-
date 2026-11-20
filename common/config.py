"""Central configuration shared by every experiment.

Keeping paths, the random seed and the split ratios in one place guarantees
that all three experiments (supervised, unsupervised, state-of-the-art) are
trained and evaluated on exactly the same data, so their numbers are comparable.
"""
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------- paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

TRAIN_CSV = DATA_DIR / "train.csv"      # labelled Kaggle tweets
KAGGLE_TEST_CSV = DATA_DIR / "test.csv"  # unlabelled Kaggle tweets
SPLIT_CSV = DATA_DIR / "splits.csv"      # id -> train/val/test, shared by the team

# ------------------------------------------------------------ data split ----
SEED = 42          # seed of the shared train/val/test split
VAL_SIZE = 0.10    # fraction of train.csv used for validation
TEST_SIZE = 0.10   # fraction of train.csv held out for the final comparison

# ------------------------------------------------------- chart colours ----
# One fixed colour per entity, reused in every figure of the project.
COLOR_NEGATIVE = "#2a78d6"   # class 0: not a disaster
COLOR_POSITIVE = "#eb6834"   # class 1: real disaster
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
CLASS_NAMES = ["Not disaster", "Disaster"]


@dataclass
class Word2VecConfig:
    """Hyperparameters of the unsupervised Word2Vec model."""
    vector_size: int = 100   # dimensionality of the word vectors
    window: int = 5          # context words on each side of the target word
    min_count: int = 2       # ignore words rarer than this
    negative: int = 10       # negative samples per positive pair
    epochs: int = 30         # passes over the (small) tweet corpus
    seed: int = SEED


@dataclass
class ClassifierConfig:
    """Hyperparameters of the BiLSTM classifier (identical for every variant)."""
    max_len: int = 32            # tweets are padded / truncated to this many tokens
    lstm_units: int = 64
    dense_units: int = 32
    embedding_dropout: float = 0.3
    dropout: float = 0.5
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 20
    patience: int = 3            # early-stopping patience on validation loss
