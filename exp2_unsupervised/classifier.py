"""The downstream supervised model: a BiLSTM tweet classifier.

The architecture is identical for every variant of the experiment. The only
things that change are how the Embedding layer is initialised (random vs.
Word2Vec) and whether it is frozen, so any difference in performance is
caused by the unsupervised pre-training alone.
"""
import keras
import numpy as np
from keras import layers

from common import config


class BiLSTMClassifier:
    """Embedding -> BiLSTM -> Dense binary classifier."""

    def __init__(self, vocab_size: int, embedding_dim: int,
                 cfg: config.ClassifierConfig,
                 embedding_matrix: np.ndarray | None = None,
                 trainable_embeddings: bool = True, seed: int = config.SEED):
        """
        embedding_matrix=None      -> random initialisation (baseline)
        embedding_matrix=<W2V>     -> transfer learning from Word2Vec
        trainable_embeddings=False -> frozen vectors (pure feature extraction)
        trainable_embeddings=True  -> vectors are fine-tuned on the labels
        """
        self.cfg = cfg
        keras.utils.set_random_seed(seed)   # same weight init / shuffling per seed
        self.model = self._build(vocab_size, embedding_dim, embedding_matrix,
                                 trainable_embeddings)

    def _build(self, vocab_size, embedding_dim, embedding_matrix, trainable):
        if embedding_matrix is None:
            initializer = "uniform"
        else:
            initializer = keras.initializers.Constant(embedding_matrix)

        model = keras.Sequential([
            layers.Input(shape=(self.cfg.max_len,), dtype="int32"),
            # mask_zero=True makes the LSTM ignore padding positions.
            layers.Embedding(vocab_size, embedding_dim, mask_zero=True,
                             embeddings_initializer=initializer,
                             trainable=trainable, name="embedding"),
            layers.SpatialDropout1D(self.cfg.embedding_dropout),
            layers.Bidirectional(layers.LSTM(self.cfg.lstm_units)),
            layers.Dropout(self.cfg.dropout),
            layers.Dense(self.cfg.dense_units, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer=keras.optimizers.Adam(self.cfg.learning_rate),
                      loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def fit(self, x_train, y_train, x_val, y_val) -> dict:
        """Train with early stopping; returns the Keras history dict."""
        stopper = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.cfg.patience,
            restore_best_weights=True)
        history = self.model.fit(
            x_train, y_train, validation_data=(x_val, y_val),
            epochs=self.cfg.max_epochs, batch_size=self.cfg.batch_size,
            callbacks=[stopper], verbose=0)
        return history.history

    def predict_proba(self, x) -> np.ndarray:
        """Probability that each tweet is a real disaster."""
        return self.model.predict(x, batch_size=256, verbose=0).ravel()
