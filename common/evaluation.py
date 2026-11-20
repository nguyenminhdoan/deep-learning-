"""Evaluation helpers shared by all experiments: metrics and standard figures."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # render straight to files; no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from . import config

METRICS = ["accuracy", "precision", "recall", "f1"]


def apply_plot_style() -> None:
    """One consistent look for every figure in the project."""
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb", "savefig.dpi": 150,
        "savefig.bbox": "tight", "font.size": 10,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
        "axes.titlecolor": "#0b0b0b", "axes.titleweight": "bold",
        "axes.titlesize": 11, "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
        "xtick.color": "#898781", "ytick.color": "#898781",
        "xtick.labelcolor": "#52514e", "ytick.labelcolor": "#52514e",
        "lines.linewidth": 2, "legend.frameon": False,
    })


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Accuracy / precision / recall / F1 for the positive ('disaster') class."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def plot_confusion_matrix(y_true, y_prob, title: str, path: Path) -> None:
    """Save a 2x2 confusion matrix with counts and row percentages."""
    cm = confusion_matrix(y_true, (np.asarray(y_prob) >= 0.5).astype(int))
    share = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.0%}", ha="center", va="center",
                    color="white" if share[i, j] > 0.5 else "#0b0b0b")
    ax.set_xticks([0, 1], config.CLASS_NAMES)
    ax.set_yticks([0, 1], config.CLASS_NAMES, rotation=90, va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)


def plot_training_curves(histories: dict, title: str, path: Path) -> None:
    """Save train/validation accuracy and loss curves.

    `histories` maps a model name to its Keras history dict. Solid lines are
    validation, dashed lines are training; each model keeps one colour.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, label in zip(axes, ["accuracy", "loss"], ["Accuracy", "Loss"]):
        for color, (name, hist) in zip(config.SERIES_COLORS, histories.items()):
            epochs = range(1, len(hist[key]) + 1)
            ax.plot(epochs, hist[key], color=color, linestyle="--", linewidth=1.2,
                    alpha=0.7)
            ax.plot(epochs, hist["val_" + key], color=color, label=name)
        ax.set_title(f"{label} per epoch")
        ax.set_xlabel("Epoch")
        ax.xaxis.get_major_locator().set_params(integer=True)
    axes[0].legend(title="solid = validation, dashed = training", fontsize=8,
                   title_fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
