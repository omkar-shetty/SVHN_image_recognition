"""
Shared plotting helpers for the SVHN notebooks.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curves(history, title: str = ""):
    """Plots loss and accuracy over epochs from a Keras History object."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title(f"{title} Loss".strip())
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title(f"{title} Accuracy".strip())
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, title: str = ""):
    """Heatmap of a confusion matrix, labels 0-9 on both axes."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=range(10), yticklabels=range(10))
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def sample_prediction_grid(x_test, y_true, y_pred, n_samples: int = 12, seed: int = 6, title: str = ""):
    """Grid of test images with predicted vs. actual labels.

    Correct predictions shown in green, incorrect in red. Uses a fixed
    seed so the same images get shown across different calls.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(x_test), size=n_samples, replace=False)

    n_cols = 4
    n_rows = int(np.ceil(n_samples / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.5))
    axes = axes.flatten()

    for ax, idx in zip(axes, indices):
        ax.imshow(x_test[idx])
        correct = y_true[idx] == y_pred[idx]
        color = "green" if correct else "red"
        ax.set_title(f"true={y_true[idx]} pred={y_pred[idx]}", color=color, fontsize=10)
        ax.axis("off")

    for ax in axes[len(indices):]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    return fig