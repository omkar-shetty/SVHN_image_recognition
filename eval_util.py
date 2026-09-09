"""
Shared evaluation helpers for the SVHN notebooks.

Multiple small and independent functions to calculate performance metrics,
where each function takes y_true/y_pred arrays, not a Keras model.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def get_predicted_labels(model, x, batch_size: int = 256) -> np.ndarray:
    """Runs model.predict() and collapses softmax output to class labels."""
    probs = model.predict(x, batch_size=batch_size, verbose=0)
    return np.argmax(probs, axis=1)


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy computed separately per digit class (0-9)."""
    accuracies = {}
    for label in sorted(np.unique(y_true)):
        mask = y_true == label
        accuracies[int(label)] = float(np.mean(y_pred[mask] == label))
    return accuracies


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Raw confusion matrix, rows = true label, columns = predicted label."""
    return confusion_matrix(y_true, y_pred, labels=list(range(10)))


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Precision/recall/F1 per class, sklearn's standard text report."""
    return classification_report(y_true, y_pred, labels=list(range(10)), zero_division=0)
