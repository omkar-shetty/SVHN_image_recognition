"""
Shared data loading for the SVHN notebooks.

The functions return numpy arrays instead of tf.data.Dataset to
support easy inspection.
"""

import numpy as np
import tensorflow_datasets as tfds


def load_image_data(validation_split: float = 0.1, dataset: str = "svhn_cropped", seed: int = 6):
    """
    Loads datasets (default SVHN (cropped)) via tensorflow_datasets.

    Returns (x_train, y_train), (x_val, y_val), (x_test, y_test).
    Images are float32, shape (N, 32, 32, 3), normalized to [0, 1].
    Labels are int, 0-9.
    """
    train_ds, test_ds = tfds.load(
        dataset,
        split=["train", "test"],
        as_supervised=True,
    )

    x_train, y_train = _to_numpy(train_ds)
    x_test, y_test = _to_numpy(test_ds)

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    rng = np.random.default_rng(seed)
    n_val = int(len(x_train) * validation_split)
    val_idx = rng.choice(len(x_train), size=n_val, replace=False)
    train_idx = np.setdiff1d(np.arange(len(x_train)), val_idx)

    x_val, y_val = x_train[val_idx], y_train[val_idx]
    x_train, y_train = x_train[train_idx], y_train[train_idx]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def _to_numpy(dataset):
    images, labels = [], []
    for image, label in tfds.as_numpy(dataset):
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)
