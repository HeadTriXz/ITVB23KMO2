import numpy as np
import tensorflow as tf

from tensorflow import keras


def load_model() -> keras.Model:
    """
    Loads the model from part two.

    :return: The loaded model.
    """
    return keras.models.load_model("../models/model.keras")


# OPGAVE 2a
def conf_matrix(labels: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Returns the confusion matrix for the given labels and predictions.

    :param labels: The true labels.
    :param pred: The predicted labels.
    :return: The confusion matrix.
    """
    return tf.math.confusion_matrix(labels, pred)


# OPGAVE 2b
def conf_els(conf: np.ndarray, labels: np.ndarray) -> list[tuple[str, int, int, int, int]]:
    """
    Returns the true positives, false positives, false negatives, and true negatives for the given confusion matrix.

    :param conf: The confusion matrix.
    :param labels: The labels.
    :return: A list of tuples containing the label, true positives, false positives, false negatives, and true negatives.
    """
    tp = np.diagonal(conf)
    fp = np.sum(conf, axis=0) - tp
    fn = np.sum(conf, axis=1) - tp
    tn = np.sum(conf) - tp - fp - fn

    return list(zip(labels, tp, fp, fn, tn))


# OPGAVE 2c
def conf_data(metrics: list[tuple[str, int, int, int, int]]) -> dict[str, float]:
    """
    Calculate the evaluation metrics based on the given metrics.

    :param metrics: The metrics from exercise 2b.
    :return: A dictionary containing the evaluation metrics.
    """
    tp = sum([x[1] for x in metrics])
    fp = sum([x[2] for x in metrics])
    fn = sum([x[3] for x in metrics])
    tn = sum([x[4] for x in metrics])

    # Calculate the metrics
    tpr = tp / (tp + fn)
    ppv = tp / (tp + fp)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)

    return {"tpr": tpr, "ppv": ppv, "tnr": tnr, "fpr": fpr}
