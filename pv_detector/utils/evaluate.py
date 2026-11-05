import numpy as np


def evaluate_FPR(true_labels, detect_results):
    """Return the false-positive rate for binary detection results."""
    true_labels = np.asarray(true_labels)
    detect_results = np.asarray(detect_results)
    total_negatives = np.sum(true_labels == 0)
    false_positives = np.sum((true_labels == 0) & (detect_results == 1))
    fpr = false_positives / total_negatives if total_negatives > 0 else 0.0
    return round(fpr, 4)


def evaluate_FNR(true_labels, detect_results):
    """Return the false-negative rate for binary detection results."""
    true_labels = np.asarray(true_labels)
    detect_results = np.asarray(detect_results)
    total_positives = np.sum(true_labels == 1)
    false_negatives = np.sum((true_labels == 1) & (detect_results == 0))
    fnr = false_negatives / total_positives if total_positives > 0 else 0.0
    return round(fnr, 4)
