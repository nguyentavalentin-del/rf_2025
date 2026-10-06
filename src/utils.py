from pathlib import Path

import numpy as np


DATA_TYPES = ("E34", "F0", "F2", "GFD", "SA")


def load_methods():
    """Load the descriptor files, preserving sample order across methods."""
    data_path = Path(__file__).resolve().parent / "data"
    methods = []
    expected_files = None

    for extension in DATA_TYPES:
        directory = data_path / extension
        files = sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == "." + extension.lower()
        )
        filenames = [path.stem.lower() for path in files]
        if expected_files is None:
            expected_files = filenames
        elif filenames != expected_files:
            raise ValueError("Descriptor files do not contain the same samples")

        descriptors = []
        for path in files:
            try:
                label = int(path.stem[1:3])
                vector = np.loadtxt(path, dtype=float, ndmin=1).tolist()
            except (OSError, ValueError) as error:
                raise ValueError("Unable to read descriptor file {}".format(path)) from error
            if not vector or not np.isfinite(vector).all():
                raise ValueError("Descriptor file {} is empty or non-finite".format(path))
            descriptors.append((label, vector))
        methods.append(descriptors)

    return tuple(methods)


def _validated_vectors(v1, v2):
    first = np.asarray(v1, dtype=float)
    second = np.asarray(v2, dtype=float)
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        raise ValueError("Vectors must be one-dimensional and have the same length")
    return first, second


def distance_euclidienne(v1, v2):
    """Return the Euclidean distance between two equal-length vectors."""
    first, second = _validated_vectors(v1, v2)
    return float(np.linalg.norm(first - second))


def distance_manhattan(v1, v2):
    """Return the Manhattan distance between two equal-length vectors."""
    first, second = _validated_vectors(v1, v2)
    return float(np.abs(first - second).sum())


def matrice_confusion(y_true, y_pred):
    """Build a confusion matrix, retaining sorted label order."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    classes = sorted(set(y_true) | set(y_pred))
    indices = {label: index for index, label in enumerate(classes)}
    matrix = [[0 for _ in classes] for _ in classes]
    for actual, predicted in zip(y_true, y_pred):
        matrix[indices[actual]][indices[predicted]] += 1
    return matrix, classes


def taux_reco(matrix):
    """Calculate accuracy from a confusion matrix."""
    if not matrix or not matrix[0]:
        raise ValueError("The confusion matrix must not be empty")
    total = sum(sum(row) for row in matrix)
    if total == 0:
        raise ValueError("The confusion matrix must contain observations")
    return sum(matrix[index][index] for index in range(len(matrix))) / total


def courbe_pre_rap(result, K=10):
    """Calculate precision/recall points from a ranked relevance list."""
    if K < 0 or K > len(result):
        raise ValueError("K must be between 0 and the number of results")
    relevant = sum(result)
    if relevant == 0 and K:
        raise ValueError("At least one result must be relevant")
    curve = []
    relevant_seen = 0
    for index in range(K):
        relevant_seen += result[index]
        curve.append((relevant_seen / (index + 1), relevant_seen / relevant))
    return curve


def f_mesure(matrix, K=None):
    """Calculate macro-averaged F1 from a confusion matrix."""
    if not matrix or len(matrix) != len(matrix[0]):
        raise ValueError("The confusion matrix must be non-empty and square")
    class_count = len(matrix) if K is None else K
    if class_count < 1:
        raise ValueError("K must be a positive number of classes")

    scores = []
    for index in range(len(matrix)):
        true_positive = matrix[index][index]
        false_positive = sum(matrix[row][index] for row in range(len(matrix))) - true_positive
        false_negative = sum(matrix[index]) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(scores) / class_count
