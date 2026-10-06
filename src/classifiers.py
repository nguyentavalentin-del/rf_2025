from collections import Counter
from numbers import Integral

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _as_feature_matrix(vectors, name):
    matrix = np.asarray(vectors, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("{} must be a non-empty 2D feature matrix".format(name))
    if not np.isfinite(matrix).all():
        raise ValueError("{} must contain only finite values".format(name))
    return matrix


def _knn_pipeline(k):
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=k, metric="manhattan"),
    )


def sse(K, centroids, clusters):
    """Calculate the sum of squared Euclidean distances within clusters."""
    if K < 0 or K > len(centroids) or K > len(clusters):
        raise ValueError("K must not exceed the number of centroids or clusters")
    total = 0.0
    for index in range(K):
        if len(clusters[index]):
            points = _as_feature_matrix(clusters[index], "cluster")
            center = np.asarray(centroids[index], dtype=float)
            if points.shape[1] != center.size:
                raise ValueError("Points and centroid must have the same dimension")
            total += float(np.square(points - center).sum())
    return total


def kmeans(vectors, K, max_iters=100, random_state=0):
    """Cluster feature vectors and return one-based cluster assignments."""
    matrix = _as_feature_matrix(vectors, "vectors")
    if not isinstance(K, Integral) or K < 1 or K > len(matrix):
        raise ValueError("K must be an integer between 1 and the number of vectors")
    if not isinstance(max_iters, Integral) or max_iters < 1:
        raise ValueError("max_iters must be a positive integer")

    model = KMeans(
        n_clusters=int(K),
        max_iter=int(max_iters),
        n_init=10,
        random_state=random_state,
    )
    return (model.fit_predict(matrix) + 1).tolist()


def knn(vectors_train, classes_train, X_test, k):
    """Classify test vectors with a scaled Manhattan k-nearest-neighbors model."""
    train = _as_feature_matrix(vectors_train, "vectors_train")
    test = np.asarray(X_test, dtype=float)
    labels = np.asarray(classes_train)

    if test.size == 0:
        return []
    if test.ndim == 1:
        test = test.reshape(1, -1)
    if test.ndim != 2 or test.shape[1] != train.shape[1]:
        raise ValueError("Training and test vectors must have the same feature dimension")
    if not np.isfinite(test).all():
        raise ValueError("X_test must contain only finite values")
    if len(labels) != len(train):
        raise ValueError("classes_train must contain one label per training vector")
    if not isinstance(k, Integral) or k < 1 or k > len(train):
        raise ValueError("k must be an integer between 1 and the training set size")
    if test.shape[0] == 0:
        return []

    model = _knn_pipeline(int(k))
    model.fit(train, labels)
    return model.predict(test).tolist()


def knn_leave_one_out(vectors, classes, k=3):
    """Return leakage-free leave-one-out predictions for one descriptor."""
    matrix = _as_feature_matrix(vectors, "vectors")
    labels = np.asarray(classes)
    if len(labels) != len(matrix):
        raise ValueError("classes must contain one label per vector")
    if len(matrix) < 2:
        raise ValueError("Leave-one-out evaluation requires at least two samples")
    if not isinstance(k, Integral) or k < 1 or k >= len(matrix):
        raise ValueError("k must be an integer between 1 and the training fold size")

    return cross_val_predict(
        _knn_pipeline(int(k)),
        matrix,
        labels,
        cv=LeaveOneOut(),
        n_jobs=1,
    ).tolist()


def vote_majoritaire_knn(methods_train, methods_test, classes_train, k=3):
    """
    Combine k-NN predictions from descriptor-specific feature matrices.

    The first descriptor wins ties, matching insertion order.
    """
    if not methods_train:
        return []
    if methods_train.keys() != methods_test.keys():
        raise ValueError("Training and test descriptors must have the same keys")

    predictions_per_method = []
    for method, vectors_train in methods_train.items():
        if len(vectors_train) != len(classes_train):
            raise ValueError("Each descriptor needs one training vector per label")
        predictions_per_method.append(
            knn(vectors_train, classes_train, methods_test[method], k)
        )

    prediction_count = len(predictions_per_method[0])
    if any(len(predictions) != prediction_count for predictions in predictions_per_method):
        raise ValueError("All descriptors must contain the same number of test vectors")

    return [
        Counter(predictions[index] for predictions in predictions_per_method)
        .most_common(1)[0][0]
        for index in range(prediction_count)
    ]
