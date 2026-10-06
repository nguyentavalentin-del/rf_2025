from collections import Counter
import time

if __package__:
    from . import classifiers, utils
else:
    import classifiers
    import utils


def _evaluate_predictions(classes, predictions, nb_classes):
    matrix, labels = utils.matrice_confusion(classes, predictions)
    print("Classes :", labels)
    print("Matrice de confusion :")
    for row in matrix:
        print(row)
    print("Taux de reconnaissance :", utils.taux_reco(matrix))
    print("F-mesure macro :", utils.f_mesure(matrix, nb_classes))


def evaluate_methods(methods, k, nb_classes):
    """Evaluate each descriptor independently with leave-one-out validation."""
    for method_name, data in methods.items():
        print("\n--- Méthode : {} ---".format(method_name))
        classes = [label for label, _ in data]
        vectors = [vector for _, vector in data]
        predictions = classifiers.knn_leave_one_out(vectors, classes, k)
        _evaluate_predictions(classes, predictions, nb_classes)


def evaluate_vote_majoritaire(methods, k, nb_classes):
    """Evaluate a descriptor majority vote with leave-one-out validation."""
    if not methods:
        raise ValueError("At least one descriptor method is required")

    method_names = list(methods)
    first_data = methods[method_names[0]]
    classes = [label for label, _ in first_data]
    predictions_per_method = []

    for method_name, data in methods.items():
        method_classes = [label for label, _ in data]
        if method_classes != classes or len(data) != len(first_data):
            raise ValueError("All descriptors must contain the same samples in the same order")
        vectors = [vector for _, vector in data]
        predictions_per_method.append(
            classifiers.knn_leave_one_out(vectors, classes, k)
        )

    predictions = [
        Counter(method_predictions[index] for method_predictions in predictions_per_method)
        .most_common(1)[0][0]
        for index in range(len(classes))
    ]
    _evaluate_predictions(classes, predictions, nb_classes)


def main():
    start = time.time()
    loaded_methods = utils.load_methods()
    all_methods = dict(zip(utils.DATA_TYPES, loaded_methods))

    k = 3
    nb_classes = 9
    print("===== Évaluation leave-one-out sur les 9 classes avec k-NN =====")
    evaluate_methods(all_methods, k, nb_classes)

    print("\n===== Vote majoritaire =====")
    evaluate_vote_majoritaire(all_methods, k, nb_classes)

    print("\n===== Évaluation sur les 5 premières classes =====")
    first_five_classes = {
        name: [sample for sample in data if sample[0] <= 5]
        for name, data in all_methods.items()
    }
    evaluate_methods(first_five_classes, k, 5)

    print("\n===== Vote majoritaire sur les 5 premières classes =====")
    evaluate_vote_majoritaire(first_five_classes, k, 5)
    print("\nTemps écoulé :", round(time.time() - start, 2), "secondes")


if __name__ == "__main__":
    main()
