import rf_2025.src.utils as utils
import rf_2025.src.classifiers as classifiers
import time

def evaluate_methods(methods, k, nb_classes):
    """
    Évalue kNN sur toutes les méthodes + affiche résultats.
    methods : dict { "E34": [(label, vector), ...], "F0": [...], ... }
    k       : nombre de voisins pour kNN interne
    nb_classes : nombre total de classes
    
    Retour : Affiche la matrice de confusion, le taux de reco et la f-mesure pour chaque méthode.
    """
    for method_name in methods:
        print(f"\n--- Méthode : {method_name} ---")
        preds = []
        classes = [c for c, _ in methods[method_name]]
        vectors = [v for _, v in methods[method_name]]
        number_of_data = len(classes)
        for i in range(number_of_data):
            test_vector = [vectors[i]]
            train_vectors = vectors[:i] + vectors[i+1:]
            y_train       = classes[:i] + classes[i+1:]
            pred = classifiers.knn(train_vectors, y_train, test_vector, k)
            preds.extend(pred)
        matrix, _ = utils.matrice_confusion(classes, preds)
        print(matrix)
        print("Taux de reconnaissance :", utils.taux_reco(matrix))
        print("F-mesure :", utils.f_mesure(matrix, nb_classes))

def evaluate_vote_majoritaire(methods, k, nb_classes):
    """
    Leave-One-Out appliqué au vote majoritaire en utilisant knn.
    
    methods : dict { "E34": [(label, vector), ...], "F0": [...], ... }
    k       : nombre de voisins pour kNN interne
    nb_classes : nombre total de classes
    
    Retour : Affiche la matrice de confusion et le taux de reco et la f-mesure.
    """

    # Toutes les méthodes doivent avoir même nombre d'échantillons
    method_names = list(methods.keys())
    n = len(methods[method_names[0]])  # nombre total de points

    # Vecteurs y_true (identiques pour toutes les méthodes)
    y_true = [methods[method_names[0]][i][0] for i in range(n)]

    preds = []

    # Leave-One-Out
    for i in range(n):

        # Construire les training/test sets pour chaque méthode
        methods_train = {}
        methods_test  = {}

        for method in method_names:
            data = methods[method]
            classes = [c for c, _ in data]
            vectors = [v for _, v in data]

            # Leave-One-Out
            train_vectors = vectors[:i] + vectors[i+1:]
            train_labels  = classes[:i] + classes[i+1:]
            test_vector   = [vectors[i]]

            methods_train[method] = train_vectors
            methods_test[method]  = test_vector

        # Appel au vote majoritaire
        pred = classifiers.vote_majoritaire_knn(methods_train, methods_test, train_labels, k)[0]
        preds.append(pred)

    # Évaluation finale
    matrix, _ = utils.matrice_confusion(y_true, preds)
    print(matrix)
    print("Taux de reconnaissance :", utils.taux_reco(matrix))
    print("F-mesure :", utils.f_mesure(matrix, nb_classes))

if __name__ == "__main__":
    start = time.time()
    # Chargement des méthodes
    E34, F0, F2, GFD, SA = utils.load_methods()
    all_methods = {
        "E34": E34,
        "F0":  F0,
        "F2":  F2,
        "GFD": GFD,
        "SA":  SA
    }

    K = 1        # k du k-NN
    NB_CLASSES = 9

    print("===== Évaluation sur les 9 classes avec knn =====")

    # Préparation des données

    # Évaluation individuelle
    evaluate_methods(all_methods, K, NB_CLASSES)
    
    # Vote majoritaire
    print("\n===== Vote majoritaire =====")
    evaluate_vote_majoritaire(all_methods, K, NB_CLASSES)

    
    print("\n====================================")
    print("===== Tests sur les 5 premières classes avec knn =====")

    # Subsets 5 classes (55 échantillons)
    all_methods_5 = {name: data[:55] for name, data in all_methods.items()}
    NB_CLASSES_5 = 5


    # Évaluation individuelle
    evaluate_methods(all_methods_5, K, NB_CLASSES_5)
    
    # Vote majoritaire
    print("\n===== Vote majoritaire =====")
    evaluate_vote_majoritaire(all_methods_5, K, NB_CLASSES_5)
    print("\nTemps écoulé :", time.time() - start, "seconds")