import rf_2025.src.utils as utils
import random
from collections import Counter

def sse(K, centroids, clusters): # Calcule la sum of squared errors
    J = 0
    for i in range(K):
        for c in clusters[i]:
            J += utils.distance_euclidienne(c,centroids[i])**2
    return J

def kmeans(vectors, K, max_iters=100):
    """Applique l'approche des k moyenne 

    Args:
        vectors (_type_): liste des vecteurs du training set
        K (_type_): nombre de clusters à partitionner
        max_iters (int, optional): nombre max d'itérations pour les calculs de centroids. Defaults to 100.

    Returns:
        _type_: _description_
    """
    random.shuffle(vectors)
    centroids = vectors[:K] #Initialise aléatoirement les centroids
    sse_old = float('inf')
    for _ in range(max_iters):
        clusters = [[] for _ in range(K)]
        preds = []
        for x in vectors:
            dists = [utils.distance_manhattan(x, c) for c in centroids] #Calcule la distance entre le vecteur et chaque centroids
            cluster_index = dists.index(min(dists))
            clusters[cluster_index].append(x)
            preds.append(cluster_index+1)
            
        new_centroids = []
        for cluster in clusters:
            if len(cluster) == 0:
                # cluster vide → on choisit un nouveau centre au hasard
                new_centroids.append(random.choice(vectors))
            else:
                centroid = [
                    sum(point[i] for point in cluster) / len(cluster) #Calclue la moyenne des vecteurs du cluster
                    for i in range(len(cluster[0]))
                ]
                new_centroids.append(centroid)
        centroids = new_centroids
        
        sse_new = sse(K, centroids, clusters)
        if abs(sse_old - sse_new) < 1e-5:   
            break
    return preds

def knn(vectors_train, classes_train, X_test, k): # KNN
    """Applique l'approche des k plus proche voisins aux vecteurs tests grace aux vecteurs train 
    et retourne les résultats

    Args:
        vectors_train (_type_): liste des vecteur du training set
        classes_train (_type_): liste des classes correspondant aux vecteurs du training set
        X_test (_type_): liste des vecteurs du test set
        k (_type_): nombre de voisins

    Returns:
        list: predictions
    """
    predictions = []
    for test_vector in X_test: # Pour he chaque vecteur de test
        distances = []
        for i in range(len(vectors_train)):
            dist = utils.distance_manhattan(test_vector, vectors_train[i])
            distances.append((dist, classes_train[i]))
        distances.sort(key=lambda x: x[0]) #trie en fonction de la distance
        k_nearest_labels = [label for _, label in distances[:k]]
        predicted_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
        predictions.append(predicted_label)
    return predictions

def vote_majoritaire_knn(methods_train, methods_test, classes_train, k=3):
    """
    methods_train : dict { "E34": X_train_E34, "GFD": X_train_GFD, ... }
    methods_test  : dict { "E34": X_test_E34,  "GFD": X_test_GFD,  ... }
    classes_train       : liste des labels du training set
    k             : nombre de voisins pour kNN
    
    Retourne : liste des classes prédites
    """
    
    # 1. On collecte les prédictions de toutes les méthodes
    predictions_per_method = []  # liste de listes
    
    for method in methods_train:
        vectors_train = methods_train[method]
        X_test  = methods_test[method]
        
        # appel à k-NN
        preds = knn(vectors_train, classes_train, X_test, k)
        predictions_per_method.append(preds)
    
    # 2. Vote majoritaire échantillon par échantillon
    n = len(predictions_per_method[0])  # nb de formes dans le test
    final_preds = []
    
    for i in range(n):
        votes = [preds[i] for preds in predictions_per_method]
        major = Counter(votes).most_common(1)[0][0]
        final_preds.append(major)
    
    return final_preds

    