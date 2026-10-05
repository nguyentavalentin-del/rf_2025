import os
def load_methods(): # Charge les données des fichier
    vector = []
    methods = []
    path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(path)
    for ext in ["E34", "F0", "F2", "GFD", "SA"]:
        met = []
        for f in sorted(os.listdir("data/" + ext)):
            if f.endswith("." + ext):
                with open(os.path.join("data/", ext, f), "r") as file:
                    for line in file:
                        vector.append(float(line))
                met.append((int(f[2]), vector))
                vector = []
        methods.append(met)
    E34, F0, F2, GFD, SA = tuple(methods)
    return E34, F0, F2, GFD, SA

def distance_euclidienne(v1, v2) : # Distance euclidienne entre les deux vecteurs v1 et v2
    r = 0
    for i in range(0, 16):
        r += abs(v1[i] - v2[i])**2
    return r**(1/2)

def distance_manhattan(v1, v2) : # Distance de Manhattan entre les deux vecteurs v1 et v2
    r = 0
    for i in range(0, 16):
        r += abs(v1[i] - v2[i])
    return r
 
def matrice_confusion(y_true, y_pred): # Construit la matrice de confusion
    classes = sorted(set(y_true) | set(y_pred))
    n = len(classes)

    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for yt, yp in zip(y_true, y_pred):
        i = classes.index(yt)
        j = classes.index(yp)  
        matrix[i][j] += 1

    return matrix, classes

def taux_reco(matrix): # Calcule le taux de reconnaisance de la matrice "matrix"
    total = 0
    correct = 0
    for i in range(len(matrix[0])):
        total += sum(matrix[i])
        correct += matrix[i][i]
    return correct/total

def courbe_pre_rap(result, K = 10):# Calcule la courbe précision/rappel
    courbe = []
    N = sum(result)
    pertinants = 0
    for i in range(K):
        pertinants += result[i]
        courbe.append((pertinants/(i+1), pertinants/N)) # (précision, rappel)
    return courbe

def f_mesure(matrix, K):
    """Calcule la f_mesure

    Args:
        matrix (_type_): matrice de confusion
        K (_type_): nombre de classes
    """
    p=0
    r=0
    for i in range(len(matrix)):
        vp = 0
        fp = 0
        fn = 0
        for j in range(len(matrix)):
            for k in range(len(matrix[j])):
                if i==j and j==k:
                    vp += matrix[j][k]
                elif i==j and j!=k:
                    fn += matrix[j][k]
                elif i!=j and i==k:
                    fp += matrix[j][k]
        if vp > 0 :
            p += vp/(vp+fp)
            r += vp/(vp+fn)
    p = p/K
    r = r/K
    return 2 * ((p * r)/(p + r))
            
                