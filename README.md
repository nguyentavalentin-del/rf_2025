# RF_2025

Projet de reconnaissance de formes à partir des descripteurs stockés dans `src/data`.
Les cinq descripteurs (`E34`, `F0`, `F2`, `GFD` et `SA`) sont évalués séparément
et combinés par vote majoritaire.

## Installation et exécution

Python 3.9 ou une version ultérieure est requis.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python3 src/main.py
```

L'évaluation utilise une validation leave-one-out et affiche la matrice de
confusion, le taux de reconnaissance et la F-mesure macro. Le paramètre `k`
est défini dans `src/main.py`.

Les algorithmes de classification et de clustering sont dans
[`src/classifiers.py`](src/classifiers.py). Les fonctions de chargement,
distances et métriques sont dans [`src/utils.py`](src/utils.py).

Pour lancer les tests :

```bash
python3 -m unittest discover -s tests
```
