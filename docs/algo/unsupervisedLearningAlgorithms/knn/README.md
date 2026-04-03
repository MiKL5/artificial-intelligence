# **L'algorithme des KNN**<a href="../../../"><img src="../../../../assets/images/atomicAi.png" alt="L'intelligence artificielle" align="right" height="64px"></a>

>>>>> *« Tell me who your neighbors are, and I'll tell you who you are. »*  
>>>>> — Principe fondateur de l'apprentissage par analogie
---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Fondements mathématiques](#2-fondements-mathématiques)
   * 2.1 [Métriques de distance](#21-métriques-de-distance)
   * 2.2 [Règle de décision](#22-règle-de-décision)
   * 2.3 [Analyse de la convergence — Théorème de Cover & Hart](#23-analyse-de-la-convergence--théorème-de-cover--hart)
3. [Complexité algorithmique](#3-complexité-algorithmique)
4. [Le choix de *k* — Biais-Variance](#4-le-choix-de-k--biais-variance)
5. [Problématiques dimensionnelles — La Malédiction de la Dimensionnalité](#5-problématiques-dimensionnelles--la-malédiction-de-la-dimensionnalité)
6. [Structures de données et optimisation](#6-structures-de-données-et-optimisation)
   * 6.1 [KD-Tree](#61-kd-tree)
   * 6.2 [Ball-Tree](#62-ball-tree)
   * 6.3 [Locality-Sensitive Hashing (LSH)](#63-locality-sensitive-hashing-lsh)
7. [Variantes et extensions](#7-variantes-et-extensions)
8. [Implémentation Python](#8-implémentation-python)
9. [Hyperparamètres & Tuning](#9-hyperparamètres--tuning)
10. [Forces, limites et cas d'usage](#10-forces-limites-et-cas-dusage)
11. [Comparaison avec d'autres algorithmes](#11-comparaison-avec-dautres-algorithmes)
12. [Références scientifiques](#12-références-scientifiques)

___

## 1. Vue d'ensemble

L'algorithme des **K plus proches voisins** (*K-Nearest Neighbors*, KNN) est un estimateur **non paramétrique**, **à apprentissage paresseux** (*lazy learning*), proposé dans sa forme initiale par Fix & Hodges en 1951. Il appartient à la famille des méthodes d'**apprentissage par instance** (*instance-based learning*) : aucun modèle explicite n'est construit lors de la phase d'entraînement ; la totalité du dataset est mémorisée et la généralisation s'opère à la prédiction.

Il supporte deux régimes d'apprentissage supervisé :

Régime | Sortie | Stratégie d'agrégation
---|---|---
**Classification** | Classe discrète | Vote majoritaire (éventuellement pondéré)
**Régression** | Valeur continue | Moyenne (éventuellement pondérée)

```
Données d'entraînement       Nouveau point x*
       ●  ●                        ?
    ●     ●     ◆              ◆
       ●     ●          →    ◆   ← k=3 voisins → classe ◆
    ●           ●
```

___

## 2. Fondements mathématiques

### 2.1 Métriques de distance

Le cœur de KNN réside dans le choix d'une **métrique de similarité** $d : \mathcal{X} \times \mathcal{X} \to \mathbb{R}_{\geq 0}$. Formellement, $d$ doit satisfaire les axiomes d'une **métrique** :

1. **Séparation** : $d(x, y) = 0 \iff x = y$
2. **Symétrie** : $d(x, y) = d(y, x)$
3. **Inégalité triangulaire** : $d(x, z) \leq d(x, y) + d(y, z)$

Les métriques les plus utilisées appartiennent à la famille **Minkowski** :

$$d_p(x, y) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{1/p}, \quad p \geq 1$$

Nom | Paramètre $p$ | Formule explicite | Remarques
---|---|---|---
**Manhattan** (L1) | $p = 1$ | $\sum_i \|x_i - y_i\|$ | Robuste aux outliers
**Euclidienne** (L2) | $p = 2$ | $\sqrt{\sum_i (x_i - y_i)^2}$ | La plus courante
**Chebyshev** (L∞) | $p \to \infty$ | $\max_i \|x_i - y_i\|$ | Sensible à la dimension dominante

**Autres métriques notables :**

- **Mahalanobis** : $d_M(x, y) = \sqrt{(x-y)^\top \Sigma^{-1} (x-y)}$ — neutralise les corrélations et les disparités d'échelle.
- **Cosinus** : $d_\cos(x, y) = 1 - \frac{x \cdot y}{\|x\|\|y\|}$ — invariante à la norme, privilégiée en NLP.
- **Hamming** : adapée aux données binaires ou catégorielles.

> ⚠️ **Invariance d'échelle** : KNN est sensible aux unités. Une **normalisation** (Min-Max, Z-score) ou une **standardisation** préalable est **obligatoire** lorsque les features sont hétérogènes.

___

### 2.2 Règle de décision

Soit un ensemble d'entraînement $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$ et un point de requête $x^*$.

**Étape 1 — Identification des k voisins :**

$$\mathcal{N}_k(x^*) = \underset{S \subseteq \mathcal{D},\, |S|=k}{\arg\min} \sum_{(x_i, y_i) \in S} d(x^*, x_i)$$

**Étape 2a — Classification (vote majoritaire non pondéré) :**

$$\hat{y} = \underset{c \in \mathcal{C}}{\arg\max} \sum_{(x_i, y_i) \in \mathcal{N}_k(x^*)} \mathbf{1}[y_i = c]$$

**Étape 2b — Classification pondérée par la distance :**

$$\hat{y} = \underset{c \in \mathcal{C}}{\arg\max} \sum_{(x_i, y_i) \in \mathcal{N}_k(x^*)} w_i \cdot \mathbf{1}[y_i = c], \quad w_i = \frac{1}{d(x^*, x_i)^2 + \varepsilon}$$

**Étape 2c — Régression :**

$$\hat{y} = \frac{1}{k} \sum_{(x_i, y_i) \in \mathcal{N}_k(x^*)} y_i \quad \text{(non pondéré)}$$

$$\hat{y} = \frac{\sum_{i} w_i y_i}{\sum_{i} w_i} \quad \text{(pondéré)}$$

___

### 2.3 Analyse de la convergence — Théorème de Cover & Hart

> **Théorème (Cover & Hart, 1967)** : Soit $R^*$ le taux d'erreur de Bayes (optimal). Le taux d'erreur asymptotique du classifieur 1-NN $R$ vérifie :
>
> $$R^* \leq R \leq R^* \left(2 - \frac{c \cdot R^*}{c - 1}\right) \leq 2R^*$$
>
> où $c$ est le nombre de classes.

Ce résultat majeur garantit que le 1-NN ne peut jamais avoir un taux d'erreur supérieur au **double de l'erreur de Bayes**, et ce **sans aucune hypothèse paramétrique** sur la distribution sous-jacente. Lorsque $N \to \infty$ et $k \to \infty$ avec $k/N \to 0$, le KNN est un **estimateur de Bayes-consistant**.

___

## 3. Complexité algorithmique

Phase | Approche naïve | Avec KD-Tree | Avec LSH
---|---|---|---
**Entraînement** | $O(1)$ | $O(N \log N)$ | $O(N)$
**Prédiction (1 requête)** | $O(Nd)$ | $O(d \log N)$ *(low-dim)* | $O(d)$ approx.
**Prédiction (Q requêtes)** | $O(QNd)$ | $O(Qd \log N)$ | $O(Qd)$ approx.
**Stockage** | $O(Nd)$ | $O(Nd)$ | $O(N)$

Avec $N$ = taille du dataset, $d$ = nombre de dimensions, $Q$ = nombre de requêtes.

> Le KD-Tree devient inefficace pour $d \gtrsim 20$ en raison de la malédiction de la dimensionnalité.

___

## 4. Le choix de *k* — Biais-Variance

Le paramètre $k$ gouverne le **compromis biais-variance** :

```
k faible (ex: k=1)                k élevé (ex: k=N)
   ↓                                    ↓
Faible biais                        Fort biais
Forte variance                      Faible variance
Sur-apprentissage                   Sous-apprentissage
Frontière de décision               Frontière de décision
très irrégulière                    très lisse
```

**Décomposition formelle** de l'erreur quadratique moyenne :

$$\text{MSE}(\hat{y}) = \underbrace{\text{Biais}^2(\hat{y})}_{\propto k} + \underbrace{\text{Var}(\hat{y})}_{\propto 1/k} + \sigma^2_\varepsilon$$

**Stratégies de sélection de *k* :**

* **Validation croisée k-fold** (recommandée) : balayage sur une grille $k \in \{1, 3, 5, \ldots, \sqrt{N}\}$
* **Règle empirique** : $k \approx \sqrt{N}$, avec préférence pour les valeurs impaires (évite les égalités en classification binaire)
* **Courbe d'erreur** : tracer l'erreur de validation en fonction de $k$ et sélectionner le coude

---

## 5. Problématiques dimensionnelles — La Malédiction de la Dimensionnalité

En grande dimension, **toutes les distances tendent à devenir équivalentes**. Formellement, pour des points uniformément distribués sur $[0,1]^d$ :

$$\frac{d_{\max} - d_{\min}}{d_{\min}} \xrightarrow{d \to \infty} 0$$

Ce phénomène, décrit par Bellman (1961) et formalisé par Beyer *et al.* (1999), rend le concept de « voisin » statistiquement vide en haute dimension.

**Conséquence pratique :** la fraction du volume de l'hypercube nécessaire pour capturer une fraction $r$ des données croît exponentiellement :

$$\ell(r, d) = r^{1/d} \xrightarrow{d \to \infty} 1$$

**Remèdes :**

Technique | Description
---|---
**PCA / t-SNE / UMAP** | Réduction de dimensionnalité avant KNN
**Sélection de features** | Élimination des features non informatives
**Métriques adaptatives** | Distance de Mahalanobis, LMNN (*Large Margin Nearest Neighbor*)
**ANN (Approximate NN)** | LSH, HNSW, FAISS — sacrifier l'exactitude pour la scalabilité

---

## 6. Structures de données et optimisation

### 6.1 KD-Tree

Un **KD-Tree** est un arbre binaire partitionnant $\mathbb{R}^d$ en hyperplans alignés sur les axes. La construction alterne cycliquement sur les dimensions en choisissant la médiane comme pivot.

```
Espace 2D :              KD-Tree :
                              [médiane x]
  ──────────────          ┌──────┴──────┐
  |    |        |      [med y]        [med y]
  |    |  ●  ●  |      ┌──┴──┐      ┌──┴──┐
  |    |        |     ●     ●       ●     ●
  |────|────────|
  | ●  |   ●   |
  |    |        |
  ──────────────
```

* **Construction** : $O(Nd\log N)$
* **Requête exacte** : $O(\log N)$ en faible dimension, $O(N)$ au pire cas
* **Optimal pour** : $d \leq 20$

### 6.2 Ball-Tree

Partitionne les données en **hypersphères imbriquées** plutôt qu'en hyperplans. Meilleure performance que le KD-Tree pour les données non uniformes et les grandes dimensions modérées ($20 < d < 100$).

* L'**inégalité triangulaire** est exploitée pour élaguer des branches entières lors de la recherche.

### 6.3 Locality-Sensitive Hashing (LSH)

Famille de fonctions de hachage $h$ telles que :

$$P[h(x) = h(y)] = f(d(x, y))$$

où $f$ est une fonction décroissante de la distance. Les points proches ont une forte probabilité de partager le même bucket. Permet une recherche **approximative** en $O(1)$ par requête (amortie).

Implémentations modernes : **FAISS** (Meta AI), **ScaNN** (Google), **HNSW** (Malkov & Yashunin, 2018).

___

## 7. Variantes et extensions

Variante | Description | Usage typique
---|---|---
**KNN pondéré** | $w_i \propto 1/d_i^2$ | Améliore la précision aux frontières
**Radius-NN** | Voisins dans un rayon $r$ fixe | Densité variable
**LMNN** | Apprend la métrique pour maximiser la marge | Classification
**Parzen-Rosenblatt** | Estimation de densité par fenêtre de Parzen | Estimation de PDF
**Condensed KNN** | Réduit le dataset aux exemples frontières | Compression mémoire
**Edited KNN** | Supprime les exemples mal classés | Débruitage
**KNN pour anomalies** | Score = $\bar{d}$ aux k voisins | Détection d'anomalies
**KNORA** | KNN pour la sélection dynamique d'ensembles | *Ensemble learning*

___

## 8. Implémentation Python

### Installation

```bash
pip install scikit-learn numpy pandas matplotlib
```

### Classification

```python
import numpy                   as     np
from   sklearn.neighbors       import KNeighborsClassifier
from   sklearn.model_selection import cross_val_score, GridSearchCV
from   sklearn.preprocessing   import StandardScaler
from   sklearn.pipeline        import Pipeline

# Pipeline complet : normalisation + KNN
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(
        n_neighbors=5,
        metric='minkowski',
        p=2,             # L2 euclidienne
        weights='distance',
        algorithm='auto',  # sklearn choisit KD-Tree ou Ball-Tree
        n_jobs=-1
    ))
])

# Recherche d'hyperparamètres
param_grid = {
    'knn__n_neighbors': range(1, 31, 2),
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan', 'minkowski']
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=10,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"Meilleurs params : {grid_search.best_params_}")
print(f"Score CV        : {grid_search.best_score_:.4f}")
```

### Régression

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics   import mean_squared_error, r2_score

knn_reg = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor(
        n_neighbors=7,
        weights='distance',
        algorithm='ball_tree',
        leaf_size=30,
        n_jobs=-1
    ))
])

knn_reg.fit(X_train, y_train)
y_pred = knn_reg.predict(X_test)

print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R²   : {r2_score(y_test, y_pred):.4f}")
```

### Sélection optimale de k par courbe d'erreur

```python
import matplotlib.pyplot as plt

k_range = range(1, 51, 2)
cv_scores = []

for k in k_range:
    knn = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=k, n_jobs=-1))
    ])
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

optimal_k = k_range[np.argmax(cv_scores)]

plt.figure(figsize=(10, 5))
plt.plot(k_range, cv_scores, marker='o', linewidth=1.5, color='steelblue')
plt.axvline(optimal_k, color='red', linestyle='--', label=f'k optimal = {optimal_k}')
plt.xlabel('Valeur de k')
plt.ylabel('Accuracy (CV 10-fold)')
plt.title('Sélection de k par validation croisée')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

### KNN avec FAISS (grandes dimensions, haute performance)

```python
import faiss
import numpy as np

# Construction de l'index (distance L2)
d = X_train.shape[1]
index = faiss.IndexFlatL2(d)

# Ajout des vecteurs d'entraînement
X_train_f32 = X_train.astype(np.float32)
index.add(X_train_f32)

# Requête : trouver les k=5 plus proches voisins
k = 5
distances, indices = index.search(X_test.astype(np.float32), k)
# indices : (N_test, k) — indices dans X_train
# distances : (N_test, k) — distances L2 au carré
```

___

## 9. Hyperparamètres & Tuning

Hyperparamètre | Valeurs typiques | Impact
---|---|--
`n_neighbors` ($k$) | $[1, \sqrt{N}]$, impair | Biais-Variance
`weights` | `uniform`, `distance` | Robustesse aux outliers
`metric` | `euclidean`, `manhattan`, `mahalanobis` | Géométrie de l'espace
`algorithm` | `auto`, `kd_tree`, `ball_tree`, `brute` | Performance à la prédiction
`leaf_size` | 20–50 | Mémoire vs vitesse de la structure
`p` (Minkowski) | 1, 2 | L1 vs L2

**Recommandations pratiques :**

* Toujours **normaliser** les features avant KNN.
* Préférer `weights='distance'` en présence de classes déséquilibrées.
* Utiliser `algorithm='ball_tree'` pour $d > 15$ et `algorithm='kd_tree'` pour $d \leq 15$.
* Pour $N > 10^5$ ou $d > 50$, basculer vers **FAISS** ou **HNSW**.

___

## 10. Forces, limites et cas d'usage

### ✅ Forces

* **Aucune hypothèse paramétrique** sur la distribution des données
* **Apprentissage trivial** ($O(1)$ à l'entraînement)
* **Adaptabilité naturelle** aux distributions multimodales
* **Interprétabilité** : la prédiction est explicable par les voisins
* **Extensible** à des tâches non conventionnelles (recommandation, anomalies)

### ❌ Limites

* **Coût de prédiction élevé** pour de grands datasets sans indexation
* **Sensibilité à la dimensionnalité** (malédiction)
* **Sensibilité aux features non pertinentes** et aux échelles
* **Mémoire** : le dataset entier doit être accessible à l'inférence
* **Pas de modèle exportable** : pas de paramètres appris

### 🎯 Cas d'usage recommandés

Domaine | Application
---|---
Systèmes de recommandation | Collaborative filtering (utilisateurs similaires)
Finance | Détection d'anomalies (fraude)
Bioinformatique | Classification de gènes, similarité de protéines
Vision | Classification d'images (baseline), image retrieval
NLP | Recherche sémantique avec embeddings
Médecine | Diagnostic assisté par analogie clinique

___

## 11. Comparaison avec d'autres algorithmes

Critère | KNN | SVM | Random Forest | Réseau de neurones
---|---|---|---|---
Entraînement | $O(1)$ | $O(N^2)$–$O(N^3)$ | $O(N d \log N)$ | $O(\text{epochs} \times N)$
Prédiction | $O(Nd)$ naïf | $O(N_{\text{sv}} d)$ | $O(T \log N)$ | $O(L \times d)$
Interprétabilité | ✅ Haute | ⚠️ Moyenne | ⚠️ Moyenne | ❌ Faible
Grande dimension | ❌ | ✅ | ✅ | ✅
Données non linéaires | ✅ | ✅ (kernel) | ✅ | ✅
Faible volume de données | ✅ | ✅ | ⚠️ | ❌
Données manquantes | ❌ | ❌ | ✅ | ⚠️

___

## 12. Références<!-- scientifiques-->

| # | Référence
:-:|---
[1] | Fix, E. & Hodges, J. L. (1951). *Discriminatory analysis: nonparametric discrimination*. USAF School of Aviation Medicine.
[2] | Cover, T. & Hart, P. (1967). *Nearest neighbor pattern classification*. IEEE Transactions on Information Theory, 13(1), 21–27.
[3] | Bellman, R. (1961). *Adaptive Control Processes: A Guided Tour*. Princeton University Press.
[4] | Beyer, K. *et al.* (1999). *When is "nearest neighbor" meaningful?* ICDT.
[5] | Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
[6] | Weinberger, K. Q. & Saul, L. K. (2009). *Distance metric learning for large margin nearest neighbor classification*. JMLR, 10, 207–244.
[7] | Johnson, J. *et al.* (2019). *Billion-scale similarity search with GPUs* (FAISS). IEEE TPAMI.
[8] | Malkov, Y. & Yashunin, D. (2018). *Efficient and robust approximate nearest neighbor search using HNSW*. IEEE TPAMI, 42(4).
[9] | Bentley, J. L. (1975). *Multidimensional binary search trees used for associative searching*. CACM, 18(9), 509–517.
[10] | Zhang, M. L. & Zhou, Z. H. (2007). *ML-KNN: A lazy learning approach to multi-label learning*. Pattern Recognition, 40(7).
