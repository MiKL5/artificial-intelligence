# Apprentissage de Dictionnaire vs Apprentissage de Dictionnaire en Ligne<a href="../../"><img src="../../../assets/images/atomicAi.png" alt="L'intelligence artificielle" align="right" height="64px"></a>

## Sommaire

1. [Le problème fondamental](#1-le-problème-fondamental)
2. [Apprentissage de dictionnaire classique (batch)](#2-apprentissage-de-dictionnaire-classique-batch)
   - [Formulation mathématique](#21-formulation-mathématique)
   - [Étape 1 — Codage parcimonieux](#22-étape-1--codage-parcimonieux)
   - [Étape 2 — Mise à jour du dictionnaire](#23-étape-2--mise-à-jour-du-dictionnaire)
   - [Algorithme K-SVD](#24-algorithme-k-svd)
   - [Complexité et limites](#25-complexité-et-limites)
3. [Apprentissage de dictionnaire en ligne](#3-apprentissage-de-dictionnaire-en-ligne)
   - [Philosophie et motivation](#31-philosophie-et-motivation)
   - [Formulation stochastique](#32-formulation-stochastique)
   - [La fonction de substitution](#33-la-fonction-de-substitution)
   - [Mécanique de la mise à jour](#34-mécanique-de-la-mise-à-jour)
   - [Pseudo-code de l'algorithme de Mairal](#35-pseudo-code-de-lalgorithme-de-mairal)
   - [Garanties de convergence](#36-garanties-de-convergence)
4. [Comparaison synthétique](#4-comparaison-synthétique)
5. [Quand choisir l'un plutôt que l'autre ?](#5-quand-choisir-lun-plutôt-que-lautre-)
6. [Implémentations de référence](#6-implémentations-de-référence)
7. [Pour aller plus loin](#7-pour-aller-plus-loin)

___

## 1. Le problème fondamental

Imaginez que vous disposiez d'un corpus de signaux — images, séquences audio, relevés biologiques — et que vous souhaitiez en extraire une **base adaptée aux données** permettant de reconstruire chaque observation avec un nombre réduit de coefficients non nuls. Ce paradigme s'appelle la **représentation parcimonieuse** (*sparse representation*).

La question centrale est la suivante :

> Étant donné un jeu de vecteurs $\mathbf{x}_1, \ldots, \mathbf{x}_n \in \mathbb{R}^d$, comment construire automatiquement un ensemble de *prototypes* $\mathbf{d}_1, \ldots, \mathbf{d}_K \in \mathbb{R}^d$ (les **atomes**) tel que chaque observation soit approximativement reconstituée par une combinaison linéaire impliquant très peu de ces atomes ?

Répondre à cette interrogation constitue l'objet de **l'apprentissage de dictionnaire** (*dictionary learning*). La différence entre la variante classique et la variante en ligne tient à la **façon dont les données sont consommées** lors de l'optimisation.

___

## 2. Apprentissage de dictionnaire classique (batch)

### 2.1 Formulation mathématique

Soit $\mathbf{X} = [\mathbf{x}_1 \mid \cdots \mid \mathbf{x}_n] \in \mathbb{R}^{d \times n}$ la matrice d'observations. On cherche conjointement :

* un **dictionnaire** $\mathbf{D} \in \mathbb{R}^{d \times K}$, dont chaque colonne $\mathbf{d}_k$ est un atome à norme unitaire,
* une **matrice de codes** $\mathbf{A} = [\boldsymbol{\alpha}_1 \mid \cdots \mid \boldsymbol{\alpha}_n] \in \mathbb{R}^{K \times n}$, dont chaque colonne $\boldsymbol{\alpha}_i$ est parcimonieuse,

en minimisant le critère global :

$$\min_{\mathbf{D} \in \mathcal{C},\, \mathbf{A}} \quad \frac{1}{n}\sum_{i=1}^{n} \left( \frac{1}{2}\|\mathbf{x}_i - \mathbf{D}\boldsymbol{\alpha}_i\|_2^2 + \lambda\|\boldsymbol{\alpha}_i\|_1 \right)$$

où $\mathcal{C} = \{\mathbf{D} \in \mathbb{R}^{d \times K} : \|\mathbf{d}_k\|_2 \leq 1,\ \forall k\}$ est une contrainte convexe qui empêche les atomes de croître indéfiniment, et $\lambda > 0$ est le paramètre de régularisation contrôlant le degré de parcimonie.

Le terme $\|\boldsymbol{\alpha}_i\|_1$ est la **pénalité Lasso** (*Least Absolute Shrinkage and Selection Operator*), convexe et favorisant les solutions creuses.

Ce problème est **bi-convexe** : convexe en $\mathbf{A}$ pour $\mathbf{D}$ fixé, et convexe en $\mathbf{D}$ pour $\mathbf{A}$ fixé — mais *pas* convexe simultanément dans les deux variables. L'approche classique exploite cette structure par une **alternance de blocs**.

___

### 2.2 Étape 1 — Codage parcimonieux

Pour un dictionnaire $\mathbf{D}$ fixé, chaque problème d'encodage se résout indépendamment :

$$\boldsymbol{\alpha}_i^* = \arg\min_{\boldsymbol{\alpha}} \frac{1}{2}\|\mathbf{x}_i - \mathbf{D}\boldsymbol{\alpha}\|_2^2 + \lambda\|\boldsymbol{\alpha}\|_1$$

Il s'agit d'un **Lasso** standard. On dispose de deux familles de solveurs :

Famille | Exemples | Caractéristique
---|---|---
**Poursuite gloutonne** | OMP, MP | Ajoute les atomes un à un ; contrôle direct du nombre de non-zéros
**Relaxation convexe** | LARS, ISTA, FISTA | Minimise le Lasso ; contrôle via $\lambda$

La **poursuite orthogonale par correspondance** (OMP — *Orthogonal Matching Pursuit*) est particulièrement prisée en traitement du signal car elle garantit une solution en au plus $s$ itérations si l'on cible une parcimonie $s$.

___

### 2.3 Étape 2 — Mise à jour du dictionnaire

Pour des codes $\mathbf{A}$ figés, le problème en $\mathbf{D}$ s'écrit :

$$\min_{\mathbf{D} \in \mathcal{C}} \|\mathbf{X} - \mathbf{D}\mathbf{A}\|_F^2$$

Deux stratégies majeures existent :

**MOD** (*Method of Optimal Directions*, Engan et al., 1999) : résout analytiquement $\mathbf{D} = \mathbf{X}\mathbf{A}^\top(\mathbf{A}\mathbf{A}^\top)^{-1}$, puis projette chaque atome sur la boule unité. Simple, mais l'inversion de $\mathbf{A}\mathbf{A}^\top$ coûte $\mathcal{O}(K^3)$ et peut être mal conditionnée.

**K-SVD** (Aharon et al., 2006) : procède atome par atome — voir la section suivante.

___

### 2.4 Algorithme K-SVD

K-SVD est l'étalon-or de l'apprentissage de dictionnaire par lots. Son idée maîtresse : mettre à jour chaque atome $\mathbf{d}_k$ *conjointement* à ses codes associés, via une **décomposition en valeurs singulières tronquée**.

**Procédure pour l'atome $k$** :

1. Former la matrice de résidus $\mathbf{E}_k = \mathbf{X} - \sum_{j \neq k} \mathbf{d}_j \boldsymbol{\alpha}_j^\top$, qui isole la contribution de l'atome $k$.
2. Restreindre $\mathbf{E}_k$ aux colonnes $\Omega_k = \{i : \alpha_{k,i} \neq 0\}$ (observations utilisant effectivement cet atome) pour obtenir $\mathbf{E}_k^R$.
3. Calculer la SVD de rang 1 : $\mathbf{E}_k^R \approx \sigma_1 \mathbf{u}_1 \mathbf{v}_1^\top$.
4. Poser $\mathbf{d}_k \leftarrow \mathbf{u}_1$ et mettre à jour les codes : $\alpha_{k,\Omega_k} \leftarrow \sigma_1 \mathbf{v}_1$.

Cette manipulation préserve la parcimonie existante tout en réduisant l'erreur de reconstruction de manière optimale au sens des moindres carrés.

```
Algorithme K-SVD
────────────────
Initialiser D aléatoirement (colonnes normalisées)
Répéter jusqu'à convergence :
    // Codage parcimonieux
    Pour i = 1 à n :
        α_i ← OMP(x_i, D, s)          # s non-zéros max
    
    // Mise à jour du dictionnaire
    Pour k = 1 à K :
        Ω_k ← {i : α_{k,i} ≠ 0}
        E_k^R ← (X - D·A + d_k·α_k^T)[:, Ω_k]
        [U, Σ, V] ← SVD(E_k^R, rang=1)
        d_k ← U[:,0]
        α_{k, Ω_k} ← Σ[0,0] · V[:,0]
Retourner D, A
```

___

### 2.5 Complexité et limites

Chaque itération de K-SVD a un coût approximatif de $\mathcal{O}(n \cdot d \cdot s \cdot K)$ pour le codage, auquel s'ajoute $\mathcal{O}(n \cdot d \cdot K)$ pour les mises à jour SVD. La **totalité des données** $\mathbf{X}$ doit résider en mémoire.

Cela entraîne trois contraintes rédhibitoires à grande échelle :

1. **Mémoire** : stocker $\mathbf{X} \in \mathbb{R}^{d \times n}$ devient prohibitif pour $n \sim 10^6$.
2. **Redondance** : parcourir l'intégralité du corpus à chaque passe est coûteux quand les observations sont corrélées.
3. **Données en flux** : impossible de traiter des séquences arrivant de manière continue (*streaming*) — le lot doit être constitué *avant* de commencer.

___

## 3. Apprentissage de dictionnaire en ligne

### 3.1 Philosophie et motivation

L'apprentissage en ligne (*online learning*) renverse la logique d'accès aux données : au lieu d'inspecter l'ensemble du corpus avant chaque révision du dictionnaire, on **consomme les observations une à une** (ou par mini-lots) et on améliore $\mathbf{D}$ immédiatement après chaque ingestion.

Cette philosophie hérite de la **descente de gradient stochastique** (SGD) introduite par Robbins et Monro (1951), mais l'adapter à l'apprentissage de dictionnaire n'est pas trivial car le problème n'est pas convexe globalement, et la mise à jour de $\mathbf{D}$ doit s'appuyer sur toute l'histoire passée pour être statistiquement cohérente.

La formulation de référence est due à **Mairal, Bach, Ponce et Sapiro** (ICML 2009).

___

### 3.2 Formulation stochastique

On réinterprète le critère empirique comme l'espérance d'une fonction de perte instantanée :

$$f(\mathbf{D}) = \mathbb{E}_{\mathbf{x}}\left[\ell(\mathbf{x}, \mathbf{D})\right] \approx \frac{1}{n}\sum_{i=1}^{n} \ell(\mathbf{x}_i, \mathbf{D})$$

où la perte locale est :

$$\ell(\mathbf{x}, \mathbf{D}) = \min_{\boldsymbol{\alpha}} \frac{1}{2}\|\mathbf{x} - \mathbf{D}\boldsymbol{\alpha}\|_2^2 + \lambda\|\boldsymbol{\alpha}\|_1$$

À l'étape $t$, on reçoit $\mathbf{x}_t$, on calcule son code optimal $\boldsymbol{\alpha}_t^*$, puis on met à jour $\mathbf{D}$ pour réduire $\ell(\mathbf{x}_t, \mathbf{D})$ tout en tenant compte des échantillons précédents.

___

### 3.3 La fonction de substitution

L'astuce centrale de Mairal et al. est de construire, à chaque instant $t$, une **fonction de substitution** (*surrogate function*) $\hat{f}_t(\mathbf{D})$ qui :

* **majore** la perte cumulée moyenne $\frac{1}{t}\sum_{i=1}^t \ell(\mathbf{x}_i, \mathbf{D})$,
* est **convexe** en $\mathbf{D}$,
* est **bon marché** à minimiser grâce au maintien de statistiques agrégées.

En développant la perte autour du code courant $\boldsymbol{\alpha}_i^* = \boldsymbol{\alpha}_i^*(\mathbf{D}_{t-1})$, on obtient une approximation quadratique :

$$\hat{f}_t(\mathbf{D}) = \frac{1}{t}\sum_{i=1}^{t} \left(\frac{1}{2}\|\mathbf{x}_i - \mathbf{D}\boldsymbol{\alpha}_i^*\|_2^2 + \lambda\|\boldsymbol{\alpha}_i^*\|_1\right)$$

Cette expression se simplifie en développant la norme carrée :

$$\hat{f}_t(\mathbf{D}) = \frac{1}{2t} \operatorname{tr}\!\left(\mathbf{D}^\top \mathbf{D} \,\mathbf{B}_t\right) - \frac{1}{t} \operatorname{tr}\!\left(\mathbf{D}^\top \mathbf{C}_t\right) + \text{cste}$$

où les **statistiques suffisantes** accumulées sont :

$$\mathbf{B}_t = \sum_{i=1}^{t} \boldsymbol{\alpha}_i^* (\boldsymbol{\alpha}_i^*)^\top \in \mathbb{R}^{K \times K}, \qquad \mathbf{C}_t = \sum_{i=1}^{t} \mathbf{x}_i (\boldsymbol{\alpha}_i^*)^\top \in \mathbb{R}^{d \times K}$$

**Point crucial** : $\mathbf{B}_t$ et $\mathbf{C}_t$ se mettent à jour en $\mathcal{O}(K^2)$ et $\mathcal{O}(dK)$ à chaque pas — indépendamment de $n$. Toute l'histoire passée est résumée dans ces deux matrices.

___

### 3.4 Mécanique de la mise à jour

La minimisation de $\hat{f}_t$ sous la contrainte $\mathbf{D} \in \mathcal{C}$ se réalise par **descente de coordonnées par blocs** sur les colonnes de $\mathbf{D}$. Pour l'atome $k$, en figeant tous les autres :

$$\mathbf{d}_k \leftarrow \frac{\mathbf{c}_k - \mathbf{D}\mathbf{b}_k + b_{kk}\mathbf{d}_k}{b_{kk}}$$

puis projection sur la boule unité : $\mathbf{d}_k \leftarrow \mathbf{d}_k / \max(1, \|\mathbf{d}_k\|_2)$.

Ici $\mathbf{c}_k$ est la $k$-ième colonne de $\mathbf{C}_t$, $\mathbf{b}_k$ la $k$-ième colonne de $\mathbf{B}_t$, et $b_{kk}$ l'élément diagonal correspondant.

Cette étape coûte $\mathcal{O}(dK)$ par passe sur toutes les colonnes — négligeable face au codage parcimonieux.

___

### 3.5 Pseudo-code de l'algorithme de Mairal

```
Algorithme Online Dictionary Learning (Mairal et al., 2009)
───────────────────────────────────────────────────────────
Entrée : flux {x_t}, D_0 (initialisé), λ, nombre d'itérations T
Initialiser A_0 ← 0_{K×K}, B_0 ← 0_{d×K}   ← statistiques

Pour t = 1 à T :
    
    // 1. Tirage (ou réception) d'un échantillon
    Obtenir x_t
    
    // 2. Codage parcimonieux avec le dictionnaire courant
    α_t* ← arg min_{α}  ½‖x_t - D_{t-1} α‖² + λ‖α‖₁
             (via LARS ou OMP)
    
    // 3. Accumulation des statistiques suffisantes
    A_t ← A_{t-1} + α_t* (α_t*)ᵀ          ∈ ℝ^{K×K}
    B_t ← B_{t-1} + x_t  (α_t*)ᵀ          ∈ ℝ^{d×K}
    
    // 4. Minimisation de la substitution (bloc-coordonnées)
    D_t ← arg min_{D ∈ C}  ½ tr(DᵀD A_t) - tr(Dᵀ B_t)
    
    // Explicitement, pour k = 1 à K :
    Pour k = 1 à K :
        u_k ← (b_k - D·a_k + a_{kk}·d_k) / a_{kk}
        d_k ← u_k / max(1, ‖u_k‖₂)

Retourner D_T
```

> **Note d'implémentation** : en pratique, on applique plusieurs passes de descente de coordonnées à l'étape 4 jusqu'à convergence locale (typiquement 5–10 tours suffisent).

___

### 3.6 Garanties de convergence

Mairal et al. démontrent que, sous des hypothèses d'ergodicité sur la distribution des données et de régularité de la perte :

* La suite $(\mathbf{D}_t)_{t \geq 1}$ **converge presque sûrement** vers un point stationnaire de $f$.
* La valeur du critère empirique $\frac{1}{t}\hat{f}_t(\mathbf{D}_t)$ **converge** vers $\inf_{\mathbf{D} \in \mathcal{C}} f(\mathbf{D})$.

La preuve s'appuie sur la théorie des **processus quasi-markoviens** et des inégalités de martingales, en montrant que la différence entre la substitution et la vraie espérance tend vers zéro.

___

## 4. Comparaison synthétique

Critère | Batch (K-SVD, MOD…) | En ligne (Mairal et al.)
---|---|---
**Accès aux données** | Intégralité du corpus à chaque itération | Un vecteur (ou mini-lot) à la fois
**Empreinte mémoire** | $\mathcal{O}(nd)$ — toute la matrice $\mathbf{X}$ | $\mathcal{O}(K^2 + dK)$ — uniquement $\mathbf{A}_t$ et $\mathbf{B}_t$
**Compatibilité streaming** | Non | Oui — convient aux flux temps-réel
**Vitesse de convergence** | Rapide sur petits volumes | Lent au départ, puis très efficace à grande échelle
**Qualité asymptotique** | Excellente sur données finies | Identique (voire meilleure) si $n \gg 1$
**Adaptabilité à la dérive** | Nulle (données statiques) | Naturelle via pondération décroissante des statistiques
**Parallélisation** | Aisée sur $n$ (codage parallèle) | Mini-lots + moyennage de Polyak-Ruppert
**Mise en œuvre** | Mature, bibliothèques abondantes | Disponible dans scikit-learn (`MiniBatchDictionaryLearning`)
**Fondements théoriques** | Convergence locale bien établie | Convergence presque sûre prouvée

**Résumé en une phrase** : la variante classique *sculpte* le dictionnaire en contemplant l'ensemble du matériau, tandis que la déclinaison en ligne *le façonne* par accumulation progressive d'expériences, résumées dans deux matrices de taille fixe.

___

## 5. Quand choisir l'un plutôt que l'autre ?

```
Taille du corpus
       │
       ▼
  n < ~10 000 ?  ──Oui──▶  Batch (K-SVD)
       │                    Précision maximale, contrôle fin
       │ Non
       ▼
  Données en flux ?  ──Oui──▶  Online obligatoire
       │
       │ Non
       ▼
  RAM suffisante ?  ──Non──▶  Online (mini-batch)
       │
       │ Oui
       ▼
  Distribution stationnaire ?
       │
   Oui │           Non
       ▼            ▼
     Batch      Online avec
    ou Online    fenêtre glissante
    (équivalent)  ou pondération
                  exponentielle
```

___

## 6. Implémentations de référence

```python
# ----- Batch -----
from sklearn.decomposition import DictionaryLearning

dl = DictionaryLearning(
    n_components=64,     # nombre d'atomes K
    alpha=1.0,           # régularisation λ
    max_iter=100,
    fit_algorithm='lars',
    transform_algorithm='omp',
    n_jobs=-1
)
codes = dl.fit_transform(X)   # X : (n_samples, n_features)
D_batch = dl.components_      # (K, d)

# ----- En ligne -----
from sklearn.decomposition import MiniBatchDictionaryLearning

odl = MiniBatchDictionaryLearning(
    n_components=64,
    alpha=1.0,
    max_iter=100,          # passes sur les mini-lots
    batch_size=256,        # taille du mini-lot
    fit_algorithm='lars',
    transform_algorithm='omp',
    n_jobs=-1
)
codes_online = odl.fit_transform(X)
D_online = odl.components_
```

> `MiniBatchDictionaryLearning` implémente fidèlement l'algorithme de Mairal et al. avec les accumulateurs $\mathbf{A}_t$ et $\mathbf{B}_t$.

___

## 7. Pour aller plus loin

* **Article fondateur du batch** : Aharon, M., Elad, M., Bruckstein, A. *K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation*. IEEE TSP, 2006.
* **Article fondateur de la variante en ligne** : Mairal, J., Bach, F., Ponce, J., Sapiro, G. *Online Learning for Matrix Factorization and Sparse Coding*. JMLR, 2010.
* **Revue exhaustive** : Rubinstein, R., Bruckstein, A. M., Elad, M. *Dictionaries for Sparse Representation Modeling*. Proceedings of the IEEE, 2010.
* **Extension neurale** : les auto-encodeurs parcimonieux (SAE) et les réseaux de déploiement (*LISTA*) peuvent être vus comme des généralisations paramétriques de ces deux régimes.
* **Lien avec les méthodes de factorisation** : NMF (*Non-negative Matrix Factorization*), ICA (*Independent Component Analysis*) et ALS (*Alternating Least Squares*) partagent la même structure bi-convexe mais imposent des contraintes différentes sur $\mathbf{D}$ et $\mathbf{A}$.