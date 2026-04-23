# Tenseurs et Algèbre Multilinéaire<a href="../../../"><img src="../../../../assets/images/atomicAi.png" alt="L'intelligence artificielle" align="right" height="64px"></a>
## I. Ontologie et Hiérarchie des Tenseurs 📊
L'**ordre** (ou rang) d'un tenseur correspond au nombre de ses dimensions (axes).

Ordre | Nom | Espace | Notation
---|---|---|---
0 | Scalaire | $\mathbb{R}$ | $s$
1 | Vecteur | $\mathbb{R}^{d_1}$ | $\mathbf{v}$
2 | Matrice | $\mathbb{R}^{d_1 \times d_2}$ | $M$
$N$ | Tenseur | $\mathbb{R}^{d_1 \times \dots \times d_N}$ | $\mathcal{T}$
### Distinction Cruciale : Covariance vs Contravariance
Au niveau doctoral, on distingue les composantes selon leur comportement lors d'un changement de base :
* **Contravariantes** ($v^i$) : Se transforment comme les coordonnées.
* **Covariantes** ($v_i$) : Se transforment comme les vecteurs de base (duaux).
## II. Opérations Tensorielles Fondamentales
### 1. Produit Tensoriel ($\otimes$)
Il permet de construire un tenseur d'ordre $p+q$ à partir de deux tenseurs d'ordres $p$ et $q$. 
Exemple : $\mathbf{a} \otimes \mathbf{b}$ crée une matrice de rang 1.
### 2. Contraction de Tenseurs
C'est la généralisation de la trace d'une matrice ou du produit matriciel. On somme sur deux indices (un haut, un bas en notation d'Einstein) pour réduire l'ordre du tenseur de 2.
### 3. Produit de Kronecker
Très utilisé en traitement du signal et pour les systèmes couplés, il permet de représenter des opérations globales sur des matrices par blocs.
## III. Décompositions Tensorielles (Analyse Spectrale Multivoie) 🧩
Tout comme la SVD décompose une matrice, nous utilisons des modèles pour les tenseurs :
1.  **Décomposition CP (Canonical Polyadic)** : Exprime un tenseur comme une somme de produits extérieurs de vecteurs. C'est l'analogue direct de la décomposition en valeurs propres.
2.  **Décomposition de Tucker** : Souvent comparée à une "ACP multidimensionnelle", elle utilise un "noyau" (core tensor) pour capturer les interactions entre les différentes dimensions.
## IV. Applications en Science des Données et Deep Learning 🧠
* **Traitement d'images et vidéos** : Une vidéo est naturellement un tenseur d'ordre 4 (Temps, Largeur, Hauteur, Canaux).
* **Compression de modèles** : Utilisation des décompositions pour réduire le nombre de paramètres des réseaux de neurones (Tensorized Neural Networks).
* **Traitement du Signal** : Séparation aveugle de sources.
>Sources et Références Académiques 📚
>* **Kolda, T. G., & Bader, B. W.** (2009). *Tensor Decompositions and Applications*.  [SIAM Review](https://www.sandia.gov/~tgkolda/pubs/pubfiles/KoldaBader2009.pdf).
>* **Cichocki, A., et al.** (2015). [_Tensor Decompositions for Signal Processing and Machine Learning_](https://ieeexplore.ieee.org/document/7038247).
>* **Hackbusch, W.** (2012). *Tensor Spaces and Numerical Tensor Calculus*. Springer.