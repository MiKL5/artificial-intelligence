# Algèbre Linéaire : Décomposition en Valeurs Singulières (SVD)<a href="../../../"><img src="../../../../assets/images/atomicAi.png" alt="L'intelligence artificielle" align="right" height="64px"></a>
En tant que Data Scientist, la SVD est notre outil le plus versatile pour l'analyse des structures latentes. Contrairement à la diagonalisation classique qui exige une matrice carrée (et souvent symétrique), la SVD s'applique à **toute** matrice $A \in \mathcal{M}_{m,n}(\mathbb{R})$, qu'elle soit rectangulaire ou singulière.
## I. Le Théorème Fondamental
Toute matrice $A$ peut être factorisée sous la forme :
$$A = U \Sigma V^T$$

Où :
* **$U$ (Matrices des vecteurs singuliers à gauche)** : Une matrice orthogonale $m \times m$ dont les colonnes sont les vecteurs propres de $AA^T$. ⬅️
* **$\Sigma$ (Valeurs singulières)** : Une matrice diagonale $m \times n$ contenant les racines carrées des valeurs propres de $A^TA$ (ou $AA^T$), classées par ordre décroissant. 📉
* **$V^T$ (Matrices des vecteurs singuliers à droite)** : Une matrice orthogonale $n \times n$ dont les colonnes sont les vecteurs propres de $A^TA$. ➡️
## II. L'Introspection Géométrique
Géométriquement, la SVD décompose n'importe quelle transformation linéaire en trois étapes successives :
1. **Rotation** dans l'espace de départ (via $V^T$).
2. **Étirement** (ou contraction) le long des axes principaux (via $\Sigma$).
3. **Rotation** dans l'espace d'arrivée (via $U$).
## III. Applications en Data Science 🧪
* **Compression d'images** : En ne conservant que les $k$ plus grandes valeurs singulières (approximation de rang faible).
* **Systèmes de Recommandation** : Identification de caractéristiques latentes (ex: styles de films/préférences utilisateurs).
* **Traitement du Langage Naturel (NLP)** : Analyse Sémantique Latente (LSA).
___
>Sources et Références Académiques
>* **Trefethen, L. N., & Bau III, D.** (1997). _Numerical Linear Algebra_. [SIAM. (L'approche algorithmique de référence)](https://people.maths.ox.ac.uk/trefethen/books.html).
>* **Golub, G. H., & Van Loan, C. F.** (2013). _Matrix Computations_. [Johns Hopkins University Press](https://jhupbooks.press.jhu.edu/title/matrix-computations).
>* **Kutz, J. N.** (2013). _Data-Driven Science and Engineering_. [Cambridge University Press](https://www.databookuw.com/).