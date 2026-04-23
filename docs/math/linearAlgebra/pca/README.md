# Analyse en Composantes Principales (ACP)<a href="../../../"><img src="../../../../assets/images/atomicAi.png" alt="L'intelligence artificielle" align="right" height="64px"></a>
L'ACP est une méthode d'**ordonnancement de la variance** qui permet de réduire la dimensionnalité d'un jeu de données en projetant les observations sur de nouveaux axes orthogonaux, appelés composantes principales.
### Le Mécanisme de Projection 📐
Mathématiquement, l'objectif est de trouver un vecteur de projection $\mathbf{w}$ qui maximise la variance des données projetées. Cela revient à résoudre un problème de maximisation sous contrainte :
$$\max_{\|\mathbf{w}\|=1} \text{Var}(X\mathbf{w}) = \max_{\|\mathbf{w}\|=1} \mathbf{w}^\top \Sigma \mathbf{w}$$
Où $\Sigma$ est la matrice de covariance des données. La solution de ce problème correspond au vecteur propre associé à la plus grande valeur propre de $\Sigma$.

### Protocole Algorithmique 🧮
1. **Standardisation** : Centrage et réduction des variables pour traiter chaque dimension sur un pied d'égalité. 📏
2. **Matrice de Covariance** : Calcul de $\Sigma = \frac{1}{n-1} X^\top X$. 🧬
3. **Analyse Spectrale** : Décomposition en valeurs propres et vecteurs propres de $\Sigma$. 💎
4. **Réduction** : Sélection des $k$ vecteurs propres associés aux plus grandes valeurs propres pour former la matrice de passage. 📊
### Sources et Références Académiques 📚
* **Jolliffe, I. T.** (2002). *Principal Component Analysis*. [Springer](https://link.springer.com/book/10.1007/b98835).
* **Shlens, J.** (2014). _[A Tutorial on Principal Component Analysis](https://arxiv.org/abs/1404.1100)_.
* **Hastie, T., et al.** (2009).  _[The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/)_.