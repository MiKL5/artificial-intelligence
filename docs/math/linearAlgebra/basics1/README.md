# **Vecteurs, matrices et opérations fondamentales**<a href="../../../"><img src="../../../../assets/images/atomicAi.png" alt="L'intelligence artificielle" align="right" height="64px"></a>

## Vecteurs et espaces vectoriels

Un **espace vectoriel** sur un corps (par exemple \(\mathbb{R}\)) est un ensemble \(V\) muni de deux opérations, l’addition de vecteurs et la multiplication par un scalaire, satisfaisant un système d’axiomes (associativité, commutativité de l’addition, existence d’un vecteur nul, etc.).
Dans le cas usuel de \(\mathbb{R}^n\), un vecteur est une liste ordonnée de \(n\) réels, que l’on écrit par exemple \(x = (x_1, \dots, x_n)\) et qui se représente géométriquement comme une flèche depuis l’origine.

Les opérations fondamentales sur les vecteurs de \(\mathbb{R}^n\) sont :
* **Addition** : \((x_1, \dots, x_n) + (y_1, \dots, y_n) = (x_1 + y_1, \dots, x_n + y_n)\).
* **Multiplication par un scalaire** : pour \(\alpha \in \mathbb{R}\), \(\alpha (x_1, \dots, x_n) = (\alpha x_1, \dots, \alpha x_n)\).

<details>
<summary>Remarque : dimension et base</summary>

La **dimension** d’un espace vectoriel est le nombre de vecteurs dans une base, c’est‑à‑dire une famille de vecteurs linéairement indépendants qui engendre tout l’espace. Pour \(\mathbb{R}^n\), la dimension est \(n\) et une base canonique est donnée par les vecteurs qui ont une seule coordonnée égale à 1 et les autres nulles.

</details>

## Matrices : définition et typologie

Une **matrice** est un tableau rectangulaire de nombres organisé en \(m\) lignes et \(n\) colonnes, que l’on note \(A \in \mathbb{R}^{m \times n}\). Les matrices permettent de représenter à la fois des données (par exemple un jeu de données tabulaire) et des applications linéaires entre espaces vectoriels de dimensions finies.

On distingue notamment :
* Les matrices **carrées** \(n \times n\), qui représentent des endomorphismes de \(\mathbb{R}^n\).
* La matrice **identité** \(I_n\), dont les coefficients diagonaux valent 1 et les autres 0, qui joue le rôle d’élément neutre pour la multiplication matricielle.

## Opérations fondamentales sur les matrices

Les opérations élémentaires sur les matrices prolongent celles définies sur les vecteurs et constituent la grammaire de l’algèbre linéaire appliquée.

### Addition et soustraction

Deux matrices \(A, B \in \mathbb{R}^{m \times n}\) peuvent être **ajoutées** ou **soustraites** terme à terme si, et seulement si, elles ont le même format :
\[
(A + B)_{ij} = a_{ij} + b_{ij}, \qquad (A - B)_{ij} = a_{ij} - b_{ij}.
\][web:26]

### Multiplication par un scalaire

Pour \(\alpha \in \mathbb{R}\) et \(A \in \mathbb{R}^{m \times n}\), la **multiplication scalaire** est définie par \((\alpha A)_{ij} = \alpha \, a_{ij}\), ce qui généralise la multiplication d’un vecteur par un scalaire.

### Produit matrice–vecteur

Si \(A \in \mathbb{R}^{m \times n}\) et \(x \in \mathbb{R}^n\), le **produit matrice–vecteur** \(Ax \in \mathbb{R}^m\) est défini par
\[
(Ax)_i = \sum_{j=1}^n a_{ij} x_j.
\][web:26][web:27]
Ce produit s’interprète comme l’application d’une transformation linéaire à un vecteur, point de départ de nombreuses applications en apprentissage automatique et en traitement du signal.

### Produit de matrices

Pour \(A \in \mathbb{R}^{m \times n}\) et \(B \in \mathbb{R}^{n \times p}\), le **produit matriciel** \(C = AB \in \mathbb{R}^{m \times p}\) est défini par
\[
C_{ik} = \sum_{j=1}^n a_{ij} b_{jk}.
\][web:23][web:26]
Cette opération est associative mais non commutative en général et correspond à la composition de deux applications linéaires successives.

### Transposée

La **transposée** \(A^T\) d’une matrice \(A \in \mathbb{R}^{m \times n}\) est la matrice \(n \times m\) obtenue en échangeant lignes et colonnes, c’est‑à‑dire \((A^T)_{ij} = a_{ji}\).
La transposée intervient de manière centrale dans la définition des produits scalaires et des matrices symétriques, très utilisées en statistiques et en optimisation.

<details>
<summary>Inverse d’une matrice (aperçu)</summary>

Une matrice carrée \(A \in \mathbb{R}^{n \times n}\) est dite **inversible** s’il existe une matrice \(A^{-1}\) telle que \(AA^{-1} = A^{-1}A = I_n\). L’existence de l’inverse équivaut à l’**injectivité** et la **surjectivité** de la transformation linéaire représentée par \(A\), et au fait que son déterminant est non nul.

</details>

> Sources  
> [Vector space – Wikipédia](https://en.wikipedia.org/wiki/Vector_space)  
> [Vector Space – Wolfram MathWorld](https://mathworld.wolfram.com/VectorSpace.html)  
> [Matrix (mathematics) – Wikipédia](https://en.wikipedia.org/wiki/Matrix_(mathematics))  
> [Linear Algebra (Gilbert Strang) – MIT OCW](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/)  
> [Vector Space – GeeksforGeeks](https://www.geeksforgeeks.org/maths/vector-space/)  
> [Matrix Operations – GeeksforGeeks](https://www.geeksforgeeks.org/maths/matrix-operations/)

