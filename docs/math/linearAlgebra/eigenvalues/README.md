# **Algèbre Linéaire Avancée : Théorie Spectrale et Morphismes**<a href="../../../"><img src="../../../../assets/images/atomicAi.png" alt="L'intelligence artificielle" align="right" height="64px"></a>
## I. Ontologie de la Transformation
En analyse de données, une matrice $A \in \mathcal{M}_n(\mathbb{K})$ ne doit pas être perçue comme un simple tableau de nombres, mais comme un **opérateur linéaire** transformant l'espace vectoriel.

La quête des **vecteurs propres** ($\mathbf{v}$) et des **valeurs propres** ($\lambda$) revient à identifier les directions d'autosimilarité de cette transformation. Contrairement aux vecteurs génériques qui subissent une rotation, un vecteur propre voit sa direction préservée : il subit uniquement une homothétie.
### Équation Fondamentale
$$A\mathbf{v} = \lambda\mathbf{v}$$
Où :
* $A$ est l'opérateur linéaire.
* $\mathbf{v} \in \mathbb{V} \setminus \{\mathbf{0}\}$ est le vecteur propre associé.
* $\lambda \in \mathbb{K}$ est le scalaire (valeur propre) représentant le facteur de dilatation.
## II. Le Spectre et le Noyau
D'un point de vue algébrique, $\lambda$ est une valeur propre de $A$ si et seulement si l'opérateur $(A - \lambda I)$ est singulier. Cela implique que son **déterminant** est nul, menant à l'équation séculaire :

$$\det(A - \lambda I) = 0$$

Ce polynôme en $\lambda$, noté $P_A(\lambda)$, est le **polynôme caractéristique**. Ses racines constituent le **spectre** de la matrice, noté $\text{Sp}(A)$.
___
>Sources et Références Académiques
>* **Strang, G.** (2016). _Introduction to Linear Algebra_. [Wellesley-Cambridge Press](https://web.mit.edu/18.06/www/).
>* **Axler, S.** (2015). _Linear Algebra Done Right_. [Springer. (Approche élégante sans déterminant privilégiant la structure des opérateurs)](https://linear.axler.net/).
>* **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). _Deep Learning_. [MIT Press. (Chapitre 2 pour une vision orientée Data Science)](https://www.deeplearningbook.org/).