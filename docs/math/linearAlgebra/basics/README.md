# Algèbre Linéaire : Vecteurs, Matrices et Fondements Opérationnels<a href="../../../"><img src="../../../../assets/images/atomicAi.png" alt="L'intelligence artificielle" align="right" height="64px"></a>
## 1. Introduction au Formalisme
L'algèbre linéaire constitue la pierre angulaire de l'édifice mathématique contemporain, agissant comme le substrat essentiel de la physique théorique, des neurosciences computationnelles et de l'intelligence artificielle.
## 2. Les Vecteurs : Éléments de l'Espace Vectoriel
Un vecteur $\mathbf{v}$ au sein d'un espace vectoriel $E$ sur un corps $\mathbb{K}$ (généralement $\mathbb{R}$ ou $\mathbb{C}$) n'est point un simple n-uplet de scalaires, mais une entité abstraite définie par ses propriétés de stabilité sous l'addition et la multiplication par un scalaire.
### Opérations Fondamentales
* **Addition Vectorielle** : Pour deux vecteurs $\mathbf{u}, \mathbf{v} \in \mathbb{K}^n$, leur somme $\mathbf{w} = \mathbf{u} + \mathbf{v}$ est définie par $w_i = u_i + v_i$ pour tout $i \in \{1, \dots, n\}$.
* **Multiplication par un Scalaire** : Soit $\alpha \in \mathbb{K}$, alors $\alpha \mathbf{v} = (\alpha v_1, \alpha v_2, \dots, \alpha v_n)$. Cette opération est une **homothétie** dans l'espace considéré.
* **Produit Scalaire (Dot Product)** : Défini par $\langle \mathbf{u}, \mathbf{v} \rangle = \sum_{i=1}^n u_i v_i$. C'est une forme bilinéaire symétrique définie positive qui induit une **norme euclidienne**.
## 3. Les Matrices : Opérateurs et Représentations
Une matrice $A \in \mathcal{M}_{m,n}(\mathbb{K})$ est une famille d'éléments $(a_{i,j})$ indexée par $i \in \{1, \dots, m\}$ (lignes) et $j \in \{1, \dots, n\}$ (colonnes). Elle représente souvent une **application linéaire** entre deux espaces de dimensions finies.
### Opérations Matricielles
#### Addition et Soustraction
L'addition est licite si et seulement si les matrices sont **isomorphes** en termes de dimensions (congruence dimensionnelle).
$$C = A + B \implies c_{i,j} = a_{i,j} + b_{i,j}$$
#### Produit Matriciel (Composition)
Le produit de $A \in \mathcal{M}_{m,n}(\mathbb{K})$ par $B \in \mathcal{M}_{n,p}(\mathbb{K})$ produit une matrice $C \in \mathcal{M}_{m,p}(\mathbb{K})$ selon la règle :
$$c_{i,j} = \sum_{k=1}^n a_{i,k} b_{k,j}$$
*Note : Le produit matriciel est non-commutatif ($AB \neq BA$ en général), une caractéristique fondamentale qui engendre la complexité des algèbres d'opérateurs.*
#### La Transposition
L'opération de transposition, notée $A^\top$, consiste à permuter les indices de ligne et de colonne : $(a^\top)_{i,j} = a_{j,i}$. Une matrice est dite **symétrique** si $A = A^\top$.
## 4. Concepts Avancés et Invariants
* **La Trace ($\text{Tr}$)** : Somme des éléments de la diagonale principale d'une matrice carrée. Elle est invariante par changement de base (similarité).
* **Le Déterminant** : Forme multilinéaire alternée qui quantifie le changement de volume lors d'une transformation linéaire. Une matrice est inversible si et seulement si son déterminant est non nul.
* **Combinaison Linéaire et Indépendance** : Un ensemble de vecteurs est dit linéairement indépendant si aucune combinaison linéaire non triviale ne s'annule. Cela définit la **base** de l'espace.
___
>Sources et Références Bibliographiques  
>L'érudition exige le recours aux sources primaires et aux traités de référence :
>
>1.  **Bourbaki, N.** (1970). *Éléments de mathématique : Algèbre, Chapitres 1 à 3*. Hermann / Springer.
>    * [Lien vers l'édition Springer](https://link.springer.com/book/10.1007/978-3-540-33850-5)
>2.  **Strang, G.** (2016). *Introduction to Linear Algebra*. Wellesley-Cambridge Press.
>    * [Site officiel du cours (MIT OCW)](https://math.mit.edu/~gs/linearalgebra/)
>3.  **Hoffman, K., & Kunze, R.** (1971). *Linear Algebra*. Prentice-Hall.
>    * [Consulter sur Archive.org (Accès public)](https://archive.org/details/linear-algebra-hoffman-kunze)
>4.  **Halmos, P. R.** (1958). *Finite-Dimensional Vector Spaces*. D. Van Nostrand Company / Springer.
>    * [Lien vers la collection Graduate Texts in Mathematics](https://link.springer.com/book/10.1007/978-1-4612-6387-6)
>5.  **Gantmacher, F. R.** (1959). *The Theory of Matrices*. Chelsea Publishing / American Mathematical Society.
>    * [Lien vers l'American Mathematical Society (Volume 1)](https://bookstore.ams.org/chel-131-e)