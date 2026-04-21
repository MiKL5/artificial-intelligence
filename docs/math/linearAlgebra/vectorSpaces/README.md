# **Espaces vectoriels et sous-espaces**
## 1. Ontologie de l'Espace Vectoriel
Un **espace vectoriel** (ou espace linéaire) sur un corps commutatif $\\mathbb{K}$ (typiquement $\\mathbb{R}$ ou $\\mathbb{C}$) est une structure algébrique $(E, +, \\cdot)$ d'une complexité élégante. Il ne s'agit point d'un simple agrégat d'objets, mais d'un ensemble muni d'une loi de composition interne (l'addition vectorielle) et d'une loi de composition externe (la multiplication par un scalaire).
### Axiomatique Fondamentale
Pour que $E$ soit qualifié d'espace vectoriel, il doit satisfaire une octologie d'axiomes garantissant la stabilité structurelle :
1. **Groupe Commutatif** : $(E, +)$ doit être un groupe abélien (associativité, commutativité, existence d'un élément neutre $\\mathbf{0}_E$, et d'un symétrique pour chaque vecteur).
2. **Distributivité des Scalaires** : $\\alpha(\\mathbf{u} + \\mathbf{v}) = \\alpha\\mathbf{u} + \\alpha\\mathbf{v}$.
3. **Distributivité Vectorielle** : $(\\alpha + \\beta)\\mathbf{u} = \\alpha\\mathbf{u} + \\beta\\mathbf{u}$.
4. **Compatibilité du Produit** : $(\\alpha\\beta)\\mathbf{u} = \\alpha(\\beta\\mathbf{u})$.
5. **Unitarité** : $1_{\\mathbb{K}} \\cdot \\mathbf{u} = \\mathbf{u}$.
## 2. Les Sous-Espaces Vectoriels (SEV)
Un sous-espace vectoriel $F$ est une partie de $E$ qui hérite de sa structure. Pour éviter une vérification fastidieuse des huit axiomes, on utilise le **critère de stabilité linéaire**.

### Critères de Caractérisation
Une partie non vide $F \\subset E$ est un sous-espace vectoriel si et seulement si elle est stable par combinaison linéaire :
$$\\forall (\\alpha, \\beta) \\in \\mathbb{K}^2, \\forall (\\mathbf{u}, \\mathbf{v}) \\in F^2, (\\alpha\\mathbf{u} + \\beta\\mathbf{v}) \\in F$$
Géométriquement, dans $\\mathbb{R}^3$, un sous-espace est soit l'origine, soit une droite passant par l'origine, soit un plan passant par l'origine, soit l'espace entier.
## 3. Sommes, Intersections et Supplémentarité
* **Intersection** : L'intersection d'une famille quelconque de sous-espaces vectoriels est immanquablement un sous-espace vectoriel.
* **Somme de Sous-Espaces** : Soient $F$ et $G$ deux SEV de $E$. La somme $F+G = \\{\\mathbf{u} + \\mathbf{v} \\mid \\mathbf{u} \\in F, \\mathbf{v} \\in G\\}$ est le plus petit SEV contenant $F \\cup G$.
* **Somme Directe ($\\oplus$)** : Si $F \\cap G = \\{\\mathbf{0}_E\\}$, la somme est dite directe. Si de surcroît $F \\oplus G = E$, alors $F$ et $G$ sont dits **supplémentaires**.
## 4. Engendrement et Dimensionnalité
La notion de **système générateur** permet de reconstruire l'intégralité de l'espace à partir d'un noyau restreint de vecteurs.
* **Famille Libre** : Un ensemble de vecteurs est linéairement indépendant si aucune combinaison linéaire non triviale ne produit le vecteur nul.
* **Base** : Une famille à la fois libre et génératrice. Selon le théorème de la base incomplète, tout espace vectoriel (de dimension finie) admet une base.
* **Dimension** : Le cardinal commun à toutes les bases de $E$. C'est un invariant topologique et algébrique majeur.
___
> Sources et Références Bibliographiques  
>La précision mathématique s'appuie sur des traités dont la validité est universellement reconnue par la communauté scientifique :
>
>1.  **Bourbaki, N.** (1970). *Éléments de mathématique : Algèbre, Chapitres 1 à 3*. Hermann / Springer.
>    * [Édition Springer - Fondements structurels](https://link.springer.com/book/10.1007/978-3-540-33850-5)
>2.  **Axler, S.** (2015). *Linear Algebra Done Right*. Springer (3rd ed.).
>    * [Site officiel de l'auteur - Approche sans déterminant](https://linear.axler.net/)
>3.  **Strang, G.** (2016). *Introduction to Linear Algebra*. Wellesley-Cambridge Press.
>    * [MIT OpenCourseWare - Ressources pédagogiques](https://math.mit.edu/~gs/linearalgebra/)
>4.  **Hoffman, K., & Kunze, R.** (1971). *Linear Algebra*. Prentice-Hall.
>    * [Consultation sur Archive.org - Le standard académique](https://archive.org/details/linear-algebra-hoffman-kunze)
>5.  **Halmos, P. R.** (1958). *Finite-Dimensional Vector Spaces*. Springer.
>    * [Lien Graduate Texts in Mathematics](https://link.springer.com/book/10.1007/978-1-4612-6387-6)