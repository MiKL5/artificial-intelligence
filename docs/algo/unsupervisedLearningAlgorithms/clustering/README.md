# **L'algorithme de clustering**<a href="../../../"><img src="../../../../assets/images/atomicAi.png" alt="L'intelligence artificielle" align="right" height="64px"></a>

Le **clustering** (ou *partitionnement de données*, *classification non supervisée*) désigne la famille des méthodes qui regroupent automatiquement un ensemble de points en **sous-ensembles homogènes** (clusters) sans étiquettes préalables. L'objectif est que les points d'un même cluster soient plus similaires entre eux qu'à ceux des autres clusters, selon une métrique définie.

Formellement, étant donné un ensemble $X = {x₁, …, xₙ} ⊂ ℝᵈ$, on cherche une partition $C = {C₁, …, Cₖ}$ de $X$ telle qu'une fonction de critère $J(C)$ soit minimisée (ou maximisée selon la formulation).

Le clustering est un problème **NP-difficile** dans le cas général : la recherche exhaustive de la partition optimale est combinatoire en $O(kⁿ / k!)$. Tous les algorithmes pratiques sont des heuristiques ou des approximations.

## Origines et chronologie

### Taxonomie numérique (années 1950–1960)

Le clustering naît dans le contexte de la **taxonomie numérique** (*numerical taxonomy*), discipline cherchant à classifier automatiquement des organismes biologiques à partir de mesures quantitatives. **Sokal & Michener** (1958) formalisent l'idée de regrouper des entités par similarité mesurée, posant les bases du clustering hiérarchique agglomératif.

> Sokal, R.R. & Michener, C.D. (1958). *A statistical method for evaluating systematic relationships*. **University of Kansas Science Bulletin**, 38, 1409–1438.

### K-means — l'algorithme fondateur (1956–1967)

L'algorithme **K-means** est attribué à plusieurs auteurs indépendants. **Stuart Lloyd** le formule dès 1957 dans un mémo interne à Bell Labs (publié seulement en 1982), dans le contexte de la **quantification vectorielle** pour la compression de signal. **Edward Forgy** (1965) en publie une variante, et **James MacQueen** (1967) lui donne son nom et sa formalisation mathématique moderne.

> Lloyd, S.P. (1982). *Least squares quantization in PCM*. [**IEEE Transactions on Information Theory**, 28(2), 129–137](https://doi.org/10.1109/TIT.1982.1056489).

> MacQueen, J. (1967). *Some methods for classification and analysis of multivariate observations*. In *Proceedings of the 5th Berkeley Symposium on Mathematical Statistics and Probability*, vol. 1, pp. 281–297. University of California Press.

### Clustering hiérarchique (1963–1973)

**Ward** (1963) propose la méthode de linkage par minimisation de la variance intra-cluster, toujours utilisée sous le nom de *Ward's method*. **Lance & Williams** (1967) unifient les différentes méthodes hiérarchiques dans un cadre algorithmique général.

> Ward, J.H. (1963). [*Hierarchical grouping to optimize an objective function*. **Journal of the American Statistical Association**, 58(301), 236–244](https://doi.org/10.1080/01621459.1963.10500845).

### DBSCAN — clustering par densité (1996)

**Ester, Kriegel, Sander & Xu** (1996) introduisent **DBSCAN** (*Density-Based Spatial Clustering of Applications with Noise*), premier algorithme capable de détecter des clusters de forme arbitraire et d'identifier explicitement les points aberrants (*noise*). Il reçoit en 2014 le **Test of Time Award** à KDD, récompensant son impact durable.

> Ester, M., Kriegel, H.-P., Sander, J. & Xu, X. (1996). *A density-based algorithm for discovering clusters in large spatial databases with noise*. [In *Proceedings of KDD 1996*, pp. 226–231. AAAI Press](https://dl.acm.org/doi/10.5555/3001460.3001507).

### Modèles de mélange gaussiens — approche probabiliste (1977 / 2000)

L'algorithme **EM** (*Expectation-Maximization*) de **Dempster, Laird & Rubin** (1977) fournit le cadre général pour l'estimation des **Gaussian Mixture Models** (GMM), formulation probabiliste du clustering. C'est une généralisation soft de K-means.

> Dempster, A.P., Laird, N.M. & Rubin, D.B. (1977). *Maximum likelihood from incomplete data via the EM algorithm*. [**Journal of the Royal Statistical Society, Series B**, 39(1), 1–38](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x).

### Spectral Clustering (2001–2002)

**Ng, Jordan & Weiss** (2001) et **Shi & Malik** (2000) formalisent le **clustering spectral**, qui transforme le problème en une décomposition en valeurs propres du Laplacien d'un graphe de similarité, permettant de découvrir des structures non convexes inaccessibles à K-means.

> Ng, A.Y., Jordan, M.I. & Weiss, Y. (2001). [*On spectral clustering: Analysis and an algorithm*. In *Advances in Neural Information Processing Systems (NeurIPS)*, 14, 849–856](https://proceedings.neurips.cc/paper/2001/hash/801272ee79cfde7fa5960571fee36b9b-Abstract.html).

> Shi, J. & Malik, J. (2000). [*Normalized cuts and image segmentation*. **IEEE Transactions on Pattern Analysis and Machine Intelligence**, 22(8), 888–905](https://doi.org/10.1109/34.868688).


## Pourquoi le clustering ?

Le clustering répond à trois besoins fondamentaux :

1. **Exploration des données** : en l'absence de labels, il révèle la structure latente d'un jeu de données — segments de clients, sous-types de maladies, régimes climatiques, communautés dans un réseau social.
2. **Compression et quantification** : K-means est à la base de la quantification vectorielle, utilisée en compression d'image (JPEG, codebooks), en indexation de bases vectorielles et en apprentissage par renforcement (discretisation d'espaces d'états).
3. **Prétraitement** : les clusters servent de features intermédiaires (Bag of Visual Words en vision, K-means features pour les réseaux de neurones peu profonds).

## Taxonomie des algorithmes

### 1. Partitionnement centroïde — K-means

**Principe** : alterner entre (a) l'affectation de chaque point au centroïde le plus proche et (b) la mise à jour des centroïdes comme moyenne des points affectés.

**Critère** : minimisation de l'inertie intra-cluster (Within-Cluster Sum of Squares, WCSS) :

$$J = Σᵢ Σ_{x ∈ Cᵢ} ‖x − μᵢ‖²$$

**Complexité** : $O(n · k · d · t)$ par itération t — linéaire en n, ce qui en fait l'algorithme de référence pour les grands jeux de données.

**Limites** : suppose des clusters convexes et isotropes, sensible à l'initialisation (solution : K-means++ de Arthur & Vassilvitskii, 2007), nécessite de fixer k a priori.

> Arthur, D. & Vassilvitskii, S. (2007). [*K-means++: The advantages of careful seeding*. In *Proceedings of SODA 2007*, pp. 1027–1035](https://dl.acm.org/doi/10.5555/1283383.1283494).

### 2. Clustering hiérarchique

**Principe** : construction d'un **dendrogramme** (arbre hiérarchique) par fusions successives (agglomératif, *bottom-up*) ou divisions (divisif, *top-down*).

Critères de linkage courants :
* **Single linkage** : distance minimale entre clusters → chaînage, clusters allongés
* **Complete linkage** : distance maximale → clusters compacts
* **Average linkage** (UPGMA) : distance moyenne
* **Ward** : minimisation de l'augmentation de variance intra-cluster

**Avantage** : pas de k à fixer a priori ; le dendrogramme offre une vision multi-échelle. **Inconvénient** : complexité O(n² log n) à O(n³), impraticable pour n > 10⁵.

### 3. Clustering par densité — DBSCAN / HDBSCAN

**Principe** : un cluster est une région de l'espace où la densité de points dépasse un seuil. Deux hyperparamètres : ε (rayon de voisinage) et minPts (nombre minimal de voisins pour être un *core point*).

**HDBSCAN** (Campello, Moulavi & Sander, 2013) étend DBSCAN en construisant une hiérarchie de clusters par densité, éliminant la sensibilité au paramètre ε.

> Campello, R.J.G.B., Moulavi, D. & Sander, J. (2013). *Density-based clustering based on hierarchical density estimates*. [In *Proceedings of PAKDD 2013*, Lecture Notes in Computer Science, vol. 7819. Springer](https://doi.org/10.1007/978-3-642-37456-2_14).

**Avantages** : détecte des clusters de forme arbitraire, identifie les outliers. **Limites** : difficultés avec des densités variables.

### 4. Modèles de mélange — GMM / EM

**Principe** : modéliser les données comme un mélange de k distributions gaussiennes. L'algorithme EM estime les paramètres (moyennes μₖ, matrices de covariance Σₖ, poids πₖ) par maximisation de la vraisemblance.

**Avantage sur K-means** : les affectations sont *soft* (probabilistes), les clusters peuvent être ellipsoïdaux et de tailles différentes. **Limite** : sensible au nombre de composantes et aux mauvaises initialisations ; risque de dégénérescence.

### 5. Clustering spectral

**Principe** : construire un graphe de similarité sur les données, calculer les k premiers vecteurs propres du Laplacien normalisé, puis appliquer K-means dans cet espace spectral de dimension réduite.

**Avantage** : capture des structures non convexes (cercles concentriques, spirales). **Limite** : complexité O(n³) pour la décomposition propre — ne passe pas à l'échelle sans approximations (Nyström, etc.).

---

## Évaluation d'un clustering

En l'absence de labels de référence (**métriques internes**) :
* **Indice de silhouette** (Rousseeuw, 1987) : compare la cohésion intra-cluster et la séparation inter-cluster. Valeur ∈ [−1, 1].
* **Indice de Davies-Bouldin** : ratio moyen dissimilarité intra / inter-cluster. Plus petit = meilleur.
* **Indice de Calinski-Harabasz** : ratio variance inter / intra. Plus grand = meilleur.

En présence de labels de référence (**métriques externes**) :
* **Rand Index** (Rand, 1971) et son ajustement ARI
* **Normalized Mutual Information** (NMI)
* **V-measure** (Rosenberg & Hirschberg, 2007)

> Rousseeuw, P.J. (1987). *Silhouettes: A graphical aid to the interpretation and validation of cluster analysis*. **Journal of Computational and Applied Mathematics**, 20, 53–65. https://doi.org/10.1016/0377-0427(87)90125-7


## Cas d'utilisation

Domaine | Application | Algorithme typique
---|---|---
Marketing | Segmentation clients (RFM, comportement) | K-means, GMM |
Bioinformatique | Clustering de gènes, sous-types tumoraux | Hiérarchique (Ward), spectral
Vision par ordinateur | Quantification vectorielle, Bag of Visual Words | K-means |
NLP | Topic clustering de documents, détection de paraphrases | K-means, HDBSCAN sur embeddings
Cybersécurité | Détection d'anomalies réseau | DBSCAN, HDBSCAN |
Astronomie | Classification de galaxies, détection de structures cosmiques | GMM, DBSCAN
Géospatial | Détection de zones urbaines, analyse de mobilité | DBSCAN, K-means |
Recommandation | Clustering d'utilisateurs pour collaborative filtering | K-means, GMM

---

## Boîte blanche, grise ou noire ?

Le clustering est majoritairement du côté de la **boîte blanche**, avec des nuances selon l'algorithme.

**Arguments pour la boîte blanche :**
* K-means, DBSCAN et le clustering hiérarchique ont des règles d'affectation **entièrement explicites et déterministes** (à l'initialisation près). On peut retracer l'appartenance de chaque point à son cluster étape par étape.
* Les centroïdes K-means sont des vecteurs dans l'espace des données : interprétables directement (profil moyen d'un segment client, spectre moyen d'un type d'étoile, etc.).
* Le dendrogramme hiérarchique offre une **traçabilité complète** de l'histoire des fusions.

**Nuances vers la boîte grise :**
* Le **clustering spectral** implique une transformation implicite de l'espace via les vecteurs propres du Laplacien, rendant les clusters moins directement interprétables dans l'espace original.
* Les **GMM** avec covariances complètes ont des paramètres interprétables (moyenne, covariance), mais l'espace de décision résultant peut être complexe.
* La **sensibilité à l'initialisation** (K-means) et au choix des hyperparamètres (ε dans DBSCAN) introduit une variabilité qui opacifie partiellement la reproductibilité.

En synthèse : pour K-means, le clustering hiérarchique et DBSCAN, l'**auditabilité est totale**. Pour le clustering spectral ou les mélanges gaussiens à haute dimension, on glisse vers une boîte grise claire — la mécanique reste formellement transparente, mais l'interprétation des résultats exige une expertise supplémentaire.

## Limites générales
* **Le nombre de clusters k** est rarement connu a priori. Les heuristiques (méthode du coude, silhouette, BIC pour les GMM) aident mais ne tranchent pas toujours clairement.
* **La malédiction de la dimensionnalité** : en haute dimension, toutes les distances tendent à être égales (concentration de la mesure), rendant les notions de proximité et de densité quasi-inopérantes. Une réduction de dimension préalable (PCA, UMAP, t-SNE) est souvent nécessaire.
* **Le clustering n'est pas une vérité absolue** : la partition obtenue dépend de la métrique choisie, de l'algorithme et de ses hyperparamètres. Deux algorithmes différents sur les mêmes données peuvent produire des résultats radicalement distincts, tous deux valides selon leur critère propre.

---

## Références complètes

1. Sokal, R.R. & Michener, C.D. (1958). *A statistical method for evaluating systematic relationships*. **University of Kansas Science Bulletin**, 38, 1409–1438.
2. [Lloyd, S.P. (1982). *Least squares quantization in PCM*. **IEEE Transactions on Information Theory**, 28(2), 129–137](https://doi.org/10.1109/TIT.1982.1056489).
3. MacQueen, J. (1967). *Some methods for classification and analysis of multivariate observations*. In *Proceedings of the 5th Berkeley Symposium*, vol. 1, 281–297. University of California Press.
4. [Ward, J.H. (1963). *Hierarchical grouping to optimize an objective function*. **Journal of the American Statistical Association**, 58(301), 236–244](https://doi.org/10.1080/01621459.1963.10500845).
5. [Ester, M., Kriegel, H.-P., Sander, J. & Xu, X. (1996). *A density-based algorithm for discovering clusters in large spatial databases with noise*. In *Proceedings of KDD 1996*, 226–231](https://dl.acm.org/doi/10.5555/3001460.3001507).
6. [Dempster, A.P., Laird, N.M. & Rubin, D.B. (1977). *Maximum likelihood from incomplete data via the EM algorithm*. **Journal of the Royal Statistical Society, Series B**, 39(1), 1–38](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x).
7. [Ng, A.Y., Jordan, M.I. & Weiss, Y. (2001). *On spectral clustering: Analysis and an algorithm*. **NeurIPS 14**, 849–856](https://proceedings.neurips.cc/paper/2001/hash/801272ee79cfde7fa5960571fee36b9b-Abstract.html).
8. [Shi, J. & Malik, J. (2000). *Normalized cuts and image segmentation*. **IEEE Transactions on Pattern Analysis and Machine Intelligence**, 22(8), 888–905](https://doi.org/10.1109/34.868688).
9. [Arthur, D. & Vassilvitskii, S. (2007). *K-means++: The advantages of careful seeding*. In *Proceedings of SODA 2007*, 1027–1035](https://dl.acm.org/doi/10.5555/1283383.1283494).
10. [Campello, R.J.G.B., Moulavi, D. & Sander, J. (2013). *Density-based clustering based on hierarchical density estimates*. **PAKDD 2013**, LNCS vol. 7819](https://doi.org/10.1007/978-3-642-37456-2_14).
11. [Rousseeuw, P.J. (1987). *Silhouettes: A graphical aid to the interpretation and validation of cluster analysis*. **Journal of Computational and Applied Mathematics**, 20, 53–65 ](https://doi.org/10.1016/0377-0427(87)90125-7).
___
[← Retour](../../../)