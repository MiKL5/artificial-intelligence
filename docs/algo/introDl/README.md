# Introduction à la Représentation Parcimonieuse et aux Dictionnaires<a href="../../"><img src="../../../assets/images/atomicAi.png" alt="L'intelligence artificielle" align="right" height="64px"></a>

<!-- > Les fondations conceptuelles et mathématiques pour aborder sereinement l'apprentissage de dictionnaire. -->
> Ce document constitue le **préalable naturel** au [comparatif](../dlVsOdl/).

___

## Sommaire

1. [Pourquoi représenter un signal ?](#1-pourquoi-représenter-un-signal-)
2. [Bases orthonormées — puissance et rigidité](#2-bases-orthonormées--puissance-et-rigidité)
3. [Redondance voulue — les trames (*frames*)](#3-redondance-voulue--les-trames-frames)
4. [Dictionnaires surcomplètes et parcimonie](#4-dictionnaires-surcomplètes-et-parcimonie)
5. [Le problème du codage parcimonieux](#5-le-problème-du-codage-parcimonieux)
   - [Formulation NP-difficile exacte](#51-formulation-np-difficile-exacte)
   - [Relaxation convexe — le Lasso](#52-relaxation-convexe--le-lasso)
   - [Approches gloutonnes — Matching Pursuit et OMP](#53-approches-gloutonnes--matching-pursuit-et-omp)
   - [Conditions de recouvrabilité — RIP et cohérence](#54-conditions-de-recouvrabilité--rip-et-cohérence)
6. [Dictionnaires analytiques vs dictionnaires appris](#6-dictionnaires-analytiques-vs-dictionnaires-appris)
7. [L'hypothèse sous-jacente — variétés et structure des données](#7-lhypothèse-sous-jacente--variétés-et-structure-des-données)
8. [La question que tout cela soulève](#8-la-question-que-tout-cela-soulève)

___

## 1. Pourquoi représenter un signal ?

Un signal numérique brut — une image en niveaux de gris de 256 × 256 pixels, un fragment audio échantillonné à 44 kHz, un spectre de masse — est un vecteur dans $\mathbb{R}^d$ avec $d$ potentiellement très grand. Dans cet espace ambiant, chaque coordonnée est un **pixel**, un **échantillon temporel**, une **fréquence de masse** : des grandeurs physiques sans relation naturelle entre elles.

Or, les signaux *réels* ne peuplent pas $\mathbb{R}^d$ de façon uniforme. Deux vérités empiriques s'imposent :

> **Observation 1 — Compressibilité.** La majorité de l'énergie d'un signal naturel est concentrée dans un petit nombre de coefficients dès lors qu'on choisit une représentation adaptée. Un portrait photographique tient en quelques kilooctets en JPEG, une symphonie en quelques mégaoctets en MP3 — sans perte perceptible.

> **Observation 2 — Structure locale.** Les signaux physiques possèdent des régularités — continuité, périodicité, symétrie, répétition de motifs — que l'espace ambiant est aveugle à voir mais qu'un changement de coordonnées peut rendre triviales.

**Changer de coordonnées**, c'est multiplier le vecteur $\mathbf{x} \in \mathbb{R}^d$ par une matrice $\mathbf{D} \in \mathbb{R}^{d \times K}$ pour obtenir des coefficients $\boldsymbol{\alpha} \in \mathbb{R}^K$ tels que $\mathbf{x} \approx \mathbf{D}\boldsymbol{\alpha}$. Le choix de $\mathbf{D}$ est l'objet de toute cette théorie.

___

## 2. Bases orthonormées — puissance et rigidité

La première idée, classique, est d'employer une **base orthonormée** : $K = d$ atomes $\{\mathbf{d}_k\}_{k=1}^d$ deux à deux orthogonaux et normalisés. La représentation est alors exacte et unique :

$$\boldsymbol{\alpha} = \mathbf{D}^\top \mathbf{x}, \qquad \mathbf{x} = \mathbf{D}\boldsymbol{\alpha}$$

Trois bases ont marqué l'histoire du traitement du signal :

Base | Atomes | Atout | Limitation
---|---|---|---
**DCT** (cosinus discrète) | Cosinus à fréquence croissante | Énergie concentrée sur images douces | Artefacts de bloc, mauvaise adaptation aux discontinuités
**Ondelettes de Haar/Daubechies** | Fonctions localisées temps-fréquence | Multi-résolution, discontinuités bien représentées | Base fixe, non adaptée au contenu
**Fourier discret (DFT)** | Exponentielles complexes | Calcul rapide (FFT), convolution triviale | Atomes globaux : mauvaise localisation temporelle

**La limite fondamentale d'une base** : la représentation est *unique*. Si les données ne s'alignent pas naturellement avec les directions de la base, les coefficients seront nombreux et diffus. Une image texturée, un signal physiologique non stationnaire, une molécule chimique : aucune base universelle ne les rendra simultanément parcimonieux.

___

## 3. Redondance voulue — les trames (*frames*)

La théorie des **trames** (*frames*), développée par Duffin et Schaeffer (1952) puis popularisée en traitement du signal par Daubechies, Mallat et Coifman, généralise la notion de base en autorisant $K > d$ : davantage d'atomes que de dimensions.

Un ensemble $\{\mathbf{d}_k\}_{k=1}^K$ forme une **trame serrée** (*tight frame*) s'il existe $A > 0$ tel que :

$$\forall \mathbf{x} \in \mathbb{R}^d, \qquad A\|\mathbf{x}\|_2^2 = \sum_{k=1}^{K} \langle \mathbf{x}, \mathbf{d}_k \rangle^2$$

La **redondance** $r = K/d > 1$ a deux effets contraires :

* ✅ **Stabilité** : avec plus d'atomes disponibles, un signal peut être représenté de multiples façons, et la représentation est robuste au bruit ou aux occultations.
* ⚠️ **Non-unicité** : la représentation n'est plus unique. L'espace des solutions $\{\boldsymbol{\alpha} : \mathbf{D}\boldsymbol{\alpha} = \mathbf{x}\}$ est un sous-espace affine de dimension $K - d > 0$.

C'est précisément cette non-unicité qui ouvre la porte à la parcimonie : parmi les infiniment nombreuses solutions, on cherche celle qui est la *plus creuse*.

___

## 4. Dictionnaires surcomplètes et parcimonie

Un **dictionnaire surcomplet** est une matrice $\mathbf{D} \in \mathbb{R}^{d \times K}$ avec $K \gg d$, dont les colonnes $\mathbf{d}_k$ (les **atomes**) sont normalisées : $\|\mathbf{d}_k\|_2 = 1$. On parle de dictionnaire **surcomplète** (*overcomplete*) ou **redondant**.

L'idée centrale peut s'énoncer simplement :

> Un signal appartenant à une certaine classe (visages humains, textures géologiques, molécules d'une famille chimique) est bien approximé par une **combinaison linéaire de quelques atomes** soigneusement choisis dans un dictionnaire suffisamment riche. Les coefficients non nuls sont peu nombreux : c'est la **parcimonie** (*sparsity*).

**Pourquoi la parcimonie est-elle désirable ?**

1. **Compression** : stocker $s \ll K$ coefficients non nuls et leurs indices coûte $\mathcal{O}(s \log K)$ bits au lieu de $\mathcal{O}(d)$.
2. **Débruitage** : le bruit, diffus dans $\mathbb{R}^d$, se retrouve réparti sur tous les coefficients, tandis que le signal utile se concentre sur $s$ ; un simple seuillage sépare les deux.
3. **Détection et classification** : la signature parcimonieuse d'un signal est une empreinte discriminante, souvent plus robuste que le vecteur brut.
4. **Inférence causale** : en biologie des systèmes ou en économétrie, un modèle parcimonieux est plus interprétable et moins sujet au sur-ajustement.

___

## 5. Le problème du codage parcimonieux

Étant donné un dictionnaire $\mathbf{D}$ **fixé** et une observation $\mathbf{x} \in \mathbb{R}^d$, trouver une représentation parcimonieuse revient à résoudre ce qu'on appelle le **problème du codage parcimonieux** (*sparse coding* ou *sparse approximation*).

### 5.1 Formulation NP-difficile exacte

La formulation la plus directe impose le nombre de coefficients non nuls via la pseudo-norme $\ell_0$ :

$$(\mathcal{P}_0) \qquad \min_{\boldsymbol{\alpha} \in \mathbb{R}^K} \|\boldsymbol{\alpha}\|_0 \quad \text{s.c.} \quad \|\mathbf{x} - \mathbf{D}\boldsymbol{\alpha}\|_2 \leq \varepsilon$$

où $\|\boldsymbol{\alpha}\|_0 = |\{k : \alpha_k \neq 0\}|$ compte le nombre de coordonnées non nulles.

Ce problème est **NP-difficile** dans le cas général (réduction au problème de couverture exacte), ce qui interdit toute résolution exacte en temps polynomial à grande dimension. La combinatoire sous-jacente est celle d'un choix parmi $\binom{K}{s}$ sous-ensembles possibles d'atomes.

___

### 5.2 Relaxation convexe — le Lasso

La solution la plus élégante pour contourner la NP-difficulté consiste à **relaxer** $\|\cdot\|_0$ en $\|\cdot\|_1$, la seule norme $\ell_p$ avec $p \leq 1$ qui soit convexe :

$$(\mathcal{P}_1) \qquad \min_{\boldsymbol{\alpha} \in \mathbb{R}^K} \frac{1}{2}\|\mathbf{x} - \mathbf{D}\boldsymbol{\alpha}\|_2^2 + \lambda\|\boldsymbol{\alpha}\|_1$$

Il s'agit du **Lasso** (*Least Absolute Shrinkage and Selection Operator*), introduit par Tibshirani (1996). Géométriquement, la boule unité $\ell_1$ (un hyperoctaèdre) possède des **sommets sur les axes** : l'optimum a donc tendance à annuler des coordonnées, favorisant la parcimonie.

**Solveurs majeurs :**

* **LARS** (*Least Angle Regression*, Efron et al., 2004) : parcourt le chemin de régularisation complet en $\mathcal{O}(Kd^2)$. Exact, efficace quand $s$ est modéré.
* **ISTA** (*Iterative Shrinkage-Thresholding Algorithm*) : descente de gradient proximal. Très simple à implémenter, convergence en $\mathcal{O}(1/t)$.
* **FISTA** (*Fast ISTA*, Beck & Teboulle, 2009) : ajoute un pas de momentum à ISTA, convergence en $\mathcal{O}(1/t^2)$ — optimal pour les méthodes du premier ordre.

La clé de ces algorithmes est l'**opérateur de seuillage doux** (*soft thresholding*) :

$$\mathcal{S}_\lambda(\alpha) = \operatorname{sgn}(\alpha) \cdot \max(|\alpha| - \lambda, 0)$$

qui est l'opérateur proximal de $\lambda\|\cdot\|_1$ et constitue la brique élémentaire de ISTA/FISTA.

___

### 5.3 Approches gloutonnes — Matching Pursuit et OMP

Lorsque l'on souhaite contrôler *directement* le nombre $s$ de coefficients non nuls plutôt que le paramètre $\lambda$, les algorithmes de **poursuite** (*pursuit*) sont plus naturels. Ils construisent le support actif $\mathcal{S}$ de façon incrémentale.

#### Matching Pursuit (MP) — Mallat & Zhang, 1993

```
MP(x, D, s)
───────────
r_0 ← x                               # résidu initial
S ← ∅                                  # support actif

Pour t = 1 à s :
    k* ← arg max_k |⟨r_{t-1}, d_k⟩|  # atome le plus corrélé
    S ← S ∪ {k*}
    α_{k*} ← α_{k*} + ⟨r_{t-1}, d_k*⟩
    r_t ← r_{t-1} - ⟨r_{t-1}, d_k*⟩ · d_k*

Retourner α
```

MP est rapide mais **ne converge pas vers le minimum** des moindres carrés sur $\mathcal{S}$ : le résidu n'est pas orthogonal aux atomes sélectionnés.

#### Orthogonal Matching Pursuit (OMP) — Pati et al., 1993 ; Tropp & Gilbert, 2007

OMP corrige ce défaut en ajoutant une **projection orthogonale** sur le sous-espace engendré par les atomes sélectionnés :

```
OMP(x, D, s)
────────────
r_0 ← x
S ← ∅

Pour t = 1 à s :
    k* ← arg max_k |⟨r_{t-1}, d_k⟩|
    S ← S ∪ {k*}
    
    // Projection sur span(D_S) — résolution par moindres carrés
    α_S ← (D_S^T D_S)^{-1} D_S^T x
    r_t ← x - D_S α_S               # résidu orthogonal à D_S

Retourner α
```

**Propriété** : à chaque itération $t$, $r_t \perp \mathbf{d}_{k}$ pour tout $k \in \mathcal{S}$. La résolution des moindres carrés coûte $\mathcal{O}(s^2 d)$, rendant OMP légèrement plus lourd que MP mais nettement plus précis.

**Coût total** : $\mathcal{O}(sKd)$ corrélations + $\mathcal{O}(s^3)$ pour les inversions triangulaires via la factorisation de Cholesky mise à jour incrémentalement.

---

### 5.4 Conditions de recouvrabilité — RIP et cohérence

Deux objets théoriques garantissent qu'une solution parcimonieuse peut être *retrouvée* par relaxation convexe ou poursuite gloutonne.

**Cohérence mutuelle** (*mutual coherence*) :

$$\mu(\mathbf{D}) = \max_{j \neq k} |\langle \mathbf{d}_j, \mathbf{d}_k \rangle|$$

Mesure à quel point les atomes sont corrélés. Un dictionnaire avec $\mu$ petit permet une identification fiable. La borne de Welch donne le minimum théorique :

$$\mu \geq \sqrt{\frac{K-d}{d(K-1)}}$$

**Propriété d'isométrie restreinte** (*RIP*, Candès & Tao, 2005) : $\mathbf{D}$ vérifie la RIP d'ordre $s$ avec constante $\delta_s$ si :

$$(1-\delta_s)\|\boldsymbol{\alpha}\|_2^2 \leq \|\mathbf{D}\boldsymbol{\alpha}\|_2^2 \leq (1+\delta_s)\|\boldsymbol{\alpha}\|_2^2$$

pour tout $\boldsymbol{\alpha}$ $s$-parcimonieux. Lorsque $\delta_{2s} < \sqrt{2} - 1$, la relaxation $\ell_1$ retrouve exactement la solution $\ell_0$ en l'absence de bruit. C'est le théorème fondamental du **compressed sensing**.

Ces deux conditions expliquent pourquoi la conception du dictionnaire n'est pas anodine : un dictionnaire trop cohérent rend les problèmes de codage mal conditionnés.

___

## 6. Dictionnaires analytiques vs dictionnaires appris

Jusqu'ici, nous avons supposé $\mathbf{D}$ **donné**. Deux philosophies coexistent.

### Dictionnaires analytiques (fixés a priori)

Construits à partir de considérations mathématiques ou physiques, **indépendants des données** :

Dictionnaire | Atomes | Domaine de prédilection
---|---|---
**DCT-II** | Cosinus multi-fréquences | Compression d'images (JPEG)
**Ondelettes** (Haar, Daubechies, sym8…) | Formes oscillantes localisées | Audio, imagerie médicale
**Gabor / STFT** | Gaussiennes modulées | Analyse temps-fréquence
**Curvelets / Shearlets** | Atomes anisotropes | Arêtes et contours d'images
**DCT-3D** | Extension spatio-temporelle | Vidéo, IRM fonctionnelle

**Avantages** : transformation rapide (FFT, algorithmes dédiés), garanties théoriques solides, pas d'entraînement requis.

**Limites** : les atomes sont conçus pour une *classe générique* de signaux. Pour un corpus spécifique — IRM cérébrales d'une population précise, spectres de diffraction d'une famille de cristaux, patches de peau saine vs lésée — un dictionnaire universel laisse une part non négligeable de la structure non exploitée.

### Dictionnaires appris (*data-driven*)

L'idée est de **laisser les données dicter la forme des atomes**. Si l'on dispose d'un corpus représentatif, on peut optimiser $\mathbf{D}$ pour minimiser l'erreur de reconstruction parcimonieuse sur ce corpus précis.

**Avantages** :
* Les atomes reflètent les **structures récurrentes** des données (Olshausen & Field, 1996, ont montré que des atomes appris sur des images naturelles ressemblent aux champs récepteurs des cellules simples du cortex visuel V1 — des barres orientées et localisées, semblables aux ondelettes de Gabor, mais *émergentes* plutôt qu'imposées).
* Performances supérieures en débruitage, restauration, et classification sur des corpus spécialisés.

**Coût** : il faut résoudre un problème d'optimisation non convexe sur l'ensemble des données — c'est précisément le sujet du [README comparatif](./README.md).

```
Comparaison schématique de la qualité de représentation
(erreur de reconstruction pour même parcimonie s)

Erreur
  │
  │  ████  Dictionnaire DCT
  │  ████
  │  ████  ████  Ondelettes
  │  ████  ████
  │  ████  ████  ████  Dictionnaire appris (batch)
  │  ████  ████  ████
  │  ████  ████  ████  ████  Dictionnaire appris (online, n→∞)
  │  ████  ████  ████  ████
  └─────────────────────────────────────────────────────────▶  n
                              (taille du corpus d'apprentissage)
```

___

## 7. L'hypothèse sous-jacente — variétés et structure des données

Pourquoi un dictionnaire appris fait-il mieux ? La réponse tient à l'**hypothèse de variété** (*manifold hypothesis*) :

> Les signaux naturels d'une classe donnée ne remplissent pas $\mathbb{R}^d$ uniformément. Ils s'organisent autour d'une **variété de faible dimension intrinsèque** $\mathcal{M} \hookrightarrow \mathbb{R}^d$, avec $\dim \mathcal{M} \ll d$.

Pour les images de visages ($d = 4096$ pixels typiquement), la dimension intrinsèque est estimée entre 20 et 100 selon les études. Pour les patches d'images naturelles de taille 8×8, elle est de l'ordre de 5 à 10.

Un dictionnaire parcimonieux est une **approximation linéaire par morceaux** de cette variété : chaque atome couvre une direction locale de $\mathcal{M}$, et une observation $\mathbf{x}$ sur $\mathcal{M}$ est décrite par les quelques atomes couvrant son voisinage.

Cette connexion relie l'apprentissage de dictionnaire à d'autres méthodes de réduction dimensionnelle non linéaire (Isomap, LLE, t-SNE), mais avec la différence décisive que la représentation reste **dans l'espace ambiant $\mathbb{R}^d$**, exploitable directement pour la reconstruction et la classification.

___

## 8. La question que tout cela soulève

Nous avons maintenant tous les ingrédients pour formuler le problème qui est au cœur du [README suivant](./README.md) :

> **Comment optimiser conjointement** le dictionnaire $\mathbf{D}$ *et* les codes $\boldsymbol{\alpha}_1, \ldots, \boldsymbol{\alpha}_n$ pour minimiser l'erreur de reconstruction globale, avec une contrainte de parcimonie sur chaque vecteur de coefficients ?

$$\min_{\mathbf{D} \in \mathcal{C},\, \mathbf{A}} \quad \frac{1}{n}\sum_{i=1}^{n} \left( \frac{1}{2}\|\mathbf{x}_i - \mathbf{D}\boldsymbol{\alpha}_i\|_2^2 + \lambda\|\boldsymbol{\alpha}_i\|_1 \right)$$

Deux familles d'algorithmes répondent à cette interrogation de façon radicalement différente selon leur rapport aux données :

- **L'approche par lots** (*batch*) — K-SVD, MOD : l'intégralité du corpus est nécessaire à chaque révision du dictionnaire.
- **L'approche en ligne** (*online*) — Mairal et al., 2009 : le dictionnaire évolue au fil des observations, sans jamais recharger le corpus complet.

→ **[Lire le README comparatif pour la suite →](./README.md)**

___

> Références
> * **Duffin, R. J. & Schaeffer, A. C.** (1952). *A class of nonharmonic Fourier series*. Trans. AMS. — Naissance de la théorie des trames.
> * **Mallat, S. & Zhang, Z.** (1993). *Matching Pursuits with Time-Frequency Dictionaries*. IEEE TSP. — Algorithme MP.
> * **Tibshirani, R.** (1996). *Regression Shrinkage and Selection via the Lasso*. JRSS-B. — Fondation du Lasso.
> * **Olshausen, B. & Field, D.** (1996). *Emergence of Simple-Cell Receptive Field Properties by Learning a Sparse Code for Natural Images*. Nature. — Première preuve que les atomes *émergent* naturellement des images naturelles.
> * **Tropp, J. & Gilbert, A.** (2007). *Signal Recovery from Random Measurements via Orthogonal Matching Pursuit*. IEEE TIT. — Garanties de recouvrabilité pour OMP.
> * **Candès, E. & Tao, T.** (2005). *Decoding by Linear Programming*. IEEE TIT. — Introduction de la RIP.
> * **Beck, A. & Teboulle, M.** (2009). *A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems*. SIAM J. Imaging Sci. — FISTA.
> * **Elad, M.** (2010). *Sparse and Redundant Representations*. Springer. — Ouvrage de référence complet sur le sujet.

___

*Lire ce document de bout en bout prend environ 25 minutes. Il constitue le socle conceptuel indispensable pour aborder les algorithmes d'apprentissage de dictionnaire sans zone d'ombre.*