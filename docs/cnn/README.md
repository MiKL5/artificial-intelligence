# **Réseau de neurones à convolution**<a href="../"><img src="../../assets/images/atomicAi.png" alt="L'intelligence artificielle" align="right" height="64px"></a>
Les **réseaux de neurones à convolution** (*Convolutional Neural Networks*) constituent une classe de modèles de *deep learning* bio-inspirés, optimisés pour le traitement de données présentant une topologie en grille, telles que les images (2D) ou les signaux temporels (1D).
---
## Fondements Théoriques
L'architecture d'un CNN repose sur le principe de l'**invariance par translation** et de la **hiérarchie des traits**. Contrairement aux réseaux denses, ils exploitent la corrélation spatiale locale pour extraire des descripteurs de plus en plus abstraits.
### L'Opérateur de Convolution
L'opération fondamentale consiste à appliquer un filtre (ou noyau) $K$ sur une entrée $I$. Mathématiquement, pour chaque position $(i, j)$, la convolution est définie par le produit de Schur suivi d'une sommation :

$$(I * K)(i, j) = \sum_{m} \sum_{n} I(i + m, j + n) \cdot K(m, n)$$
### Caractéristiques de Conception
* **Champs Récepteurs Locaux :** Unité de traitement connectée uniquement à une portion restreinte de l'entrée.
* **Partage des Poids :** Les mêmes coefficients de filtre sont appliqués sur l'ensemble du volume d'entrée, réduisant drastiquement le nombre de paramètres.
* **Sous-échantillonnage (Pooling) :** Opération de réduction de résolution (ex: *Max-Pooling*) visant à assurer une robustesse face aux micro-déformations spatiales.

<div align="center"><br><hr><br><img src="../../assets/images/4doc/defineCnn.webp"><br><br><hr></div>

>Référence  
>L'évolution des CNN peut être tracée à travers ces publications fondamentales qui ont redéfini l'état de l'art en vision par ordinateur :
>1.  **LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).** *Gradient-based learning applied to document recognition.* Proceedings of the IEEE. 
    > *L'introduction de LeNet-5 et la démonstration de l'efficacité de la rétropropagation pour l'apprentissage de caractéristiques visuelles.*
>2.  **Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).** *ImageNet classification with deep convolutional neural networks.* Advances in Neural Information Processing Systems (NeurIPS). 
    > *La publication d'AlexNet, marquant le début de l'ère du Deep Learning grâce à l'utilisation des GPU.*
>3.  **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning.* MIT Press. 
    > *L'ouvrage de référence synthétisant les bases mathématiques et algorithmiques des réseaux de neurones.*
>4.  **He, K., Zhang, X., Ren, S., & Sun, J. (2016).** *Deep Residual Learning for Image Recognition.* CVPR. 
    > *L'introduction des connexions résiduelles (ResNets), permettant l'entraînement de réseaux dépassant les 100 couches de profondeur.*
___
[← Retour](../../)