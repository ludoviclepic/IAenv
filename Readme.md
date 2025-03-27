# 🌾 UNet 3D pour la Classification de Séries Temporelles Agricoles

## Présentation

Ce projet propose une extension de l'article **"Pixel-wise Agricultural Image Time Series Classification"** de Vincent et al.  
Plutôt que de modifier la méthode originale, nous avons conçu une **approche complémentaire** à l’aide d’un **UNet 3D**, pour classifier les cultures à partir d’images satellites multi-temporelles. L’objectif est d’intégrer à la fois les dimensions **spatiales et temporelles**, afin d’assister la classification par prototypes en fournissant un contexte global.

---

## 🔬 Contribution scientifique

Nous introduisons une **architecture auxiliaire** basée sur un **UNet 3D**, capable d’extraire des cartes de segmentation spatio-temporelles. Celles-ci peuvent ensuite guider l'affectation des pixels aux prototypes, renforçant ainsi la **cohérence spatiale** et **l’interprétabilité** de la classification.

### Pourquoi un UNet 3D ?

- Exploite les **informations spatiales et temporelles** simultanément.
- Conserve une **interprétabilité forte**.
- **Convergence rapide** (~8 epochs).
- Faible coût d’entraînement, **parallélisable** et peu énergivore.
- Compatible avec des méthodes de classification par prototypes déformables.

---

## 🧼 Données et Prétraitement

Nous utilisons le jeu de données **PASTIS**, identique à celui de l'article de référence. Un **nettoyage rigoureux** a été effectué pour garantir la qualité des données.

### Méthodologie de nettoyage

- Calcul de la **médiane temporelle** de chaque pixel.
- Masquage des pixels déviant de plus de **3 écarts-types**.
- Suppression des timestamps où **>50 %** des pixels sont masqués.

---

## 🧠 Entraînement & Évaluation

Le UNet 3D prédit une carte 3D (pixels × temps × classes). Pour comparer aux annotations statiques, nous utilisons la **médiane temporelle** des prédictions.

- **Loss** : `CrossEntropyLoss` (avec pondération des classes)
- **Validation croisée** pour éviter l'overfitting
- **Convergence rapide** : 8 epochs

---

## 📊 Résultats

- **IoU** : ~40 % (vs 60 % dans l’état de l’art)
- Entraînement rapide et simple
- Compatible avec la méthode de prototypes pour un **contrôle multi-échelle** (local & global)

---

## 🧩 Vers une approche hybride

Notre objectif à terme : **fusionner** la segmentation par UNet 3D avec la classification par **prototypes déformables** :

- Le UNet fournit un **contexte spatial global**
- Les prototypes assurent une **classification fine et locale**

Cette combinaison permettrait d’obtenir un modèle **interprétable, robuste et rapide**.

---

## 🚀 Perspectives

- Intégration de **Transformers visuels** pour capter des dépendances globales (relief, proximité maritime...)
- Extension à des régions climatiques variées
- Approfondir l’explicabilité des décisions du modèle

---

## 📚 Référence

Ce projet est basé sur :

**Vincent, E., Ponce, J., & Aubry, M.**  
*Pixel-wise Agricultural Image Time Series Classification*  
📄 [arXiv 2303.12533](https://arxiv.org/abs/2303.12533)  
🔗 [Code officiel](https://github.com/ElliotVincent/AgriITSC)
