# Scoring de risque crédit — Fintech

Projet de data science pour prédire le risque de défaut de paiement de clients fintech.  
Pipeline complet : génération des données → EDA → prétraitement → modèles ML → évaluation → export Power BI.

---

## Contexte et objectif

Une entreprise fintech souhaite automatiser l'évaluation du risque crédit de ses clients.  
L'objectif est de prédire la variable binaire `default` (1 = défaut de paiement, 0 = remboursement normal) à partir de variables socio-démographiques et comportementales.

---

## Dataset

**Données synthétiques — 8 000 clients × 11 variables explicatives**

Le jeu de données a été généré synthétiquement (`scripts/01_generate_data.py`) à partir de distributions statistiques réalistes (log-normale, bêta, exponentielle, Poisson) et d'un modèle logistique calibré sur des pratiques métier bancaires/fintech standard.

> **Note importante :** Ce dataset ne correspond pas à un ancien fichier qui n'a pas pu être retrouvé. Il a été recréé de zéro à partir des statistiques disponibles et d'une logique métier réaliste. Les performances des modèles reflètent la séparabilité réelle de ces données synthétiques — elles n'ont pas été optimisées pour atteindre une AUC cible.

| Variable | Type | Description |
|---|---|---|
| `age` | int | Âge du client (18–74 ans) |
| `revenu_mensuel` | float | Revenu mensuel en € |
| `anciennete_compte_mois` | int | Ancienneté du compte en mois |
| `nb_transactions_mois` | int | Nombre de transactions par mois |
| `montant_moyen_transaction` | float | Montant moyen d'une transaction en € |
| `taux_endettement_pct` | float | Taux d'endettement en % |
| `nb_incidents_paiement_12m` | int | Incidents de paiement sur 12 mois |
| `montant_credit_demande` | float | Montant du crédit demandé en € |
| `type_contrat` | str | Type de contrat (CDI / CDD / Indépendant / Sans emploi) |
| `score_bureau_externe` | int | Score de crédit externe (324–850) |
| `utilisation_carte_pct` | float | Taux d'utilisation de la carte de crédit en % |
| `default` | int | **Cible** — 1 = défaut, 0 = remboursement |

**Statistiques clés :**
- Taux de défaut : **19.62 %** (1 570 défauts / 8 000 clients)
- Split entraînement / test : 6 000 / 2 000 (75 % / 25 %, stratifié)
- Graine fixe : `np.random.seed(42)` — résultats reproductibles

---

## Résultats

| Modèle | AUC (test) | AUC (cross-val 5-fold) | F1-score |
|---|---|---|---|
| **Régression Logistique** | **0.6554** | **0.6818** | **0.3724** |
| Random Forest | 0.6474 | 0.6684 | 0.3762 |
| Gradient Boosting | 0.6430 | 0.6672 | 0.0917 |
| Baseline (classe majoritaire) | 0.5000 | 0.5000 | 0.0000 |

**Meilleur modèle : Régression Logistique** — sélectionné selon l'AUC (test), avec un gain de +0.1554 par rapport à la baseline aléatoire.

> **À noter :** le Random Forest obtient légèrement le meilleur F1-score (0.3762 vs 0.3724 pour la Régression Logistique). En pratique, le choix entre les deux dépend de l'arbitrage métier : l'AUC mesure la qualité de classement global, le F1 reflète l'équilibre précision/rappel sur la classe minoritaire (défaut).

Tous les modèles utilisent `class_weight="balanced"` pour compenser le déséquilibre de classes (~80/20).

---

## Segmentation Power BI

Les probabilités de défaut sont découpées en **3 niveaux de risque** par tertiles calculés sur le jeu d'entraînement uniquement (aucune fuite de données test).

| Segment | Seuil | Clients (test, n=2 000) |
|---|---|---|
| Faible risque | proba < 0.388 | 668 |
| Risque moyen | 0.388 ≤ proba < 0.533 | 666 |
| Risque élevé | proba ≥ 0.533 | 666 |

> Ces segments sont **relatifs** : ils classent les clients les uns par rapport aux autres sur ce dataset synthétique. Ils ne correspondent pas à des probabilités absolues de défaut calibrées.

---

## Méthodologie

```
Génération des données
        ↓
Analyse exploratoire (EDA) — distributions, corrélations, taux de défaut par segment
        ↓
Prétraitement — StandardScaler (numériques) + OneHotEncoder (type_contrat)
        ↓
Entraînement — 3 modèles + baseline, validation croisée 5-fold (entraînement uniquement)
        ↓
Évaluation — AUC ROC, F1-score, matrice de confusion (jeu de test)
        ↓
Export — probabilités + segments de risque → Power BI
```

---

## Structure du projet

```
credit-risk-scoring/
├── notebook/
│   └── Scoring_Risque_Credit_Fintech.ipynb   # Pipeline complet
├── scripts/
│   └── 01_generate_data.py                   # Générateur du dataset synthétique
├── data/
│   ├── credit_scoring_dataset.csv            # Dataset (8 000 lignes)
│   └── export_powerbi_credit_scoring.csv     # Export avec probabilités et segments
├── figures/
│   ├── fig_correlation.png
│   ├── fig_roc_curves.png
│   ├── fig_confusion_matrix.png
│   └── fig_feature_importance.png
├── requirements.txt
└── README.md
```

---

## Reproduction

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. (Optionnel) Régénérer le dataset
python scripts/01_generate_data.py

# 3. Exécuter le notebook
jupyter notebook notebook/Scoring_Risque_Credit_Fintech.ipynb
```

---

## Stack technique

- **Python 3.10+**
- **pandas / numpy** — manipulation des données
- **scikit-learn** — Pipeline, ColumnTransformer, modèles ML, métriques
- **matplotlib / seaborn** — visualisations
- **Jupyter Notebook** — environnement interactif
- **Power BI** — dashboard de suivi du risque (données exportées en CSV)
