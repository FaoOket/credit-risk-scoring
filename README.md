# Scoring de risque crédit — *Give Me Some Credit* (Kaggle)

Pipeline complet de machine learning sur des **données réelles** pour prédire
le risque de défaut de paiement de clients fintech/bancaires.

---

## Contexte et objectif

Prédire la variable binaire `SeriousDlqin2yrs` (1 = incident de paiement grave dans les 2 ans,
0 = aucun incident) à partir du profil financier d'un client.

Il s'agit d'un problème classique de **scoring crédit**, utilisé par les banques et établissements
de crédit pour décider d'accorder ou non un prêt, et à quel taux.

---

## Données réelles

**Source :** [Give Me Some Credit — Kaggle](https://www.kaggle.com/competitions/GiveMeSomeCredit/data)
— dataset de référence du scoring crédit, utilisé en compétition internationale.

> **Pour reproduire :** télécharger `cs-training.csv` depuis la page Kaggle ci-dessus
> et le placer dans `data/cs-training.csv`. Le fichier n'est pas versionné
> (licence Kaggle, redistribution non autorisée).

**150 000 clients réels — taux de défaut : 6.68 %**

| Variable | Description |
|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | Taux d'utilisation des lignes de crédit renouvelable |
| `age` | Âge du client |
| `NumberOfTime30-59DaysPastDueNotWorse` | Retards de paiement 30–59 jours (12–24 mois) |
| `DebtRatio` | Ratio d'endettement (charges / revenus) |
| `MonthlyIncome` | Revenu mensuel (€) — ~20 % de valeurs manquantes |
| `NumberOfOpenCreditLinesAndLoans` | Nombre de crédits ouverts |
| `NumberOfTimes90DaysLate` | Retards > 90 jours |
| `NumberRealEstateLoansOrLines` | Nombre de crédits immobiliers |
| `NumberOfTime60-89DaysPastDueNotWorse` | Retards de paiement 60–89 jours |
| `NumberOfDependents` | Nombre de personnes à charge — ~2 % de valeurs manquantes |

---

## Nettoyage des données

Trois défauts classiques des données réelles corrigés avant modélisation :

1. **Valeurs manquantes** — `MonthlyIncome` (~20 %) et `NumberOfDependents` (~2 %) :
   imputées par la **médiane dans le pipeline** (aucune fuite de données vers le test).
2. **Codes sentinelles 96/98** — colonnes de retards de paiement : valeurs techniques
   sans signification → plafonnées à 20.
3. **Valeurs extrêmes** — `RevolvingUtilizationOfUnsecuredLines` et `DebtRatio` :
   plafonnées au **99ᵉ centile** pour neutraliser les saisies aberrantes.
4. **Âge** : plancher à 18 ans.

---

## Résultats

| Modèle | AUC (test) | AUC (CV 5-fold) | F1-score |
|---|---|---|---|
| **Gradient Boosting** *(retenu par AUC)* | **0.8676** | **0.8632** | 0.3019 |
| Random Forest *(meilleur F1)* | 0.8646 | 0.8603 | **0.3301** |
| Régression logistique | 0.8556 | 0.8506 | 0.3273 |
| Baseline (classe majoritaire) | 0.5000 | 0.5000 | 0.0000 |

**Meilleur modèle : Gradient Boosting** — sélectionné selon l'AUC (test).
Gain de **+0.3676** par rapport à la baseline aléatoire.

> Le Random Forest obtient légèrement le meilleur F1-score (0.3301 vs 0.3019).
> Le choix entre les deux dépend de l'arbitrage métier : AUC pour la qualité
> de classement globale, F1 pour l'équilibre précision/rappel sur les défauts.

> **Rappel :** l'AUC est un score entre 0 (inversé) et **1 (parfait)**, 0,50 correspondant
> au hasard pur. Ce n'est pas un pourcentage de bonnes réponses.

---

## Segmentation Power BI

Les probabilités de défaut sont découpées en 3 niveaux de risque par **tertiles**
calculés sur le jeu d'entraînement uniquement (aucune fuite vers le test) :

| Segment | Clients test (n=37 500) |
|---|---|
| Faible risque | ≈ 12 480 |
| Risque moyen | ≈ 12 457 |
| Risque élevé | ≈ 12 563 |

> Ces segments sont **relatifs** au portefeuille : ils classent les clients les uns
> par rapport aux autres. Le seuil entre segments varie si le portefeuille change.

---

## Méthodologie

```
Données réelles (150 000 clients, Kaggle)
        ↓
Nettoyage — valeurs manquantes, codes 96/98, valeurs extrêmes
        ↓
EDA — distribution de la cible, corrélations
        ↓
Prétraitement dans pipeline — imputation médiane + StandardScaler
        ↓
Entraînement — 3 modèles + baseline, validation croisée 5-fold (train uniquement)
        ↓
Évaluation — AUC ROC, F1-score, matrice de confusion (test)
        ↓
Export — probabilités + segments de risque → Power BI (non versionné)
```

---

## Structure du projet

```
credit-risk-scoring/
├── notebook/
│   └── Scoring_GiveMeSomeCredit.ipynb    # Pipeline complet avec sorties
├── data/
│   └── cs-training.csv                   # À télécharger depuis Kaggle (non versionné)
├── figures/
│   ├── fig_target.png
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

# 2. Télécharger cs-training.csv depuis Kaggle → data/cs-training.csv

# 3. Exécuter le notebook
jupyter notebook notebook/Scoring_GiveMeSomeCredit.ipynb
```

---

## Stack technique

- **Python 3.10+**
- **pandas / numpy** — manipulation et nettoyage des données
- **scikit-learn** — Pipeline, SimpleImputer, ColumnTransformer, modèles ML, métriques
- **matplotlib / seaborn** — visualisations
- **Jupyter Notebook** — environnement interactif
- **Power BI** — dashboard de suivi du risque (export CSV)

