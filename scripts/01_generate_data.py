"""
Génération du dataset synthétique de scoring de risque crédit.

Calibré sur les statistiques du notebook (describe()) et les proportions
du test set (export_powerbi_credit_scoring.csv, 2 000 lignes).
Graine fixe : np.random.seed(42) — reproduction exacte garantie.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ── Racine du projet ──────────────────────────────────────────────────────────
def _find_root():
    for p in [Path(__file__).resolve().parent.parent]:
        if (p / "data").exists() and (p / "notebook").exists():
            return p
    return Path(__file__).resolve().parent.parent

PROJECT_ROOT = _find_root()
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Paramètres ────────────────────────────────────────────────────────────────
N = 8_000
np.random.seed(42)

# ── Variables explicatives ────────────────────────────────────────────────────

# age : normale tronquée, entier, [18, 74]
age = np.clip(np.round(np.random.normal(34.7, 9.6, N)).astype(int), 18, 74)

# revenu_mensuel : log-normale (mean≈2010, std≈943), min=800
# sigma² = log(1 + (943/2010)²) ≈ 0.199  →  sigma≈0.446, mu≈7.507
revenu_mensuel = np.round(
    np.clip(np.random.lognormal(7.507, 0.446, N), 800, 14000), 2
)

# anciennete_compte_mois : exponentielle (mean≈24, std≈24), entier, [1, 225]
anciennete_compte_mois = np.clip(
    np.round(np.random.exponential(23.7, N)).astype(int), 1, 225
)

# nb_transactions_mois : normale, entier, [4, 35]
nb_transactions_mois = np.clip(
    np.round(np.random.normal(17.9, 4.2, N)).astype(int), 4, 35
)

# montant_moyen_transaction : log-normale (mean≈46, std≈42)
# sigma² = log(1 + (42/46)²) ≈ 0.607  →  sigma≈0.779, mu≈3.528
montant_moyen_transaction = np.round(
    np.clip(np.random.lognormal(3.528, 0.779, N), 5.0, 600.0), 2
)

# taux_endettement_pct : Beta(2, 5) × 100  (mean≈28.6 %, std≈15.9 %)
taux_endettement_pct = np.round(
    np.clip(np.random.beta(2, 5, N) * 100, 0.17, 89.60), 2
)

# nb_incidents_paiement_12m : Poisson(0.39), entier, [0, 5]
nb_incidents_paiement_12m = np.clip(
    np.random.poisson(0.39, N).astype(int), 0, 5
)

# montant_credit_demande : log-normale (mean≈4364, std≈2885)
# sigma² = log(1 + (2885/4364)²) ≈ 0.363  →  sigma≈0.603, mu≈8.199
montant_credit_demande = np.round(
    np.clip(np.random.lognormal(8.199, 0.603, N), 500.0, 40000.0), 2
)

# type_contrat : proportions issues du test set (2 000 lignes)
type_contrat = np.random.choice(
    ["CDI", "CDD", "Indépendant", "Sans emploi"],
    size=N,
    p=[0.549, 0.213, 0.141, 0.097],
)

# score_bureau_externe : normale(649, 89), entier, [324, 850]
score_bureau_externe = np.clip(
    np.round(np.random.normal(649, 88.6, N)).astype(int), 324, 850
)

# utilisation_carte_pct : Beta(2, 3) × 100  (mean≈40 %, std≈20 %)
utilisation_carte_pct = np.round(
    np.clip(np.random.beta(2, 3, N) * 100, 0.40, 97.43), 2
)

# ── Génération de la cible (défaut) ──────────────────────────────────────────
# Logit calibré sur les facteurs de risque métier :
#   - score_bureau élevé  → bon payeur (protection)
#   - taux_endettement    → sur-endettement = risque
#   - nb_incidents        → historique négatif = risque
#   - type_contrat        → stabilité emploi = protection partielle
#   - revenu              → capacité de remboursement = protection
#   - utilisation_carte   → signal comportemental léger

score_risk      = (850 - score_bureau_externe) / (850 - 324)   # ↑ = risque
debt_risk       = taux_endettement_pct / 100                    # ↑ = risque
incident_risk   = nb_incidents_paiement_12m / 5                 # ↑ = risque
income_prot     = np.log1p(revenu_mensuel) / np.log1p(14000)    # ↑ = protection
util_risk       = utilisation_carte_pct / 100                   # ↑ = risque (faible)

employ_risk = np.select(
    [type_contrat == "Sans emploi",
     type_contrat == "CDD",
     type_contrat == "Indépendant"],
    [1.0, 0.45, 0.25],
    default=0.0,
)

logit = (
    -2.52
    + 2.6  * score_risk
    + 1.9  * debt_risk
    + 2.1  * incident_risk
    + 1.1  * employ_risk
    - 1.6  * income_prot
    + 0.45 * util_risk
    + np.random.normal(0, 0.5, N)   # bruit aléatoire réaliste
)

prob_default = 1.0 / (1.0 + np.exp(-logit))
default = (np.random.random(N) < prob_default).astype(int)

# ── Assemblage du DataFrame ───────────────────────────────────────────────────
df = pd.DataFrame({
    "age":                        age,
    "revenu_mensuel":             revenu_mensuel,
    "anciennete_compte_mois":     anciennete_compte_mois,
    "nb_transactions_mois":       nb_transactions_mois,
    "montant_moyen_transaction":  montant_moyen_transaction,
    "taux_endettement_pct":       taux_endettement_pct,
    "nb_incidents_paiement_12m":  nb_incidents_paiement_12m,
    "montant_credit_demande":     montant_credit_demande,
    "type_contrat":               type_contrat,
    "score_bureau_externe":       score_bureau_externe,
    "utilisation_carte_pct":      utilisation_carte_pct,
    "default":                    default,
})

# ── Export ────────────────────────────────────────────────────────────────────
out_path = DATA_DIR / "credit_scoring_dataset.csv"
df.to_csv(out_path, index=False)

print(f"Dataset généré : {out_path}")
print(f"Dimensions     : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"Taux de défaut : {df['default'].mean():.4f}  ({df['default'].sum()} défauts)")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nStatistiques :")
print(df.describe().T[["mean", "std", "min", "max"]].round(3))
print(f"\ntype_contrat :")
print(df["type_contrat"].value_counts())
