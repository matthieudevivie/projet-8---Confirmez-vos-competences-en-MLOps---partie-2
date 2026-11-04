"""Indicateurs du tableau de bord de monitoring : les calculs seuls, sans affichage (testés par pytest).

Les seuils reprennent la règle d'alerte de notebooks/analyse_drift.ipynb (section 5).
"""

import numpy as np
import pandas as pd

PSI_A_SURVEILLER, PSI_ALERTE = 0.1, 0.25       # dérive d'une variable
ECART_MANQUANTS_PTS = 10                         # écart de valeurs manquantes, en points
PSI_SCORES, ECART_REFUS_PTS = 0.1, 5             # dérive des décisions


def calculer_psi(reference, production, n_tranches=10):
    """PSI d'une variable : < 0,1 stable · 0,1 à 0,25 à surveiller · > 0,25 dérive critique."""
    ref, prod = reference.dropna(), production.dropna()
    if ref.nunique() <= 2:
        # Variable 0/1 : une tranche par valeur (des déciles tomberaient tous sur 0 ou 1 → une seule tranche)
        valeurs = sorted(ref.unique())
        part_ref = ref.value_counts(normalize=True).reindex(valeurs).to_numpy()
        part_prod = prod.value_counts(normalize=True).reindex(valeurs, fill_value=0).to_numpy()
    else:
        # Variable numérique : tranches de 10 % des clients de RÉFÉRENCE (déciles)
        bornes = np.unique(np.quantile(ref, np.linspace(0, 1, n_tranches + 1)))
        bornes[0], bornes[-1] = -np.inf, np.inf          # les valeurs hors bornes tombent dans la 1re/dernière tranche
        part_ref = np.histogram(ref, bins=bornes)[0] / len(ref)
        part_prod = np.histogram(prod, bins=bornes)[0] / len(prod)
    part_ref = np.clip(part_ref, 1e-6, None)             # évite log(0) si une tranche est vide
    part_prod = np.clip(part_prod, 1e-6, None)
    return float(np.sum((part_prod - part_ref) * np.log(part_prod / part_ref)))


def niveau(psi, ecart_manquants_pts=0.0):
    """Niveau d'une variable selon la règle d'alerte."""
    if psi > PSI_ALERTE or abs(ecart_manquants_pts) > ECART_MANQUANTS_PTS:
        return "alerte"
    if psi > PSI_A_SURVEILLER:
        return "à surveiller"
    return "stable"


def entrees_en_colonnes(appels):
    """Les variables journalisées (colonne JSON « entrees ») des prédictions réussies, une colonne par variable."""
    reussis = appels[appels["statut_http"] == 200]
    return pd.DataFrame(reussis["entrees"].tolist(), index=reussis.index).astype(float)  # None → NaN


def indicateurs_cles(appels, seuil):
    """Volume, erreurs, décisions et temps de réponse d'un ensemble d'appels."""
    reussis = appels[appels["statut_http"] == 200]
    return {
        "appels": len(appels),
        "taux_erreur_pct": (appels["statut_http"] != 200).mean() * 100,
        "taux_refus_pct": (reussis["probabilite_defaut"] >= seuil).mean() * 100,
        "latence_p50_ms": reussis["duree_ms"].quantile(0.50),
        "latence_p95_ms": reussis["duree_ms"].quantile(0.95),
        "latence_p99_ms": reussis["duree_ms"].quantile(0.99),
        "inference_moyenne_ms": reussis["duree_inference_ms"].mean(),
    }


def tableau_derive(reference, production):
    """Pour chaque variable surveillée : PSI, écart de valeurs manquantes et niveau d'alerte."""
    lignes = []
    for nom in reference.columns:
        psi = calculer_psi(reference[nom], production[nom])
        ecart = (production[nom].isna().mean() - reference[nom].isna().mean()) * 100
        lignes.append({"variable": nom, "psi": psi, "ecart_manquants_pts": ecart, "niveau": niveau(psi, ecart)})
    return pd.DataFrame(lignes).set_index("variable")


def derive_des_scores(scores_reference, scores_production, seuil):
    """PSI des probabilités prédites et variation du taux de refus : les décisions changent-elles ?"""
    psi = calculer_psi(scores_reference, scores_production)
    ecart_refus = ((scores_production >= seuil).mean() - (scores_reference >= seuil).mean()) * 100
    return {"psi": psi, "ecart_refus_pts": ecart_refus,
            "alerte": psi > PSI_SCORES or abs(ecart_refus) > ECART_REFUS_PTS}
