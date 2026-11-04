"""Tests des indicateurs du tableau de bord : sur de petites données construites à la main."""

import numpy as np
import pandas as pd
import pytest

from dashboard.indicateurs import calculer_psi, derive_des_scores, indicateurs_cles, niveau, tableau_derive


@pytest.fixture
def appels():
    """Quatre appels : trois prédictions réussies (dont un refus) et une requête invalide."""
    return pd.DataFrame({
        "statut_http": [200, 200, 200, 422],
        "duree_ms": [4.0, 5.0, 6.0, 1.0],
        "duree_inference_ms": [3.0, 4.0, 5.0, None],
        "probabilite_defaut": [0.02, 0.05, 0.30, None],
    })


def test_psi_nul_sur_deux_jeux_identiques():
    valeurs = pd.Series(np.random.default_rng(0).normal(size=1000))
    assert calculer_psi(valeurs, valeurs) < 1e-9


def test_psi_eleve_sur_une_variable_decalee():
    rng = np.random.default_rng(0)
    assert calculer_psi(pd.Series(rng.normal(size=1000)), pd.Series(rng.normal(2, 1, size=1000))) > 0.25


def test_psi_d_une_variable_0_1():
    reference = pd.Series([0.0] * 50 + [1.0] * 50)
    assert calculer_psi(reference, pd.Series([0.0] * 90 + [1.0] * 10)) > 0.25


@pytest.mark.parametrize("psi, ecart, attendu", [
    (0.05, 0, "stable"),
    (0.15, 0, "à surveiller"),
    (0.30, 0, "alerte"),
    (0.01, -13.6, "alerte"),   # comme EXT_SOURCE_1 : distribution stable, mais beaucoup moins de manquants
])
def test_niveau(psi, ecart, attendu):
    assert niveau(psi, ecart) == attendu


def test_indicateurs_cles(appels):
    kpi = indicateurs_cles(appels, seuil=0.09)
    assert kpi["appels"] == 4
    assert kpi["taux_erreur_pct"] == 25.0
    assert kpi["taux_refus_pct"] == pytest.approx(100 / 3)       # 1 refus sur 3 prédictions : la 422 ne compte pas
    assert kpi["latence_p50_ms"] == 5.0                          # la 422 (1 ms) ne fausse pas la latence
    assert kpi["inference_moyenne_ms"] == 4.0


def test_tableau_derive_repere_les_manquants():
    reference = pd.DataFrame({"x": np.linspace(0, 1, 100)})
    production = reference.copy()
    production.loc[:29, "x"] = np.nan                           # 30 % de manquants en plus
    tableau = tableau_derive(reference, production)
    assert tableau.loc["x", "ecart_manquants_pts"] == pytest.approx(30)
    assert tableau.loc["x", "niveau"] == "alerte"


def test_derive_des_scores():
    reference = pd.Series([0.02] * 70 + [0.20] * 30)                  # 30 % de refus
    assert not derive_des_scores(reference, reference, seuil=0.09)["alerte"]
    production = pd.Series([0.02] * 50 + [0.20] * 50)                 # 50 % de refus
    derive = derive_des_scores(reference, production, seuil=0.09)
    assert derive["ecart_refus_pts"] == pytest.approx(20)
    assert derive["alerte"]
