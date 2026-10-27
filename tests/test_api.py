"""Tests d'intégration de l'API : de la requête HTTP à la réponse."""

import pytest

from api import schemas


def test_health(api, monkeypatch):
    monkeypatch.delenv("COMMIT", raising=False)  # le test ne dépend pas de l'environnement de la machine
    reponse = api.get("/health")
    assert reponse.status_code == 200
    assert reponse.json() == {"statut": "ok", "modele": "scoring_credit_lgbm", "version": 10, "commit": "local"}


def test_health_indique_le_commit_deploye(api, monkeypatch):
    monkeypatch.setenv("COMMIT", "7d6a990")
    assert api.get("/health").json()["commit"] == "7d6a990"


def test_la_racine_renvoie_vers_la_documentation(api):
    reponse = api.get("/", follow_redirects=False)
    assert reponse.status_code == 307
    assert reponse.headers["location"] == "/docs"


def test_prediction_d_un_client_valide(api, clients_exemple):
    reponse = api.post("/predict", json=clients_exemple[2])
    assert reponse.status_code == 200
    assert reponse.json() == {"probabilite_defaut": 0.1983, "decision": "refuse", "seuil": 0.09, "version_modele": 10}


@pytest.mark.parametrize("variable, valeur, erreur", [
    ("DAYS_BIRTH", 5 * 365, "less_than_equal"),           # un âge de -5 ans
    ("DAYS_BIRTH", -16 * 365, "less_than_equal"),         # mineur
    ("AMT_INCOME_TOTAL", 0, "greater_than"),              # revenu nul
    ("AMT_CREDIT", -5000, "greater_than"),
    ("DAYS_EMPLOYED", 365243, "less_than_equal"),         # valeur aberrante non préparée
    ("EXT_SOURCE_2", 1.4, "less_than_equal"),
    ("AMT_CREDIT", "beaucoup", "float_parsing"),          # texte au lieu d'un nombre
    ("CNT_CHILDREN", 1.5, "int_from_float"),              # entier attendu
    ("AMT_INCOME_TOTAL", None, "float_type"),             # obligatoire mis à null
])
def test_valeurs_refusees(api, clients_exemple, variable, valeur, erreur):
    client = dict(clients_exemple[0], **{variable: valeur})
    reponse = api.post("/predict", json=client)
    assert reponse.status_code == 422
    detail = reponse.json()["detail"][0]
    assert detail["loc"] == ["body", variable]
    assert detail["type"] == erreur


def test_champ_obligatoire_absent(api, clients_exemple):
    client = dict(clients_exemple[0])
    del client["AMT_INCOME_TOTAL"]
    reponse = api.post("/predict", json=client)
    assert reponse.status_code == 422
    assert reponse.json()["detail"][0]["type"] == "missing"


def test_champ_inconnu_refuse(api, clients_exemple):
    reponse = api.post("/predict", json=dict(clients_exemple[0], AGE=35))
    assert reponse.status_code == 422
    assert reponse.json()["detail"][0]["type"] == "extra_forbidden"


@pytest.mark.parametrize("variable, valeur", [
    ("AMT_ANNUITY", None),             # facultatif absent
    ("AMT_INCOME_TOTAL", 10_000_000),  # inhabituel mais possible : accepté, le monitoring surveillera
])
def test_valeurs_acceptees(api, clients_exemple, variable, valeur):
    reponse = api.post("/predict", json=dict(clients_exemple[0], **{variable: valeur}))
    assert reponse.status_code == 200


def test_regle_sur_variable_inconnue_bloque_le_demarrage(monkeypatch):
    monkeypatch.setitem(schemas.REGLES_METIER, "DAYS_BIRHT", schemas.Field(le=0))  # faute de frappe
    with pytest.raises(ValueError, match="DAYS_BIRHT"):
        schemas.construire_schema_client()
