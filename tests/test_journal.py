"""Tests de la journalisation des appels : chaque appel à /predict laisse une trace en base."""

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.journal import Appel, Journal
from api.main import app


@pytest.fixture
def api_journalisee(tmp_path, monkeypatch):
    """L'API branchée sur une base SQLite jetable, neuve pour chaque test."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'journal.db'}")
    with TestClient(
        app, raise_server_exceptions=False
    ) as client_http:  # une panne doit renvoyer 500, pas planter le test
        yield client_http


def appels_enregistres():
    with Session(app.state.journal.moteur) as session:
        return session.scalars(select(Appel)).all()


def test_une_prediction_est_enregistree(api_journalisee, clients_exemple):
    api_journalisee.post("/predict", params={"id_client": 100001}, json=clients_exemple[2])
    [appel] = appels_enregistres()
    assert appel.statut_http == 200
    assert appel.id_client == 100001
    assert (appel.probabilite_defaut, appel.decision, appel.version_modele) == (0.1983, "refuse", 10)
    assert list(appel.entrees) == app.state.variables_surveillees  # les 12 variables surveillées, pas les 800
    assert appel.entrees["EXT_SOURCE_3"] == clients_exemple[2]["EXT_SOURCE_3"]
    assert 0 < appel.duree_inference_ms < appel.duree_ms


def test_un_refus_est_enregistre_avec_sa_raison(api_journalisee, clients_exemple):
    api_journalisee.post("/predict", json=dict(clients_exemple[0], AGE=35))
    [appel] = appels_enregistres()
    assert appel.statut_http == 422
    assert appel.erreur[0]["type"] == "extra_forbidden"
    assert appel.erreur[0]["loc"] == ["body", "AGE"]  # la raison suffit à comprendre le refus
    assert "input" not in appel.erreur[0]  # sans recopier les valeurs envoyées
    assert appel.entrees is None


def test_une_panne_est_enregistree(api_journalisee, clients_exemple, monkeypatch):
    def panne(client):
        raise RuntimeError("modèle indisponible")

    monkeypatch.setattr(app.state.modele, "predire", panne)
    reponse = api_journalisee.post("/predict", json=clients_exemple[0])
    assert reponse.status_code == 500
    [appel] = appels_enregistres()
    assert appel.statut_http == 500
    assert appel.erreur == [{"type": "RuntimeError", "msg": "modèle indisponible"}]


def test_les_autres_routes_ne_sont_pas_journalisees(api_journalisee):
    api_journalisee.get("/health")
    assert appels_enregistres() == []


def test_sans_base_seul_le_log_json_est_ecrit(caplog):
    journal = Journal(None)  # comme sur Render : pas de DATABASE_URL
    assert not journal.actif
    journal.enregistrer(statut_http=200, duree_ms=12.5, entrees={"EXT_SOURCE_3": 0.6})
    ligne = json.loads(caplog.records[-1].getMessage())
    assert ligne == {"statut_http": 200, "duree_ms": 12.5, "entrees": {"EXT_SOURCE_3": 0.6}}
