"""Données et objets partagés par tous les tests."""

import json
import math

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.modele import DOSSIER_MODELE, ModeleScoring


@pytest.fixture(scope="session")
def clients_exemple():
    """Les trois clients enregistrés par MLflow avec le modèle (NaN → None, comme en JSON)."""
    exemple = json.loads((DOSSIER_MODELE / "input_example.json").read_text())
    return [
        {nom: None if isinstance(valeur, float) and math.isnan(valeur) else valeur
         for nom, valeur in zip(exemple["columns"], ligne)}
        for ligne in exemple["data"]
    ]


@pytest.fixture(scope="session")
def modele():
    return ModeleScoring()


@pytest.fixture(scope="session")
def api():
    """Un client HTTP de test ; le with déclenche le démarrage (lifespan), donc le chargement du modèle."""
    with TestClient(app) as client_http:
        yield client_http
