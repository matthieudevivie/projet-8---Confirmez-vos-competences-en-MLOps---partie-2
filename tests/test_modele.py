"""Tests unitaires du modèle : chargement, prédiction, décision."""

import json
import shutil

import pytest

from api.modele import DOSSIER_MODELE, ModeleScoring

# Probabilités obtenues dans l'environnement d'entraînement (projet 6) pour les trois clients d'exemple
REFERENCE = [0.0487, 0.0807, 0.1983]


def test_le_modele_attend_800_variables(modele):
    assert len(modele.variables) == 800


def test_le_seuil_vient_des_metadonnees(modele):
    assert modele.seuil == 0.09


def test_predictions_identiques_au_projet_6(modele, clients_exemple):
    probabilites = [modele.predire(client) for client in clients_exemple]
    assert probabilites == pytest.approx(REFERENCE, abs=1e-4)


def test_valeurs_manquantes_acceptees(modele, clients_exemple):
    client = {nom: None for nom in clients_exemple[0]}  # aucune information
    assert 0 <= modele.predire(client) <= 1


@pytest.mark.parametrize("probabilite, decision", [
    (0.0899, "accorde"),
    (0.09, "refuse"),  # au seuil exactement : on refuse
    (0.5, "refuse"),
])
def test_decision_autour_du_seuil(modele, probabilite, decision):
    assert modele.decider(probabilite) == decision


def test_refuse_un_modele_altere(tmp_path):
    dossier = tmp_path / "model"
    shutil.copytree(DOSSIER_MODELE, dossier)
    metadata = json.loads((dossier / "metadata.json").read_text())
    metadata["sha256_model_pkl"] = "0" * 64
    (dossier / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="empreinte"):
        ModeleScoring(dossier)
