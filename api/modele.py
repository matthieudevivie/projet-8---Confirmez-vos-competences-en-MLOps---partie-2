"""Chargement du modèle de scoring et calcul des prédictions."""

import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd
import yaml

DOSSIER_MODELE = Path(__file__).resolve().parent.parent / "model"


def lire_signature(dossier=DOSSIER_MODELE):
    """Variables attendues par le modèle, lues dans le fichier MLmodel écrit par MLflow."""
    mlmodel = yaml.safe_load((dossier / "MLmodel").read_text())
    return json.loads(mlmodel["signature"]["inputs"])


class ModeleScoring:
    """Le modèle, son seuil et ses métadonnées, chargés une seule fois."""

    def __init__(self, dossier=DOSSIER_MODELE):
        self.metadata = json.loads((dossier / "metadata.json").read_text())
        contenu = (dossier / "model.pkl").read_bytes()
        empreinte = hashlib.sha256(contenu).hexdigest()
        if empreinte != self.metadata["sha256_model_pkl"]:
            raise RuntimeError("model.pkl ne correspond pas à l'empreinte de metadata.json")
        self.modele = pickle.loads(contenu)
        self.seuil = self.metadata["seuil_decision"]
        self.variables = list(self.modele.feature_name_)

    def predire(self, client: dict) -> float:
        """Probabilité de défaut d'un client (dictionnaire variable → valeur)."""
        X = pd.DataFrame([client], columns=self.variables).astype(float)  # None → NaN
        return float(self.modele.predict_proba(X)[0, 1])

    def decider(self, probabilite: float) -> str:
        return "refuse" if probabilite >= self.seuil else "accorde"
