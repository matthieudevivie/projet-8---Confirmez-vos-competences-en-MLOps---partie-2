"""Chargement du modèle de scoring et calcul des prédictions."""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
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
        self.types = {variable["name"]: variable["type"] for variable in lire_signature(dossier)}

    def variables_surveillees(self, n_top=10, n_categorielles=2) -> list[str]:
        """Les variables dont on surveille la dérive : le top du gain, plus des variables 0/1 pour le chi²."""
        importances = pd.Series(
            self.modele.booster_.feature_importance(importance_type="gain"), index=self.variables
        ).sort_values(ascending=False)
        top = importances.head(n_top).index.tolist()
        categorielles = [nom for nom in importances.index if self.types[nom] == "boolean"][:n_categorielles]
        return top + categorielles

    def predire(self, client: dict) -> float:
        """Probabilité de défaut d'un client (dictionnaire variable → valeur)."""
        # Une ligne numpy dans l'ordre des variables du modèle : pandas coûtait 75 % du temps
        X = np.array([[client.get(nom) for nom in self.variables]], dtype=np.float64)  # None → NaN
        return float(self.modele.booster_.predict(X)[0])  # le booster renvoie directement P(défaut)

    def decider(self, probabilite: float) -> str:
        """Décision d'octroi : refusé si la probabilité de défaut atteint le seuil."""
        return "refuse" if probabilite >= self.seuil else "accorde"
