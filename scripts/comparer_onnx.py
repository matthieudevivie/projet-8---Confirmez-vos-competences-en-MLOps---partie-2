"""ONNX Runtime face à la version déployée (LightGBM + numpy) : mêmes prédictions ? plus rapide ?

Les deux bibliothèques ne servent qu'à ce test : elles ne sont pas dans pyproject.toml. Lancement :
    uv run --with onnxmltools --with onnxruntime python -m scripts.comparer_onnx
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as rt
import pandas as pd
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

from api.modele import ModeleScoring
from scripts.mesure_latence import chronometrer

RACINE = Path(__file__).resolve().parent.parent
modele = ModeleScoring()
client = json.loads((RACINE / "data" / "client_exemple.json").read_text())

# 1. Conversion : l'entrée est un tableau float32 de n lignes (None = nombre libre) × 800 variables
onx = convert_lightgbm(
    modele.modele,  # un LGBMClassifier simple, pas un Pipeline : il se convertit directement
    initial_types=[("entree", FloatTensorType([None, len(modele.variables)]))],
    zipmap=False,  # probabilités en tableau simple, pas en liste de dictionnaires
)
octets = onx.SerializeToString()
session = rt.InferenceSession(octets, providers=["CPUExecutionProvider"])  # chargé en mémoire, sans fichier
taille_pkl = (RACINE / "model" / "model.pkl").stat().st_size
print(f"Taille du modèle : pickle {taille_pkl / 1e6:.1f} Mo | ONNX {len(octets) / 1e6:.1f} Mo")


def predire_onnx(entrees: dict) -> float:
    """Même chemin que ModeleScoring.predire, mais avec ONNX Runtime (qui attend du float32)."""
    ligne = np.array([[entrees.get(nom) for nom in modele.variables]], dtype=np.float32)
    # sortie [1] = probabilités ; colonne 1 = P(défaut)
    return float(session.run(None, {"entree": ligne})[1][0, 1])


# 2. Précision : 5 000 clients de référence, toutes les prédictions d'un coup
X = pd.read_parquet(RACINE / "data" / "reference.parquet", columns=modele.variables)
X = X.sample(5000, random_state=0).astype(float).to_numpy()
proba_lgbm = modele.modele.booster_.predict(X)
proba_onnx = session.run(None, {"entree": X.astype(np.float32)})[1][:, 1]
decisions_differentes = ((proba_lgbm >= modele.seuil) != (proba_onnx >= modele.seuil)).sum()
print(f"Écart max de probabilité : {np.abs(proba_onnx - proba_lgbm).max():.2e}")
print(f"Décisions différentes : {decisions_differentes} / {len(X)}")


print("predire() déployé (LightGBM)", chronometrer(lambda: modele.predire(client), repetitions=1000))
print("predire_onnx()              ", chronometrer(lambda: predire_onnx(client), repetitions=1000))
