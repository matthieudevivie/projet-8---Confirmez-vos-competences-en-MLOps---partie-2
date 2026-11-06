"""Où passe le temps d'une prédiction ? Chronométrage des étapes de ModeleScoring.predire."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from api.modele import ModeleScoring

RACINE = Path(__file__).resolve().parent.parent
modele = ModeleScoring()  # chargé une fois, comme dans l'API
client = json.loads((RACINE / "data" / "client_exemple.json").read_text())


def chronometrer(fonction, repetitions=300):
    """Médiane et 95e centile, en millisecondes, de `repetitions` appels."""
    fonction()  # un premier appel « à blanc » : il ne compte pas
    temps = []
    for _ in range(repetitions):
        debut = time.perf_counter()
        fonction()
        temps.append((time.perf_counter() - debut) * 1000)
    return f"médiane {np.median(temps):6.3f} ms | p95 {np.percentile(temps, 95):6.3f} ms"


X = pd.DataFrame([client], columns=modele.variables).astype(float)

print("predire() actuel          ", chronometrer(lambda: modele.predire(client)))
print("Ancienne méthode (pandas) :")
print("  1. dict → DataFrame     ", chronometrer(lambda: pd.DataFrame([client], columns=modele.variables).astype(float)))
print("  2. predict_proba(df)    ", chronometrer(lambda: modele.modele.predict_proba(X)))