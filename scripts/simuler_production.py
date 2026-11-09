"""Simule la production : envoie des clients de application_test, un par un, à l'API lancée en local.

L'API journalise chaque appel (dans logs/appels.db). Dans un premier terminal :
    uv run --env-file .env uvicorn api.main:app --port 8000
puis dans un second :
    uv run python scripts/simuler_production.py --clients 5000
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
URL_API = "http://localhost:8000"


def requetes(production):
    """Une requête par client : son identifiant (passé dans l'URL) et ses 800 variables (le corps JSON)."""
    variables = production.drop(columns="SK_ID_CURR")
    # types Python natifs, NaN → null en JSON
    variables = variables.astype(object).where(variables.notna(), None)
    return zip(production["SK_ID_CURR"].tolist(), variables.to_dict(orient="records"))


def main():
    parser = argparse.ArgumentParser(description="Envoie des clients de production à l'API locale.")
    parser.add_argument(
        "--clients", type=int, default=5000, help="nombre de clients, tirés au hasard (défaut : 5000)"
    )
    parser.add_argument(
        "--taux-erreurs",
        type=float,
        default=0.02,
        help="part de requêtes rendues invalides exprès, pour mesurer le taux d'erreur (défaut : 0.02)",
    )
    args = parser.parse_args()

    production = pd.read_parquet(RACINE / "data" / "production.parquet")
    production = production.sample(n=min(args.clients, len(production)), random_state=42)
    hasard = np.random.default_rng(42)  # graine fixe : la même simulation à chaque exécution

    with httpx.Client(base_url=URL_API, timeout=10) as http:
        try:
            http.get("/health").raise_for_status()
        except httpx.HTTPError:
            sys.exit(f"L'API ne répond pas sur {URL_API} : lance-la d'abord (voir l'en-tête du script).")

        statuts = Counter()
        debut = time.perf_counter()
        for numero, (id_client, corps) in enumerate(requetes(production), start=1):
            if hasard.random() < args.taux_erreurs:
                del corps["AMT_INCOME_TOTAL"]  # une saisie incomplète : l'API doit répondre 422
            reponse = http.post("/predict", params={"id_client": id_client}, json=corps)
            statuts[reponse.status_code] += 1
            if numero % 500 == 0:
                print(f"{numero} clients envoyés en {time.perf_counter() - debut:.0f} s : {dict(statuts)}")

    print(
        f"Terminé : {sum(statuts.values())} appels en {time.perf_counter() - debut:.0f} s, statuts {dict(statuts)}"
    )


if __name__ == "__main__":
    main()
