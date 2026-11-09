"""API de scoring crédit : point d'entrée FastAPI."""

import os
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from api.journal import Journal
from api.modele import ModeleScoring
from api.schemas import Client, Prediction

RAPPORT_DRIFT = Path(__file__).resolve().parent.parent / "rapports" / "rapport_drift.html"


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Au démarrage : charge le modèle et ouvre le journal ; à l'arrêt : ferme le journal."""
    application.state.modele = ModeleScoring()  # chargé UNE fois, au démarrage
    application.state.variables_surveillees = application.state.modele.variables_surveillees()
    application.state.journal = Journal(os.environ.get("DATABASE_URL"))
    yield
    application.state.journal.fermer()


app = FastAPI(
    title="API de scoring crédit — Prêt à Dépenser",
    description="Renvoie la probabilité de défaut d'un client et la décision d'octroi.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def journaliser_les_predictions(request: Request, call_next):
    """Mesure et enregistre chaque appel à /predict, y compris les refus (422) et les pannes (500)."""
    if request.url.path != "/predict":
        return await call_next(request)
    date_appel = datetime.now(UTC)
    debut = time.perf_counter()
    details = {}
    try:
        reponse = await call_next(request)
        statut = reponse.status_code
        details = getattr(request.state, "journal", {})  # rempli par /predict ou par le refus de validation
    except Exception as exc:
        statut = 500
        details = {"erreur": [{"type": type(exc).__name__, "msg": str(exc)}]}
        raise
    finally:
        id_client = request.query_params.get("id_client", "")
        await run_in_threadpool(  # écriture en base hors de la boucle principale : l'API reste réactive
            request.app.state.journal.enregistrer,
            date_appel=date_appel,
            id_client=int(id_client) if id_client.isdigit() else None,
            statut_http=statut,
            duree_ms=(time.perf_counter() - debut) * 1000,
            **details,
        )
    return reponse


@app.exception_handler(RequestValidationError)
async def refus_de_validation(request: Request, exc: RequestValidationError):
    """Garde la raison du refus, puis renvoie l'erreur 422 habituelle de FastAPI."""
    request.state.journal = {
        # sans "input" : Pydantic y recopie la valeur refusée, voire toute la requête pour un champ manquant
        "erreur": jsonable_encoder([{k: v for k, v in e.items() if k != "input"} for e in exc.errors()]),
    }
    return await request_validation_exception_handler(request, exc)


@app.get("/", include_in_schema=False)
def accueil():
    """La racine renvoie vers la documentation interactive (page d'accueil du service)."""
    return RedirectResponse("/docs")


@app.get("/health")
def health(request: Request):
    """Vérifie que l'API répond, et indique le modèle chargé et le commit du code déployé."""
    modele = request.app.state.modele
    return {
        "statut": "ok",
        "modele": modele.metadata["nom"],
        "version": modele.metadata["version_registre"],
        "commit": os.environ.get("COMMIT", "local"),
    }


@app.get("/drift", response_class=FileResponse)
def rapport_drift():
    """Le dernier rapport de data drift (Evidently), produit par notebooks/analyse_drift.ipynb."""
    if not RAPPORT_DRIFT.exists():
        raise HTTPException(status_code=404, detail="Aucun rapport de drift n'a encore été produit.")
    return FileResponse(RAPPORT_DRIFT, media_type="text/html")


# id_client n'est pas lu ici mais par le middleware : le déclarer l'affiche dans la documentation /docs
@app.post("/predict", response_model=Prediction)
def predict(
    client: Client,
    request: Request,
    id_client: int | None = None,  # pylint: disable=unused-argument
):
    """Score d'un client à partir de ses variables. `id_client` (facultatif) sert seulement au suivi."""
    modele = request.app.state.modele
    entrees = client.model_dump()
    debut = time.perf_counter()
    probabilite = modele.predire(entrees)
    duree_inference_ms = (time.perf_counter() - debut) * 1000
    prediction = Prediction(
        probabilite_defaut=round(probabilite, 4),
        decision=modele.decider(probabilite),
        seuil=modele.seuil,
        version_modele=modele.metadata["version_registre"],
    )
    request.state.journal = {
        "entrees": {nom: entrees[nom] for nom in request.app.state.variables_surveillees},
        "duree_inference_ms": duree_inference_ms,
        **prediction.model_dump(),
    }
    return prediction
