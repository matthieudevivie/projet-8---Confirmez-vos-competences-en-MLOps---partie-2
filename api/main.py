"""API de scoring crédit : point d'entrée FastAPI."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse


from api.modele import ModeleScoring
from api.schemas import Client, Prediction


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.modele = ModeleScoring()  # chargé UNE fois, au démarrage
    yield


app = FastAPI(
    title="API de scoring crédit — Prêt à Dépenser",
    description="Renvoie la probabilité de défaut d'un client et la décision d'octroi.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def accueil():
    """La racine renvoie vers la documentation interactive (page d'accueil du service)."""
    return RedirectResponse("/docs")



@app.get("/health")
def health(request: Request):
    """Vérifie que l'API répond et indique le modèle chargé."""
    modele = request.app.state.modele
    return {"statut": "ok", "modele": modele.metadata["nom"], "version": modele.metadata["version_registre"]}


@app.post("/predict", response_model=Prediction)
def predict(client: Client, request: Request):
    """Score d'un client à partir de ses variables."""
    modele = request.app.state.modele
    probabilite = modele.predire(client.model_dump())
    return Prediction(
        probabilite_defaut=round(probabilite, 4),
        decision=modele.decider(probabilite),
        seuil=modele.seuil,
        version_modele=modele.metadata["version_registre"],
    )
