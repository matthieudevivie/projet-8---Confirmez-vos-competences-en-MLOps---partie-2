"""Schémas de validation des entrées et des sorties de l'API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, create_model

from api.modele import lire_signature

TYPES = {"long": int, "double": float, "boolean": bool}


def construire_schema_client():
    """Un champ par variable du modèle : obligatoire ou facultatif, avec son type."""
    champs = {}
    for variable in lire_signature():
        type_python = TYPES[variable["type"]]
        if variable["required"]:
            champs[variable["name"]] = (type_python, ...)
        else:
            champs[variable["name"]] = (type_python | None, None)
    return create_model(
        "Client",
        __config__=ConfigDict(extra="forbid"),  # un nom de variable inconnu est refusé
        **champs,
    )


Client = construire_schema_client()


class Prediction(BaseModel):
    probabilite_defaut: float
    decision: Literal["accorde", "refuse"]
    seuil: float
    version_modele: int
