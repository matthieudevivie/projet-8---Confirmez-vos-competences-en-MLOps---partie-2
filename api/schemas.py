"""Schémas de validation des entrées et des sorties de l'API."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from api.modele import lire_signature

TYPES = {"long": int, "double": float, "boolean": bool}
ANS = 365

# Règles métier : les valeurs IMPOSSIBLES des variables saisies, refusées avec une erreur 422.
# Les dates sont comptées en jours avant la demande, donc négatives.
REGLES_METIER = {
    "DAYS_BIRTH": Field(ge=-100 * ANS, le=-18 * ANS, description="Âge en jours avant la demande (négatif) : entre 18 et 100 ans"),
    "AMT_INCOME_TOTAL": Field(gt=0, description="Revenu annuel, strictement positif"),
    "AMT_CREDIT": Field(gt=0, description="Montant du crédit, strictement positif"),
    "AMT_ANNUITY": Field(gt=0, description="Annuité, strictement positive"),
    "AMT_GOODS_PRICE": Field(gt=0, description="Prix du bien financé, strictement positif"),
    "CNT_CHILDREN": Field(ge=0, le=20, description="Nombre d'enfants"),
    "CNT_FAM_MEMBERS": Field(ge=1, le=25, description="Taille du foyer, client compris"),
    "DAYS_EMPLOYED": Field(le=0, description="Ancienneté dans l'emploi en jours (négatif) ; la valeur aberrante 365243 doit arriver à null"),
    "DAYS_REGISTRATION": Field(le=0, description="Jours depuis le changement d'enregistrement (négatif)"),
    "DAYS_ID_PUBLISH": Field(le=0, description="Jours depuis le changement de pièce d'identité (négatif)"),
    "DAYS_LAST_PHONE_CHANGE": Field(le=0, description="Jours depuis le changement de téléphone (négatif)"),
    "EXT_SOURCE_1": Field(ge=0, le=1, description="Score externe normalisé"),
    "EXT_SOURCE_2": Field(ge=0, le=1, description="Score externe normalisé"),
    "EXT_SOURCE_3": Field(ge=0, le=1, description="Score externe normalisé"),
    "REGION_RATING_CLIENT": Field(ge=1, le=3, description="Note de la région, de 1 à 3"),
    "REGION_RATING_CLIENT_W_CITY": Field(ge=1, le=3, description="Note de la région et de la ville, de 1 à 3"),
    "OWN_CAR_AGE": Field(ge=0, description="Âge du véhicule en années"),
    "HOUR_APPR_PROCESS_START": Field(ge=0, le=23, description="Heure de la demande"),
}


def construire_schema_client():
    """Un champ par variable du modèle : obligatoire ou facultatif, avec son type et ses règles métier."""
    signature = lire_signature()
    inconnues = set(REGLES_METIER) - {variable["name"] for variable in signature}
    if inconnues:
        raise ValueError(f"Règles métier sur des variables absentes du modèle : {sorted(inconnues)}")
    champs = {}
    for variable in signature:
        nom = variable["name"]
        type_valide = Annotated[TYPES[variable["type"]], REGLES_METIER.get(nom, Field())]
        if variable["required"]:
            champs[nom] = (type_valide, ...)
        else:
            champs[nom] = (type_valide | None, None)
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
