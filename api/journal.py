"""Journal des appels de production : une ligne JSON dans les logs, et une ligne en base si elle est configurée.

La base est désignée par la variable d'environnement DATABASE_URL (en local : un fichier SQLite). Sans elle
(sur Render, dont le disque est effacé à chaque redémarrage), seuls les logs JSON sont écrits.
"""

import json
import logging
from datetime import datetime

from sqlalchemy import JSON, DateTime, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

logger = logging.getLogger("scoring.appels")
logger.setLevel(logging.INFO)
if not logger.handlers:
    sortie = logging.StreamHandler()  # vers la sortie standard, que Render affiche dans ses logs
    sortie.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sortie)


class Base(DeclarativeBase):
    pass


class Appel(Base):
    """Un appel à /predict, réussi ou refusé."""

    __tablename__ = "appels"

    id: Mapped[int] = mapped_column(primary_key=True)
    date_appel: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    id_client: Mapped[int | None]
    statut_http: Mapped[int]  # 200 réussi, 422 refusé, 500 panne
    duree_ms: Mapped[float]  # latence totale vue par l'API
    duree_inference_ms: Mapped[float | None]  # le seul calcul du modèle
    probabilite_defaut: Mapped[float | None]
    decision: Mapped[str | None] = mapped_column(String(10))
    seuil: Mapped[float | None]
    version_modele: Mapped[int | None]
    entrees: Mapped[dict | None] = mapped_column(JSON)  # les seules variables surveillées, pas les 800
    erreur: Mapped[list | None] = mapped_column(JSON)  # le détail du refus ou de la panne


class Journal:
    def __init__(self, url: str | None = None):
        self.moteur = create_engine(url) if url else None
        if self.moteur is not None:
            Base.metadata.create_all(self.moteur)  # crée la table au premier démarrage, sinon ne fait rien

    @property
    def actif(self) -> bool:
        return self.moteur is not None

    def enregistrer(self, **appel):
        logger.info(json.dumps(appel, default=str))  # default=str : la date devient du texte
        if self.moteur is not None:
            with Session(self.moteur) as session:
                session.add(Appel(**appel))
                session.commit()

    def fermer(self):
        if self.moteur is not None:
            self.moteur.dispose()
