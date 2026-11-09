"""Tableau de bord de monitoring de l'API de scoring.

Lit les appels journalisés (logs/appels.db) et les compare aux données d'entraînement
(data/reference.parquet).
Usage : uv run streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

RACINE = Path(__file__).resolve().parent.parent
# streamlit lance ce fichier comme un script : on rend api/ et dashboard/ importables
sys.path.append(str(RACINE))
# pylint: disable=wrong-import-position
from api.modele import ModeleScoring
from dashboard.indicateurs import derive_des_scores, entrees_en_colonnes, indicateurs_cles, tableau_derive

# pylint: enable=wrong-import-position

BASE = RACINE / "logs" / "appels.db"
REFERENCE = RACINE / "data" / "reference.parquet"
URL_RAPPORT = "https://scoring-credit.onrender.com/drift"
COULEURS = {"stable": "#d4edda", "à surveiller": "#fff3cd", "alerte": "#f8d7da"}


@st.cache_resource
def charger_modele():
    """Le modèle, chargé une seule fois pour toutes les sessions Streamlit."""
    return ModeleScoring()  # le même modèle que l'API, empreinte vérifiée


@st.cache_data
def charger_reference(_modele):
    """Variables surveillées et scores des données d'entraînement : calculés une fois, gardés en cache."""
    variables = _modele.variables_surveillees()
    variables_ref = pd.read_parquet(REFERENCE, columns=variables).astype(float)
    X = pd.read_parquet(REFERENCE, columns=_modele.variables).sample(50_000, random_state=42)
    scores = pd.Series(_modele.modele.predict_proba(X)[:, 1])  # même échantillon que le notebook
    return variables_ref, scores


@st.cache_data(ttl=60)  # relu au plus toutes les minutes : les nouveaux appels apparaissent sans redémarrer
def charger_appels():
    """Les appels journalisés, du plus ancien au plus récent."""
    return pd.read_sql_table("appels", f"sqlite:///{BASE}").sort_values("date_appel")


def distribution(scores, source, bornes):
    """Part des scores (en %) dans chaque tranche, pour l'histogramme comparé."""
    parts = np.histogram(scores.clip(upper=bornes[-1]), bins=bornes)[0] / len(scores) * 100
    return pd.DataFrame({"score": bornes[:-1], "part_pct": parts, "source": source})


st.set_page_config(page_title="Monitoring du scoring crédit", layout="wide")
st.title("Monitoring de l'API de scoring crédit")

if not BASE.exists():
    st.error(
        f"Aucune base d'appels trouvée ({BASE.relative_to(RACINE)}) : "
        "lancez l'API avec DATABASE_URL (voir le README)."
    )
    st.stop()

modele = charger_modele()
appels_tous = charger_appels()
with st.spinner("Calcul des scores de référence (une seule fois)…"):
    reference, scores_reference = charger_reference(modele)

# --- Fenêtre analysée -------------------------------------------------------------------------
n_max = len(appels_tous)
taille = st.sidebar.slider(
    "Fenêtre : derniers appels", min_value=min(100, n_max), max_value=n_max, value=min(500, n_max), step=100
)
appels = appels_tous.tail(taille)
st.sidebar.caption(
    f"{appels['date_appel'].min():%d/%m/%Y %H:%M} → {appels['date_appel'].max():%d/%m/%Y %H:%M}  \n"
    f"{n_max:,} appels journalisés au total".replace(",", " ")
)
if taille < 500:
    st.sidebar.warning("Sous 500 appels, le PSI déclenche de fausses alertes (voir le notebook, section 7).")

# --- Indicateurs clés -------------------------------------------------------------------------
kpi = indicateurs_cles(appels, modele.seuil)
colonnes = st.columns(6)
colonnes[0].metric("Appels", f"{kpi['appels']:,}".replace(",", " "))
colonnes[1].metric("Taux d'erreur", f"{kpi['taux_erreur_pct']:.1f} %")
colonnes[2].metric("Taux de refus", f"{kpi['taux_refus_pct']:.1f} %")
colonnes[3].metric("Latence médiane", f"{kpi['latence_p50_ms']:.1f} ms")
colonnes[4].metric("Latence p99", f"{kpi['latence_p99_ms']:.1f} ms")
colonnes[5].metric("Inférence moyenne", f"{kpi['inference_moyenne_ms']:.1f} ms")

# --- Les décisions ont-elles changé ? ---------------------------------------------------------
st.header("Distribution des scores")
scores_production = appels.loc[appels["statut_http"] == 200, "probabilite_defaut"]
derive = derive_des_scores(scores_reference, scores_production, modele.seuil)
message = (
    f"PSI des scores : {derive['psi']:.3f} · "
    f"taux de refus : {derive['ecart_refus_pts']:+.1f} points par rapport à la référence"
)

tranches = np.linspace(0, 0.5, 51)
donnees = pd.concat(
    [
        distribution(scores_reference, "référence", tranches),
        distribution(scores_production, "production", tranches),
    ]
)
courbes = (
    alt.Chart(donnees)
    .mark_line(interpolate="step-after")
    .encode(
        x=alt.X("score:Q", title="probabilité de défaut prédite (au-delà de 0,5 : regroupée à droite)"),
        y=alt.Y("part_pct:Q", title="part des clients (%)"),
        color=alt.Color("source:N", title=None),
    )
)
seuil = (
    alt.Chart(pd.DataFrame({"seuil": [modele.seuil]}))
    .mark_rule(strokeDash=[4, 4])
    .encode(x="seuil:Q", color=alt.value("red"))
)
st.altair_chart(courbes + seuil, width="stretch")
st.caption(f"Ligne rouge : seuil de décision ({modele.seuil}). Au-dessus, le crédit est refusé.")

# --- Temps de réponse -------------------------------------------------------------------------
st.header("Temps de réponse")
reussis = appels[appels["statut_http"] == 200].reset_index(drop=True)
temps = pd.DataFrame(
    {
        "latence totale (ms)": reussis["duree_ms"].rolling(50, min_periods=1).median(),
        "inférence du modèle (ms)": reussis["duree_inference_ms"].rolling(50, min_periods=1).median(),
    }
)
st.line_chart(
    temps,
    x_label="appels réussis, du plus ancien au plus récent",
    y_label="médiane glissante sur 50 appels (ms)",
)
erreurs = appels["statut_http"].value_counts().rename_axis("statut HTTP").to_frame("appels")
st.dataframe(erreurs)

# --- Dérive des variables ---------------------------------------------------------------------
st.header("Dérive des variables surveillées")
production = entrees_en_colonnes(appels)
tableau = tableau_derive(reference, production)
st.dataframe(
    tableau.style.format({"psi": "{:.3f}", "ecart_manquants_pts": "{:+.1f}"}).map(
        lambda n: f"background-color: {COULEURS[n]}; color: black", subset=["niveau"]
    ),
    column_config={"psi": "PSI", "ecart_manquants_pts": "écart de manquants (points)", "niveau": "niveau"},
    height=35 * (len(tableau) + 1) + 3,  # toutes les variables visibles, sans barre de défilement
)
st.caption(
    "Alerte : PSI > 0,25 ou écart de valeurs manquantes > 10 points · à surveiller : PSI entre 0,1 et 0,25. "
    f"Détail par variable : [rapport Evidently]({URL_RAPPORT})."
)
