"""Prépare les données de référence et de production pour l'analyse du drift.

Reproduit à l'identique la préparation du projet 6 (notebook preparation_des_donnees.ipynb),
appliquée en une seule fois aux clients d'entraînement (application_train) et aux nouveaux
clients (application_test), puis produit deux fichiers dans data/ :
- reference.parquet  : le X_train du projet 6 (ce que le modèle a vu à l'entraînement)
- production.parquet : les clients de application_test, qu'on enverra un par un à l'API

Usage : uv run python scripts/preparer_donnees.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RACINE = Path(__file__).resolve().parent.parent
sys.path.append(str(RACINE))
from api.modele import DOSSIER_MODELE, ModeleScoring  # pylint: disable=wrong-import-position

DOSSIER_BRUT = RACINE.parent / "projet6" / "data"  # les CSV Kaggle, déjà téléchargés au projet 6
DOSSIER_SORTIE = RACINE / "data"


# --- Fonctions reprises telles quelles du projet 6 --------------------------------


def one_hot_encoder(df, nan_as_category=True):
    """Encode les colonnes texte en 0/1 ; renvoie le tableau et la liste des nouvelles colonnes."""
    original_columns = list(df.columns)
    categorical_columns = df.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    df[new_columns] = df[new_columns].astype("int8")
    return df, new_columns


def bureau_and_balance(nan_as_category=True):  # pylint: disable=too-many-locals
    """Historique des crédits dans les autres organismes, agrégé par client."""
    bureau = pd.read_csv(DOSSIER_BRUT / "bureau.csv")
    bb = pd.read_csv(DOSSIER_BRUT / "bureau_balance.csv")
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    bb_aggregations = {"MONTHS_BALANCE": ["min", "max", "size"]}
    for col in bb_cat:
        bb_aggregations[col] = ["mean"]
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how="left", on="SK_ID_BUREAU")
    bureau = bureau.drop(columns=["SK_ID_BUREAU"])
    num_aggregations = {
        "DAYS_CREDIT": ["min", "max", "mean", "var"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "DAYS_CREDIT_UPDATE": ["mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM_LIMIT": ["mean", "sum"],
        "AMT_ANNUITY": ["max", "mean"],
        "CNT_CREDIT_PROLONG": ["sum"],
        "MONTHS_BALANCE_MIN": ["min"],
        "MONTHS_BALANCE_MAX": ["max"],
        "MONTHS_BALANCE_SIZE": ["mean", "sum"],
    }
    cat_aggregations = {}
    for cat in bureau_cat:
        cat_aggregations[cat] = ["mean"]
    for cat in bb_cat:
        cat_aggregations[cat + "_MEAN"] = ["mean"]
    bureau_agg = bureau.groupby("SK_ID_CURR").agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(["BURO_" + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    active = bureau[bureau["CREDIT_ACTIVE_Active"] == 1]
    active_agg = active.groupby("SK_ID_CURR").agg(num_aggregations)
    active_agg.columns = pd.Index(
        ["ACTIVE_" + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()]
    )
    bureau_agg = bureau_agg.join(active_agg, how="left", on="SK_ID_CURR")
    closed = bureau[bureau["CREDIT_ACTIVE_Closed"] == 1]
    closed_agg = closed.groupby("SK_ID_CURR").agg(num_aggregations)
    closed_agg.columns = pd.Index(
        ["CLOSED_" + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()]
    )
    bureau_agg = bureau_agg.join(closed_agg, how="left", on="SK_ID_CURR")
    return bureau_agg


def pos_cash():
    """Soldes mensuels des crédits à la consommation et en point de vente, agrégés par client."""
    pos = pd.read_csv(DOSSIER_BRUT / "POS_CASH_balance.csv")
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)
    aggregations = {
        "MONTHS_BALANCE": ["max", "mean", "size"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
    }
    for cat in cat_cols:
        aggregations[cat] = ["mean"]
    pos_agg = pos.groupby("SK_ID_CURR").agg(aggregations)
    pos_agg.columns = pd.Index(["POS_" + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    pos_agg["POS_COUNT"] = pos.groupby("SK_ID_CURR").size()
    return pos_agg


def installments_payments():
    """Historique des remboursements (retards, écarts de montant), agrégé par client."""
    ins = pd.read_csv(DOSSIER_BRUT / "installments_payments.csv")
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)
    ins["PAYMENT_PERC"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"]
    ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
    ins["DBD"] = (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
    aggregations = {
        "NUM_INSTALMENT_VERSION": ["nunique"],
        "DPD": ["max", "mean", "sum"],
        "DBD": ["max", "mean", "sum"],
        "PAYMENT_PERC": ["max", "mean", "sum", "var"],
        "PAYMENT_DIFF": ["max", "mean", "sum", "var"],
        "AMT_INSTALMENT": ["max", "mean", "sum"],
        "AMT_PAYMENT": ["min", "max", "mean", "sum"],
        "DAYS_ENTRY_PAYMENT": ["max", "mean", "sum"],
    }
    for cat in cat_cols:
        aggregations[cat] = ["mean"]
    ins_agg = ins.groupby("SK_ID_CURR").agg(aggregations)
    ins_agg.columns = pd.Index(["INSTAL_" + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    ins_agg["INSTAL_COUNT"] = ins.groupby("SK_ID_CURR").size()
    return ins_agg


def credit_card_balance():
    """Soldes mensuels des cartes de crédit, agrégés par client."""
    cc = pd.read_csv(DOSSIER_BRUT / "credit_card_balance.csv")
    cc, _ = one_hot_encoder(cc, nan_as_category=True)
    cc = cc.drop(columns=["SK_ID_PREV"])
    cc_agg = cc.groupby("SK_ID_CURR").agg(["min", "max", "mean", "sum", "var"])
    cc_agg.columns = pd.Index(["CC_" + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    cc_agg["CC_COUNT"] = cc.groupby("SK_ID_CURR").size()
    return cc_agg


def previous_applications():
    """Demandes de crédit précédentes chez Home Credit, agrégées par client."""
    prev = pd.read_csv(DOSSIER_BRUT / "previous_application.csv")
    prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)
    for col in [
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ]:
        prev[col] = prev[col].replace(365243, np.nan)
    prev["APP_CREDIT_PERC"] = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]
    num_aggregations = {
        "AMT_ANNUITY": ["min", "max", "mean"],
        "AMT_APPLICATION": ["min", "max", "mean"],
        "AMT_CREDIT": ["min", "max", "mean"],
        "APP_CREDIT_PERC": ["min", "max", "mean", "var"],
        "AMT_DOWN_PAYMENT": ["min", "max", "mean"],
        "AMT_GOODS_PRICE": ["min", "max", "mean"],
        "HOUR_APPR_PROCESS_START": ["min", "max", "mean"],
        "RATE_DOWN_PAYMENT": ["min", "max", "mean"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "sum"],
    }
    cat_aggregations = {cat: ["mean"] for cat in cat_cols}
    prev_agg = prev.groupby("SK_ID_CURR").agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(["PREV_" + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    approved = prev[prev["NAME_CONTRACT_STATUS_Approved"] == 1]
    approved_agg = approved.groupby("SK_ID_CURR").agg(num_aggregations)
    approved_agg.columns = pd.Index(
        ["APPROVED_" + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()]
    )
    prev_agg = prev_agg.join(approved_agg, how="left", on="SK_ID_CURR")
    refused = prev[prev["NAME_CONTRACT_STATUS_Refused"] == 1]
    refused_agg = refused.groupby("SK_ID_CURR").agg(num_aggregations)
    refused_agg.columns = pd.Index(
        ["REFUSED_" + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()]
    )
    prev_agg = prev_agg.join(refused_agg, how="left", on="SK_ID_CURR")
    return prev_agg


# --- Préparation de la table principale, commune aux deux jeux ---------------------


def preparer_application():
    """train + test empilés, puis encodés ensemble : les deux jeux ont ainsi les mêmes colonnes."""
    train = pd.read_csv(DOSSIER_BRUT / "application_train.csv")
    test = pd.read_csv(DOSSIER_BRUT / "application_test.csv")
    df = pd.concat([train, test], ignore_index=True)  # TARGET vaut NaN pour les clients du test

    colonnes_texte = df.select_dtypes(exclude=["number", "bool"]).columns
    df = pd.get_dummies(df, columns=colonnes_texte, dummy_na=False)

    df["DAYS_EMPLOYED_ANOM"] = df["DAYS_EMPLOYED"] == 365243
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    df["CREDIT_INCOME_PERC"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
    df["ANNUITY_INCOME_PERC"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    return df


def main():
    """Reconstruit les 800 variables du projet 6 et écrit les jeux de référence et de production."""
    modele = ModeleScoring()

    print("1/4 Table principale…")
    df = preparer_application()

    print("2/4 Tables d'historique (quelques minutes)…")
    for agregats in (
        pos_cash(),
        installments_payments(),
        credit_card_balance(),
        previous_applications(),
        bureau_and_balance(),
    ):
        df = df.join(agregats, how="left", on="SK_ID_CURR")
        del agregats  # libère la mémoire au fur et à mesure

    print("3/4 Nettoyage final, comme au projet 6…")
    df = df.replace([np.inf, -np.inf], np.nan)
    df.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", str(c)) for c in df.columns]
    colonnes_objet = df.select_dtypes("object").columns
    df[colonnes_objet] = df[colonnes_objet].apply(pd.to_numeric, errors="coerce")

    absentes = set(modele.variables) - set(df.columns)
    if absentes:
        raise ValueError(f"Variables du modèle introuvables après préparation : {sorted(absentes)}")

    print("4/4 Séparation référence / production…")
    est_train = df["TARGET"].notna()

    # Référence = le X_train du projet 6 : même découpage, mêmes paramètres, donc mêmes clients
    train = df[est_train]
    X_train, _, _, _ = train_test_split(
        train[modele.variables], train["TARGET"], test_size=0.2, stratify=train["TARGET"], random_state=42
    )
    reference = X_train.join(train["SK_ID_CURR"])

    # Production = les nouveaux clients, avec leur identifiant, sur les 800 variables du modèle
    production = df.loc[~est_train, ["SK_ID_CURR", *modele.variables]]

    # Contrôle de non-régression : MLflow a enregistré avec le modèle les 3 premiers clients du
    # X_train du projet 6 (input_example.json). On doit les retrouver à l'identique, sur les 800
    # variables : sinon la préparation a dérivé, et le script s'arrête avant d'écrire quoi que ce soit.
    exemple = json.loads((DOSSIER_MODELE / "input_example.json").read_text())
    attendu = pd.DataFrame(exemple["data"], columns=exemple["columns"]).astype(float)
    obtenu = X_train.head(3)[exemple["columns"]].astype(float).reset_index(drop=True)
    pd.testing.assert_frame_equal(obtenu, attendu)
    print("Contrôle OK : les 3 clients de référence sont identiques à ceux du projet 6")

    DOSSIER_SORTIE.mkdir(exist_ok=True)
    reference.to_parquet(DOSSIER_SORTIE / "reference.parquet")
    production.to_parquet(DOSSIER_SORTIE / "production.parquet")
    print(f"reference.parquet  : {reference.shape}")
    print(f"production.parquet : {production.shape}")


if __name__ == "__main__":
    main()
