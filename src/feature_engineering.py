# src/feature_engineering.py
import numpy as np
import pandas as pd

DERIVED = [
    "AGE_BIN","AGE_YEARS","ANNUITY_INCOME_RATIO","CHILDREN_RATIO",
    "CREDIT_GOODS_RATIO","CREDIT_INCOME_RATIO","CREDIT_TERM_MONTHS",
    "DOC_COUNT","EMPLOY_TO_AGE_RATIO","EMPLOY_YEARS","EXT_SOURCES_MEAN",
    "EXT_SOURCES_NA","EXT_SOURCES_SUM","INCOME_PER_PERSON",
    "MISSING_COUNT_ROW","OWN_CAR_BOOL","OWN_REALTY_BOOL","PAYMENT_RATE","REG_YEARS"
]

def _safe_div(a, b):
    return np.where((b is None) | (b==0), np.nan, a/b)

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée les variables dérivées attendues par le modèle à partir des champs 'base'.
    Les colonnes manquantes sont ajoutées avec NaN (puis imputées plus tard)."""
    df = df.copy()

    # Champs utiles (peuvent ne pas tous exister)
    get = lambda c: df[c] if c in df.columns else np.nan

    # Âge
    if "AGE_YEARS" not in df.columns:
        if "DAYS_BIRTH" in df.columns:
            df["AGE_YEARS"] = (-get("DAYS_BIRTH") / 365.25)
        elif "AGE_YEARS" not in df.columns:
            df["AGE_YEARS"] = np.nan

    # Binning âge (approx. lisible – mêmes libellés en prod que lors du train)
    # Adapte si tu as les bornes exactes ; sinon ces labels resteront stables.
    bins = [0,25,35,45,55,65,120]
    labels = ["(0,25]","(25,35]","(35,45]","(45,55]","(55,65]","(65,120]"]
    df["AGE_BIN"] = pd.cut(df["AGE_YEARS"], bins=bins, labels=labels, include_lowest=True).astype("object")

    # Emploi (jours négatifs dans dataset Home Credit)
    if "EMPLOY_YEARS" not in df.columns and "DAYS_EMPLOYED" in df.columns:
        df["EMPLOY_YEARS"] = (-get("DAYS_EMPLOYED") / 365.25)

    if "EMPLOY_TO_AGE_RATIO" not in df.columns:
        df["EMPLOY_TO_AGE_RATIO"] = get("EMPLOY_YEARS") / df["AGE_YEARS"]

    # EXT sources
    ext1, ext2, ext3 = get("EXT_SOURCE_1"), get("EXT_SOURCE_2"), get("EXT_SOURCE_3")
    ext_mat = np.vstack([ext1, ext2, ext3]).T if hasattr(ext1, "__len__") else np.array([[np.nan, np.nan, np.nan]])
    df["EXT_SOURCES_MEAN"] = np.nanmean(ext_mat, axis=1)
    df["EXT_SOURCES_SUM"]  = np.nansum(ext_mat, axis=1)
    df["EXT_SOURCES_NA"]   = np.isnan(ext_mat).sum(axis=1)

    # Ratios crédit
    df["CREDIT_GOODS_RATIO"]  = _safe_div(get("AMT_CREDIT"), get("AMT_GOODS_PRICE"))
    df["PAYMENT_RATE"]        = _safe_div(get("AMT_ANNUITY"), get("AMT_CREDIT"))
    df["CREDIT_TERM_MONTHS"]  = _safe_div(get("AMT_CREDIT"), get("AMT_ANNUITY"))
    df["CREDIT_INCOME_RATIO"] = _safe_div(get("AMT_CREDIT"), get("AMT_INCOME_TOTAL"))
    df["ANNUITY_INCOME_RATIO"]= _safe_div(get("AMT_ANNUITY"), get("AMT_INCOME_TOTAL"))

    # Démographie / ménage
    df["INCOME_PER_PERSON"] = _safe_div(get("AMT_INCOME_TOTAL"), get("CNT_FAM_MEMBERS"))
    df["CHILDREN_RATIO"]    = _safe_div(get("CNT_CHILDREN"), get("CNT_FAM_MEMBERS"))

    # Enregistrements en années
    if "REG_YEARS" not in df.columns and "DAYS_REGISTRATION" in df.columns:
        df["REG_YEARS"] = (-get("DAYS_REGISTRATION") / 365.25)

    # Flags voiture / immobilier -> bool (0/1)
    df["OWN_CAR_BOOL"]    = (get("FLAG_OWN_CAR").astype(str).str.upper() == "Y").astype(float) if "FLAG_OWN_CAR" in df else np.nan
    df["OWN_REALTY_BOOL"] = (get("FLAG_OWN_REALTY").astype(str).str.upper() == "Y").astype(float) if "FLAG_OWN_REALTY" in df else np.nan

    # DOC_COUNT (si non saisi directement, somme des flags documents si présents)
    if "DOC_COUNT" not in df.columns:
        doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
        if doc_cols:
            df["DOC_COUNT"] = df[doc_cols].sum(axis=1)
        else:
            df["DOC_COUNT"] = np.nan

    # Nombre de manquants sur la ligne (appliqué après ajout features)
    df["MISSING_COUNT_ROW"] = df.isna().sum(axis=1)

    # S’assure que toutes les colonnes dérivées existent
    for c in DERIVED:
        if c not in df.columns:
            df[c] = np.nan
    return df
