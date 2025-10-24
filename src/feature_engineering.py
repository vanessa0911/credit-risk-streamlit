# src/feature_engineering.py
import numpy as np
import pandas as pd
from typing import List

# Liste (indicative) des features dérivées produites ici
DERIVED: List[str] = [
    "AGE_BIN", "AGE_YEARS", "ANNUITY_INCOME_RATIO", "CHILDREN_RATIO",
    "CREDIT_GOODS_RATIO", "CREDIT_INCOME_RATIO", "CREDIT_TERM_MONTHS",
    "DOC_COUNT", "EMPLOY_TO_AGE_RATIO", "EMPLOY_YEARS", "EXT_SOURCES_MEAN",
    "EXT_SOURCES_NA", "EXT_SOURCES_SUM", "INCOME_PER_PERSON",
    "MISSING_COUNT_ROW", "OWN_CAR_BOOL", "OWN_REALTY_BOOL", "PAYMENT_RATE",
    "REG_YEARS"
]

def _safe_div(a, b):
    """Division élément par élément, retourne NaN si b==0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((b == 0) | pd.isna(b), np.nan, a / b)

def _col(df: pd.DataFrame, name: str, default=np.nan, as_str=False) -> pd.Series:
    """Retourne une série de longueur len(df). Si la colonne n'existe pas, remplit avec 'default'."""
    if name in df.columns:
        s = df[name]
    else:
        s = pd.Series([default] * len(df), index=df.index)
    if as_str:
        s = s.astype(str)
    return s

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée de façon robuste les variables dérivées attendues par le modèle,
    même si les colonnes 'base' sont absentes. Ne lève pas d'erreur :
    remplit avec NaN/valeurs par défaut de bonne longueur.
    """
    df = df.copy()

    # ----- Bases utiles (séries de bonne longueur) -----
    DAYS_BIRTH        = _col(df, "DAYS_BIRTH")
    DAYS_EMPLOYED     = _col(df, "DAYS_EMPLOYED")
    DAYS_REGISTRATION = _col(df, "DAYS_REGISTRATION")
    AMT_CREDIT        = _col(df, "AMT_CREDIT", default=np.nan)
    AMT_ANNUITY       = _col(df, "AMT_ANNUITY", default=np.nan)
    AMT_GOODS_PRICE   = _col(df, "AMT_GOODS_PRICE", default=np.nan)
    AMT_INCOME_TOTAL  = _col(df, "AMT_INCOME_TOTAL", default=np.nan)
    CNT_CHILDREN      = _col(df, "CNT_CHILDREN", default=0)
    CNT_FAM_MEMBERS   = _col(df, "CNT_FAM_MEMBERS", default=1)
    FLAG_OWN_CAR      = _col(df, "FLAG_OWN_CAR", default="N", as_str=True).str.upper()
    FLAG_OWN_REALTY   = _col(df, "FLAG_OWN_REALTY", default="N", as_str=True).str.upper()
    DAYS_ID_PUBLISH   = _col(df, "DAYS_ID_PUBLISH", default=np.nan)
    CODE_GENDER       = _col(df, "CODE_GENDER", default="XNA", as_str=True)

    # EXT sources -> séries longueur len(df)
    EXT_SOURCE_1 = _col(df, "EXT_SOURCE_1")
    EXT_SOURCE_2 = _col(df, "EXT_SOURCE_2")
    EXT_SOURCE_3 = _col(df, "EXT_SOURCE_3")

    # ----- Dérivées -----
    # Âge en années (datasets HomeCredit: jours négatifs)
    AGE_YEARS = np.where(pd.isna(DAYS_BIRTH), np.nan, -DAYS_BIRTH / 365.25)
    df["AGE_YEARS"] = AGE_YEARS

    # Bin âge (bornes génériques)
    bins = [0, 25, 35, 45, 55, 65, 120]
    labels = ["(0,25]", "(25,35]", "(35,45]", "(45,55]", "(55,65]", "(65,120]"]
    df["AGE_BIN"] = pd.Categorical(pd.cut(df["AGE_YEARS"], bins=bins, labels=labels, include_lowest=True), categories=labels)

    # Emploi
    EMPLOY_YEARS = np.where(pd.isna(DAYS_EMPLOYED), np.nan, -DAYS_EMPLOYED / 365.25)
    df["EMPLOY_YEARS"] = EMPLOY_YEARS
    df["EMPLOY_TO_AGE_RATIO"] = np.where(pd.isna(AGE_YEARS) | (AGE_YEARS == 0), np.nan, EMPLOY_YEARS / AGE_YEARS)

    # EXT sources agrégées (matrice n x 3)
    ext_mat = np.vstack([
        EXT_SOURCE_1.to_numpy(), EXT_SOURCE_2.to_numpy(), EXT_SOURCE_3.to_numpy()
    ]).T
    df["EXT_SOURCES_MEAN"] = np.nanmean(ext_mat, axis=1)
    df["EXT_SOURCES_SUM"]  = np.nansum(ext_mat, axis=1)
    df["EXT_SOURCES_NA"]   = np.isnan(ext_mat).sum(axis=1)

    # Ratios crédit
    df["CREDIT_GOODS_RATIO"]  = _safe_div(AMT_CREDIT, AMT_GOODS_PRICE)
    df["PAYMENT_RATE"]        = _safe_div(AMT_ANNUITY, AMT_CREDIT)
    df["CREDIT_TERM_MONTHS"]  = _safe_div(AMT_CREDIT, AMT_ANNUITY)
    df["CREDIT_INCOME_RATIO"] = _safe_div(AMT_CREDIT, AMT_INCOME_TOTAL)
    df["ANNUITY_INCOME_RATIO"]= _safe_div(AMT_ANNUITY, AMT_INCOME_TOTAL)

    # Démographie / ménage
    df["INCOME_PER_PERSON"] = _safe_div(AMT_INCOME_TOTAL, CNT_FAM_MEMBERS)
    df["CHILDREN_RATIO"]    = _safe_div(CNT_CHILDREN, CNT_FAM_MEMBERS)

    # Enregistrement (années)
    df["REG_YEARS"] = np.where(pd.isna(DAYS_REGISTRATION), np.nan, -DAYS_REGISTRATION / 365.25)

    # Flags voiture / immobilier -> bool (0/1)
    df["OWN_CAR_BOOL"]    = (FLAG_OWN_CAR == "Y").astype(float)
    df["OWN_REALTY_BOOL"] = (FLAG_OWN_REALTY == "Y").astype(float)

    # DOC_COUNT : somme des flags documents si présents, sinon NaN
    if "DOC_COUNT" in df.columns:
        # garder la colonne telle quelle si elle existe déjà
        pass
    else:
        doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
        if len(doc_cols) > 0:
            df["DOC_COUNT"] = df[doc_cols].sum(axis=1)
        else:
            df["DOC_COUNT"] = np.nan

    # Nombre de manquants par ligne (après ajouts ci-dessus)
    df["MISSING_COUNT_ROW"] = df.isna().sum(axis=1)

    # S'assure que toutes les colonnes DERIVED existent
    for c in DERIVED:
        if c not in df.columns:
            df[c] = np.nan

    # Nettoie les infinis
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df
