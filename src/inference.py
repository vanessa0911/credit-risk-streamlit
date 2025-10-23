# src/inference.py
from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from typing import Dict, List
from .feature_engineering import add_derived_features

ARTIFACTS = Path("artifacts")

def _read_metadata():
    meta_p = ARTIFACTS / "metadata.json"
    if not meta_p.exists():
        raise FileNotFoundError("metadata.json introuvable dans artifacts/")
    with open(meta_p, "r", encoding="utf-8") as f:
        return json.load(f)

def expected_columns() -> List[str]:
    meta = _read_metadata()
    return meta.get("expected_input_columns")  # 139 colonnes

def decision_threshold():
    meta = _read_metadata()
    return float(meta.get("decision_threshold", {}).get("t_selected", 0.5))

def load_models() -> Dict[str, object]:
    models = {}
    for name in ["model_calibrated_isotonic.joblib",
                 "model_calibrated_sigmoid.joblib",
                 "model_baseline_logreg.joblib"]:
        p = ARTIFACTS / name
        if p.exists():
            models[name] = joblib.load(p)
    return models

def prepare_features(one_row: pd.DataFrame) -> pd.DataFrame:
    """Construit TOUTES les features attendues par le modèle, dans le bon ordre."""
    df = add_derived_features(one_row)
    cols = expected_columns()
    # Ajout colonnes manquantes
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    # Ordre exact
    df = df[cols]
    # Imputation simple (fallback)
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(0)
        else:
            df[c] = df[c].fillna("Unknown")
    return df

def predict_proba_row(one_row: pd.DataFrame, model) -> float:
    X = prepare_features(one_row)
    return float(model.predict_proba(X)[:, 1][0])
