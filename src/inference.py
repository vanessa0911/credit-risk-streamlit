from pathlib import Path
import joblib, numpy as np, pandas as pd
from typing import Dict
from .preprocessing import align_features, simple_impute

ARTIFACTS = Path("artifacts")

def load_feature_names() -> list:
    p = ARTIFACTS / "feature_names.npy"
    if p.exists():
        return list(np.load(p, allow_pickle=True))
    raise FileNotFoundError("feature_names.npy introuvable dans artifacts/")

def load_models() -> Dict[str, object]:
    models = {}
    for name in ["model_calibrated_isotonic.joblib",
                 "model_calibrated_sigmoid.joblib",
                 "model_baseline_logreg.joblib"]:
        p = ARTIFACTS / name
        if p.exists():
            models[name] = joblib.load(p)
    return models

def predict_proba(df_new: pd.DataFrame, model) -> pd.Series:
    feats = load_feature_names()
    X = simple_impute(align_features(df_new, feats))
    return pd.Series(model.predict_proba(X)[:, 1], index=df_new.index, name="proba_default")
