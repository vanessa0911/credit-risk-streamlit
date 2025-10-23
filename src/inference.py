from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from .preprocessing import align_features, simple_impute

ARTIFACTS = Path("artifacts")

def load_feature_names() -> list:
    p = ARTIFACTS / "feature_names.npy"
    if not p.exists():
        raise FileNotFoundError("feature_names.npy introuvable dans artifacts/")
    return list(np.load(p, allow_pickle=True))

def load_models() -> dict:
    models = {}
    for name in ["model_calibrated_isotonic.joblib",
                 "model_calibrated_sigmoid.joblib",
                 "model_baseline_logreg.joblib"]:
        fp = ARTIFACTS / name
        if fp.exists():
            models[name] = joblib.load(fp)
    return models

def predict_proba(df_new: pd.DataFrame, model) -> pd.Series:
    feats = load_feature_names()
    X = simple_impute(align_features(df_new, feats))
    return pd.Series(model.predict_proba(X)[:, 1], index=df_new.index, name="proba_default")
