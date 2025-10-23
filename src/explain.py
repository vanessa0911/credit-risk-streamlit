# src/explain.py
from pathlib import Path
import pandas as pd, json
ARTIFACTS = Path("artifacts")

def load_global_importance():
    p = ARTIFACTS / "global_importance.csv"
    return pd.read_csv(p) if p.exists() else None

def load_interpretability_summary():
    p = ARTIFACTS / "interpretability_summary.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None
