import pandas as pd
from typing import List

def align_features(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]

def simple_impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "")
    return df
