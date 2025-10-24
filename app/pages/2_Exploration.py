# app/pages/2_Exploration.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

from src.inference import load_models, expected_columns
from src.feature_engineering import add_derived_features

st.set_page_config(page_title="Exploration clients", layout="wide")
st.title("🔎 Exploration des données clients")

# ---------------------------------------------------------
# 0) Utilitaires
# ---------------------------------------------------------
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

ROOT = repo_root()

def read_any(path: Path) -> pd.DataFrame:
    """Lit .csv et .csv.gz selon l'extension."""
    name = str(path).lower()
    if name.endswith(".csv.gz") or name.endswith(".gz"):
        return pd.read_csv(path, compression="gzip", low_memory=False)
    return pd.read_csv(path, low_memory=False)

def list_candidate_csvs() -> list[Path]:
    """
    Priorités sans limite de taille :
      1) data/processed/sample_clients.csv
      2) data/processed/sample_clients.csv.gz
      3) local_example.csv (racine)
    Puis scan 'raisonnable' (toutes tailles) :
      4) data/processed/*.csv / *.csv.gz
      5) artifacts/*.csv
      6) ./*.csv
    """
    cands: list[Path] = []
    # Explicites en priorité (pas de filtre de taille)
    explicit = [
        ROOT / "data/processed/sample_clients.csv",
        ROOT / "data/processed/sample_clients.csv.gz",
        ROOT / "local_example.csv",
    ]
    for p in explicit:
        if p.exists() and p.is_file():
            cands.append(p)

    # Fallbacks (toutes tailles)
    for folder in ["data/processed", "artifacts", "."]:
        base = ROOT / folder
        if base.exists() and base.is_dir():
            for pat in ("*.csv", "*.csv.gz"):
                cands.extend(sorted(base.glob(pat)))

    # unicité
    uniq, seen = [], set()
    for p in cands:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq

@st.cache_data
def load_csv_cached(path: str) -> pd.DataFrame:
    return read_any(Path(path))

# ---------------------------------------------------------
# 1) Sélection / chargement du dataset
# ---------------------------------------------------------
col_top = st.columns([1, 1, 2, 1])
with col_top[3]:
    if st.button("🔄 Recharger"):
        st.cache_data.clear()
        st.rerun()

cands = list_candidate_csvs()
if not cands:
    st.warning(
        "Aucun dataset exploitable trouvé.\n\n"
        "Recherché (dans l'ordre) :\n"
        "• data/processed/sample_clients.csv(.gz)\n"
        "• ./local_example.csv\n"
        "• data/processed/*.csv(.gz)\n"
        "• artifacts/*.csv\n"
        "• ./*.csv\n"
        "Astuce : génère d’abord `data/processed/sample_clients.csv.gz` depuis la page 3, "
        "ou décompresse-le en `.csv`."
    )
    st.stop()

choices = [str(p.relative_to(ROOT)) for p in cands]
default_idx = 0  # sample_clients prioritaire si présent
sel = st.selectbox("Source de données détectée :", choices, index=default_idx)
src_path = ROOT / sel

df_raw = load_csv_cached(str(src_path))

st.success(f"Dataset chargé : **{sel}** — {len(df_raw)} lignes, {df_raw.shape[1]} colonnes.")

with st.expander("🔍 Infos (debug rapide)"):
    st.write("Racine du repo :", str(ROOT))
    st.write("CSV détectés (ordre d'essai) :", choices[:10], "…")
    st.write("Colonnes (aperçu) :", list(df_raw.columns)[:20])

# ---------------------------------------------------------
# 2) Identifiant client
# ---------------------------------------------------------
cand_ids = [c for c in ["SK_ID_CURR", "client_id", "ID", "customer_id", "id"] if c in df_raw.columns]
CLIENT_ID = cand_ids[0] if cand_ids else None
if CLIENT_ID is None:
    df_raw = df_raw.reset_index(drop=False).rename(columns={"index": "row_id"})
    CLIENT_ID = "row_id"

# ---------------------------------------------------------
# 3) Features pour le modèle (robustes aux colonnes manquantes)
# ---------------------------------------------------------
@st.cache_data
def build_features_for_model(df: pd.DataFrame) -> pd.DataFrame:
    from pandas.api.types import is_numeric_dtype, is_categorical_dtype
    df2 = add_derived_features(df)
    cols = expected_columns()

    # Ajout colonnes manquantes + ordre
    for c in cols:
        if c not in df2.columns:
            df2[c] = np.nan
    df2 = df2[cols]

    # Imputation robuste
    for c in df2.columns:
        s = df2[c]
        if is_numeric_dtype(s):
            df2[c] = pd.to_numeric(s, errors="coerce").fillna(0)
        else:
            if is_categorical_dtype(s):
                df2[c] = s.cat.add_categories(["Unknown"]).fillna("Unknown")
            else:
                df2[c] = s.fillna("Unknown")
            # Tentative de conversion numérique si pertinent
            s_num = pd.to_numeric(df2[c], errors="coerce")
            if s_num.notna().any():
                df2[c] = s_num.fillna(0)
            else:
                df2[c] = df2[c].astype(str)

    df2.replace([np.inf, -np.inf], 0, inplace=True)
    return df2

@st.cache_data
def derived_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    d = add_derived_features(df)
    keep = [c for c in ["DOC_COUNT", "AMT_CREDIT", "AMT_ANNUITY", "PAYMENT_RATE"] if c in d.columns]
    return d[keep] if keep else pd.DataFrame(index=df.index)

# ---------------------------------------------------------
# 4) Modèle + probas
# ---------------------------------------------------------
@st.cache_resource
def _models():
    return load_models()

models = _models()
if not models:
    st.error("Aucun modèle dans artifacts/. Ajoute les .joblib + metadata.json puis relance.")
    st.stop()

col_model = st.columns([1, 3])
with col_model[0]:
    model_name = st.selectbox("Modèle", list(models.keys()))
model = models[model_name]

with st.spinner("Préparation des features et calcul des probabilités…"):
    X = build_features_for_model(df_raw)
    proba = model.predict_proba(X)[:, 1]

df_plot = df_raw.copy()
df_plot["proba_default"] = proba

# Joindre dérivées utiles si dispo
df_add = derived_for_plot(df_raw)
if not df_add.empty:
    df_plot = df_plot.join(df_add, how="left")

# ---------------------------------------------------------
# 5) CHART #1 — Top par métrique
# ---------------------------------------------------------
st.markdown("### 1) 🏆 Top par métrique")

metric_options = []
for col in df_plot.columns:
    low = col.lower()
    if "transaction" in low or low in {"transaction_count", "transactions", "nb_transactions", "txn", "n_transactions"}:
        metric_options.append(col)
for col in ["DOC_COUNT", "AMT_CREDIT", "AMT_INCOME_TOTAL"]:
    if col in df_plot.columns and col not in metric_options:
        metric_options.append(col)
if "proba_default" not in metric_options:
    metric_options.append("proba_default")

metric_col = st.selectbox("Métrique à classer (Top N)", metric_options, index=0)

top_n = st.slider("Afficher le Top N", 5, min(50, len(df_plot)), min(10, len(df_plot)), 1)
df_top = df_plot[[CLIENT_ID, metric_col]].copy()
df_top = df_top.sort_values(metric_col, ascending=False).head(top_n)

if df_top[metric_col].notna().any():
    fig1 = px.bar(
        df_top,
        x=metric_col,
        y=CLIENT_ID,
        orientation="h",
        title=f"Top {top_n} par {metric_col}",
        hover_data=[CLIENT_ID, metric_col],
    )
    fig1.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
    st.plotly_chart(fig1, use_container_width=True)
    st.caption(
        "Lecture : barres = valeur de la métrique pour les clients en tête. "
        "Si la métrique est `proba_default`, c’est la probabilité de défaut calculée."
    )
else:
    st.info("Aucune valeur exploitable pour cette métrique.")
    st.dataframe(df_top, use_container_width=True)

chosen = st.selectbox("Focus client (affichage de la ligne d’origine)", df_top[CLIENT_ID].astype(str).tolist())
st.write(df_plot[df_plot[CLIENT_ID].astype(str) == str(chosen)].head(1))

# ---------------------------------------------------------
# 6) CHART #2 — Carte Montant ↔ Risque OU histogramme
# ---------------------------------------------------------
st.markdown("### 2) 📌 Carte Montant ↔ Risque (ou distribution des probabilités)")

x_col = "AMT_CREDIT" if "AMT_CREDIT" in df_plot.columns else None
y_col = "AMT_ANNUITY" if "AMT_ANNUITY" in df_plot.columns else ("PAYMENT_RATE" if "PAYMENT_RATE" in df_plot.columns else None)

if x_col and y_col and df_plot[x_col].notna().any() and df_plot[y_col].notna().any():
    colf1, colf2 = st.columns(2)
    with colf1:
        xmin, xmax = float(df_plot[x_col].min()), float(df_plot[x_col].max())
        x_range = st.slider(f"Filtre {x_col}", xmin, xmax, (xmin, xmax))
    with colf2:
        ymin, ymax = float(df_plot[y_col].min()), float(df_plot[y_col].max())
        y_range = st.slider(f"Filtre {y_col}", ymin, ymax, (ymin, ymax))

    mask = (df_plot[x_col].between(*x_range)) & (df_plot[y_col].between(*y_range))
    df_scatter = df_plot.loc[mask, [CLIENT_ID, x_col, y_col, "proba_default"]].copy()

    fig2 = px.scatter(
        df_scatter,
        x=x_col, y=y_col,
        color="proba_default",
        hover_data=[CLIENT_ID, "proba_default"],
        title=f"{x_col} vs {y_col} – couleur = probabilité de défaut",
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Chaque point = un client. X = montant de crédit, Y = annuité (ou `PAYMENT_RATE`). "
        "La couleur indique la probabilité de défaut."
    )
else:
    fig2 = px.histogram(df_plot, x="proba_default", nbins=30, title="Distribution des probabilités de défaut")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Répartition des probabilités de défaut sur le dataset chargé. "
        "Utile si `AMT_CREDIT` / `AMT_ANNUITY` ne sont pas disponibles."
    )
