# app/pages/2_Exploration.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

# Réutilise la logique existante du repo
from src.inference import load_models, expected_columns
from src.feature_engineering import add_derived_features

st.set_page_config(page_title="Exploration clients", layout="wide")
st.title("🔎 Exploration des données clients")

# ---------------------------------------------------------
# 0) Utilitaires
# ---------------------------------------------------------
def repo_root() -> Path:
    """Ce fichier est <repo>/app/pages/... -> remonte à la racine repo."""
    return Path(__file__).resolve().parents[2]

def list_candidate_csvs(max_size=20_000_000) -> list:
    """
    Ordre de priorité:
      1) ./local_example.csv
      2) data/processed/sample_clients.csv
      3) tous les petits CSV dans data/processed/, artifacts/, puis racine
    """
    root = repo_root()
    candidates = []

    p_local = root / "local_example.csv"
    if p_local.exists() and p_local.is_file():
        candidates.append(p_local)

    p_sample = root / "data/processed/sample_clients.csv"
    if p_sample.exists() and p_sample.is_file():
        candidates.append(p_sample)

    for folder in ["data/processed", "artifacts", "."]:
        base = (root / folder).resolve()
        if base.exists() and base.is_dir():
            for p in sorted(base.glob("*.csv")):
                if p in candidates:
                    continue
                try:
                    if p.stat().st_size <= max_size:
                        candidates.append(p)
                except Exception:
                    continue

    # unicité
    seen, uniq = set(), []
    for p in candidates:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

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
        "Aucun dataset client exploitable trouvé.\n\n"
        "Recherché automatiquement :\n"
        "• ./local_example.csv (racine du repo)\n"
        "• data/processed/sample_clients.csv\n"
        "• data/processed/*.csv\n"
        "• artifacts/*.csv\n"
        "• ./*.csv (racine)"
    )
    st.stop()

choices = [str(p.relative_to(repo_root())) for p in cands]
sel = st.selectbox("Source de données détectée :", choices, index=0)
src_path = repo_root() / sel
df_raw = load_csv(src_path)

st.success(f"Dataset chargé : **{sel}** — {len(df_raw)} lignes, {df_raw.shape[1]} colonnes.")

with st.expander("🔍 Infos (debug rapide)"):
    st.write("Racine du repo :", str(repo_root()))
    st.write("CSV détectés :", choices)
    st.write("Colonnes (aperçu) :", list(df_raw.columns)[:20])

# ---------------------------------------------------------
# 2) Identifiant client
# ---------------------------------------------------------
cand_ids = [c for c in ["SK_ID_CURR", "client_id", "ID", "customer_id", "id"] if c in df_raw.columns]
CLIENT_ID = cand_ids[0] if cand_ids else None
if CLIENT_ID is None:
    df_raw = df_raw.reset_index().rename(columns={"index": "row_id"})
    CLIENT_ID = "row_id"

# ---------------------------------------------------------
# 3) Features pour le modèle (batch, robustes aux colonnes manquantes)
# ---------------------------------------------------------
@st.cache_data
def build_features_for_model(df: pd.DataFrame) -> pd.DataFrame:
    df2 = add_derived_features(df)
    cols = expected_columns()  # colonnes exactes attendues (metadata.json)
    for c in cols:
        if c not in df2.columns:
            df2[c] = np.nan
    df2 = df2[cols]
    # Imputation simple
    for c in df2.columns:
        if pd.api.types.is_numeric_dtype(df2[c]):
            df2[c] = df2[c].fillna(0)
        else:
            df2[c] = df2[c].fillna("Unknown")
    df2.replace([np.inf, -np.inf], 0, inplace=True)
    return df2

@st.cache_data
def derived_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    """On veut certaines colonnes dérivées potentielles pour les graphes (si existantes)."""
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
    st.error("Aucun modèle disponible dans artifacts/. Ajoute les .joblib + metadata.json puis relance.")
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

# On ajoute (si possible) quelques dérivées utiles pour les graphes
df_add = derived_for_plot(df_raw)
if not df_add.empty:
    df_plot = df_plot.join(df_add, how="left")

# ---------------------------------------------------------
# 5) CHART #1 — Top clients par activité (transactions/documents/crédit)
#    Fallback final: Top par probabilité si aucune colonne métrique n'existe
# ---------------------------------------------------------
st.markdown("### 1) 🏆 Top clients par activité")

def pick_activity_column(cols):
    low = [c.lower() for c in cols]
    # candidats contenant "transaction"
    for i, c in enumerate(low):
        if ("transaction" in c) or (c in {"transaction_count", "transactions", "nb_transactions", "txn", "n_transactions"}):
            return cols[i]
    # fallbacks successifs
    for candidate in ["DOC_COUNT", "AMT_CREDIT", "AMT_INCOME_TOTAL"]:
        if candidate in cols:
            return candidate
    return None

metric_col = pick_activity_column(df_plot.columns)

if metric_col is None:
    # Fallback: Top par proba_default (toujours dispo)
    st.info("Aucune colonne 'transactions' ni proxy (DOC_COUNT/AMT_CREDIT/AMT_INCOME_TOTAL). Affichage du Top par probabilité.")
    metric_col = "proba_default"

top_n = st.slider("Afficher le Top N", 5, min(50, len(df_plot)), min(10, len(df_plot)), 1)
df_top = df_plot[[CLIENT_ID, metric_col]].copy()
df_top = df_top.sort_values(metric_col, ascending=False).head(top_n)

fig1 = px.bar(
    df_top,
    x=metric_col,
    y=CLIENT_ID,
    orientation="h",
    title=f"Top {top_n} clients par {metric_col}",
    hover_data=[CLIENT_ID, metric_col],
)
fig1.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
st.plotly_chart(fig1, use_container_width=True)

chosen = st.selectbox("Focus client (détails ligne brute)", df_top[CLIENT_ID].astype(str).tolist())
st.write(df_plot[df_plot[CLIENT_ID].astype(str) == str(chosen)].head(1))

# ---------------------------------------------------------
# 6) CHART #2 — Carte Montant ↔ Risque (sinon histogramme des probas)
# ---------------------------------------------------------
st.markdown("### 2) 📌 Carte Montant ↔ Risque (couleur = probabilité de défaut)")

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
else:
    st.info("Colonnes nécessaires indisponibles pour le scatter. Affichage de la distribution des probabilités.")
    fig2 = px.histogram(df_plot, x="proba_default", nbins=30, title="Distribution des probabilités de défaut")
    st.plotly_chart(fig2, use_container_width=True)
