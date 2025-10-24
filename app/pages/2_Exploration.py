# app/pages/2_Exploration_Clients.py
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

############################################
# 1) Charger un dataset déjà présent dans le repo
#    Ordre de recherche :
#    - ./local_example.csv (racine du repo)
#    - data/processed/ (premier petit CSV trouvé)
#    - artifacts/ (premier petit CSV trouvé)
############################################
SEARCH_ORDER = [
    Path("local_example.csv"),                    # racine du repo
    Path("data/processed/sample_clients.csv"),    # nom explicite si présent
]

@st.cache_data
def load_any_dataset():
    # a) chemins explicites
    for p in SEARCH_ORDER:
        if p.exists():
            try:
                df = pd.read_csv(p)
                return df, str(p)
            except Exception:
                pass

    # b) fallback : premier "petit" CSV trouvé dans data/processed puis artifacts puis racine
    for root in ["data/processed", "artifacts", "."]:
        pr = Path(root)
        if pr.exists():
            candidates = [p for p in pr.glob("*.csv") if p.stat().st_size <= 5_000_000]
            for p in sorted(candidates):
                try:
                    df = pd.read_csv(p)
                    return df, f"{root}/{p.name}"
                except Exception:
                    continue
    return None, None

df_raw, src_name = load_any_dataset()

if df_raw is None or df_raw.empty:
    st.warning("Aucun dataset client exploitable trouvé dans le repo "
               "(recherché : ./local_example.csv, data/processed/*.csv, artifacts/*.csv).")
    st.stop()

st.success(f"Dataset chargé : **{src_name}** – {len(df_raw)} lignes, {df_raw.shape[1]} colonnes.")

############################################
# 2) Détection d'un identifiant client
############################################
CAND_ID = [c for c in ["SK_ID_CURR", "client_id", "ID", "customer_id", "id"] if c in df_raw.columns]
CLIENT_ID = CAND_ID[0] if CAND_ID else None
if CLIENT_ID is None:
    df_raw = df_raw.reset_index().rename(columns={"index": "row_id"})
    CLIENT_ID = "row_id"

############################################
# 3) Construire les features attendues par le modèle (batch)
############################################
@st.cache_data
def build_features_for_model(df: pd.DataFrame) -> pd.DataFrame:
    df2 = add_derived_features(df)
    cols = expected_columns()  # colonnes exactes attendues par le modèle (metadata.json)
    for c in cols:
        if c not in df2.columns:
            df2[c] = np.nan
    df2 = df2[cols]
    # Imputation légère (fallback sûr)
    for c in df2.columns:
        if pd.api.types.is_numeric_dtype(df2[c]):
            df2[c] = df2[c].fillna(0)
        else:
            df2[c] = df2[c].fillna("Unknown")
    return df2

############################################
# 4) Choix du modèle + calcul des probabilités (pour colorer/ordonner)
############################################
@st.cache_resource
def _models():
    return load_models()

models = _models()
if not models:
    st.error("Aucun modèle disponible dans artifacts/. Ajoute les .joblib + metadata.json puis relance.")
    st.stop()

col_hdr = st.columns([1, 1, 2])
with col_hdr[0]:
    model_name = st.selectbox("Modèle", list(models.keys()))
model = models[model_name]

with st.spinner("Préparation des features et calcul des probabilités…"):
    X = build_features_for_model(df_raw)
    proba = model.predict_proba(X)[:, 1]

df_plot = df_raw.copy()
df_plot["proba_default"] = proba

############################################
# 5) CHART #1 — Top clients par activité
#    Répond à : “quel est le client avec le plus de transactions ?”
#    Heuristique :
#      - si une colonne 'transactions' existe (txn/transaction_count...), on l’utilise
#      - sinon fallback sur DOC_COUNT
#      - sinon fallback sur AMT_CREDIT
#      - sinon AMT_INCOME_TOTAL
############################################
st.markdown("### 1) 🏆 Top clients par activité")

def pick_activity_column(cols):
    low = [c.lower() for c in cols]
    # candidats "transactions"
    for i, c in enumerate(low):
        if ("transaction" in c) or (c in {"transaction_count", "transactions", "nb_transactions", "txn", "n_transactions"}):
            return cols[i]
    # fallback successifs
    for candidate in ["DOC_COUNT", "AMT_CREDIT", "AMT_INCOME_TOTAL"]:
        if candidate in cols:
            return candidate
    return None

metric_col = pick_activity_column(df_plot.columns)

if metric_col is None:
    st.info("Aucune colonne 'transactions' ni proxy (DOC_COUNT/AMT_CREDIT/AMT_INCOME_TOTAL) trouvée dans le dataset.")
else:
    top_n = st.slider("Afficher le Top N", 5, min(50, len(df_plot)), 10, 1)
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

############################################
# 6) CHART #2 — Carte Montant ↔ Risque (colorée par proba)
############################################
st.markdown("### 2) 📌 Carte Montant ↔ Risque (couleur = probabilité de défaut)")

x_col = "AMT_CREDIT" if "AMT_CREDIT" in df_plot.columns else None
y_col = "AMT_ANNUITY" if "AMT_ANNUITY" in df_plot.columns else ("PAYMENT_RATE" if "PAYMENT_RATE" in df_plot.columns else None)

if x_col and y_col:
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
    st.info("Colonnes nécessaires non trouvées pour le scatter (AMT_CREDIT et AMT_ANNUITY/PAYMENT_RATE).")

############################################
# 7) (Optionnel) Distribution des probabilités
############################################
with st.expander("📊 Distribution des probabilités (optionnel)"):
    fig3 = px.histogram(df_plot, x="proba_default", nbins=30, title="Distribution des probabilités de défaut")
    st.plotly_chart(fig3, use_container_width=True)
