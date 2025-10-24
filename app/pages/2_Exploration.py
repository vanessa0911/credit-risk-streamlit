# app/pages/2_Exploration.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

st.set_page_config(page_title="Exploration clients", layout="wide")
st.title("🔎 Exploration des clients & transactions")

DATA_DEFAULT = Path("data/processed/transactions_sample.csv")

@st.cache_data
def load_transactions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _prepare(df)

def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Colonnes minimales attendues
    needed = ["client_id", "tx_id", "tx_amount", "tx_date"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}. Attendues: {needed}")
    # Types
    df["tx_date"] = pd.to_datetime(df["tx_date"], errors="coerce")
    df = df.dropna(subset=["tx_date"])
    if not np.issubdtype(df["tx_amount"].dtype, np.number):
        df["tx_amount"] = pd.to_numeric(df["tx_amount"], errors="coerce")
    df = df.dropna(subset=["tx_amount"])
    # Catégorie optionnelle
    if "category" not in df.columns:
        df["category"] = "NA"
    return df

# --- Source de données ---
left, right = st.columns([2,1])
with left:
    if DATA_DEFAULT.exists():
        st.success("Dataset détecté : data/processed/transactions_sample.csv")
        df = load_transactions(DATA_DEFAULT)
    else:
        uploaded = st.file_uploader("Charger un CSV de transactions (exploration uniquement)", type=["csv"])
        if uploaded:
            df = _prepare(pd.read_csv(uploaded))
        else:
            st.info("Aucun fichier chargé. Ajoute data/processed/transactions_sample.csv ou charge un CSV.")
            st.stop()

with right:
    st.metric("Clients uniques", f"{df['client_id'].nunique():,}")
    st.metric("Transactions", f"{len(df):,}")
    st.metric("Montant total", f"{df['tx_amount'].sum():,.0f}")

st.divider()

# --- Filtres globaux ---
min_date, max_date = df["tx_date"].min().date(), df["tx_date"].max().date()
c1, c2, c3 = st.columns([1.2, 1.2, 2])
with c1:
    d_start = st.date_input("Date début", value=min_date, min_value=min_date, max_value=max_date)
with c2:
    d_end = st.date_input("Date fin", value=max_date, min_value=min_date, max_value=max_date)
with c3:
    cats = sorted(df["category"].astype(str).unique())
    sel_cats = st.multiselect("Catégories", cats, default=cats)

if d_start > d_end:
    st.error("La date de début est après la date de fin.")
    st.stop()

mask = (df["tx_date"].dt.date >= d_start) & (df["tx_date"].dt.date <= d_end) & (df["category"].isin(sel_cats))
dff = df.loc[mask].copy()

# =========================
# GRAPHIQUE 1 : Top clients
# =========================
st.subheader("🏆 Top clients")
colA, colB, colC = st.columns([1.2, 1.2, 1])
with colA:
    metric = st.radio("Mesure", ["Nombre de transactions", "Montant total"], index=0, horizontal=True)
with colB:
    top_n = st.slider("Top N", min_value=5, max_value=50, value=15, step=5)
with colC:
    sort_desc = st.checkbox("Tri décroissant", value=True)

if metric == "Nombre de transactions":
    agg = dff.groupby("client_id")["tx_id"].nunique().rename("transactions").reset_index()
    y = "transactions"
else:
    agg = dff.groupby("client_id")["tx_amount"].sum().rename("montant_total").reset_index()
    y = "montant_total"

agg = agg.sort_values(y, ascending=not sort_desc).head(top_n)

# Réponse à la question "quel est le client avec le plus de transactions ?"
if metric == "Nombre de transactions" and len(agg):
    top_client = agg.iloc[0]["client_id"]
    top_val = int(agg.iloc[0][y])
    st.info(f"👉 Client avec le plus de transactions : **{top_client}** ({top_val} transactions)")

fig1 = px.bar(
    agg, x="client_id", y=y, hover_data=agg.columns,
    title=f"Top {top_n} clients — {y.replace('_', ' ')}",
)
fig1.update_layout(xaxis_title="client_id", yaxis_title=y)
st.plotly_chart(fig1, use_container_width=True)

with st.expander("Données agrégées (download)"):
    st.dataframe(agg, use_container_width=True, hide_index=True)
    st.download_button("Télécharger l’agrégat (CSV)", agg.to_csv(index=False).encode("utf-8"), "top_clients.csv")

st.divider()

# ===============================
# GRAPHIQUE 2 : Série temporelle
# ===============================
st.subheader("⏱️ Transactions dans le temps (client)")
clients_sorted = dff["client_id"].value_counts().index.astype(str).tolist()
sel_client = st.selectbox("Choisir un client", clients_sorted[:200] if clients_sorted else [])

if sel_client:
    dfc = dff[dff["client_id"].astype(str) == sel_client].copy()
    # Agrégation par période (mois par défaut)
    dfc["_month"] = dfc["tx_date"].dt.to_period("M").dt.to_timestamp()
    ts = dfc.groupby("_month").agg(
        nb_tx=("tx_id", "nunique"),
        montant_total=("tx_amount", "sum"),
        montant_moyen=("tx_amount", "mean"),
    ).reset_index()

    tab = st.radio("Afficher", ["Nombre de transactions", "Montant total", "Montant moyen"], horizontal=True)
    y_col = {"Nombre de transactions": "nb_tx", "Montant total": "montant_total", "Montant moyen": "montant_moyen"}[tab]

    fig2 = px.line(ts, x="_month", y=y_col, markers=True, title=f"Client {sel_client} — {tab}")
    fig2.update_layout(xaxis_title="mois", yaxis_title=y_col)
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Détails client (mois)"):
        st.dataframe(ts, use_container_width=True, hide_index=True)
        st.download_button(
            f"Télécharger (CSV) – {sel_client}",
            ts.to_csv(index=False).encode("utf-8"),
            f"client_{sel_client}_timeseries.csv"
        )

st.divider()

# ==============================
# (Bonus) GRAPHIQUE 3 : Portefeuille
# ==============================
with st.expander("📊 Vue portefeuille (optionnel)"):
    cloud = dff.groupby("client_id").agg(
        nb_tx=("tx_id", "nunique"),
        montant_total=("tx_amount", "sum"),
        montant_moyen=("tx_amount", "mean"),
    ).reset_index()

    fig3 = px.scatter(
        cloud, x="nb_tx", y="montant_moyen",
        size="montant_total", hover_name="client_id",
        title="Portefeuille clients — nb_tx vs montant_moyen (taille = montant_total)",
    )
    fig3.update_layout(xaxis_title="Nombre de transactions", yaxis_title="Montant moyen")
    st.plotly_chart(fig3, use_container_width=True)
