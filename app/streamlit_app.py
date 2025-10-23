# app/streamlit_app.py
import streamlit as st
import pandas as pd
from src.inference import load_models, predict_proba
from src.explain import load_global_importance, load_interpretability_summary

st.set_page_config(page_title="Scoring crédit", layout="wide")
st.title("💳 Estimation du risque de défaut")

@st.cache_resource
def _models():
    return load_models()

models = _models()
if not models:
    st.warning("Aucun modèle trouvé dans artifacts/. Dépose tes .joblib + feature_names.npy")
else:
    st.success(f"Modèles disponibles : {', '.join(models.keys())}")

uploaded = st.file_uploader("Importer un CSV de nouveaux clients", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Aperçu :", df.head())

    choice = st.selectbox("Choisir le modèle", list(models.keys()))
    proba = predict_proba(df, models[choice])
    out = df.copy()
    out["proba_default"] = proba

    seuil = st.slider("Seuil de refus (par défaut 0.20)", 0.0, 1.0, 0.20, 0.01)
    out["decision"] = (out["proba_default"] >= seuil).map({True: "Refuser", False: "Accorder"})

    st.write("Résultats :", out.head())
    st.download_button("Télécharger les scores", out.to_csv(index=False), "scores.csv")

st.subheader("🧠 Explications globales")
gi = load_global_importance()
if gi is not None:
    st.write(gi.head(20))
else:
    st.info("global_importance.csv non trouvé dans artifacts/.")
