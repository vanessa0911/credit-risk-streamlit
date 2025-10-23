# app/streamlit_app.py
import streamlit as st
import pandas as pd
from pathlib import Path
from src.inference import load_models, predict_proba_row, decision_threshold
from src.explain import load_global_importance

st.set_page_config(page_title="Scoring crédit", layout="wide")
st.title("💳 Estimation du risque de défaut")

# --- modèles ---
@st.cache_resource
def _models():
    return load_models()

models = _models()
if not models:
    st.error("Aucun modèle dans artifacts/. Vérifie les .joblib + metadata.json, puis `git lfs pull`.")
    st.stop()

model_name = st.selectbox("Modèle", list(models.keys()))
model = models[model_name]
thr_default = decision_threshold()

# --- variables prioritaires d'après tes fichiers d'analyse ---
gi = load_global_importance()
if gi is not None:
    PRIORITY = gi.sort_values("rank").head(15)["raw_feature"].tolist()
else:
    PRIORITY = []  # fallback

st.markdown("### 🏁 Variables prioritaires (à remplir en premier)")

# Champs 'base' minimaux pour calculer les priorités (alignés avec tes top features)
col1, col2, col3 = st.columns(3)
with col1:
    AGE_YEARS = st.number_input("Âge (années)", min_value=18.0, max_value=99.0, value=40.0, step=1.0)
    EMPLOY_YEARS = st.number_input("Ancienneté emploi (années)", min_value=0.0, max_value=60.0, value=5.0, step=0.5)
    DAYS_ID_PUBLISH = st.number_input("Jours depuis changement ID", min_value=0, max_value=10000, value=1000, step=10)
with col2:
    AMT_CREDIT = st.number_input("Montant crédit", min_value=0.0, value=200000.0, step=1000.0)
    AMT_ANNUITY = st.number_input("Annuité mensuelle", min_value=0.0, value=1200.0, step=10.0)
    AMT_GOODS_PRICE = st.number_input("Prix du bien", min_value=0.0, value=220000.0, step=1000.0)
with col3:
    AMT_INCOME_TOTAL = st.number_input("Revenu annuel", min_value=0.0, value=60000.0, step=1000.0)
    EXT_SOURCE_1 = st.number_input("EXT_SOURCE_1 (si connu)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    EXT_SOURCE_2 = st.number_input("EXT_SOURCE_2 (si connu)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    EXT_SOURCE_3 = st.number_input("EXT_SOURCE_3 (si connu)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# Catégoriels clés dans le top (ex. CODE_GENDER)
CODE_GENDER = st.selectbox("Genre (CODE_GENDER)", ["F","M","XNA"], index=0)

# Autres bases utiles
colA, colB = st.columns(2)
with colA:
    CNT_CHILDREN = st.number_input("Nombre d'enfants", min_value=0, max_value=20, value=0, step=1)
    CNT_FAM_MEMBERS = st.number_input("Membres du foyer", min_value=1.0, max_value=20.0, value=1.0, step=1.0)
with colB:
    DAYS_EMPLOYED = st.number_input("Jours d'emploi (positif=chômage, négatif=emploi)", value=-int(EMPLOY_YEARS*365.25), step=10)
    DAYS_BIRTH = st.number_input("Jours depuis naissance (négatif)", value=-int(AGE_YEARS*365.25), step=10)

# Calcul et affichage des variables prioritaires (lecture seule)
import numpy as np
def _safe_div(a,b): return (a / b) if (b and b!=0) else np.nan
priority_values = {
    "EXT_SOURCES_MEAN": np.nanmean([EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3]),
    "EXT_SOURCES_SUM": np.nansum([EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3]),
    "EXT_SOURCE_3": EXT_SOURCE_3,
    "CREDIT_GOODS_RATIO": _safe_div(AMT_CREDIT, AMT_GOODS_PRICE),
    "PAYMENT_RATE": _safe_div(AMT_ANNUITY, AMT_CREDIT),
    "AMT_ANNUITY": AMT_ANNUITY,
    "CREDIT_TERM_MONTHS": _safe_div(AMT_CREDIT, AMT_ANNUITY),
    "AMT_GOODS_PRICE": AMT_GOODS_PRICE,
    "DAYS_ID_PUBLISH": DAYS_ID_PUBLISH,
    "EMPLOY_TO_AGE_RATIO": (EMPLOY_YEARS / AGE_YEARS) if AGE_YEARS else np.nan,
    "AGE_YEARS": AGE_YEARS,
    "EXT_SOURCE_2": EXT_SOURCE_2,
    "CODE_GENDER": CODE_GENDER,
    "DAYS_LAST_PHONE_CHANGE": np.nan,   # champ saisi plus bas si besoin
    "DOC_COUNT": np.nan                 # champ saisi plus bas si besoin
}

if PRIORITY:
    st.info("Ces valeurs se mettent à jour automatiquement à partir des champs saisis.")
    df_prior = pd.DataFrame(
        [{"feature": f, "valeur": priority_values.get(f, "—")} for f in PRIORITY]
    )
    st.dataframe(df_prior, hide_index=True, use_container_width=True)

st.markdown("### 🧩 Autres variables (optionnel)")
with st.expander("Afficher / saisir les autres variables du modèle"):
    # Saisies simples — ajoute ce dont tu as besoin, par ex.:
    DAYS_LAST_PHONE_CHANGE = st.number_input("Jours depuis dernier changement de téléphone", min_value=0, max_value=10000, value=500, step=10)
    DOC_COUNT_IN = st.number_input("Nombre de documents fournis (DOC_COUNT)", min_value=0, max_value=50, value=10, step=1)
    FLAG_OWN_CAR = st.selectbox("Possède une voiture (FLAG_OWN_CAR)", ["N","Y"], index=0)
    FLAG_OWN_REALTY = st.selectbox("Possède un bien immobilier (FLAG_OWN_REALTY)", ["N","Y"], index=1)
    # ... tu pourras compléter ici avec d'autres champs métier utiles.

# Bouton prédire
if st.button("Calculer la probabilité de défaut"):
    # Construis une seule ligne avec toutes les colonnes "base"
    row = pd.DataFrame([{
        "AGE_YEARS": AGE_YEARS,
        "EMPLOY_YEARS": EMPLOY_YEARS,
        "DAYS_EMPLOYED": DAYS_EMPLOYED,
        "DAYS_BIRTH": DAYS_BIRTH,
        "DAYS_ID_PUBLISH": DAYS_ID_PUBLISH,
        "AMT_CREDIT": AMT_CREDIT,
        "AMT_ANNUITY": AMT_ANNUITY,
        "AMT_GOODS_PRICE": AMT_GOODS_PRICE,
        "AMT_INCOME_TOTAL": AMT_INCOME_TOTAL,
        "CNT_CHILDREN": CNT_CHILDREN,
        "CNT_FAM_MEMBERS": CNT_FAM_MEMBERS,
        "EXT_SOURCE_1": EXT_SOURCE_1,
        "EXT_SOURCE_2": EXT_SOURCE_2,
        "EXT_SOURCE_3": EXT_SOURCE_3,
        "CODE_GENDER": CODE_GENDER,
        "DAYS_LAST_PHONE_CHANGE": DAYS_LAST_PHONE_CHANGE,
        "DOC_COUNT": DOC_COUNT_IN,
        "FLAG_OWN_CAR": FLAG_OWN_CAR,
        "FLAG_OWN_REALTY": FLAG_OWN_REALTY,
    }])

    proba = predict_proba_row(row, model)
    st.metric("Probabilité de défaut", f"{proba:.3f}")
    seuil = st.slider("Seuil de refus", 0.0, 1.0, float(thr_default), 0.01)
    decision = "Refuser" if proba >= seuil else "Accorder"
    st.success(f"Décision proposée : **{decision}** (seuil={seuil:.2f})")
