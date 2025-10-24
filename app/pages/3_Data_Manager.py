# app/pages/3_Data_Manager.py
import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype
from src.inference import expected_columns  # colonnes que le modèle attend

st.set_page_config(page_title="Gestion des données clients", layout="wide")
st.title("🧰 Gestion des données clients (échantillonnage & compression)")

# ------------------------------
# 0) Chemins et garde-fous
# ------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"

def assert_raw_is_dir():
    if RAW_DIR.exists() and not RAW_DIR.is_dir():
        st.error(
            f"⚠️ `{RAW_DIR.relative_to(ROOT)}` **existe mais n'est pas un dossier**.\n\n"
            "Corrige via le terminal :\n"
            f"```\ncd {ROOT}\nmv data/raw data/raw.bak\nmkdir -p data/raw data/processed\n```\n"
            "Puis relance cette page."
        )
        st.stop()

assert_raw_is_dir()  # ne crée rien; vérifie seulement

st.markdown("""
Cette page crée une **version légère** (≤ 25 MB) de tes données clients à partir d’un gros fichier :
1) **Choisis** un CSV/CSV.GZ dans `data/raw/` (ou uploade-le si la taille le permet).
2) **Sélection** des colonnes utiles (celles du modèle, ajustables).
3) **Échantillonnage** automatique pour tenir sous la taille cible.
4) **Compression** en `data/processed/sample_clients.csv.gz` (prêt pour GitHub).
""")

# ------------------------------
# 1) Source (upload OU data/raw)
# ------------------------------
st.subheader("1) Choisir la source")

uploaded = st.file_uploader(
    "Uploader un CSV ou CSV.GZ (si possible). Astuce: pour éviter l'erreur 413, dépose le fichier via l'explorateur dans `data/raw/`.",
    type=["csv", "gz"], accept_multiple_files=False,
)

existing = []
if RAW_DIR.is_dir():
    existing = sorted(list(RAW_DIR.glob("*.csv"))) + sorted(list(RAW_DIR.glob("*.csv.gz")))

choice = st.selectbox(
    "… ou sélectionner un fichier déjà présent dans `data/raw/`",
    ["— Aucun —"] + [str(p.relative_to(ROOT)) for p in existing],
    index=0
)

def read_any(path_or_buffer, name: str) -> pd.DataFrame:
    nm = name.lower()
    if nm.endswith(".csv.gz") or nm.endswith(".gz"):
        return pd.read_csv(path_or_buffer, compression="gzip", low_memory=False)
    return pd.read_csv(path_or_buffer, low_memory=False)

df, source_name = None, None
if uploaded is not None:
    source_name = uploaded.name
    with st.spinner(f"Lecture de {source_name}…"):
        df = read_any(uploaded, source_name)
elif choice != "— Aucun —":
    source_name = choice
    with st.spinner(f"Lecture de {choice}…"):
        df = read_any(ROOT / choice, choice)

if df is None:
    st.info("Dépose un fichier **ou** choisis-en un dans `data/raw/` pour continuer.")
    st.stop()

st.success(f"Source chargée : **{source_name}** — {len(df):,} lignes, {df.shape[1]} colonnes")
with st.expander("Aperçu (10 premières lignes)"):
    st.dataframe(df.head(10), use_container_width=True)

# ------------------------------
# 2) Colonnes à garder
# ------------------------------
st.subheader("2) Colonnes à garder")

expected = expected_columns() or []
expected_set = set(expected)

id_candidates = [c for c in ["SK_ID_CURR", "client_id", "ID", "customer_id", "id"] if c in df.columns]
keep = [c for c in df.columns if c in expected_set]
for cid in id_candidates:
    if cid not in keep:
        keep.append(cid)
if not keep:
    st.warning("Aucune colonne du modèle trouvée. On sélectionne **toutes** les colonnes par défaut.")
    keep = list(df.columns)

sel_cols = st.multiselect("Colonnes conservées (tu peux ajuster) :", options=list(df.columns), default=keep)
df_sel = df[sel_cols].copy()

# Downcast numérique pour réduire la taille
for c in df_sel.columns:
    if is_numeric_dtype(df_sel[c]):
        as_int = pd.to_numeric(df_sel[c], errors="coerce", downcast="integer")
        if as_int.notna().sum() >= df_sel[c].notna().sum() * 0.9:
            df_sel[c] = as_int
        else:
            df_sel[c] = pd.to_numeric(df_sel[c], errors="coerce", downcast="float")

st.write(f"Colonnes retenues : **{len(sel_cols)}**")

# ------------------------------
# 3) Taille cible & échantillonnage
# ------------------------------
st.subheader("3) Échantillonnage & taille cible")
target_mb = st.slider("Taille maximale du fichier compressé (.csv.gz)", 5, 25, 20, 1)
target_bytes = target_mb * 1024 * 1024

strat_col = st.selectbox(
    "Stratifier l’échantillon (optionnel si une colonne cible existe)",
    ["— Aucune —"] + [c for c in df_sel.columns if c.lower() in {"target", "default", "y"}],
    index=0
)
if strat_col == "— Aucune —":
    strat_col = None

def estimate_rows_for_target(df_in: pd.DataFrame, target_bytes: int, sample_rows: int = 5000) -> int:
    n = len(df_in)
    test = df_in if n <= sample_rows else df_in.sample(sample_rows, random_state=42)
    buf = io.BytesIO()
    test.to_csv(buf, index=False, compression="gzip")
    bytes_per_row = max(1, buf.tell() / len(test))
    est = int(target_bytes / bytes_per_row)
    return max(1, min(n, est))

est_n = estimate_rows_for_target(df_sel, target_bytes)
st.write(f"**Estimation automatique** : ~ **{est_n:,}** lignes pour ≈ {target_mb} MB.")

n_rows = st.number_input("Forcer le nombre de lignes (optionnel)", min_value=1, max_value=len(df_sel), value=est_n, step=1)

with st.spinner("Création de l’échantillon…"):
    if n_rows >= len(df_sel):
        df_out = df_sel
    else:
        if strat_col and strat_col in df_sel.columns:
            df_out = (
                df_sel.groupby(strat_col, group_keys=False)
                .apply(lambda g: g.sample(max(1, int(np.ceil(len(g) * n_rows / len(df_sel)))), random_state=42))
            )
            if len(df_out) > n_rows:
                df_out = df_out.sample(n_rows, random_state=42)
        else:
            df_out = df_sel.sample(n_rows, random_state=42)

st.success(f"Échantillon créé : **{len(df_out):,}** lignes, {df_out.shape[1]} colonnes")

with st.expander("Aperçu de l’échantillon"):
    st.dataframe(df_out.head(20), use_container_width=True)

# ------------------------------
# 4) Sauvegarde compressée
# ------------------------------
st.subheader("4) Sauvegarder en .csv.gz (prêt pour GitHub)")
out_path = PROC_DIR / "sample_clients.csv.gz"

if st.button("💾 Enregistrer `data/processed/sample_clients.csv.gz`"):
    # On crée **seulement** le dossier processed si besoin
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with st.spinner("Écriture du fichier compressé…"):
        df_out.to_csv(out_path, index=False, compression="gzip")

    size = out_path.stat().st_size
    st.success(f"Fichier écrit : **{out_path.relative_to(ROOT)}** — **{size/1024/1024:.2f} MB**")
    st.download_button(
        "⬇️ Télécharger ce fichier",
        data=out_path.read_bytes(),
        file_name="sample_clients.csv.gz",
        mime="application/gzip",
    )

st.caption("Astuce : pour éviter les erreurs 413, uploade ton fichier via l’explorateur Codespaces dans `data/raw/`.")
