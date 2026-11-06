import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent

st.set_page_config(page_title="Dashboard Prediksi", layout="wide")


@st.cache_resource
def load_model():
    with open(BASE_DIR / "model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_meta():
    with open(BASE_DIR / "model_meta.json") as f:
        return json.load(f)


model = load_model()
meta = load_meta()

NUM = meta["numerical"]
CAT = meta["categorical"]
FEATURES = meta["feature_order"]
POS = meta["positive_label"]
NEG = meta["negative_label"]


def clean_input(df):
    """Samakan tipe data dengan saat training."""
    df = df[FEATURES].copy()
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CAT:
        df[c] = df[c].astype("object").where(df[c].isna(), df[c].astype(str))
    return df


def manual_input():
    st.sidebar.header("Input Manual")
    values = {}
    for col in FEATURES:
        if col in NUM:
            info = NUM[col]
            lo, hi, med = info["min"], info["max"], info["median"]
            if info["is_int"]:
                values[col] = st.sidebar.number_input(
                    col, min_value=int(lo), max_value=int(hi), value=int(med), step=1)
            else:
                values[col] = st.sidebar.number_input(
                    col, min_value=float(lo), max_value=float(hi), value=float(med),
                    step=round((hi - lo) / 100, 3) or 0.1)
        else:
            values[col] = st.sidebar.selectbox(col, CAT[col])
    return pd.DataFrame(values, index=[0])


# ---------- Tampilan ----------
st.title("Dashboard Prediksi: " + str(meta["target"]).replace("_", " ").title())
st.write("Dibuat oleh: **(isi nama kelompok / anggota)**")
st.caption(f"Model: {meta['model_name']} | Kelas positif: {POS} | Kelas negatif: {NEG}. "
           "Aplikasi ini untuk keperluan tugas kuliah, bukan alat diagnosis.")

with st.expander("Performa model pada data uji"):
    m = meta["metrics"]
    cols = st.columns(5)
    for col, key in zip(cols, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]):
        if key in m:
            col.metric(key, f"{m[key]:.3f}")

st.sidebar.header("Fitur Input")
uploaded = st.sidebar.file_uploader("Upload file CSV (opsional)", type=["csv"])

input_df = None
if uploaded is not None:
    raw = pd.read_csv(uploaded)
    missing = [c for c in FEATURES if c not in raw.columns]
    if missing:
        st.error(f"Kolom berikut tidak ada di CSV: {missing}")
    else:
        input_df = clean_input(raw)
else:
    input_df = clean_input(manual_input())

st.sidebar.header("Threshold Prediksi")
mode = st.sidebar.radio("Pilih threshold", ["Standar (0.5)", f"Terbaik dari ROC ({meta['best_threshold']:.2f})"])
threshold = 0.5 if mode.startswith("Standar") else float(meta["best_threshold"])

if input_df is not None:
    st.subheader("Data input")
    st.dataframe(input_df, use_container_width=True)

    if st.sidebar.button("Predict!"):
        with st.spinner("Memproses..."):
            proba = model.predict_proba(input_df)[:, 1]
            pred = (proba >= threshold).astype(int)

        if len(input_df) == 1:
            label = POS if pred[0] == 1 else NEG
            st.subheader("Hasil prediksi")
            if pred[0] == 1:
                st.error(f"Prediksi: {label}")
            else:
                st.success(f"Prediksi: {label}")
            st.metric(f"Probabilitas kelas '{POS}'", f"{proba[0]*100:.1f}%")
            st.progress(float(proba[0]))
            st.caption(f"Threshold yang dipakai: {threshold:.2f}")
        else:
            result = input_df.copy()
            result["prediksi"] = [POS if p == 1 else NEG for p in pred]
            result["probabilitas_positif"] = proba.round(4)
            st.subheader("Hasil prediksi (batch)")
            st.dataframe(result, use_container_width=True)
            st.bar_chart(result["prediksi"].value_counts())
            st.download_button("Download hasil (CSV)", result.to_csv(index=False).encode("utf-8"),
                               "hasil_prediksi.csv", "text/csv")
