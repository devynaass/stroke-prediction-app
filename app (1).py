import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# =====================================================================
# KONFIGURASI (ubah bagian ini saja)
# =====================================================================
APP_TITLE = "Prediksi Risiko Stroke"
APP_ICON = "🧠"
CREATORS = "(isi nama kelompok / anggota)"

BASE_DIR = Path(__file__).parent
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

LABELS = {
    "gender": "Jenis kelamin",
    "age": "Usia (tahun)",
    "hypertension": "Riwayat hipertensi",
    "heart_disease": "Riwayat penyakit jantung",
    "ever_married": "Pernah menikah",
    "work_type": "Jenis pekerjaan",
    "Residence_type": "Tempat tinggal",
    "avg_glucose_level": "Rata-rata kadar glukosa (mg/dL)",
    "bmi": "BMI (Indeks Massa Tubuh)",
    "smoking_status": "Status merokok",
}
HELPS = {
    "avg_glucose_level": "Kadar glukosa darah rata-rata. Umumnya di bawah 140 mg/dL dianggap normal.",
    "bmi": "Berat (kg) dibagi tinggi (m) kuadrat. Rentang normal sekitar 18,5 - 24,9.",
    "hypertension": "Apakah pasien pernah didiagnosis tekanan darah tinggi?",
    "heart_disease": "Apakah pasien memiliki riwayat penyakit jantung?",
}
VALUE_LABELS = {
    "gender": {"Male": "Laki-laki", "Female": "Perempuan", "Other": "Lainnya"},
    "hypertension": {"0": "Tidak", "1": "Ya"},
    "heart_disease": {"0": "Tidak", "1": "Ya"},
    "ever_married": {"Yes": "Ya", "No": "Tidak"},
    "work_type": {"Private": "Swasta", "Self-employed": "Wiraswasta",
                  "Govt_job": "PNS / Pemerintah", "children": "Anak-anak",
                  "Never_worked": "Belum pernah bekerja"},
    "Residence_type": {"Urban": "Perkotaan", "Rural": "Pedesaan"},
    "smoking_status": {"formerly smoked": "Pernah merokok", "never smoked": "Tidak pernah merokok",
                       "smokes": "Perokok aktif", "Unknown": "Tidak diketahui"},
}
GROUPS = [
    ("👤 Data diri", ["gender", "age", "ever_married", "Residence_type"]),
    ("❤️ Riwayat kesehatan", ["hypertension", "heart_disease", "avg_glucose_level", "bmi"]),
    ("🚬 Gaya hidup & pekerjaan", ["work_type", "smoking_status"]),
]
STEP = {"age": 1.0, "avg_glucose_level": 0.5, "bmi": 0.1}
PRESETS = {
    "Pilih contoh profil...": None,
    "Dewasa muda, sehat": {"gender": "Female", "age": 25.0, "ever_married": "No", "Residence_type": "Urban",
                           "hypertension": "0", "heart_disease": "0", "avg_glucose_level": 80.0, "bmi": 22.0,
                           "work_type": "Private", "smoking_status": "never smoked"},
    "Lansia dengan hipertensi": {"gender": "Male", "age": 72.0, "ever_married": "Yes", "Residence_type": "Rural",
                                 "hypertension": "1", "heart_disease": "1", "avg_glucose_level": 215.0, "bmi": 33.0,
                                 "work_type": "Self-employed", "smoking_status": "formerly smoked"},
}
# Hasil dari notebook (boleh disesuaikan dengan angka kelompokmu)
MODEL_COMPARISON = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "Decision Tree", "MLP"],
    "ROC-AUC (uji)": [0.840, 0.832, 0.822, 0.780],
    "CV AUC (mean)": [0.838, 0.837, 0.811, 0.790],
    "Recall @0.5": [0.02, 0.00, 0.00, 0.06],
    "Recall @threshold optimal": [0.80, 0.78, 0.80, 0.70],
}).set_index("Model")

# =====================================================================
# STYLE
# =====================================================================
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
#MainMenu, footer {visibility: hidden;}
.hero {background: linear-gradient(135deg,#4f46e5 0%,#7c3aed 55%,#db2777 100%);
       padding: 2rem 2.3rem; border-radius: 22px; color: #fff; margin-bottom: 1.2rem;
       box-shadow: 0 12px 30px rgba(79,70,229,.25);}
.hero h1 {margin: 0; font-size: 2.2rem; color: #fff; line-height: 1.2;}
.hero p {margin: .5rem 0 0 0; opacity: .92; font-size: 1.02rem;}
.chip {display: inline-block; background: rgba(255,255,255,.18); padding: .28rem .8rem;
       border-radius: 999px; font-size: .8rem; margin: .8rem .45rem 0 0;}
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 16px; padding: 1rem 1.2rem;
       background: rgba(128,128,128,.06);}
.card h4 {margin: 0 0 .3rem 0;}
.kpi {text-align: center; border: 1px solid rgba(128,128,128,.25); border-radius: 16px;
      padding: .9rem .5rem; background: rgba(128,128,128,.06);}
.kpi .v {font-size: 1.7rem; font-weight: 700;}
.kpi .l {font-size: .8rem; opacity: .7;}
.result {border-radius: 20px; padding: 1.5rem 1.7rem; color: #fff; margin-bottom: .8rem;
         box-shadow: 0 10px 26px rgba(0,0,0,.18);}
.r-low {background: linear-gradient(135deg,#059669,#10b981);}
.r-mid {background: linear-gradient(135deg,#d97706,#f59e0b);}
.r-high {background: linear-gradient(135deg,#dc2626,#f43f5e);}
.result .tag {font-size: .85rem; letter-spacing: .06em; text-transform: uppercase; opacity: .9;}
.result .big {font-size: 2.6rem; font-weight: 800; line-height: 1.1; margin: .2rem 0;}
.result .sub {opacity: .95; font-size: .98rem;}
.gauge {position: relative; height: 14px; border-radius: 999px; margin: 1.6rem 0 2rem 0;
        background: linear-gradient(90deg,#10b981 0%,#f59e0b 50%,#ef4444 100%);}
.mk {position: absolute; top: -7px; width: 5px; height: 28px; border-radius: 3px;}
.mk.you {background: #111; box-shadow: 0 0 0 2px #fff;}
.mk.thr {background: rgba(128,128,128,.9); width: 3px;}
.mk-l {position: absolute; top: 26px; font-size: .72rem; transform: translateX(-50%); white-space: nowrap; opacity: .8;}
.fx-row {display: flex; align-items: center; margin: .3rem 0; font-size: .88rem;}
.fx-name {width: 46%; padding-right: .5rem;}
.fx-bar {flex: 1; height: 10px; background: rgba(128,128,128,.15); border-radius: 6px; position: relative;}
.fx-fill {position: absolute; top: 0; height: 10px; border-radius: 6px;}
.fx-val {width: 70px; text-align: right; font-variant-numeric: tabular-nums; opacity: .8;}
.note {font-size: .8rem; opacity: .7;}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# LOAD MODEL
# =====================================================================
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
NUM, CAT, FEATURES = meta["numerical"], meta["categorical"], meta["feature_order"]
POS, NEG = meta["positive_label"], meta["negative_label"]
BEST_THR = float(meta["best_threshold"])
METRICS = meta.get("metrics", {})


def fmt_label(col):
    return LABELS.get(col, col.replace("_", " ").title())


def fmt_value(col, v):
    return VALUE_LABELS.get(col, {}).get(str(v), str(v))


def clean_input(df):
    df = df[FEATURES].copy()
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CAT:
        df[c] = df[c].astype("object").where(df[c].isna(), df[c].astype(str))
    return df


def band(p, thr):
    if p >= thr:
        return "high"
    if p >= 0.5 * thr:
        return "mid"
    return "low"


BAND_INFO = {
    "low": ("r-low", "Risiko rendah", "Probabilitas masih jauh di bawah batas deteksi model.", "Rendah"),
    "mid": ("r-mid", "Waspada", "Probabilitas mendekati batas deteksi model. Perhatikan faktor risikonya.", "Waspada"),
    "high": ("r-high", "Risiko tinggi", "Probabilitas melewati batas deteksi, jadi model memprediksi kelas positif.", "Tinggi"),
}


def explain(row, p_now):
    """What-if: seberapa besar probabilitas berubah jika satu fitur diganti nilai 'tipikal'."""
    effects = {}
    for col in FEATURES:
        if col in NUM:
            alt = row.copy()
            alt[col] = NUM[col]["median"]
            p_ref = float(model.predict_proba(alt)[0, 1])
        else:
            probs = []
            for opt in CAT[col]:
                alt = row.copy()
                alt[col] = opt
                probs.append(float(model.predict_proba(alt)[0, 1]))
            p_ref = float(np.mean(probs))
        effects[col] = p_now - p_ref
    return pd.Series(effects).sort_values(key=lambda s: s.abs(), ascending=False)


def render_effects(effects, top=6):
    eff = effects.head(top)
    mx = float(eff.abs().max()) or 1.0
    rows = []
    for col, v in eff.items():
        w = abs(v) / mx * 50
        color = "#ef4444" if v > 0 else "#10b981"
        style = f"left:50%;width:{w}%;background:{color};" if v > 0 else f"left:{50 - w}%;width:{w}%;background:{color};"
        rows.append(
            f"<div class='fx-row'><div class='fx-name'>{fmt_label(col)}</div>"
            f"<div class='fx-bar'><div class='fx-fill' style='{style}'></div></div>"
            f"<div class='fx-val'>{v*100:+.1f} pp</div></div>")
    st.markdown("".join(rows), unsafe_allow_html=True)
    st.markdown("<div class='note'>Merah menaikkan risiko, hijau menurunkan, dibandingkan nilai tipikal "
                "(median / rata-rata kategori). pp = poin persentase probabilitas.</div>", unsafe_allow_html=True)


def render_result(row, thr):
    p = float(model.predict_proba(row)[0, 1])
    b = band(p, thr)
    css, title, text, _ = BAND_INFO[b]
    verdict = POS if p >= thr else NEG
    st.markdown(
        f"<div class='result {css}'><div class='tag'>{title}</div>"
        f"<div class='big'>{p*100:.1f}%</div>"
        f"<div class='sub'>Probabilitas kelas positif ({POS}). Prediksi model: <b>{verdict}</b>.<br>{text}</div></div>",
        unsafe_allow_html=True)

    scale = max(3 * thr, min(1.0, p * 1.25))
    you = min(p / scale * 100, 99)
    tpos = min(thr / scale * 100, 99)
    st.markdown(
        f"<div class='gauge'>"
        f"<div class='mk thr' style='left:{tpos}%'></div><div class='mk-l' style='left:{tpos}%'>batas {thr*100:.1f}%</div>"
        f"<div class='mk you' style='left:{you}%'></div></div>",
        unsafe_allow_html=True)

    st.markdown("**Faktor yang paling berpengaruh pada hasil ini**")
    try:
        render_effects(explain(row, p))
    except Exception:
        st.caption("Penjelasan faktor tidak tersedia untuk model ini.")

    if p >= thr:
        st.warning("Hasil ini hanya indikasi awal dari model statistik. Silakan konsultasikan dengan tenaga medis.")
    return p


# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown(f"### {APP_ICON} {APP_TITLE}")
    st.caption("Dashboard prediksi dengan model klasifikasi terbaik dari perbandingan 4 model.")
    st.markdown("#### Pengaturan threshold")
    mode = st.radio("Batas keputusan",
                    [f"Optimal ({BEST_THR:.3f}) - disarankan", "Standar (0.5)", "Kustom"],
                    help="Data stroke sangat tidak seimbang. Pada 0.5 model hampir tidak pernah memprediksi stroke, "
                         "sedangkan threshold optimal menaikkan kemampuan mendeteksi kasus (recall).")
    if mode.startswith("Optimal"):
        THR = BEST_THR
    elif mode.startswith("Standar"):
        THR = 0.5
    else:
        THR = st.slider("Threshold kustom", 0.01, 0.99, float(round(BEST_THR, 2)), 0.01)
    st.info(f"Threshold aktif: **{THR:.3f}**")
    st.caption("Aplikasi ini untuk tugas kuliah dan bukan alat diagnosis medis.")

# =====================================================================
# HERO
# =====================================================================
auc = METRICS.get("ROC-AUC")
chips = f"<span class='chip'>Model: {meta['model_name']}</span>"
if auc is not None:
    chips += f"<span class='chip'>ROC-AUC {auc:.3f}</span>"
chips += f"<span class='chip'>Kelas positif: {POS}</span><span class='chip'>{len(FEATURES)} fitur input</span>"
st.markdown(
    f"<div class='hero'><h1>{APP_ICON} {APP_TITLE}</h1>"
    f"<p>Masukkan data pasien, dan model machine learning akan memperkirakan risikonya.</p>"
    f"{chips}<p style='font-size:.85rem;margin-top:.9rem'>Dibuat oleh: <b>{CREATORS}</b></p></div>",
    unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🩺 Prediksi Individu", "📂 Prediksi Massal (CSV)", "📊 Tentang Model"])

# =====================================================================
# TAB 1: INDIVIDU
# =====================================================================
def apply_preset():
    preset = PRESETS.get(st.session_state.get("preset"))
    if not preset:
        return
    for col, val in preset.items():
        if col in NUM or (col in CAT and val in CAT[col]):
            st.session_state[f"in_{col}"] = val


def render_input(col):
    label, help_, key = fmt_label(col), HELPS.get(col), f"in_{col}"
    if col in NUM:
        info = NUM[col]
        if info["is_int"]:
            lo, hi = int(info["min"]), int(info["max"])
            return st.slider(label, lo, hi, int(np.clip(info["median"], lo, hi)), 1, help=help_, key=key)
        lo, hi = float(np.floor(info["min"])), float(np.ceil(info["max"]))
        step = STEP.get(col, 1.0 if hi - lo >= 20 else 0.1)
        med = float(np.clip(info["median"], lo, hi))
        fmt = "%.0f" if step >= 1 else "%.1f"
        return st.slider(label, lo, hi, med, step, help=help_, key=key, format=fmt)
    opts = CAT[col]
    if len(opts) <= 3:
        return st.radio(label, opts, horizontal=True, help=help_, key=key,
                        format_func=lambda v, c=col: fmt_value(c, v))
    return st.selectbox(label, opts, help=help_, key=key,
                        format_func=lambda v, c=col: fmt_value(c, v))


with tab1:
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.selectbox("Mulai cepat dengan contoh profil", list(PRESETS.keys()), key="preset", on_change=apply_preset)
        grouped = {c for _, cols in GROUPS for c in cols}
        groups = GROUPS + ([("📋 Lainnya", [c for c in FEATURES if c not in grouped])]
                           if any(c not in grouped for c in FEATURES) else [])
        values = {}
        with st.form("form_prediksi"):
            for title, cols in groups:
                cols = [c for c in cols if c in FEATURES]
                if not cols:
                    continue
                st.markdown(f"**{title}**")
                grid = st.columns(2)
                for i, c in enumerate(cols):
                    with grid[i % 2]:
                        values[c] = render_input(c)
            submitted = st.form_submit_button("🔍 Prediksi sekarang", type="primary")
        if submitted:
            st.session_state["last_input"] = clean_input(pd.DataFrame(values, index=[0]))

    with right:
        if "last_input" in st.session_state:
            last = st.session_state["last_input"]
            render_result(last, THR)
            with st.expander("Lihat data yang dimasukkan"):
                show = pd.DataFrame({"Fitur": [fmt_label(c) for c in FEATURES],
                                     "Nilai": [fmt_value(c, last.iloc[0][c]) for c in FEATURES]})
                st.dataframe(show, hide_index=True)
        else:
            st.markdown(
                "<div class='card'><h4>Hasil akan tampil di sini</h4>"
                "Isi data di sebelah kiri atau pilih contoh profil, lalu klik <b>Prediksi sekarang</b>.</div>",
                unsafe_allow_html=True)

# =====================================================================
# TAB 2: BATCH
# =====================================================================
with tab2:
    st.markdown("Upload file CSV berisi banyak pasien sekaligus. Kolom harus sama dengan template.")
    template = pd.DataFrame([{c: (NUM[c]["median"] if c in NUM else CAT[c][0]) for c in FEATURES}])
    st.download_button("⬇️ Download template CSV", template.to_csv(index=False).encode("utf-8"),
                       "template_input.csv", "text/csv")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        raw = pd.read_csv(up)
        missing = [c for c in FEATURES if c not in raw.columns]
        if missing:
            st.error(f"Kolom berikut tidak ada di CSV: {missing}")
        else:
            data = clean_input(raw)
            proba = model.predict_proba(data)[:, 1]
            out = raw.copy()
            out["probabilitas_%"] = (proba * 100).round(2)
            out["prediksi"] = np.where(proba >= THR, POS, NEG)
            out["tingkat_risiko"] = [BAND_INFO[band(p, THR)][3] for p in proba]
            n_pos = int((proba >= THR).sum())
            k1, k2, k3 = st.columns(3)
            k1.markdown(f"<div class='kpi'><div class='v'>{len(out)}</div><div class='l'>Total data</div></div>", unsafe_allow_html=True)
            k2.markdown(f"<div class='kpi'><div class='v'>{n_pos}</div><div class='l'>Terprediksi {POS}</div></div>", unsafe_allow_html=True)
            k3.markdown(f"<div class='kpi'><div class='v'>{n_pos/len(out)*100:.1f}%</div><div class='l'>Proporsi positif</div></div>", unsafe_allow_html=True)
            st.markdown("")
            st.dataframe(out)
            st.bar_chart(out["tingkat_risiko"].value_counts())
            st.download_button("⬇️ Download hasil prediksi", out.to_csv(index=False).encode("utf-8"),
                               "hasil_prediksi.csv", "text/csv", type="primary")

# =====================================================================
# TAB 3: TENTANG MODEL
# =====================================================================
with tab3:
    st.markdown("#### Kenapa model ini dipilih?")
    st.write(f"Dari 4 model (Logistic Regression, Decision Tree, Random Forest, MLP), "
             f"**{meta['model_name']}** punya ROC-AUC tertinggi pada data uji dan stabil pada cross-validation.")
    cols = st.columns(4)
    for col, key in zip(cols, ["Accuracy", "Precision", "Recall", "ROC-AUC"]):
        if key in METRICS:
            col.markdown(f"<div class='kpi'><div class='v'>{METRICS[key]:.3f}</div>"
                         f"<div class='l'>{key} (threshold 0.5)</div></div>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("#### Perbandingan 4 model")
    st.dataframe(MODEL_COMPARISON)
    st.markdown("#### Cara membaca threshold")
    st.markdown(
        f"- Data sangat **tidak seimbang** (sekitar 5% kasus positif), sehingga akurasi tinggi bisa menyesatkan.\n"
        f"- Pada threshold **0.5**, model hampir selalu menjawab *{NEG}* (recall sangat rendah).\n"
        f"- Pada threshold **optimal ({BEST_THR:.3f})**, recall naik drastis, tetapi precision turun "
        f"(lebih banyak peringatan yang ternyata salah).\n"
        f"- Untuk deteksi dini penyakit, melewatkan kasus biasanya lebih berbahaya daripada peringatan keliru.")
    st.markdown("#### Alur model")
    st.markdown("Imputasi missing value → scaling fitur numerik → one-hot encoding fitur kategorikal → klasifikasi. "
                "Semua langkah tersimpan dalam satu pipeline sehingga input dashboard diproses persis seperti saat training.")
    st.caption("Keterbatasan: hanya 50 kasus positif pada data uji, sehingga angka bisa bergeser pada data lain. "
               "Dashboard ini bukan pengganti diagnosis dokter.")
