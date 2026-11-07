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
APP_ICON = "🌻"
CREATORS = "Kelompok 1 PSD RB -- Devyna - Dharu - Ridho"

BASE_DIR = Path(__file__).parent
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

LABELS = {
    "gender": "Jenis kelamin", "age": "Usia (tahun)", "hypertension": "Riwayat hipertensi",
    "heart_disease": "Riwayat penyakit jantung", "ever_married": "Pernah menikah",
    "work_type": "Jenis pekerjaan", "Residence_type": "Tempat tinggal",
    "avg_glucose_level": "Rata-rata kadar glukosa (mg/dL)", "bmi": "BMI (Indeks Massa Tubuh)",
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
    ("Data diri", ["gender", "age", "ever_married", "Residence_type"]),
    ("Riwayat kesehatan", ["hypertension", "heart_disease", "avg_glucose_level", "bmi"]),
    ("Gaya hidup & pekerjaan", ["work_type", "smoking_status"]),
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
# BUNGA MATAHARI (SVG kecil, dipakai sebagai hiasan & penanda)
# =====================================================================
def sunflower(size=40, mood="happy", petals=12):
    pet = "".join(
        f'<ellipse cx="50" cy="21" rx="8.5" ry="17" transform="rotate({i * 360 / petals:.1f} 50 50)" '
        f'fill="{"#FFD23F" if i % 2 == 0 else "#FFBE0B"}" stroke="#E99A00" stroke-width="1.2"/>'
        for i in range(petals))
    face = ""
    if size >= 26 and mood:
        mouth = {"happy": "M41 56 Q50 65 59 56", "flat": "M43 59 L57 59", "worried": "M42 62 Q50 54 58 62"}[mood]
        face = ('<circle cx="42" cy="47" r="2.6" fill="#3B2412"/><circle cx="58" cy="47" r="2.6" fill="#3B2412"/>'
                '<circle cx="36" cy="54" r="3.4" fill="#F4A6A0" opacity=".75"/>'
                '<circle cx="64" cy="54" r="3.4" fill="#F4A6A0" opacity=".75"/>'
                f'<path d="{mouth}" stroke="#3B2412" stroke-width="2.4" fill="none" stroke-linecap="round"/>')
    return (f'<svg viewBox="0 0 100 100" width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg">'
            f'{pet}<circle cx="50" cy="50" r="21" fill="#8B5A2B" stroke="#6B3F1D" stroke-width="2"/>{face}</svg>')


# =====================================================================
# STYLE
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@600;700;800&family=Nunito:wght@400;600;700;800&display=swap');
:root {--ink:#4A2E14; --sun:#FFC83D; --sun-d:#F0A202; --cream:#FFF9E8; --line:#F5D77A; --leaf:#4C9A4A;}
.stApp {font-family:'Nunito',sans-serif; color:var(--ink); background-color:var(--cream);
        background-image:radial-gradient(#FFE6A3 1.3px, transparent 1.3px); background-size:24px 24px;}
.block-container {padding-top:1.4rem; max-width:1250px;}
#MainMenu, footer {visibility:hidden;}
h1,h2,h3,h4 {font-family:'Baloo 2',sans-serif !important; color:var(--ink);}
.stMarkdown, .stMarkdown p, label, [data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"] {color:var(--ink);}
[data-testid="stSidebar"] {background:#FFF1C2; border-right:3px dotted var(--line);}

/* tab berbentuk pil */
.stTabs [data-baseweb="tab-list"] {gap:8px; border-bottom:none;}
.stTabs [data-baseweb="tab"] {background:#fff; border:2px solid var(--line); border-radius:999px;
        padding:.45rem 1.2rem; font-weight:800; height:auto;}
.stTabs [aria-selected="true"] {background:var(--sun); border-color:var(--sun-d);}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display:none;}

/* tombol */
button[kind="primary"], button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"] {
        background:#F59E0B; color:#3B2412; border:none; border-radius:999px; font-weight:800;
        box-shadow:0 4px 0 #C77700; padding:.55rem 1.4rem;}
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover,
button[data-testid="stBaseButton-primaryFormSubmit"]:hover {background:#FFB52E; color:#3B2412; transform:translateY(-1px);}
.stDownloadButton button {border-radius:999px; border:2px solid var(--sun-d); font-weight:700;}
[data-testid="stForm"] {background:#fff; border:2px dashed var(--sun-d); border-radius:22px; padding:1.1rem 1.3rem;}

/* hero */
.hero {position:relative; overflow:hidden; background:linear-gradient(120deg,#FFE066 0%,#FFC83D 55%,#FFAE1A 100%);
       border-radius:28px; padding:1.9rem 2.2rem 1.6rem; margin-bottom:1rem; border:3px solid #fff;
       box-shadow:0 10px 0 #F0A202, 0 18px 30px rgba(240,162,2,.25);}
.hero h1 {margin:0; font-size:2.5rem; line-height:1.1; color:var(--ink);}
.hero p {margin:.45rem 0 0; font-size:1.05rem; font-weight:600; max-width:60%;}
.chip {display:inline-block; background:rgba(255,255,255,.75); color:var(--ink); padding:.25rem .85rem;
       border-radius:999px; font-size:.82rem; font-weight:700; margin:.9rem .45rem 0 0;}
.hf span {position:absolute; display:block;}
@media (max-width:760px){.hf{display:none}.hero p{max-width:100%}}
.sway {transform-origin:50% 90%; animation:sway 4.5s ease-in-out infinite;}
.d2 {animation-delay:-1.5s;} .d3 {animation-delay:-3s;}
@keyframes sway {0%,100%{transform:rotate(-5deg)} 50%{transform:rotate(5deg)}}
@media (prefers-reduced-motion:reduce){.sway{animation:none}}

/* kartu & KPI */
.card {border:2px solid var(--line); border-radius:20px; padding:1rem 1.3rem; background:#fff;}
.card h4 {margin:0 0 .3rem;}
.kpi {text-align:center; background:#fff; border:2px solid var(--line); border-top:8px solid var(--sun);
      border-radius:18px; padding:.7rem .5rem;}
.kpi .v {font-family:'Baloo 2',sans-serif; font-size:1.7rem; font-weight:800; line-height:1.2;}
.kpi .l {font-size:.78rem; font-weight:700; opacity:.7;}
.sec {display:flex; align-items:center; gap:.4rem; font-family:'Baloo 2',sans-serif; font-weight:700;
      font-size:1.1rem; margin:.5rem 0 .2rem;}

/* hasil */
.result {position:relative; border-radius:24px; padding:1.4rem 1.6rem; margin-bottom:.6rem; border:3px solid #fff;
         box-shadow:0 8px 0 rgba(0,0,0,.12);}
.r-low {background:linear-gradient(135deg,#2E9E5B,#62C98A); color:#fff;}
.r-mid {background:linear-gradient(135deg,#FFB703,#FFD45C); color:var(--ink);}
.r-high {background:linear-gradient(135deg,#E4572E,#F58A5B); color:#fff;}
.result .tag {font-size:.85rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase;}
.result .big {font-family:'Baloo 2',sans-serif; font-size:3rem; font-weight:800; line-height:1.05;}
.result .sub {font-weight:600; max-width:78%;}
.badge {position:absolute; right:16px; top:16px; background:rgba(255,255,255,.9); border-radius:50%;
        width:84px; height:84px; display:flex; align-items:center; justify-content:center;}
.gauge {position:relative; height:16px; border-radius:999px; margin:1.8rem 0 2.2rem;
        background:linear-gradient(90deg,#62C98A 0%,#FFC83D 50%,#E4572E 100%); box-shadow:inset 0 0 0 2px #fff;}
.mk {position:absolute; transform:translateX(-50%);}
.mk.thr {top:-8px; width:0; height:32px; border-left:3px dashed #4A2E14;}
.mk.you {top:-13px;}
.mk-l {position:absolute; top:28px; font-size:.72rem; font-weight:700; transform:translateX(-50%); white-space:nowrap;}
.fx-row {display:flex; align-items:center; margin:.3rem 0; font-size:.9rem; font-weight:600;}
.fx-name {width:46%; padding-right:.5rem;}
.fx-bar {flex:1; height:12px; background:#F6E8BF; border-radius:8px; position:relative;}
.fx-fill {position:absolute; top:0; height:12px; border-radius:8px;}
.fx-val {width:72px; text-align:right; font-variant-numeric:tabular-nums;}
.note {font-size:.8rem; opacity:.75;}
.flow {display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; margin:.4rem 0 1rem;}
.flow .s {background:#fff; border:2px solid var(--line); border-radius:999px; padding:.3rem .9rem; font-weight:700;}
.flow .a {font-weight:800; color:var(--sun-d);}
.meadow {display:flex; justify-content:space-around; align-items:flex-end; border-bottom:10px solid #6DBB5A;
         border-radius:0 0 8px 8px; margin:2.2rem 0 .4rem; padding:0 6%;}
.stem {width:4px; background:var(--leaf); border-radius:3px; margin:-4px auto 0;}
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
    "low": ("r-low", "Risiko rendah", "Probabilitas masih jauh di bawah batas deteksi model.", "Rendah", "happy"),
    "mid": ("r-mid", "Waspada", "Probabilitas mendekati batas deteksi model. Perhatikan faktor risikonya.", "Waspada", "flat"),
    "high": ("r-high", "Risiko tinggi", "Probabilitas melewati batas deteksi, jadi model memprediksi kelas positif.", "Tinggi", "worried"),
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
        color = "#E4572E" if v > 0 else "#2E9E5B"
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
    css, title, text, _, mood = BAND_INFO[b]
    verdict = POS if p >= thr else NEG
    st.markdown(
        f"<div class='result {css}'><div class='badge'>{sunflower(64, mood)}</div>"
        f"<div class='tag'>{title}</div><div class='big'>{p*100:.1f}%</div>"
        f"<div class='sub'>Probabilitas kelas positif ({POS}), atau {p/thr:.1f}x dari batas keputusan. "
        f"Prediksi model: <b>{verdict}</b>.<br>{text}</div></div>",
        unsafe_allow_html=True)

    scale = max(3 * thr, min(1.0, p * 1.25))
    you = min(p / scale * 100, 97)
    tpos = min(thr / scale * 100, 99)
    st.markdown(
        f"<div class='gauge'>"
        f"<div class='mk thr' style='left:{tpos}%'></div><div class='mk-l' style='left:{tpos}%'>batas {thr*100:.1f}%</div>"
        f"<div class='mk you' style='left:{you}%'>{sunflower(40, mood)}</div></div>",
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
    st.markdown(f"<div class='sec'>{sunflower(34)} {APP_TITLE}</div>", unsafe_allow_html=True)
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
chips = f"<span class='chip'>Model: {meta['model_name']}</span><span class='chip'>Kelas positif: {POS}</span>"
flowers = (f"<div class='hf'><span class='sway' style='right:3%;top:10%'>{sunflower(104)}</span>"
           f"<span class='sway d2' style='right:19%;top:58%'>{sunflower(56)}</span>"
           f"<span class='sway d3' style='right:27%;top:8%'>{sunflower(38)}</span></div>")
st.markdown(
    f"<div class='hero'>{flowers}<h1>{APP_TITLE}</h1>"
    f"<p>Masukkan data pasien, lalu model machine learning akan memperkirakan risikonya.</p>"
    f"{chips}<div style='font-size:.88rem;margin-top:.9rem;font-weight:600'>Dibuat oleh: <b>{CREATORS}</b></div></div>",
    unsafe_allow_html=True)

auc = METRICS.get("ROC-AUC")
k = st.columns(4)
kpis = [(meta["model_name"], "Model terpilih"),
        (f"{auc:.3f}" if auc is not None else "-", "ROC-AUC data uji"),
        (f"{THR:.3f}", "Threshold aktif"),
        (str(len(FEATURES)), "Fitur input")]
for c, (v, l) in zip(k, kpis):
    c.markdown(f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{l}</div></div>", unsafe_allow_html=True)
st.markdown("")

tab1, tab2, tab3 = st.tabs(["Prediksi Individu", "Prediksi Massal (CSV)", "Tentang Model"])

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
        groups = GROUPS + ([("Lainnya", [c for c in FEATURES if c not in grouped])]
                           if any(c not in grouped for c in FEATURES) else [])
        values = {}
        with st.form("form_prediksi"):
            for title, cols in groups:
                cols = [c for c in cols if c in FEATURES]
                if not cols:
                    continue
                st.markdown(f"<div class='sec'>{sunflower(24, None)} {title}</div>", unsafe_allow_html=True)
                grid = st.columns(2)
                for i, c in enumerate(cols):
                    with grid[i % 2]:
                        values[c] = render_input(c)
            submitted = st.form_submit_button("Prediksi sekarang", type="primary")
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
                f"<div class='card' style='text-align:center;padding:2rem 1.3rem'>{sunflower(90)}"
                "<h4>Hasil akan tampil di sini</h4>"
                "Isi data di sebelah kiri atau pilih contoh profil, lalu klik <b>Prediksi sekarang</b>.</div>",
                unsafe_allow_html=True)

# =====================================================================
# TAB 2: BATCH
# =====================================================================
with tab2:
    st.markdown("Upload file CSV berisi banyak pasien sekaligus. Kolom harus sama dengan template.")
    template = pd.DataFrame([{c: (NUM[c]["median"] if c in NUM else CAT[c][0]) for c in FEATURES}])
    st.download_button("Download template CSV", template.to_csv(index=False).encode("utf-8"),
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
            order = ["Rendah", "Waspada", "Tinggi"]
            st.bar_chart(out["tingkat_risiko"].value_counts().reindex(order, fill_value=0), color="#F5A623")
            st.download_button("Download hasil prediksi", out.to_csv(index=False).encode("utf-8"),
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
    c1, c2 = st.columns([1.3, 1], gap="large")
    c1.dataframe(MODEL_COMPARISON)
    c2.bar_chart(MODEL_COMPARISON["ROC-AUC (uji)"], color="#F5A623")
    st.markdown("#### Cara membaca threshold")
    st.markdown(
        f"- Data sangat **tidak seimbang** (sekitar 5% kasus positif), sehingga akurasi tinggi bisa menyesatkan.\n"
        f"- Pada threshold **0.5**, model hampir selalu menjawab *{NEG}* (recall sangat rendah).\n"
        f"- Pada threshold **optimal ({BEST_THR:.3f})**, recall naik drastis, tetapi precision turun "
        f"(lebih banyak peringatan yang ternyata salah).\n"
        f"- Untuk deteksi dini penyakit, melewatkan kasus biasanya lebih berbahaya daripada peringatan keliru.")
    st.markdown("#### Alur model")
    st.markdown("<div class='flow'><span class='s'>Imputasi missing value</span><span class='a'>&rarr;</span>"
                "<span class='s'>Scaling fitur numerik</span><span class='a'>&rarr;</span>"
                "<span class='s'>One-hot encoding kategorikal</span><span class='a'>&rarr;</span>"
                "<span class='s'>Klasifikasi</span></div>", unsafe_allow_html=True)
    st.markdown("Semua langkah tersimpan dalam satu pipeline sehingga input dashboard diproses persis seperti saat training.")
    st.caption("Keterbatasan: hanya 50 kasus positif pada data uji, sehingga angka bisa bergeser pada data lain. "
               "Dashboard ini bukan pengganti diagnosis dokter.")

# =====================================================================
# FOOTER: padang bunga matahari
# =====================================================================
plants = "".join(
    f"<div class='sway' style='animation-delay:-{i * .6:.1f}s'>{sunflower(s)}<div class='stem' style='height:{h}px'></div></div>"
    for i, (s, h) in enumerate([(30, 24), (22, 12), (38, 34), (26, 16), (34, 28), (22, 10), (36, 30), (26, 18)]))
st.markdown(f"<div class='meadow'>{plants}</div>", unsafe_allow_html=True)
st.caption("Dibuat dengan Streamlit. Bukan alat diagnosis medis.")
