import streamlit as st
import pandas as pd
import pickle
import joblib

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(page_title="Prediksi Injury Severity", page_icon="🚗", layout="centered")

st.title("🚗 Prediksi Injury Severity")
st.write(
    "Aplikasi ini memprediksi tingkat keparahan cedera (injury severity) "
    "berdasarkan data kecelakaan lalu lintas menggunakan model XGBoost "
    "yang sudah dilatih sebelumnya."
)

# =========================================================
# LOAD MODEL
# =========================================================
MODEL_PATH = "injury_severity_model.pkl"

@st.cache_resource
def load_model(path):
    # Coba joblib dulu (umum dipakai untuk model sklearn/XGBoost),
    # baru fallback ke pickle biasa kalau gagal.
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(
        f"File model '{MODEL_PATH}' tidak ditemukan. "
        "Pastikan file .pkl berada satu folder dengan app.py ini."
    )
    st.stop()
except Exception as e:
    st.error(
        f"Gagal memuat model: {e}\n\n"
        "Kemungkinan penyebab: file .pkl korup (misalnya rusak saat di-commit/transfer), "
        "atau formatnya bukan pickle/joblib yang valid. "
        "Coba re-export model dari notebook training dengan joblib.dump(model, 'injury_severity_model.pkl') "
        "dan pastikan file ditransfer dalam mode biner (jangan lewat Git tanpa binary handling, "
        "jangan lewat email/chat yang bisa mengubah encoding file)."
    )
    st.stop()

# =========================================================
# DAFTAR OPSI (sesuai unique value hasil eksplorasi data)
# =========================================================
PROVINSI_LIST = [
    "Kalimantan Timur", "Jawa Timur", "Jawa Tengah", "Nusa Tenggara Barat", "Aceh",
    "Kalimantan Barat", "Bali", "Bangka Belitung", "Jawa Barat", "Lampung",
    "Sulawesi Selatan", "DI Yogyakarta", "DKI Jakarta", "Kalimantan Selatan",
    "Bengkulu", "Riau", "Maluku", "Sumatera Selatan", "Maluku Utara",
    "Kepulauan Riau", "Jambi", "Gorontalo", "Sulawesi Utara", "Sumatera Utara",
    "Papua", "Banten", "Sulawesi Tenggara", "Sumatera Barat", "Sulawesi Tengah",
    "Sulawesi Barat", "Kalimantan Tengah", "Nusa Tenggara Timur",
    "Kalimantan Utara", "Papua Barat",
]

JENIS_KECELAKAAN_LIST = ["Lalu Lintas Jalan", "Penumpang Angkutan Umum"]

JENIS_KENDARAAN_LIST = [
    "Sepeda Motor", "Truk/Angkutan Barang", "Lainnya",
    "Mobil Penumpang", "Angkutan Umum/Bus",
]

GENDER_LIST = ["PRIA", "WANITA"]

# "None" muncul sebagai salah satu kategori valid pada kolom ini di data asli
JENIS_KLAIM_LIST = ["Lalu Lintas Jalan", "Penumpang Angkutan Umum", "None"]

# =========================================================
# FORM INPUT
# =========================================================
st.header("Input Data")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        usia = st.number_input(
            "Usia (tahun)", min_value=1, max_value=67, value=30, step=1,
        )
        usia_kendaraan_tahun = st.number_input(
            "Usia Kendaraan (tahun)", min_value=0, max_value=24, value=5, step=1,
        )
        jumlah_kendaraan_terlibat = st.number_input(
            "Jumlah Kendaraan Terlibat", min_value=1, max_value=3, value=1, step=1,
        )
        jumlah_klaim = st.number_input(
            "Jumlah Klaim (Rp)",
            min_value=500000, max_value=8633000, value=500000, step=50000,
        )

    with col2:
        provinsi = st.selectbox("Provinsi", options=PROVINSI_LIST)
        jenis_kecelakaan = st.selectbox("Jenis Kecelakaan", options=JENIS_KECELAKAAN_LIST)
        jenis_kendaraan = st.selectbox("Jenis Kendaraan", options=JENIS_KENDARAAN_LIST)
        gender = st.selectbox("Gender", options=GENDER_LIST)
        jenis_klaim = st.selectbox("Jenis Klaim", options=JENIS_KLAIM_LIST)

    submitted = st.form_submit_button("🔍 Prediksi")

# =========================================================
# PREDIKSI
# =========================================================
if submitted:
    # jenis_klaim "None" (string) dikembalikan ke None asli agar konsisten dgn data training
    jenis_klaim_value = None if jenis_klaim == "None" else jenis_klaim

    input_df = pd.DataFrame(
        [{
            "usia": usia,
            "usia_kendaraan_tahun": usia_kendaraan_tahun,
            "jumlah_kendaraan_terlibat": jumlah_kendaraan_terlibat,
            "jumlah_klaim": jumlah_klaim,
            "provinsi": provinsi,
            "jenis_kecelakaan": jenis_kecelakaan,
            "jenis_kendaraan": jenis_kendaraan,
            "gender": gender,
            "jenis_klaim": jenis_klaim_value,
        }]
    )

    st.subheader("Data Input")
    st.dataframe(input_df, use_container_width=True)

    try:
        prediction = model.predict(input_df)
        st.subheader("Hasil Prediksi")
        st.success(f"Prediksi Injury Severity: **{prediction[0]}**")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)
            proba_df = pd.DataFrame(proba, columns=model.classes_)
            st.write("Probabilitas tiap kelas:")
            st.dataframe(proba_df, use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi error saat melakukan prediksi: {e}")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "Model: Pipeline (ColumnTransformer + XGBClassifier) — "
    "numeric: SimpleImputer + RobustScaler, nominal: SimpleImputer + OneHotEncoder."
)
