# ==========================================================
# IMPORT LIBRARY
# ==========================================================
import streamlit as st
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import time
import textwrap
# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Career Recommendation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================================
# LOAD CSS
# ==========================================================

def load_css():
    css_path = Path("assets/style.css")

    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True
            )

load_css()

# ==========================================================
# LOAD MODEL
# ==========================================================
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "model" / "model_knn.pkl")
    scaler = joblib.load(BASE_DIR / "model" / "scaler.pkl")
    label_encoder = joblib.load(BASE_DIR / "model" / "label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, label_encoder = load_model()
# ==========================================================
# SESSION STATE
# ==========================================================

DEFAULT_STATE = {

    "nama": "",
    "angkatan": "2021",
    "jenis_kelamin": "Laki-laki",
    "semester": "8",

    "karier": None,
    "confidence": 0,
    "probabilitas": None,
    "ranking": None,

    "hasil": False
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Session pertanyaan
for i in range(1, 26):
    st.session_state.setdefault(f"q{i}", None)

# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🎓 Sistem Rekomendasi Karier Mahasiswa Sistem Informasi</div>
        <p class="hero-description">
            Aplikasi ini membantu mahasiswa memperoleh rekomendasi karier
            berdasarkan hasil assessment
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ==========================================================
# LAYOUT
# ==========================================================

left, right = st.columns(
    [9, 1],
    gap="large"
)

# ==========================================================
# PANEL KIRI
# ==========================================================
with left:
    st.markdown("## 👤 Profil Mahasiswa")

    col1, col2 = st.columns(2)

    with col1:
        nama = st.text_input(
            "Nama Lengkap *",
            value=st.session_state["nama"],
            placeholder="Masukkan nama lengkap"
        )

    with col2:
        angkatan = st.selectbox(
            "Angkatan *",
            [
                "2021",
                "2022",
                "2023",
                "2024",
                "2025"
            ],
            index=[
                "2021",
                "2022",
                "2023",
                "2024",
                "2025"
            ].index(st.session_state["angkatan"])
        )

    col3, col4 = st.columns(2)
    with col3:

        jenis_kelamin = st.selectbox(
            "Jenis Kelamin *",
            [
                "Laki-laki",
                "Perempuan"
            ]
        )

    with col4:
        semester = st.selectbox(
            "Semester *",
            [
                "6",
                "7",
                "8",
                "9",
                "10"
            ]
        )

    # Simpan Session
    st.session_state["nama"] = nama
    st.session_state["angkatan"] = angkatan
    st.session_state["jenis_kelamin"] = jenis_kelamin
    st.session_state["semester"] = semester

st.divider()

# Set konfigurasi halaman
st.set_page_config(page_title="Sistem Rekomendasi Karier", page_icon="📝", layout="wide")

# ==========================================================
# CUSTOM CSS UNTUK VISUAL YANG LEBIH BAGUS
# ==========================================================
st.markdown("""
<style>
    /* Styling Header */
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    
    /* Card Petunjuk Rating */
    .info-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 25px;
    }
    .rating-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 10px;
    }
    .rating-badge {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.88rem;
        font-weight: 500;
        color: #334155;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }

    /* Card untuk setiap Pertanyaan */
    .question-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 20px 8px 20px;
        margin-bottom: 16px;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .question-card:hover {
        border-color: #3B82F6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.08);
    }
    .question-text {
        font-weight: 600;
        font-size: 1rem;
        color: #0F172A;
        margin-bottom: 8px;
    }

    /* Kustomisasi Radio Button agar seperti Pill */
    div[role="radiogroup"] {
        gap: 12px !important;
    }
    div[role="radiogroup"] label {
        background-color: #F1F5F9 !important;
        padding: 6px 16px !important;
        border-radius: 8px !important;
        border: 1px solid #E2E8F0 !important;
        transition: all 0.2s ease !important;
    }
    div[role="radiogroup"] label:hover {
        background-color: #E2E8F0 !important;
        border-color: #CBD5E1 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER & PETUNJUK
# ==========================================================
st.markdown('<div class="main-title">📝 Assessment Karier</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
    <span style="color: #475569; font-weight: 500;">Silakan berikan penilaian yang paling sesuai dengan kondisi Anda:</span>
    <div class="rating-grid">
        <div class="rating-badge">⭐ 1 = Sangat Tidak Sesuai</div>
        <div class="rating-badge">⭐⭐ 2 = Tidak Sesuai</div>
        <div class="rating-badge">⭐⭐⭐ 3 = Cukup Sesuai</div>
        <div class="rating-badge">⭐⭐⭐⭐ 4 = Sesuai</div>
        <div class="rating-badge">⭐⭐⭐⭐⭐ 5 = Sangat Sesuai</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# FUNGSI RADIO
# ==========================================================
def tampil_radio(no, pertanyaan):
    st.markdown(f'<div class="question-card"><div class="question-text">{no}. {pertanyaan}</div>', unsafe_allow_html=True)
    
    st.radio(
        label=pertanyaan,
        options=[1, 2, 3, 4, 5],
        key=f"q{no}",
        horizontal=True,
        index=None,
        label_visibility="collapsed"  # Menyembunyikan label bawaan Streamlit agar tidak double
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# TAB ASSESSMENT
# ==========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Bagian 1",
    " Bagian 2",
    " Bagian 3",
    " Bagian 4",
    " Bagian 5"
])

# ======================================================
# BAGIAN 1 - SOFTWARE DEVELOPER
# ======================================================
with tab1:
    tampil_radio(1, "Saya suka menulis kode program menggunakan bahasa pemrograman untuk membangun aplikasi.")
    tampil_radio(2, "Saya senang membuat dokumentasi program sebagai bagian dari proses pengembangan perangkat lunak.")
    tampil_radio(3, "Saya menikmati mencari dan memperbaiki kesalahan (bug) pada kode program.")
    tampil_radio(4, "Saya tertarik mengembangkan sistem atau aplikasi sesuai kebutuhan pengguna.")
    tampil_radio(5, "Saya senang melakukan pengujian (testing) untuk memastikan aplikasi berjalan dengan baik.")

# ======================================================
# BAGIAN 2 - DATA ANALYST
# ======================================================
with tab2:
    tampil_radio(6, "Saya senang mengolah data menggunakan Excel, SQL, atau tools analisis data lainnya.")
    tampil_radio(7, "Saya tertarik menganalisis data untuk menemukan pola, tren, atau insight.")
    tampil_radio(8, "Saya suka menggunakan Python atau bahasa pemrograman lainnya untuk mengolah data.")
    tampil_radio(9, "Saya senang melakukan pengolahan data sebelum proses analisis.")
    tampil_radio(10, "Saya menikmati menyusun laporan berdasarkan hasil analisis data.")

# ======================================================
# BAGIAN 3 - SYSTEM ANALYST
# ======================================================
with tab3:
    tampil_radio(11, "Saya senang menganalisis kebutuhan pengguna sebelum sistem dikembangkan.")
    tampil_radio(12, "Saya suka mendokumentasikan kebutuhan pengguna ke dalam spesifikasi kebutuhan sistem.")
    tampil_radio(13, "Saya tertarik mempelajari konsep SDLC, basis data, dan dasar pemrograman.")
    tampil_radio(14, "Saya senang membuat perancangan basis data menggunakan Entity Relationship Diagram (ERD).")
    tampil_radio(15, "Saya menikmati melakukan analisis dan perancangan sistem sesuai kebutuhan pengguna.")

# ======================================================
# BAGIAN 4 - IT SUPPORT
# ======================================================
with tab4:
    tampil_radio(16, "Saya senang melakukan instalasi dan konfigurasi sistem operasi maupun perangkat lunak.")
    tampil_radio(17, "Saya menikmati melakukan maintenance dan troubleshooting pada komputer atau perangkat pendukung.")
    tampil_radio(18, "Saya tertarik mempelajari konfigurasi dan pemeliharaan jaringan komputer.")
    tampil_radio(19, "Saya senang membantu pengguna menyelesaikan permasalahan perangkat keras maupun perangkat lunak.")
    tampil_radio(20, "Saya suka menangani permasalahan teknis yang berkaitan dengan perangkat komputer.")

# ======================================================
# BAGIAN 5 - UI/UX DESIGNER
# ======================================================
with tab5:
    tampil_radio(21, "Saya tertarik melakukan user research sebelum merancang antarmuka aplikasi.")
    tampil_radio(22, "Saya senang memanfaatkan masukan pengguna untuk meningkatkan kualitas desain.")
    tampil_radio(23, "Saya suka membuat wireframe sebelum mendesain aplikasi atau website.")
    tampil_radio(24, "Saya senang membuat desain antarmuka yang menarik dan mudah digunakan.")
    tampil_radio(25, "Saya tertarik melakukan usability testing untuk mengevaluasi pengalaman pengguna.")

# ==========================================================
# PROGRESS
# ==========================================================

jumlah_jawab = sum(
    st.session_state[f"q{i}"] is not None
    for i in range(1,26)
)

progress = jumlah_jawab / 25

st.progress(progress)

st.caption(
    f"Progress Assessment : {jumlah_jawab}/25 Pertanyaan"
)

# ==========================================================
# VALIDASI
# ==========================================================

belum = []

for i in range(1, 26):
    if st.session_state[f"q{i}"] is None:
        belum.append(i)

if belum:
    st.error("⚠️ Assessment belum lengkap.")
    st.warning(f"Masih ada **{len(belum)} pertanyaan** yang belum dijawab.")
    st.stop()
   
# ==========================================================
# PREDIKSI
# ==========================================================
prediksi = st.button(
    "🚀 Lihat Rekomendasi Karier",
    use_container_width=True,
    type="primary"
)

if prediksi:

    belum = []

    for i in range(1, 26):
        if st.session_state[f"q{i}"] is None:
            belum.append(i)

    if len(belum) > 0:
        st.error("⚠️ Masih ada pertanyaan assessment yang belum dijawab.")
        st.stop()

    jenis_kelamin = 1 if st.session_state["jenis_kelamin"] == "Laki-laki" else 0

    # Samakan encoding dengan saat training
    angkatan_map = {
        "2021": 0,
        "2022": 1,
        "2023": 2,
        "2024": 3,
        "2025": 4
    }

    tahun_kelulusan = angkatan_map[st.session_state["angkatan"]]

    data_input = [[
        jenis_kelamin,
        tahun_kelulusan,

        st.session_state["q1"],
        st.session_state["q2"],
        st.session_state["q3"],
        st.session_state["q4"],
        st.session_state["q5"],
        st.session_state["q6"],
        st.session_state["q7"],
        st.session_state["q8"],
        st.session_state["q9"],
        st.session_state["q10"],
        st.session_state["q11"],
        st.session_state["q12"],
        st.session_state["q13"],
        st.session_state["q14"],
        st.session_state["q15"],
        st.session_state["q16"],
        st.session_state["q17"],
        st.session_state["q18"],
        st.session_state["q19"],
        st.session_state["q20"],
        st.session_state["q21"],
        st.session_state["q22"],
        st.session_state["q23"],
        st.session_state["q24"],
        st.session_state["q25"]
    ]]

    # Validasi jumlah fitur
    if len(data_input[0]) != 27:
        st.error(f"Jumlah fitur tidak sesuai. Ditemukan {len(data_input[0])} fitur, seharusnya 27.")
        st.stop()

    # Validasi tidak boleh ada nilai kosong
    if any(x is None for x in data_input[0]):
        st.error("Masih ada pertanyaan yang belum dijawab.")
        st.stop()

    try:
        data_scaled = scaler.transform(data_input)
    except Exception as e:
        st.error(str(e))
        st.stop()

if prediksi:

    hasil = model.predict(data_scaled)
    probabilitas = model.predict_proba(data_scaled)

    confidence = float(np.max(probabilitas) * 100)

    karier = label_encoder.inverse_transform(hasil)[0]

    st.session_state["probabilitas"] = probabilitas
    st.session_state["confidence"] = confidence
    st.session_state["Bidang Pekerjaan"] = karier

# ==========================================================
# DEFINE DEFAULT SKOR (Agar tidak NameError)
# ==========================================================
skor_programmer = skor_programmer if 'skor_programmer' in locals() else 0
skor_data_analyst = skor_data_analyst if 'skor_data_analyst' in locals() else 0
skor_system_analyst = skor_system_analyst if 'skor_system_analyst' in locals() else 0
skor_it_support = skor_it_support if 'skor_it_support' in locals() else 0
skor_uiux = skor_uiux if 'skor_uiux' in locals() else 0
# ============================================
# DASHBOARD HASIL PREDIKSI
# ==========================================================
if st.session_state["probabilitas"] is None:
    st.stop()

prob = st.session_state["probabilitas"][0]

ranking = sorted(
    zip(label_encoder.classes_, prob),
    key=lambda x: x[1],
    reverse=True
)

profesi_utama = ranking[0][0]
nilai_utama = ranking[0][1] * 100

profesi_kedua = ranking[1][0]
nilai_kedua = ranking[1][1] * 100

selisih = nilai_utama - nilai_kedua

st.divider()

st.header("🎯 Hasil Rekomendasi Karier")

st.markdown("""
Sistem telah menganalisis jawaban assessment menggunakan algoritma
**K-Nearest Neighbor (KNN)** berdasarkan hasil assessment dan profil
pendukung yang Anda berikan. Berikut merupakan hasil interpretasi
rekomendasi karier yang diperoleh.
""")

if st.session_state.get("confidence") is None:
    st.stop()

confidence = st.session_state["confidence"]

if confidence >= 85:

    st.success("""
Profil Anda menunjukkan tingkat kecocokan yang **sangat baik** terhadap
profesi yang direkomendasikan. Hal ini mengindikasikan bahwa kompetensi,
minat, dan kemampuan yang dimiliki telah sesuai dengan karakteristik
profesi tersebut.

Anda dapat mulai mempersiapkan diri memasuki dunia kerja dengan
membangun portofolio, mengikuti program magang, maupun memperoleh
sertifikasi yang relevan.
""")

elif confidence >= 70:

    st.info("""
Profil Anda menunjukkan tingkat kecocokan yang **baik** terhadap profesi
yang direkomendasikan. Kompetensi dasar yang dimiliki sudah cukup sesuai,
namun masih terdapat beberapa aspek yang dapat dikembangkan agar lebih
siap bersaing di dunia kerja.
""")

elif confidence >= 50:

    st.warning("""
Profil Anda memiliki **potensi** pada profesi yang direkomendasikan.
Meskipun demikian, masih diperlukan peningkatan kompetensi melalui
pelatihan, penyelesaian proyek, maupun pengalaman praktik agar kesiapan
karier semakin optimal.
""")

else:

    st.error("""
Profil Anda masih memerlukan pengembangan lebih lanjut agar sesuai dengan
kebutuhan profesi yang direkomendasikan. Disarankan untuk memperdalam
kompetensi dasar, mengikuti pelatihan, dan melakukan assessment kembali
setelah kemampuan meningkat.
""")

# ==========================================================
# REKOMENDASI PENGEMBANGAN
# ==========================================================
    st.subheader("💡 Rekomendasi Pengembangan")

    if confidence >= 85:

        rekomendasi = [
            "Bangun portofolio profesional.",
            "Ikuti proses rekrutmen atau program magang.",
            "Perbanyak pengalaman proyek nyata.",
            "Ikuti sertifikasi untuk meningkatkan nilai jual."
        ]

    elif confidence >= 70:

        rekomendasi = [
            "Perdalam kompetensi teknis sesuai profesi.",
            "Bangun lebih banyak proyek sebagai portofolio.",
            "Ikuti sertifikasi yang relevan.",
            "Perluas pengalaman melalui magang atau freelance."
        ]

    elif confidence >= 50:

        rekomendasi = [
            "Pelajari kembali materi dasar sesuai bidang.",
            "Ikuti kursus atau bootcamp.",
            "Latihan mengerjakan studi kasus.",
            "Bangun portofolio sederhana.",
            "Ikuti kegiatan magang apabila memungkinkan."
        ]

    else:

        rekomendasi = [
            "Pelajari kembali kompetensi dasar profesi ini.",
            "Ikuti kursus online secara bertahap.",
            "Kembangkan kemampuan teknis melalui latihan rutin.",
            "Eksplorasi profesi lain yang memiliki nilai kecocokan lebih tinggi.",
            "Lakukan assessment kembali setelah kemampuan meningkat."
        ]


    for i, item in enumerate(rekomendasi, start=1):
        st.write(f"{i}. {item}")

# ==========================================================
# VISUALISASI HASIL & INTERPRETASI
# ==========================================================
col1, col2 = st.columns([1.1, 0.9], gap="large")

# ----------------------------------------------------------
# KOLOM 1: PIE CHART
# ----------------------------------------------------------
with col1:
    st.markdown("### 📈 Tingkat Kecocokan Profesi")

    pie_df = pd.DataFrame({
        "Profesi": label_encoder.classes_,
        "Persentase": prob * 100
    })

    fig = px.pie(
        pie_df,
        names="Profesi",
        values="Persentase",
        hole=0.45,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Kecocokan: %{value:.2f}%<extra></extra>"
    )

    fig.update_layout(
        showlegend=False,
        margin=dict(t=20, b=20, l=10, r=10),
        height=320
    )

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# KOLOM 2: INTERPRETASI HASIL
# ----------------------------------------------------------
with col2:
    st.markdown("### 📋 Interpretasi Hasil")

    # Penentuan Tingkat & Warna Badge
    if confidence >= 90:
        tingkat, bg_color, text_color = "Sangat Tinggi", "#E8F5E9", "#2E7D32"
    elif confidence >= 80:
        tingkat, bg_color, text_color = "Tinggi", "#E3F2FD", "#1565C0"
    elif confidence >= 70:
        tingkat, bg_color, text_color = "Cukup Tinggi", "#FFF8E1", "#F57F17"
    elif confidence >= 60:
        tingkat, bg_color, text_color = "Sedang", "#FFF3E0", "#E65100"
    else:
        tingkat, bg_color, text_color = "Perlu Evaluasi", "#FFEBEE", "#C62828"

    # Card Informasi Utama
    st.markdown(
        f"""
        <div style="background-color: {bg_color}; padding: 12px 16px; border-radius: 8px; border-left: 5px solid {text_color}; margin-bottom: 15px;">
            <span style="color: {text_color}; font-weight: bold; font-size: 0.9em;">TINGKAT KEYAKINAN MODEL</span>
            <h3 style="margin: 0; color: {text_color}; font-size: 1.4em;">🎯 {tingkat} ({confidence:.2f}%)</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Ringkasan Nilai Pendukung & Deskripsi
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric(label="Rekomendasi Utama", value=profesi_utama)
    with col_stat2:
        st.metric(label="Hasil Rekomendasi", value=f"{confidence:.2f}%")

    st.markdown(
        f"""
        <div style="font-size: 0.88em; color: #4A5568; line-height: 1.5; margin-top: 10px;">
        Model <b>K-Nearest Neighbor (KNN)</b> mendeteksi tingkat kemiripan karakteristik Anda sebesar <b>{confidence:.2f}%</b> terhadap posisi <b>{profesi_utama}</b>.<br><br>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================================================
# RANKING REKOMENDASI & REKOMENDASI PENGEMBANGAN
# ==========================================================

col_rank1, col_rank2 = st.columns(2, gap="medium")

# ==========================================================
# RANKING & REKOMENDASI PENGEMBANGAN (SAFE VERSION)
# ==========================================================

# Keamanan jika variabel ranking belum didefinisikan
if 'ranking' not in locals():
    ranking = ["Software Developer", "IT Support", "Data Analyst", "System Analyst", "UI/UX Design"]

col1, col2 = st.columns(2, gap="medium")

# ----------------------------------------------------------
# 1. RANKING REKOMENDASI KARIER
# ----------------------------------------------------------
with col1:
    emoji_list = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    ranking_items = []

    for i, item in enumerate(ranking[:5]):
        # Ekstrak nama profesi baik jika bentuknya ("Software", 90) maupun cuma "Software"
        profesi = item[0] if isinstance(item, (tuple, list)) else item
        
        icon = emoji_list[i] if i < len(emoji_list) else "🔹"
        border_color = "#1E88E5" if i == 0 else "#cbd5e1"
        font_style = "font-weight: bold; color: #1E88E5;" if i == 0 else "color: #333;"
        
        item_html = f'<div style="display: flex; align-items: center; justify-content: space-between; padding: 10px 14px; margin-bottom: 8px; background-color: #f8f9fa; border-radius: 8px; border-left: 4px solid {border_color};"><span style="font-size: 0.95em; {font_style}">{icon} {profesi}</span><span style="font-size: 0.8em; color: #64748b; background: #e2e8f0; padding: 2px 8px; border-radius: 12px;">Peringkat {i+1}</span></div>'
        ranking_items.append(item_html)

    card_ranking_html = f'''
    <div style="background: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 4px 12px rgba(0,0,0,0.03);">
        <h3 style="margin-top:0; margin-bottom:15px; font-size: 1.2em; color: #1e293b;">🏆 Ranking Rekomendasi Karier</h3>
        <hr style="margin-bottom: 15px; border: 0; border-top: 1px solid #eee;">
        {"".join(ranking_items)}
    </div>
    '''
    
    st.markdown(card_ranking_html, unsafe_allow_html=True)

# ----------------------------------------------------------
# 2. REKOMENDASI PENGEMBANGAN
# ----------------------------------------------------------
with col2:
    conf = confidence if 'confidence' in locals() else 80

    if conf >= 85:
        rekomendasi = [
            "Bangun portofolio profesional.",
            "Ikuti proses rekrutmen atau program magang.",
            "Perbanyak pengalaman proyek nyata.",
            "Ikuti sertifikasi untuk meningkatkan nilai jual."
        ]
    elif conf >= 70:
        rekomendasi = [
            "Perdalam kompetensi teknis sesuai profesi.",
            "Bangun lebih banyak proyek sebagai portofolio.",
            "Ikuti sertifikasi yang relevan.",
            "Perluas pengalaman melalui magang atau freelance."
        ]
    elif conf >= 50:
        rekomendasi = [
            "Pelajari kembali materi dasar sesuai bidang.",
            "Ikuti kursus atau bootcamp.",
            "Latihan mengerjakan studi kasus.",
            "Bangun portofolio sederhana.",
            "Ikuti kegiatan magang apabila memungkinkan."
        ]
    else:
        rekomendasi = [
            "Pelajari kembali kompetensi dasar profesi ini.",
            "Ikuti kursus online secara bertahap.",
            "Kembangkan kemampuan teknis melalui latihan rutin.",
            "Eksplorasi profesi lain yang memiliki nilai kecocokan lebih tinggi.",
            "Lakukan assessment kembali setelah kemampuan meningkat."
        ]

    rekomendasi_items = []
    for i, item in enumerate(rekomendasi, start=1):
        item_html = f'<div style="display: flex; align-items: flex-start; gap: 12px; margin-bottom: 10px; padding: 10px 12px; background: #f0fdf4; border-radius: 8px; border: 1px solid #dcfce7;"><span style="background: #22c55e; color: white; border-radius: 50%; width: 22px; height: 22px; display: flex; align-items: center; justify-content: center; font-size: 0.75em; font-weight: bold; flex-shrink: 0; margin-top: 2px;">{i}</span><span style="font-size: 0.9em; color: #166534; line-height: 1.4;">{item}</span></div>'
        rekomendasi_items.append(item_html)

    card_rekomendasi_html = f'''
    <div style="background: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 4px 12px rgba(0,0,0,0.03);">
        <h3 style="margin-top:0; margin-bottom:15px; font-size: 1.2em; color: #1e293b;">💡 Rekomendasi Pengembangan</h3>
        <hr style="margin-bottom: 15px; border: 0; border-top: 1px solid #eee;">
        {"".join(rekomendasi_items)}
    </div>
    '''
    
    st.markdown(card_rekomendasi_html, unsafe_allow_html=True)

# ==========================================================
# CARD INFORMASI KARIER
# ==========================================================

career_info = {
# ==========================================================
# Software Developer
# ==========================================================
    "Software Developer": {

        "tugas": [
            "Menganalisis kebutuhan pengguna sebelum proses pengembangan aplikasi.",
            "Merancang dan mengembangkan aplikasi atau sistem perangkat lunak.",
            "Melakukan pengujian (testing) dan debugging untuk memastikan kualitas aplikasi.",
            "Memelihara serta meningkatkan performa dan keamanan sistem.",
            "Berkolaborasi dengan tim pengembang selama proses pengembangan perangkat lunak."
        ],

        "skills": [
            "Programming – Mengembangkan aplikasi menggunakan bahasa pemrograman.",
            "Object-Oriented Programming (OOP) – Membangun aplikasi yang terstruktur dan mudah dikembangkan.",
            "SQL – Mengelola serta mengambil data dari database.",
            "Git & GitHub – Mengelola versi kode dan kolaborasi pengembangan.",
            "Algorithms & Data Structures – Merancang solusi yang efisien untuk berbagai permasalahan.",
            "Problem Solving – Menganalisis dan menyelesaikan permasalahan teknis.",
        ],

        "tools": [
            "Visual Studio Code – Editor kode untuk mengembangkan aplikasi.",
            "Git – Mengelola versi source code.",
            "GitHub – Menyimpan source code dan kolaborasi tim.",
            "MySQL – Mengelola database aplikasi.",
            "Postman – Menguji dan mendokumentasikan API.",
            "Docker – Menjalankan aplikasi dalam container."
        ],

        "belajar": [
            {
                "nama": "freeCodeCamp",
                "deskripsi": "Tutorial pemrograman dan pengembangan web secara gratis.",
                "url": "https://www.freecodecamp.org"
            },
            {
                "nama": "W3Schools",
                "deskripsi": "Belajar HTML, CSS, JavaScript, SQL, dan berbagai bahasa pemrograman.",
                "url": "https://www.w3schools.com"
            },
            {
                "nama": "Coursera",
                "deskripsi": "Kursus Software Engineering dan Programming dari berbagai universitas.",
                "url": "https://www.coursera.org"
            },
            {
                "nama": "Dicoding",
                "deskripsi": "Kelas pengembangan aplikasi Android, Web, dan Cloud berbahasa Indonesia.",
                "url": "https://www.dicoding.com"
            },
            {
                "nama": "YouTube",
                "deskripsi": "Belajar melalui channel seperti Programming with Mosh, Bro Code, dan Web Programming UNPAS.",
                "url": "https://www.youtube.com"
            },
            {
                "nama": "GitHub",
                "deskripsi": "Mempelajari project open-source dan berlatih kolaborasi pengembangan perangkat lunak.",
                "url": "https://github.com"
            }

                ],

        "sertifikasi": [
            {
                "nama": "Meta Back-End Developer Professional Certificate",
                "penyelenggara": "Coursera (Meta)",
                "url": "https://www.coursera.org/professional-certificates/meta-back-end-developer"
            },
            {
                "nama": "Meta Front-End Developer Professional Certificate",
                "penyelenggara": "Coursera (Meta)",
                "url": "https://www.coursera.org/professional-certificates/meta-front-end-developer"
            },
            {
                "nama": "IBM Full Stack Software Developer Professional Certificate",
                "penyelenggara": "Coursera (IBM)",
                "url": "https://www.coursera.org/professional-certificates/ibm-full-stack-cloud-developer"
            },
            {
                "nama": "Microsoft Certified: Azure Developer Associate",
                "penyelenggara": "Microsoft Learn",
                "url": "https://learn.microsoft.com/credentials/certifications/azure-developer/"
            },
            {
                "nama": "AWS Certified Developer – Associate",
                "penyelenggara": "Amazon Web Services (AWS)",
                "url": "https://aws.amazon.com/certification/certified-developer-associate/"
            }
        ],
        "pengembangan_kompetensi": [
            "Pelajari bahasa pemrograman seperti Python, Java, atau JavaScript secara mendalam.",
            "Kuasai framework pengembangan sesuai bidang yang diminati, seperti Laravel, React, atau Spring Boot.",
            "Bangun portofolio melalui proyek pribadi maupun kontribusi pada proyek open-source.",
            "Pelajari konsep database, API, dan version control menggunakan Git.",
            "Ikuti pelatihan atau sertifikasi untuk meningkatkan kompetensi di bidang pengembangan perangkat lunak."
        ],
        "gaji_prospek": {
            "entry_level": "Rp6.000.000 – Rp10.000.000/bulan",
            "mid_level": "Rp10.000.000 – Rp18.000.000/bulan",
            "senior": "Rp18.000.000 – Rp35.000.000+/bulan",
            "prospek": "Sangat tinggi. Software Developer dibutuhkan di berbagai sektor, seperti perusahaan teknologi, startup, perbankan, e-commerce, kesehatan, manufaktur, hingga instansi pemerintahan.",
            "catatan": "Estimasi gaji dapat berbeda tergantung lokasi, pengalaman, teknologi yang dikuasai, dan kebijakan perusahaan."
        },
        "peluang_karier": [
            {
                "platform": "LinkedIn Jobs",
                "deskripsi": "Platform profesional untuk mencari lowongan kerja dan membangun jaringan karier.",
                "url": "https://www.linkedin.com/jobs/"
            },
            {
                "platform": "JobStreet",
                "deskripsi": "Portal lowongan kerja dengan berbagai peluang karier di Indonesia.",
                "url": "https://www.jobstreet.co.id/"
            },
            {
                "platform": "Glints",
                "deskripsi": "Platform karier yang menyediakan lowongan kerja, magang, dan pengembangan karier.",
                "url": "https://glints.com/id"
            },
            {
                "platform": "Kalibrr",
                "deskripsi": "Portal rekrutmen yang menghubungkan pencari kerja dengan perusahaan di berbagai industri.",
                "url": "https://www.kalibrr.com/"
            },
            {
                "platform": "Indeed",
                "deskripsi": "Mesin pencari lowongan kerja yang mengumpulkan informasi dari berbagai perusahaan.",
                "url": "https://id.indeed.com/"
            }
        ]
    },
# ==========================================================
# DATA ANALYST
# ==========================================================
    "Data Analyst": {
        "tugas": [
            "Mengumpulkan, membersihkan, dan memvalidasi data dari berbagai sumber.",
            "Menganalisis data untuk menemukan pola, tren, dan insight yang mendukung pengambilan keputusan.",
            "Membuat visualisasi data dalam bentuk dashboard maupun laporan analisis.",
            "Menyampaikan hasil analisis kepada stakeholder sebagai dasar pengambilan keputusan."
        ],

        "skills": [
            "SQL – Mengambil, mengelola, dan memanipulasi data dari database.",
            "Python – Mengolah, menganalisis, dan memvisualisasikan data.",
            "Data Visualization – Menyajikan hasil analisis dalam bentuk grafik atau dashboard.",
            "Statistical Analysis – Menganalisis data menggunakan metode statistik.",
            "Critical Thinking – Menginterpretasikan data untuk menghasilkan insight yang tepat.",
            "Problem Solving – Menemukan solusi berdasarkan hasil analisis data."
        ],

        "tools": [
            "Microsoft Excel – Mengolah, membersihkan, dan menganalisis data dalam berbagai format.",
            "Power BI – Membuat dashboard interaktif dan visualisasi data.",
            "Jupyter Notebook – Menulis serta menjalankan kode Python untuk analisis data.",
            "Google Colab – Melakukan analisis data berbasis cloud menggunakan Python.",
            "MySQL – Mengelola dan mengambil data dari database relasional.",
            "Tableau – Membuat visualisasi data yang informatif dan mudah dipahami."
        ],
            "belajar": [
        {
            "platform": "Kaggle Learn",
            "url": "https://www.kaggle.com/learn"
        },
        {
            "platform": "Microsoft Learn",
            "url": "https://learn.microsoft.com/en-us/training/career-paths/data-analyst"
        },
        {
            "platform": "Coursera",
            "url": "https://www.coursera.org/browse/data-science/data-analysis"
        },
        {
            "platform": "DataCamp",
            "url": "https://www.datacamp.com/"
        },
        {
            "platform": "Google Colab",
            "url": "https://colab.research.google.com/"
        }
    ],
            "sertifikasi": [
        {
            "nama": "Google Data Analytics Professional Certificate",
            "penyelenggara": "Coursera (Google)",
            "url": "https://www.coursera.org/professional-certificates/google-data-analytics"
        },
        {
            "nama": "IBM Data Analyst Professional Certificate",
            "penyelenggara": "Coursera (IBM)",
            "url": "https://www.coursera.org/professional-certificates/ibm-data-analyst"
        },
        {
            "nama": "Microsoft Certified: Power BI Data Analyst Associate",
            "penyelenggara": "Microsoft Learn",
            "url": "https://learn.microsoft.com/credentials/certifications/power-bi-data-analyst-associate/"
        },
        {
            "nama": "Tableau Desktop Specialist",
            "penyelenggara": "Tableau",
            "url": "https://www.tableau.com/learn/certification"
        },
        {
            "nama": "Databricks Certified Data Analyst Associate",
            "penyelenggara": "Databricks Academy",
            "url": "https://www.databricks.com/learn/certification"
        }
    ],
        "pengembangan_kompetensi": [
            "Pelajari SQL untuk mengolah, mengambil, dan menganalisis data dari database.",
            "Kuasai Microsoft Excel, Python (Pandas, NumPy), serta tools visualisasi seperti Power BI atau Tableau.",
            "Pelajari dasar statistik, data cleaning, dan exploratory data analysis (EDA) untuk menghasilkan analisis yang akurat.",
            "Bangun portofolio melalui proyek analisis data menggunakan dataset publik dan sajikan hasilnya dalam bentuk dashboard atau laporan.",
            "Ikuti pelatihan atau sertifikasi di bidang analisis data untuk meningkatkan kompetensi dan daya saing di dunia kerja."
        ],
        "gaji_prospek": {
            "entry_level": "Rp5.500.000 – Rp9.000.000/bulan",
            "mid_level": "Rp9.000.000 – Rp16.000.000/bulan",
            "senior": "Rp16.000.000 – Rp30.000.000+/bulan",
            "prospek": "Sangat baik. Data Analyst dibutuhkan di berbagai sektor, seperti perbankan, fintech, e-commerce, telekomunikasi, kesehatan, manufaktur, konsultan, dan instansi pemerintahan untuk mendukung pengambilan keputusan berbasis data.",
            "catatan": "Estimasi gaji dapat berbeda tergantung lokasi, pengalaman, kemampuan analisis data, tools yang dikuasai, dan kebijakan perusahaan."
        },
        "peluang_karier": [
            {
                "platform": "LinkedIn Jobs",
                "deskripsi": "Menyediakan berbagai lowongan Data Analyst dari perusahaan nasional maupun internasional.",
                "url": "https://www.linkedin.com/jobs/"
            },
            {
                "platform": "JobStreet",
                "deskripsi": "Portal lowongan kerja dengan banyak posisi Data Analyst di berbagai industri.",
                "url": "https://www.jobstreet.co.id/"
            },
            {
                "platform": "Glints",
                "deskripsi": "Menyediakan peluang kerja dan magang untuk Data Analyst serta bidang data lainnya.",
                "url": "https://glints.com/id"
            },
            {
                "platform": "Indeed",
                "deskripsi": "Mesin pencari lowongan kerja yang menawarkan berbagai posisi Data Analyst.",
                "url": "https://id.indeed.com/"
            },
            {
                "platform": "Kalibrr",
                "deskripsi": "Portal rekrutmen yang menyediakan berbagai peluang karier di bidang analisis data.",
                "url": "https://www.kalibrr.com/"
            }
        ]
    },
# ==========================================================
# SYSTEM ANALYST
# ==========================================================
    "System Analyst": {
        "tugas": [
            "Menganalisis kebutuhan pengguna dan proses bisnis organisasi.",
            "Menyusun spesifikasi kebutuhan sistem sebagai acuan pengembangan aplikasi.",
            "Merancang solusi sistem yang sesuai dengan kebutuhan pengguna dan organisasi.",
            "Berkoordinasi dengan pengguna dan tim pengembang selama proses pengembangan sistem."
        ],

        "skills": [
            "Business Analysis – Memahami proses bisnis dan kebutuhan organisasi.",
            "Requirement Analysis – Mengidentifikasi serta mendokumentasikan kebutuhan pengguna.",
            "System Analysis – Menganalisis dan merancang solusi sistem yang sesuai.",
            "SQL – Memahami struktur data dan melakukan pengelolaan database dasar.",
            "Problem Solving – Menyelesaikan permasalahan sistem secara efektif.",
            "Communication – Berkomunikasi dengan pengguna dan tim pengembang."
        ],
        "tools": [
            "Draw.io – Membuat diagram UML, flowchart, dan rancangan sistem.",
            "Microsoft Visio – Mendesain diagram proses bisnis dan arsitektur sistem.",
            "Figma – Membuat rancangan antarmuka sebagai acuan pengembangan sistem.",
            "Jira – Mengelola kebutuhan, tugas, dan progres pengembangan proyek.",
            "MySQL – Menganalisis struktur dan data pada database sistem.",
            "Microsoft Office – Menyusun dokumentasi kebutuhan dan spesifikasi sistem."
        ],
            "belajar": [
        {
            "platform": "Coursera",
            "url": "https://www.coursera.org/"
        },
        {
            "platform": "IIBA",
            "url": "https://www.iiba.org/"
        },
        {
            "platform": "Microsoft Learn",
            "url": "https://learn.microsoft.com/en-us/training/"
        },
        {
            "platform": "Udemy",
            "url": "https://www.udemy.com/"
        },
        {
            "platform": "Visual Paradigm",
            "url": "https://online.visual-paradigm.com/"
        }
    ],
            "sertifikasi": [
        {
            "nama": "Entry Certificate in Business Analysis (ECBA)",
            "penyelenggara": "IIBA",
            "url": "https://www.iiba.org/business-analysis-certifications/ecba/"
        },
        {
            "nama": "Certification of Capability in Business Analysis (CCBA)",
            "penyelenggara": "IIBA",
            "url": "https://www.iiba.org/business-analysis-certifications/ccba/"
        },
        {
            "nama": "Certified Business Analysis Professional (CBAP)",
            "penyelenggara": "IIBA",
            "url": "https://www.iiba.org/business-analysis-certifications/cbap/"
        },
        {
            "nama": "PMI Professional in Business Analysis (PMI-PBA)",
            "penyelenggara": "Project Management Institute (PMI)",
            "url": "https://www.pmi.org/certifications/business-analysis-pba"
        }
    ],
        "pengembangan_kompetensi": [
            "Pelajari teknik analisis kebutuhan sistem dan dokumentasi kebutuhan pengguna secara mendalam.",
            "Kuasai pemodelan sistem menggunakan UML, BPMN, dan Entity Relationship Diagram (ERD).",
            "Pahami Software Development Life Cycle (SDLC), metodologi Agile, serta proses pengembangan perangkat lunak.",
            "Tingkatkan kemampuan komunikasi dan problem solving untuk menjembatani kebutuhan pengguna dengan tim pengembang.",
            "Ikuti pelatihan atau sertifikasi di bidang analisis sistem dan manajemen proyek untuk memperkuat kompetensi profesional."
        ],
        "gaji_prospek": {
            "entry_level": "Rp6.000.000 – Rp9.500.000/bulan",
            "mid_level": "Rp9.500.000 – Rp17.000.000/bulan",
            "senior": "Rp17.000.000 – Rp30.000.000+/bulan",
            "prospek": "Baik. System Analyst banyak dibutuhkan di perusahaan teknologi, perbankan, asuransi, telekomunikasi, manufaktur, konsultan IT, dan instansi pemerintahan untuk menganalisis serta merancang solusi sistem informasi.",
            "catatan": "Estimasi gaji dapat berbeda tergantung lokasi, pengalaman, kompleksitas proyek, dan kebijakan perusahaan."
        },
        "peluang_karier": [
            {
                "platform": "LinkedIn Jobs",
                "deskripsi": "Menyediakan berbagai lowongan System Analyst dari berbagai perusahaan.",
                "url": "https://www.linkedin.com/jobs/"
            },
            {
                "platform": "JobStreet",
                "deskripsi": "Portal lowongan kerja untuk posisi System Analyst dan Business Analyst.",
                "url": "https://www.jobstreet.co.id/"
            },
            {
                "platform": "Glints",
                "deskripsi": "Menawarkan peluang karier di bidang analisis dan pengembangan sistem.",
                "url": "https://glints.com/id"
            },
            {
                "platform": "Kalibrr",
                "deskripsi": "Portal rekrutmen yang menyediakan berbagai posisi System Analyst.",
                "url": "https://www.kalibrr.com/"
            },
            {
                "platform": "Indeed",
                "deskripsi": "Mengumpulkan lowongan kerja System Analyst dari berbagai perusahaan.",
                "url": "https://id.indeed.com/"
            }
        ]
    },
# ==========================================================
# IT SUPPORT
# ==========================================================
    "IT Support": {
        "tugas": [
            "Memberikan dukungan teknis kepada pengguna terkait perangkat keras, perangkat lunak, dan jaringan.",
            "Melakukan instalasi, konfigurasi, serta pemeliharaan perangkat komputer dan sistem operasi.",
            "Mengidentifikasi dan menyelesaikan permasalahan teknis (troubleshooting) pada sistem dan jaringan.",
            "Melakukan pemantauan serta dokumentasi terhadap permasalahan dan solusi yang telah dilakukan."
        ],

        "skills": [
            "Computer Hardware – Memahami instalasi, perawatan, dan perbaikan perangkat keras.",
            "Computer Networking – Mengelola konfigurasi dan pemeliharaan jaringan komputer.",
            "Operating System – Menginstal, mengonfigurasi, dan memelihara sistem operasi.",
            "Troubleshooting – Mengidentifikasi serta menyelesaikan permasalahan teknis pada perangkat dan sistem.",
            "Communication – Memberikan penjelasan teknis kepada pengguna secara jelas.",
            "Customer Service – Memberikan layanan dan dukungan teknis yang profesional kepada pengguna."
        ],
        "tools": [
            "TeamViewer – Memberikan dukungan teknis secara remote kepada pengguna.",
            "AnyDesk – Mengakses dan mengendalikan komputer pengguna dari jarak jauh.",
            "Wireshark – Menganalisis lalu lintas jaringan untuk proses troubleshooting.",
            "Cisco Packet Tracer – Mensimulasikan konfigurasi dan pengujian jaringan komputer.",
            "Windows Server – Mengelola layanan server dan pengguna dalam jaringan.",
            "Microsoft Active Directory – Mengelola akun pengguna, komputer, dan hak akses pada jaringan."
        ],
            "belajar": [
        {
            "platform": "Cisco Networking Academy",
            "url": "https://www.netacad.com/"
        },
        {
            "platform": "Microsoft Learn",
            "url": "https://learn.microsoft.com/en-us/training/"
        },
        {
            "platform": "Coursera",
            "url": "https://www.coursera.org/professional-certificates/microsoft-it-support-specialist"
        },
        {
            "platform": "CompTIA",
            "url": "https://www.comptia.org/"
        },
        {
            "platform": "Professor Messer",
            "url": "https://www.professormesser.com/"
        }
    ],
            "sertifikasi": [
        {
            "nama": "Google IT Support Professional Certificate",
            "penyelenggara": "Coursera (Google)",
            "url": "https://www.coursera.org/professional-certificates/google-it-support"
        },
        {
            "nama": "CompTIA A+",
            "penyelenggara": "CompTIA",
            "url": "https://www.comptia.org/certifications/a"
        },
        {
            "nama": "CompTIA Network+",
            "penyelenggara": "CompTIA",
            "url": "https://www.comptia.org/certifications/network"
        },
        {
            "nama": "Cisco Certified Network Associate (CCNA)",
            "penyelenggara": "Cisco",
            "url": "https://www.cisco.com/site/us/en/learn/training-certifications/certifications/enterprise/ccna/index.html"
        },
        {
            "nama": "Microsoft Certified: Windows Server Hybrid Administrator Associate",
            "penyelenggara": "Microsoft Learn",
            "url": "https://learn.microsoft.com/credentials/certifications/windows-server-hybrid-administrator/"
        }
    ],
    "pengembangan_kompetensi": [
        "Pelajari instalasi, konfigurasi, dan pemeliharaan sistem operasi Windows maupun Linux.",
        "Kuasai dasar jaringan komputer, troubleshooting perangkat keras dan perangkat lunak, serta konfigurasi perangkat jaringan.",
        "Pahami konsep keamanan informasi, backup data, dan pengelolaan akun pengguna.",
        "Latih kemampuan komunikasi dan pelayanan pengguna untuk memberikan dukungan teknis secara efektif.",
        "Ikuti pelatihan atau sertifikasi seperti CompTIA A+, Network+, atau Microsoft untuk meningkatkan kompetensi di bidang IT Support."
    ],
    "gaji_prospek": {
        "entry_level": "Rp4.500.000 – Rp7.500.000/bulan",
        "mid_level": "Rp7.500.000 – Rp12.000.000/bulan",
        "senior": "Rp12.000.000 – Rp20.000.000+/bulan",
        "prospek": "Baik. IT Support dibutuhkan oleh hampir semua organisasi, termasuk perusahaan teknologi, perbankan, pendidikan, rumah sakit, manufaktur, retail, dan instansi pemerintahan untuk menjaga operasional infrastruktur teknologi informasi.",
        "catatan": "Estimasi gaji dapat berbeda tergantung lokasi, pengalaman, sertifikasi yang dimiliki, dan kebijakan perusahaan."
    },
    "peluang_karier": [
        {
            "platform": "JobStreet",
            "deskripsi": "Portal lowongan kerja dengan banyak posisi IT Support dan Technical Support.",
            "url": "https://www.jobstreet.co.id/"
        },
        {
            "platform": "Glints",
            "deskripsi": "Menawarkan peluang kerja dan magang di bidang dukungan teknis.",
            "url": "https://glints.com/id"
        },
        {
            "platform": "Kalibrr",
            "deskripsi": "Portal rekrutmen yang menyediakan berbagai posisi IT Support.",
            "url": "https://www.kalibrr.com/"
        },
        {
            "platform": "Indeed",
            "deskripsi": "Mesin pencari lowongan kerja untuk berbagai posisi IT Support.",
            "url": "https://id.indeed.com/"
        }
    ]
 },
# ==========================================================
# UI/UX DESIGNER
# ==========================================================
    "UI/UX Designer": {
        "tugas": [
            "Melakukan riset untuk memahami kebutuhan dan perilaku pengguna.",
            "Membuat wireframe, prototype, dan desain antarmuka aplikasi.",
            "Melakukan pengujian usability untuk meningkatkan pengalaman pengguna.",
            "Berkolaborasi dengan tim pengembang agar desain dapat diimplementasikan dengan baik."
        ],
        "skills": [
            "User Research – Memahami kebutuhan, perilaku, dan pengalaman pengguna.",
            "Wireframing – Membuat rancangan awal struktur antarmuka aplikasi.",
            "Prototyping – Membuat prototype interaktif sebelum proses pengembangan.",
            "UI Design – Mendesain tampilan antarmuka yang menarik dan konsisten.",
            "UX Design – Meningkatkan kenyamanan dan pengalaman pengguna saat menggunakan aplikasi.",
            "Creativity – Menghasilkan ide dan solusi desain yang inovatif."
        ],
        "tools": [
            "Figma – Mendesain antarmuka dan membuat prototype aplikasi.",
            "Adobe XD – Membuat desain UI dan prototype interaktif.",
            "FigJam – Berkolaborasi dalam brainstorming dan user flow.",
            "Miro – Membuat user journey, wireframe, dan alur proses desain.",
            "Canva – Membuat aset visual dan presentasi desain.",
            "Maze – Melakukan usability testing terhadap prototype desain."
        ],
    "belajar": [
        {
            "platform": "Google UX Design (Coursera)",
            "deskripsi": "Program pembelajaran UX Design dari Google mulai dari dasar hingga pembuatan portofolio.",
            "url": "https://www.coursera.org/professional-certificates/google-ux-design"
        },
        {
            "platform": "Figma Learn",
            "deskripsi": "Dokumentasi dan tutorial resmi Figma untuk mempelajari desain antarmuka, prototyping, serta kolaborasi desain.",
            "url": "https://help.figma.com/"
        },
        {
            "platform": "Interaction Design Foundation",
            "deskripsi": "Platform pembelajaran UX/UI yang menyediakan materi mengenai user research, interaction design, usability, dan design thinking.",
            "url": "https://www.interaction-design.org/"
        },
        {
            "platform": "Adobe XD Learn",
            "deskripsi": "Tutorial resmi Adobe untuk mempelajari pembuatan desain antarmuka, wireframe, dan prototype menggunakan Adobe XD.",
            "url": "https://helpx.adobe.com/xd/tutorials.html"
        },
        {
            "platform": "NN/g (Nielsen Norman Group)",
            "deskripsi": "Menyediakan artikel, panduan, dan hasil riset mengenai User Experience (UX), usability, serta praktik terbaik dalam desain produk digital.",
            "url": "https://www.nngroup.com/articles/"
        }
    ],
            "sertifikasi": [
        {
            "nama": "Google UX Design Professional Certificate",
            "penyelenggara": "Coursera (Google)",
            "url": "https://www.coursera.org/professional-certificates/google-ux-design"
        },
        {
            "nama": "Professional Diploma in UX Design",
            "penyelenggara": "UX Design Institute",
            "url": "https://www.uxdesigninstitute.com/"
        },
        {
            "nama": "Adobe Certified Professional",
            "penyelenggara": "Adobe",
            "url": "https://certifiedprofessional.adobe.com/"
        },
        {
            "nama": "Interaction Design Foundation Courses",
            "penyelenggara": "Interaction Design Foundation",
            "url": "https://www.interaction-design.org/courses"
        }
    ],
        "pengembangan_kompetensi": [
            "Pelajari prinsip desain antarmuka, pengalaman pengguna, serta design thinking untuk menghasilkan solusi yang berorientasi pada pengguna.",
            "Kuasai tools desain seperti Figma atau Adobe XD untuk membuat wireframe, prototype, dan desain antarmuka.",
            "Pelajari user research, usability testing, serta analisis kebutuhan pengguna untuk meningkatkan kualitas desain.",
            "Bangun portofolio melalui proyek desain aplikasi atau website yang menampilkan proses desain dari riset hingga prototipe.",
            "Ikuti pelatihan atau sertifikasi UI/UX Design untuk memperdalam keterampilan dan mengikuti perkembangan tren desain."
        ],
        "gaji_prospek": {
            "entry_level": "Rp5.500.000 – Rp9.000.000/bulan",
            "mid_level": "Rp9.000.000 – Rp16.000.000/bulan",
            "senior": "Rp16.000.000 – Rp28.000.000+/bulan",
            "prospek": "Sangat baik. UI/UX Designer dibutuhkan oleh startup, perusahaan teknologi, e-commerce, fintech, agensi digital, perusahaan pengembang aplikasi, dan berbagai organisasi yang berfokus pada pengalaman pengguna.",
            "catatan": "Estimasi gaji dapat berbeda tergantung lokasi, pengalaman, kualitas portofolio, kemampuan desain, dan kebijakan perusahaan."
        },
            "peluang_karier": [
            {
                "platform": "LinkedIn Jobs",
                "deskripsi": "Menyediakan berbagai lowongan UI/UX Designer dari startup hingga perusahaan besar.",
                "url": "https://www.linkedin.com/jobs/"
            },
            {
                "platform": "JobStreet",
                "deskripsi": "Portal lowongan kerja dengan banyak peluang untuk UI/UX Designer.",
                "url": "https://www.jobstreet.co.id/"
            },
            {
                "platform": "Glints",
                "deskripsi": "Menawarkan peluang kerja dan magang di bidang UI/UX Design.",
                "url": "https://glints.com/id"
            },
            {
                "platform": "Kalibrr",
                "deskripsi": "Portal rekrutmen yang menyediakan berbagai posisi UI/UX Designer.",
                "url": "https://www.kalibrr.com/"
            },
            {
                "platform": "Indeed",
                "deskripsi": "Mengumpulkan lowongan UI/UX Designer dari berbagai perusahaan.",
                "url": "https://id.indeed.com/"
            }
        ]
    },
}
# ==========================================================
# CARD INFORMASI KARIER
# ==========================================================
career_mapping = {
    "UI/UX Design": "UI/UX Designer",
    "UI/UX Designer": "UI/UX Designer",
    "Software Development": "Software Developer",
    "Software Developer": "Software Developer",
    "Data Analysis": "Data Analyst",
    "Data Analyst": "Data Analyst"
}

target_key = career_mapping.get(karier, karier)
info = career_info.get(target_key, {})

st.markdown("## 💼 Informasi Karier")

col1, col2, col3 = st.columns(3)

with col1:
    tugas_html = "".join([f"<li>✅ {tugas}</li>" for tugas in info.get("tugas", [])])
    st.markdown(f'<div class="career-card card-blue"><div><h3>📌 Tugas & Tanggung Jawab</h3><p>Berikut beberapa tugas utama yang dilakukan oleh seorang <b>{karier}</b>.</p><hr><ul>{tugas_html}</ul></div><div class="card-footer bg-blue">💡 Tugas dapat berbeda pada setiap perusahaan sesuai kebutuhan bisnis.</div></div>', unsafe_allow_html=True)

with col2:
    tools_html = "".join([f"<li>💻 {tool}</li>" for tool in info.get("tools", [])])
    st.markdown(f'<div class="career-card card-green"><div><h3>🛠 Tools yang Direkomendasikan</h3><p>Beberapa tools yang umum digunakan oleh seorang <b>{karier}</b>.</p><hr><ul>{tools_html}</ul></div><div class="card-footer bg-green">🚀 Kuasai minimal 3–5 tools agar lebih siap memasuki dunia kerja.</div></div>', unsafe_allow_html=True)

with col3:
    skills_html = "".join([f"<li>⭐ {skill}</li>" for skill in info.get("skills", [])])
    st.markdown(f'<div class="career-card card-orange"><div><h3>💡 Skills yang Dibutuhkan</h3><p>Kemampuan yang perlu dimiliki sebagai seorang <b>{karier}</b>.</p><hr><ul>{skills_html}</ul></div><div class="card-footer bg-orange">🎯 Hard skill dan soft skill sama pentingnya untuk menunjang karier.</div></div>', unsafe_allow_html=True)

# ==========================================================
# CARD 2 - PENGEMBANGAN DIRI
# ==========================================================

st.markdown("## 🚀 Pengembangan Diri")

# ⚠️ PENTING: Bikin kolom baru agar tidak numpuk dengan bagian atas
col4, col5, col6 = st.columns(3)

# =========================
# 1. SUMBER BELAJAR
# =========================
with col4:
    items = []
    for x in info.get("belajar", []):
        if isinstance(x, dict):
            nama = x.get("platform") or x.get("nama") or x.get("judul") or "Sumber Belajar"
            
            # Cek berbagai kemungkinan key deskripsi
            desc = x.get("deskripsi") or x.get("keterangan") or x.get("detail") or ""
            
            sub_html = f'<p style="margin:2px 0 6px 0; font-size:0.85em; color:#666;">{desc}</p>' if desc else ''
            items.append(f'<div class="resource-item"><a href="{x.get("url", "#")}" target="_blank" class="resource-link link-blue">🔗 {nama}</a>{sub_html}</div>')
            
        elif isinstance(x, str):
            items.append(f'<div class="resource-item"><p style="margin:2px 0;">🔗 {x}</p></div>')

    st.markdown(f'<div class="career-card card-blue"><div><h4>📚 Sumber Belajar</h4><hr>{"".join(items)}</div></div>', unsafe_allow_html=True)

# =========================
# 2. SERTIFIKASI
# =========================
with col5:
    items = []
    for x in info.get("sertifikasi", []):
        if isinstance(x, dict):
            nama = x.get("nama") or x.get("judul") or x.get("platform") or "Sertifikasi"
            sub = x.get("penyelenggara") or x.get("deskripsi") or ""
            sub_html = f'<p style="margin:2px 0 0 0; font-size:0.85em; color:#666;">{sub}</p>' if sub else ''
            items.append(f'<div class="resource-item"><a href="{x.get("url", "#")}" target="_blank" class="resource-link link-green">🏅 {nama}</a>{sub_html}</div>')
        elif isinstance(x, str):
            items.append(f'<div class="resource-item"><p style="margin:2px 0;">🏅 {x}</p></div>')

    st.markdown(f'<div class="career-card card-green"><div><h4>🏅 Sertifikasi</h4><hr>{"".join(items)}</div></div>', unsafe_allow_html=True)

# =========================
# 3. ROADMAP
# =========================
with col6:
    if karier in ["Software Developer", "Software Engineering"]:
        roadmap = "https://roadmap.sh/backend"
    elif karier in ["Data Analyst", "Data Analysis"]:
        roadmap = "https://roadmap.sh/data-analyst"
    elif karier in ["System Analyst"]:
        roadmap = "https://roadmap.sh/software-design-and-architecture"
    elif karier in ["IT Support"]:
        roadmap = "https://roadmap.sh/linux"
    elif karier in ["UI/UX Designer", "UI/UX Design"]:
        roadmap = "https://roadmap.sh/ux-design"
    else:
        roadmap = "https://roadmap.sh"

    st.markdown(f'<div class="career-card card-orange"><div><h4>🗺️ Roadmap</h4><hr><p>Ikuti panduan jalur belajar terstruktur untuk mendukung karier Anda:</p><a href="{roadmap}" target="_blank" class="btn-roadmap">🗺️ Lihat Roadmap Belajar</a></div><div class="card-footer bg-orange">💡 Konsistensi belajar dan membangun portofolio sangat penting!</div></div>', unsafe_allow_html=True)

# ==========================================================
# CARD 3 - PENGEMBANGAN KARIER
# ==========================================================
st.markdown("## 📈 Pengembangan Karier")

col7, col8, col9 = st.columns(3)

# =========================
# 1. PENGEMBANGAN KOMPETENSI
# =========================
with col7:
    items = []
    # Ambil list dari key "pengembangan_kompetensi" atau "kompetensi"
    kompetensi_list = info.get("pengembangan_kompetensi") or info.get("kompetensi") or []
    
    for x in kompetensi_list:
        if isinstance(x, str):
            items.append(f'<div class="resource-item"><p style="margin:4px 0; font-size:0.88em;">🔹 {x}</p></div>')
        elif isinstance(x, dict):
            nama = x.get("nama") or x.get("judul") or ""
            desc = x.get("deskripsi") or x.get("keterangan") or ""
            sub_html = f'<p style="margin:2px 0 0 0; font-size:0.83em; color:#666;">{desc}</p>' if desc else ''
            items.append(f'<div class="resource-item"><p style="margin:2px 0; font-weight:600; font-size:0.88em;">🔹 {nama}</p>{sub_html}</div>')

    st.markdown(f'''
    <div class="career-card card-blue">
        <div>
            <h4>🎯 Pengembangan Kompetensi</h4>
            <hr>
            {"".join(items) if items else "<p style='font-size:0.88em; color:#666;'>Kembangkan keterampilan teknis dan non-teknis Anda secara berkala.</p>"}
        </div>
        <div class="card-footer bg-blue">
            💡 Tingkatkan kompetensi ini untuk mempercepat kenaikan jenjang karier.
        </div>
    </div>
    ''', unsafe_allow_html=True)

# =========================
# 2. GAJI DAN PROSPEK KARIER
# =========================
with col8:
    gp = info.get("gaji_prospek", {})
    
    entry = gp.get("entry_level") or info.get("gaji_entry") or "-"
    mid = gp.get("mid_level") or info.get("gaji_mid") or "-"
    senior = gp.get("senior") or info.get("gaji_senior") or "-"
    prospek = gp.get("prospek") or info.get("prospek") or "-"
    catatan = gp.get("catatan") or "Estimasi gaji dapat berbeda tergantung lokasi dan perusahaan."

    st.markdown(f'''
    <div class="career-card card-green">
        <div>
            <h4>💰 Gaji & Prospek Karier</h4>
            <hr>
            <div class="resource-item" style="margin-bottom:10px;">
                <p style="margin:0; font-size:0.85em; color:#555; font-weight:bold;">Estimasi Gaji:</p>
                <p style="margin:3px 0 0 0; font-size:0.85em;">• <b>Entry:</b> {entry}</p>
                <p style="margin:2px 0 0 0; font-size:0.85em;">• <b>Mid:</b> {mid}</p>
                <p style="margin:2px 0 0 0; font-size:0.85em;">• <b>Senior:</b> {senior}</p>
            </div>
            <div class="resource-item">
                <p style="margin:0; font-size:0.85em; color:#555; font-weight:bold;">Prospek Karier:</p>
                <p style="margin:3px 0 0 0; font-size:0.85em; color:#333;">📈 {prospek}</p>
            </div>
        </div>
        <div class="card-footer bg-green">
            📌 {catatan}
        </div>
    </div>
    ''', unsafe_allow_html=True)

# =========================
# 3. PELUANG KARIER
# =========================
with col9:
    items = []
    # Ambil list dari key "peluang_karier" atau "lowongan"
    peluang_list = info.get("peluang_karier") or info.get("lowongan") or []
    
    for x in peluang_list:
        if isinstance(x, dict):
            platform = x.get("platform") or x.get("nama") or "Platform Kerja"
            url = x.get("url", "#")
            desc = x.get("deskripsi") or x.get("keterangan") or ""
            sub_html = f'<p style="margin:2px 0 0 0; font-size:0.83em; color:#666;">{desc}</p>' if desc else ''
            items.append(f'<div class="resource-item"><a href="{url}" target="_blank" class="resource-link link-orange">🔍 {platform}</a>{sub_html}</div>')

    st.markdown(f'''
    <div class="career-card card-orange">
        <div>
            <h4>💼 Peluang Karier</h4>
            <hr>
            {"".join(items) if items else "<p style='font-size:0.88em; color:#666;'>Cari peluang kerja di portal terpercaya.</p>"}
        </div>
        <div class="card-footer bg-orange">
            🚀 Pantau lowongan secara berkala dan siapkan CV terbaik Anda!
        </div>
    </div>
    ''', unsafe_allow_html=True)

# ==========================================================
# LOGIC & TEKS ANALISIS
# ==========================================================
# Pastikan variabel memiliki nilai aman
nama_user = nama if 'nama' in locals() and nama else "Pengguna"
nama_file = nama_user.replace(" ", "_")

if confidence >= 85:
    status_badge = "🟢 Sangat Cocok"
    analisis = (
        "Profil Anda menunjukkan tingkat kecocokan yang sangat baik terhadap profesi "
        "yang direkomendasikan. Kompetensi, minat, dan kemampuan yang dimiliki telah "
        "sesuai dengan karakteristik profesi tersebut."
    )
elif confidence >= 70:
    status_badge = "🔵 Cukup Cocok"
    analisis = (
        "Profil Anda menunjukkan tingkat kecocokan yang baik terhadap profesi yang "
        "direkomendasikan. Kompetensi dasar sudah cukup sesuai, namun masih perlu "
        "dikembangkan agar lebih siap memasuki dunia kerja."
    )
elif confidence >= 50:
    status_badge = "🟡 Berpotensi"
    analisis = (
        "Profil Anda memiliki potensi pada profesi yang direkomendasikan. Disarankan "
        "untuk meningkatkan kompetensi melalui pelatihan, proyek, maupun pengalaman praktik."
    )
else:
    status_badge = "🟠 Perlu Pengembangan"
    analisis = (
        "Profil Anda masih memerlukan pengembangan lebih lanjut agar sesuai dengan "
        "kebutuhan profesi yang direkomendasikan."
    )
# ==========================================================
# TAMPILAN KESIMPULAN (Native Streamlit Components)
# ==========================================================
st.divider()
st.subheader("📌 Kesimpulan Hasil Rekomendasi")

# 1. Main Hero Card (Menggunakan Container Border Native Streamlit)
with st.container(border=True):
    col_title, col_status = st.columns([3, 1])
    with col_title:
        st.caption("REKOMENDASI PROFESI UTAMA")
        st.title(f"🎯 {karier}")
    with col_status:
        st.write("") # Spacing
        st.subheader(status_badge)

# 2. Metric & Progress Bar (Di dalam Card Terpisah)
with st.container(border=True):
    col_metric, col_desc = st.columns([1, 2], gap="large")

    with col_metric:
        st.metric(
            label="Tingkat Keyakinan (KNN)",
            value=f"{confidence:.2f}%",
            delta="Akurasi Tinggi" if confidence >= 70 else "Akurasi Sedang"
        )

    with col_desc:
        st.write("**Kesesuaian Profil:**")
        # Menghindari error nilai progress > 100 atau < 0
        progress_val = max(0, min(int(confidence), 100))
        st.progress(progress_val)
        st.caption("Dihitung berdasarkan analisis algoritma K-Nearest Neighbor (KNN).")

# 3. Card Analisis Detail
with st.container(border=True):
    st.markdown("**💡 Analisis Kesiapan:**")
    st.write(analisis)

# 4. Info Saran & Terima Kasih
st.info(
    "**💡 Saran Pengembangan Karier:** Hasil rekomendasi ini dapat dijadikan sebagai referensi awal. "
    "Anda disarankan untuk terus meningkatkan kompetensi, mengembangkan keterampilan teknis, "
    "membangun portofolio, serta mengikuti pelatihan/sertifikasi yang relevan agar siap bersaing di dunia kerja."
)

st.success("🎉 Terima kasih telah menggunakan Career Recommendation System!")

st.markdown("### 📄 Unduh Laporan")

st.info(
    "📄 Jika Anda ingin menyimpan hasil rekomendasi, silakan gunakan fitur "
    "**Print (Ctrl + P)** pada browser, kemudian pilih **Save as PDF** atau "
    "cetak menggunakan printer sesuai kebutuhan."
)
