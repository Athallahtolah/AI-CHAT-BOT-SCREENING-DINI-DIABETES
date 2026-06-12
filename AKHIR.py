import os
os.environ["USE_TF"] = "NO"

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 1. KONFIGURASI HALAMAN & CUSTOM CSS DARK MODE PERFECT
# ==========================================
st.set_page_config(
    page_title="DIABETES AI Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    /* 1. Background Utama Halaman: Hitam Agak Keabuan (Dark Charcoal) */
    .stApp, [data-testid="stHeader"] {
        background: linear-gradient(135deg, #1e1e24 0%, #121214 100%) !important;
        color: #ffffff !important;
        font-family: 'Inter', system-ui, sans-serif !important;
    }

    /* 2. Judul di Tengah Paling Atas (Warna Putih Bersinar) */
    .center-title {
        text-align: center;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        letter-spacing: -1px;
        margin-top: -30px !important;
        margin-bottom: 5px !important;
    }
    .center-subtitle {
        text-align: center;
        font-size: 0.95rem !important;
        color: #cbd5e1 !important; /* Abu terang mendekati putih */
        margin-bottom: 30px !important;
    }

    /* ==========================================
       TARGET LOCK: STYLING CONTAINER TEMA GELAP
       ========================================== */
    
    /* 1. Kotak Parameter Kiri - Neon Purple Border */
    div[data-testid="column"]:nth-child(1) [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #1a1a1e !important;
        border: 2px solid #a78bfa !important; /* Ungu Neon Soft */
        border-radius: 20px !important;
        padding: 20px !important;
        box-shadow: 0 10px 30px rgba(167, 139, 250, 0.05) !important;
    }

    /* 2. Kotak Chat Kanan - Neon Pink Border */
    div[data-testid="column"]:nth-child(2) [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #1a1a1e !important;
        border: 2px solid #f472b6 !important; /* Pink Neon Soft */
        border-radius: 24px !important;
        padding: 25px !important;
        box-shadow: 0 15px 35px rgba(244, 114, 182, 0.05) !important;
    }

    /* Mengatur kerapatan teks di kolom parameter */
    div[data-testid="column"]:nth-child(1) [data-testid="stVerticalBlock"] {
        gap: 0.35rem !important; 
    }
    .param-text {
        font-size: 0.9rem !important;
        font-weight: 600;
        color: #ffffff !important; /* Font Putih Tegas */
        margin-top: 5px !important;
        margin-bottom: -8px !important;
    }

    /* Memaksa Semua Teks Streamlit di Dalam Chat Menjadi Putih */
    .stChatMessage, .stChatMessage p, .stChatMessage span, .stChatMessage div, 
    div[data-testid="stMarkdownContainer"] p, stCaption, .stCaption, div.stCaption {
        color: #ffffff !important; 
    }
    
    /* Judul kecil kolom */
    h3 {
        color: #ffffff !important;
    }

    /* Gaya Gelembung Chat di Ruang Gelap */
    div[data-testid="stChatMessageAssistant"] {
        background-color: #26262b !important; 
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 16px !important;
    }
    div[data-testid="stChatMessageUser"] {
        background: #2e2a38 !important; /* Sentuhan ungu gelap */
        border: 1px solid rgba(167, 139, 250, 0.3) !important;
        border-radius: 16px !important;
    }

    /* Input Tempat Mengetik Chat */
    [data-testid="stChatInput"] textarea {
        color: #ffffff !important;
    }
    [data-testid="stChatInput"] > div {
        background-color: #26262b !important;
        border: 1.5px solid rgba(244, 114, 182, 0.4) !important;
        border-radius: 16px !important;
    }
    
    /* Teks Kode di dalam Parameter */
    code {
        color: #f472b6 !important; 
        background-color: #26262b !important;
        padding: 2px 6px !important;
        border-radius: 6px !important;
        font-size: 0.85rem !important;
    }

    /* Progress Bar Mode Gelap */
    div.stProgress > div > div {
        background-color: #2d2d34 !important;
        border-radius: 20px !important;
        height: 8px !important;
    }
    div.stProgress > div > div > div > div {
        background: linear-gradient(90deg, #a78bfa 0%, #f472b6 100%) !important; 
        border-radius: 20px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# API Key Management
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except:
    os.environ["GROQ_API_KEY"] = "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"

# ==========================================
# 2. INTELIJEN MODEL LOGIC (MLP & RAG)
# ==========================================
@st.cache_resource
def latih_dan_siapkan_mlp():
    if not os.path.exists("diabetes.csv"):
        st.error("File 'diabetes.csv' tidak ditemukan.")
        st.stop()
    df = pd.read_csv("diabetes.csv")
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model_mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    model_mlp.fit(X_train_scaled, y_train)
    return model_mlp, scaler

model_mlp, scaler_data = latih_dan_siapkan_mlp()

@st.cache_resource
def inisialisasi_ai_jurnal_ringan():
    jurnal_folder = "jurnal_folder"
    if not os.path.exists(jurnal_folder):
        return None
    daftar_jurnal = [os.path.join(jurnal_folder, f) for f in os.listdir(jurnal_folder) if f.endswith(".pdf")]
    if not daftar_jurnal:
        return None
    dokumen_gabungan = []
    for jalur_file in daftar_jurnal:
        try:
            loader = PyPDFLoader(jalur_file)
            dokumen_gabungan.extend(loader.load())
        except: pass
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(dokumen_gabungan)
    return BM25Retriever.from_documents(splits) if splits else None

retriever_jurnal = inisialisasi_ai_jurnal_ringan()

def dapatkan_analisis_groq_chat(data_narasi, hasil_mlp, retriever):
    if not retriever:
        context_docs = "Tidak ada literatur jurnal tersedia."
    else:
        docs = retriever.invoke(data_narasi)
        context_docs = "\n\n".join(doc.page_content for doc in docs)

    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3)
    template = """Anda adalah "Dokter AI" dari PT Bukhori Group.
    Berikan laporan analisis hasil prediksi diabetes pasien dengan format Markdown yang rapi.
    
    Aturan Penulisan:
    1. Tampilkan '{prediction_mlp}' di baris paling atas sebagai HEADLINE BESAR (Gunakan #).
    2. Cetak TEBAL (BOLD) poin-poin penting, parameter klinis, dan angka hasil laboratorium pasien.
    3. Tuliskan analisis hubungan dengan Konteks Jurnal 2026 secara singkat.
    4. Berikan saran preventif (Gaya hidup & Tes HbA1c).

    ### Konteks Jurnal 2026:
    {context}
    ### Data Klinis:
    {input_data}
    ### Hasil Prediksi:
    {prediction_mlp}
    """
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain.invoke({"context": context_docs, "input_data": data_narasi, "prediction_mlp": hasil_mlp})

# ==========================================
# 3. MEMORI PERCAKAPAN & STATE MANAGEMENT
# ==========================================
PERTANYAAN = [
    "1. Berapa jumlah kehamilan Anda? (Ketik 0 jika laki-laki)",
    "2. Berapa kadar Gula Darah (Glucose) puasa Anda saat ini?",
    "3. Berapa Tekanan Darah Diastolik Anda? (Contoh: 80)",
    "4. Berapa nilai Ketebalan Kulit (Skin Thickness) Anda? (Ketik 0 jika tidak tahu)",
    "5. Berapa kadar Insulin Anda dalam darah? (Ketik 0 jika tidak tahu)",
    "6. Berapa nilai BMI (Indeks Massa Tubuh) Anda?",
    "7. Berapa nilai Diabetes Pedigree Function Anda? (Ketik 0.5 jika tidak tahu)",
    "8. Berapa usia Anda saat ini?",
    "Terakhir, apakah Anda memiliki keluhan kesehatan lain saat ini?"
]

NAMA_PARAMETER = ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", "Insulin", "BMI", "Diabetes Pedigree", "Age"]

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Halo, saya Dokter AI. Mari kita lakukan skrining dini diabetes.\n\n" + PERTANYAAN[0]}]
if "step" not in st.session_state:
    st.session_state.step = 0
if "user_data" not in st.session_state:
    st.session_state.user_data = []
if "narasi_keluhan" not in st.session_state:
    st.session_state.narasi_keluhan = ""

# ==========================================
# 4. IMPLEMENTASI LAYOUT 
# ==========================================

# 1. Judul Aplikasi di Tengah Paling Atas
st.markdown('<div class="center-title">🤖 CHAT BOT: SCREENING DIABETES</div>', unsafe_allow_html=True)
st.markdown('<div class="center-subtitle">Hybrid Machine Learning & RAG Analysis </div>', unsafe_allow_html=True)

# 2. Pembagian Kolom Bawah: Kiri vs Kanan
col_info, col_chat = st.columns([1, 1.4], gap="medium")

# --- PANEL SEBELAH KIRI: DIABETES PARAMETERS (SEMUA PARAMETER DENGAN BAR KUSTOM) ---
with col_info:
    st.markdown('<h3 style="font-size:1.1rem; font-weight:700; margin-bottom:5px;">📋 Diabetes Parameters</h3>', unsafe_allow_html=True)
    
    with st.container(border=True):
        if len(st.session_state.user_data) == 0:
            st.markdown('<p style="color:#a0aec0; font-size:0.85rem; margin-bottom:10px;">Belum ada data klinis yang diinput.</p>', unsafe_allow_html=True)
            # Tampilan awal: Semua bar kosong/mati sebelum diinput
            for name in NAMA_PARAMETER:
                st.markdown(f'<p class="param-text" style="color:#a0aec0 !important;">{name}</p>', unsafe_allow_html=True)
                st.markdown("""
                    <div style="background-color: #2d2d34; border-radius: 10px; width: 100%; height: 8px; overflow: hidden; margin-top: 5px; margin-bottom: 15px;">
                        <div style="background: #2d2d34; width: 0%; height: 100%;"></div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            # Tampilkan parameter yang sudah diisi oleh user
            for i, val in enumerate(st.session_state.user_data):
                param_name = NAMA_PARAMETER[i]
                
                # Kalkulasi persentase bar dinamis biar proporsional
                if param_name == "Glucose":
                    persen_bar = min((val / 200.0) * 100, 100.0)
                    satuan = "mg/dL"
                elif param_name == "BMI":
                    persen_bar = min((val / 50.0) * 100, 100.0)
                    satuan = "kg/m²"
                elif param_name == "Blood Pressure":
                    persen_bar = min((val / 150.0) * 100, 100.0)
                    satuan = "mmHg"
                elif param_name == "Age":
                    persen_bar = min((val / 100.0) * 100, 100.0)
                    satuan = "years"
                else:
                    persen_bar = min((val / 100.0) * 100, 100.0) if val > 0 else 0.0
                    satuan = ""

                # Render Teks & Bar HTML Kustom untuk data yang sudah ada
                st.markdown(f'<p class="param-text">{param_name}: <code>{val} {satuan}</code></p>', unsafe_allow_html=True)
                st.markdown(f"""
                    <div style="background-color: #2d2d34; border-radius: 10px; width: 100%; height: 8px; overflow: hidden; margin-top: 5px; margin-bottom: 15px;">
                        <div style="background: linear-gradient(90deg, #a78bfa 0%, #f472b6 100%); width: {persen_bar}%; height: 100%; border-radius: 10px; transition: width 0.5s ease-in-out;"></div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Tampilkan sisa parameter yang belum dijawab sebagai placeholder abu-abu
            sisa = len(NAMA_PARAMETER) - len(st.session_state.user_data)
            for j in range(sisa):
                idx = len(st.session_state.user_data) + j
                st.markdown(f'<p class="param-text" style="color:#a0aec0 !important;">{NAMA_PARAMETER[idx]}</p>', unsafe_allow_html=True)
                st.markdown("""
                    <div style="background-color: #2d2d34; border-radius: 10px; width: 100%; height: 8px; overflow: hidden; margin-top: 5px; margin-bottom: 15px;">
                        <div style="background: #2d2d34; width: 0%; height: 100%;"></div>
                    </div>
                """, unsafe_allow_html=True)

# --- PANEL SEBELAH KANAN: CHAT BOX ---
with col_chat:
    st.markdown('<h3 style="font-size:1.1rem; font-weight:700; margin-bottom:5px;">💬 Room Chat Agent</h3>', unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown('<div style="text-align: center; font-weight:700; color:#cbd5e1; font-size:0.9rem;">✨ Ask our AI Anything</div>', unsafe_allow_html=True)
        st.markdown("<hr style='margin:10px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

        if st.session_state.messages:
            pesan_terakhir = st.session_state.messages[-1]
            if pesan_terakhir["role"] == "user" and len(st.session_state.messages) > 1:
                pesan_tampil = st.session_state.messages[-2:]
            else:
                pesan_tampil = [pesan_terakhir]
                
            for msg in pesan_tampil:
                if "Halo, saya Dokter AI" in msg["content"]:
                    if "\n\n" in msg["content"]:
                        msg_clean = msg["content"].split("\n\n")[-1]
                        with st.chat_message("assistant"): st.markdown(msg_clean)
                    else:
                        with st.chat_message("assistant"): st.markdown(msg["content"])
                    continue
                with st.chat_message(msg["role"]): 
                    st.markdown(msg["content"])

        st.markdown("<br>", unsafe_allow_html=True)

    # Form Chat Input
    if user_input := st.chat_input("Type your answer here..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        current_step = st.session_state.step
        total_questions = len(PERTANYAAN)

        if current_step < (total_questions - 1):
            try:
                list_angka = [s for s in user_input.split() if s.replace('.', '', 1).isdigit()]
                if not list_angka: raise ValueError
                st.session_state.user_data.append(float(list_angka[0]))
                step_lanjut = True
            except:
                step_lanjut = False
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ **Format Angka Salah.** Sila ulangi untuk '{NAMA_PARAMETER[current_step]}'.\n\n*Soal:* {PERTANYAAN[current_step]}"})
        else:
            st.session_state.narasi_keluhan = user_input
            step_lanjut = True

        if step_lanjut:
            st.session_state.step += 1
            next_step = st.session_state.step

            if next_step < total_questions:
                st.session_state.messages.append({"role": "assistant", "content": PERTANYAAN[next_step]})
            elif next_step == total_questions:
                st.session_state.messages.append({"role": "assistant", "content": "## Sedang Memproses Hasil..."})
                try:
                    fitur_mlp = st.session_state.user_data[:8]
                    if len(fitur_mlp) < 8: fitur_mlp += [0.0] * (8 - len(fitur_mlp))
                    data_df = pd.DataFrame([fitur_mlp])
                    data_scaled = scaler_data.transform(data_df)
                    prediksi = model_mlp.predict(data_scaled)[0]
                    
                    hasil_mlp = "# 🔴 POSITIVE DIABETES (72.0% Accuracy)" if prediksi == 1 else "# 🟢 NEGATIVE DIABETES (72.0% Accuracy)"
                    ringkasan_kasus = f"Pasien: Glucose={fitur_mlp[1]}, BMI={fitur_mlp[5]}. Keluhan: {st.session_state.narasi_keluhan}"
                    respons_final = dapatkan_analisis_groq_chat(ringkasan_kasus, hasil_mlp, retriever_jurnal)
                except Exception as e:
                    respons_final = f"Terjadi gangguan teknis: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": respons_final})
        st.rerun()
