import os
os.environ["USE_TF"] = "NO"

import streamlit as st
import pandas as pd
from operator import itemgetter
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
# 1. JUDUL & KONFIGURASI FORMAL
# ==========================================

try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except:
    # Jika di laptop (lokal), pakai kode API Key kamu langsung di bawah ini
    os.environ["GROQ_API_KEY"] = "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"
st.set_page_config(page_title="Sistem Deteksi Diabetes", layout="wide")

st.title("📊 Sistem Pakar Hybrid: Prediksi Diabetes Melitus Menggunakan MLP Classifier dan Retrieval-Augmented Generation (RAG)")
st.write("Aplikasi skrining dini diabetes beserta analisis preventif berbasis korpus jurnal ilmiah.")
st.markdown("---")

# ==========================================
# 2. TRAINING MODEL MLP (72% OPTIMAL)
# ==========================================
@st.cache_resource
def latih_dan_siapkan_mlp():
    df = pd.read_csv("diabetes.csv")
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model_mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16), activation='relu', solver='adam',
        alpha=0.01, learning_rate_init=0.001, max_iter=500, random_state=42
    )
    model_mlp.fit(X_train_scaled, y_train)
    return model_mlp, scaler

model_mlp, scaler_data = latih_dan_siapkan_mlp()

# ==========================================
# 3. INITIALIZE RAG JURNAL
# ==========================================
@st.cache_resource
def inisialisasi_ai_jurnal_ringan():
    daftar_jurnal = ["jurnal_folder/jurnal_neuropati.pdf", "jurnal_folder/jurnal_jantung.pdf", "jurnal_folder/jurnal_yoyo.pdf"]
    dokumen_gabungan = []
    for jalur_file in daftar_jurnal:
        try:
            loader = PyPDFLoader(jalur_file)
            dokumen_gabungan.extend(loader.load())
        except: pass
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(dokumen_gabungan)
    retriever = BM25Retriever.from_documents(splits)
    retriever.k = 3
    return retriever

retriever_jurnal = inisialisasi_ai_jurnal_ringan()

# Dapatkan Analisis RAG Formal & Interaktif
def dapatkan_analisis_groq_chat(data_narasi, hasil_mlp, retriever):
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3)
    template = """Anda adalah "Dokter AI", pakar medis virtual yang ramah dan ilmiah.
    Berikan laporan analisis hasil prediksi diabetes pasien dengan gaya chat yang santun, empati, dan terstruktur menggunakan Markdown.
    
    Aturan:
    1. Tampilkan hasil prediksi MLP di baris paling atas secara mencolok.
    2. Berikan poin analisis data klinis pasien berdasarkan keluhan yang disampaikan.
    3. Hubungkan secara ilmiah keluhan pasien (seperti riwayat diet, gejala, dll) dengan fakta dari Konteks Jurnal 2026.
    4. Berikan saran preventif konkret (gaya hidup, tes lab HbA1c).

    Konteks Jurnal 2026: {context}
    Narasi Pasien: {input_data}
    Hasil Prediksi Awal MLP: {prediction_mlp}
    """
    prompt = ChatPromptTemplate.from_template(template)
    format_docs = lambda docs: "\n\n".join(doc.page_content for doc in docs)
    rag_chain = (
        {"context": itemgetter("input_data") | retriever | format_docs, "input_data": itemgetter("input_data"), "prediction_mlp": itemgetter("prediction_mlp")}
        | prompt | llm | StrOutputParser()
    )
    return rag_chain.invoke({"input_data": data_narasi, "prediction_mlp": hasil_mlp})

# ==========================================
# 4. MEMORI PERCAKAPAN & STATE WAWANCARA
# ==========================================
# Daftar pertanyaan berurutan untuk mengisi 8 parameter Pima Indians
PERTANYAAN = [
    "1. Berapa jumlah kehamilan Anda? (Ketik 0 jika Anda laki-laki)",
    "2. Berapa kadar Gula Darah (Glucose) Anda saat puasa / setelah makan?",
    "3. Berapa Tekanan Darah (Diastolic) Anda? (Contoh: 80)",
    "4. Berapa nilai Ketebalan Kulit (Skin Thickness) tricep Anda? (Ketik 0 jika tidak tahu)",
    "5. Berapa kadar Insulin Anda dalam darah? (Ketik 0 jika tidak tahu)",
    "6. Berapa nilai BMI (Indeks Massa Tubuh) Anda? (Ketik 0 jika tidak tahu)",
    "7. Berapa nilai Diabetes Pedigree Function (faktor keturunan) Anda? (Ketik 0.5 jika tidak tahu)",
    "8. Berapa usia Anda saat ini?",
    "Terima kasih! Terakhir, apakah Anda memiliki keluhan lain? (Misal: berat badan naik turun/diet yoyo, sering kesemutan, atau mudah lelah?)"
]

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Halo, saya Dokter AI. Saya akan membantu Anda melakukan skrining risiko diabetes secara mandiri. Mari kita mulai dengan beberapa pertanyaan klinis.\n\n" + PERTANYAAN[0]}]
if "step" not in st.session_state:
    st.session_state.step = 0
if "user_data" not in st.session_state:
    st.session_state.user_data = []
if "narasi_keluhan" not in st.session_state:
    st.session_state.narasi_keluhan = ""

# Tampilkan riwayat chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 5. AKSI INPUT CHAT USER (VERSI ANTI-ASAL)
# ==========================================
if user_input := st.chat_input("Ketik jawaban Anda di sini..."):
    current_step = st.session_state.step
    input_valid = True
    angka_terdeteksi = 0.0

    # 1. Validasi Angka untuk Langkah 0 sampai 7
    if current_step < 8:
        try:
            # Cari angka di dalam teks input user
            list_angka = [s for s in user_input.split() if s.replace('.', '', 1).isdigit()]
            if len(list_angka) > 0:
                angka_terdeteksi = float(list_angka[0])
            else:
                # Jika tidak ada angka sama sekali di ketikan user
                input_valid = False
        except:
            input_valid = False
    else:
        # Langkah ke-9 (Keluhan tambahan bebas teks murni)
        st.session_state.narasi_keluhan = user_input

    # 2. Jika Input Valid, Jalankan Alur Chat
    if input_valid:
        # Tampilkan chat user ke layar
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        if current_step < 8:
            st.session_state.user_data.append(angka_terdeteksi)

        # Naikkan langkah ke pertanyaan berikutnya
        st.session_state.step += 1
        next_step = st.session_state.step

        with st.chat_message("assistant"):
            if next_step < len(PERTANYAAN):
                respons_bot = PERTANYAAN[next_step]
                st.markdown(respons_bot)
                st.session_state.messages.append({"role": "assistant", "content": respons_bot})
            
            elif next_step == len(PERTANYAAN):
                with st.spinner("Dokter AI sedang memproses data klinis Anda pada model MLP & RAG Jurnal..."):
                    try:
                        fitur_mlp = st.session_state.user_data[:8]
                        data_df = pd.DataFrame([fitur_mlp])
                        data_scaled = scaler_data.transform(data_df)
                        prediksi = model_mlp.predict(data_scaled)[0]
                        
                        hasil_mlp = "🚨 HASIL PREDIKSI MLP: POSITIVE DIABETES (72.0% Accuracy)" if prediksi == 1 else "🟢 HASIL PREDIKSI MLP: NEGATIVE DIABETES (72.0% Accuracy)"
                        ringkasan_kasus = f"Data parameter medis: {fitur_mlp}. Keluhan tambahan pasien: {st.session_state.narasi_keluhan}"
                        respons_final = dapatkan_analisis_groq_chat(ringkasan_kasus, hasil_mlp, retriever_jurnal)
                    except Exception as e:
                        respons_final = f"Terjadi kesalahan sistem: {str(e)}."
                    
                    st.markdown(respons_final)
                    st.session_state.messages.append({"role": "assistant", "content": respons_final})
                    st.info("Obrolan selesai. Jika ingin mencoba lagi, silakan refresh (F5) halaman browser Anda.")

    # 3. Jika Input SALAH (Ketik Asal), Tolak dan Minta Ketik Ulang
    else:
        # Tampilkan chat asal user agar tetap interaktif
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        # Peringatan dari Dokter AI
        pesan_peringatan = f"⚠️ **Format Salah.** Mohon masukkan nilai berupa **angka** yang valid untuk parameter ini.\n\n*Ulangi Pertanyaan:* {PERTANYAAN[current_step]}"
        with st.chat_message("assistant"):
            st.markdown(pesan_peringatan)
        st.session_state.messages.append({"role": "assistant", "content": pesan_peringatan})