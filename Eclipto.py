import streamlit as st
import PyPDF2
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
from streamlit_lottie import st_lottie



# Fungsi untuk membersihkan dan mempersiapkan teks
def clean_text(text):
    # Implementasikan pembersihan teks dasar jika diperlukan
    return text.strip()
# Fungsi untuk memuat animasi Lottie
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Fungsi untuk membaca teks dari file PDF
def read_pdf(file):
    try:
        return extract_text(file)
    except:
        reader = PyPDF2.PdfFileReader(file)
        text = ''
        for page in range(reader.numPages):
            text += reader.getPage(page).extractText()
        return text

# Fungsi untuk menghitung cosine similarity
def calculate_cosine_similarity(text1, text2):
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return np.round(cosine_sim[0][0] * 100, 2)
    except Exception as e:
        st.error(f"Error dalam menghitung cosine similarity: {e}")
        return None

# Streamlit app
st.image("IOH.jpg", width=200)
st.title("Teana - CV Matcher IOH")

col1, col2, col3 = st.columns([1,2,1])
with col2:  # Animasi akan ditampilkan di kolom tenga
     lotieurl="https://lottie.host/6b443cd1-428a-4de0-bbf9-c2cd373fd44d/aSJ9in8K2A.json"
     lottie_json = load_lottieurl(lotieurl)
     st_lottie(lottie_json, height=200, width=200)


# Pilihan antara mengunggah PDF atau memasukkan teks
option_cv = st.radio("Pilih metode input untuk CV:", ('Unggah PDF', 'Masukkan Teks'))
if option_cv == 'Unggah PDF':
    cv_file = st.file_uploader("Upload CV (PDF format)", type="pdf")
    if cv_file:
        cv_text = clean_text(read_pdf(cv_file))
else:
    cv_text = clean_text(st.text_area("Masukkan Teks CV"))

option_jd = st.radio("Pilih metode input untuk Deskripsi Pekerjaan:", ('Unggah PDF', 'Masukkan Teks'))
if option_jd == 'Unggah PDF':
    job_desc_file = st.file_uploader("Upload Job Description (PDF format)", type="pdf")
    if job_desc_file:
        job_desc_text = clean_text(read_pdf(job_desc_file))
else:
    job_desc_text = clean_text(st.text_area("Masukkan Teks Deskripsi Pekerjaan"))

if st.button("Calculate Similarity"):
    if cv_text and job_desc_text:
        similarity = calculate_cosine_similarity(cv_text, job_desc_text)
        if similarity is not None:
            st.write(f"Similarity Score: {similarity}%")
    else:
        st.error("Please provide both CV and Job Description.")
