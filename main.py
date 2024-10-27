import os
from dotenv import dotenv_values
import streamlit as st
from groq import Groq
import tempfile

# Konfigurasi page
st.set_page_config(
    page_title="Gathika: Analisa Audio Dengan AI",
    page_icon="🎙️",
    layout="wide"
)

# Handling environment variables
try:
    secrets = dotenv_values(".env")  # untuk development environment
    GROQ_API_KEY = secrets["GROQ_API_KEY"]
except:
    secrets = st.secrets  # untuk streamlit cloud deployment
    GROQ_API_KEY = secrets["GROQ_API_KEY"]

# Save API key to environment variable
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Inisialisasi Groq client
client = Groq()

# Fungsi untuk transkripsi audio
def transcribe_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        with open(tmp_file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(tmp_file_path, file.read()),
                model="whisper-large-v3-turbo",
                language="id",
                response_format="verbose_json"
            )
        os.unlink(tmp_file_path)
        return transcription.text
    except Exception as e:
        os.unlink(tmp_file_path)
        raise e

# Fungsi untuk analisis teks
def analyze_text(text):
    messages = [
        {
            "role": "system",
            "content": "Anda adalah asisten yang ahli dalam menganalisis teks. Berikan analisis yang mencakup: Ringkasan utama, Poin-poin penting, Topik utama yang dibahas, Konteks dan implikasi penting, Rekomendasi atau tindak lanjut (jika relevan)"
        },
        {
            "role": "user",
            "content": f"Analisis teks berikut ini:\n\n{text}"
        }
    ]
    
    analysis = ""
    stream = client.chat.completions.create(
        model="llama-3.2-1b-preview",
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            analysis += chunk.choices[0].delta.content
    
    return analysis

# Title
st.title("🎙️ Gathika Audio Transcription & Analysis")
st.write("Upload file audio untuk ditranskrip dan dianalisis secara otomatis")

# File uploader
allowed_types = ['flac', 'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'ogg', 'wav', 'webm']
uploaded_file = st.file_uploader(
    "Upload file audio Anda", 
    type=allowed_types,
    help=f"Format yang didukung: {', '.join(allowed_types)}. Maksimal 25MB"
)

if uploaded_file is not None:
    # Check file size
    if uploaded_file.size > 25 * 1024 * 1024:  # 25MB in bytes
        st.error("File terlalu besar. Maksimal ukuran file adalah 25MB")
    else:
        # Create tabs
        tab1, tab2 = st.tabs(["📊 Hasil Analisis", "📝 Hasil Transkripsi"])
        
        with st.spinner('Memproses file audio...'):
            try:
                # Transcribe audio
                transcription = transcribe_audio(uploaded_file)
                
                # Analyze transcription
                with st.spinner('Menganalisis transkripsi...'):
                    analysis = analyze_text(transcription)
                
                # Display results in tabs
                with tab1:
                    st.markdown("### 📊 Analisis")
                    st.write(analysis)
                
                with tab2:
                    st.markdown("### 📝 Transkripsi")
                    st.write(transcription)
                
                # Success message
                st.success('Proses selesai!')
                
            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")

# Footer
st.markdown("---")
st.caption("Dibuat oleh F- dengan Streamlit & Groq AI")
