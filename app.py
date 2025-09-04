import os
import glob
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from utils.load_config import LoadConfig
from utils.summarizer import Summarizer
from utils.generate_mcqs import MCQGenerator
from utils.chat_with_file import chat_with_file
from utils.quiz_engine import QuizEngine
from utils.session import reset_app_session

# === Load environment variables ===
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# === Load config ===
CONFIG = LoadConfig()

# === Streamlit Config ===
st.set_page_config(page_title="HELPY Assistant", layout="wide")

# === Google Font for Calligraphy Title ===
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Dancing+Script&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# === Load Custom CSS ===
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# === Main Title ===
st.markdown("<h1 class='main-title'>Helpy: Your Personal Assistant 📚 </h1>", unsafe_allow_html=True)
st.markdown("---")

# === Initialize Session State ===
if "file_path" not in st.session_state:
    st.session_state.file_path = None
    st.session_state.file_text = ""
    st.session_state.questions = []
    st.session_state.current_question = 0
    st.session_state.score = 0
    st.session_state.answered = False
    st.session_state.score_history = []
    st.session_state.active_tab = None
    st.session_state.chat_history = []

# === Define Upload Directory ===
upload_dir = os.path.join("data", "uploads")
os.makedirs(upload_dir, exist_ok=True)

# === Upload UI (Before Upload) ===
if not st.session_state.file_path:
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="landing-wrapper">
                <div class='pre-upload-box'>
                    <h3>📥 Ready to Study Smarter?</h3>
                    <p>Upload your lecture notes, handouts, past questions or slides to get started.</p>
                </div>
                <div class='upload-centered'>
            """, unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                label="Upload your file",
                type=["pdf", "docx", "pptx", "xlsx", "txt"],
                label_visibility="collapsed"
            )

            st.markdown("</div></div>", unsafe_allow_html=True)

    if uploaded_file:
        try:
            # Clean existing files in upload_dir
            for old_file in glob.glob(os.path.join(upload_dir, "*")):
                os.remove(old_file)

            file_name = uploaded_file.name
            file_path = os.path.join(upload_dir, file_name)

            # Save new file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.session_state.file_path = file_path
            st.toast("✅ File uploaded successfully!", icon="📄")

            # Optional PDF text extraction (basic preview only)
            try:
                if file_name.endswith(".pdf"):
                    from PyPDF2 import PdfReader
                    reader = PdfReader(file_path)
                    text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
                    st.session_state.file_text = text
            except Exception as e:
                st.warning(f"PDF preview extraction failed: {e}")

        except Exception as e:
            st.error(f"❌ File upload failed: {e}")

# === Sidebar Menu (After Upload) ===
if st.session_state.file_path:
    with st.sidebar:
        st.markdown("<div class='sidebar-section-title'>🧭 CHOOSE A FEATURE</div>", unsafe_allow_html=True)

        if st.button("📝 Summarize"):
            st.session_state.active_tab = "summarize"
        if st.button("❓ Self-Test"):
            st.session_state.active_tab = "self_test"
        if st.button("💬 Chat With File"):
            st.session_state.active_tab = "chat"

        st.markdown("---")
        if st.button("📥 Upload new document"):
            reset_app_session()

# === Main Feature Logic ===
if st.session_state.file_path and st.session_state.active_tab:

    if st.session_state.active_tab == "summarize":
        st.subheader("📋 Summary")
        with st.spinner("Generating summary..."):
            summary = Summarizer.summarize_file(st.session_state.file_path)
        with st.expander("🔍 View Summary", expanded=True):
            st.markdown(f"<div class='summary-box'>{summary}</div>", unsafe_allow_html=True)

    elif st.session_state.active_tab == "chat":
        st.markdown("<div class='chat-wrapper'>", unsafe_allow_html=True)
        chat_with_file(st.session_state.file_path)
        st.markdown("</div>", unsafe_allow_html=True)

    elif st.session_state.active_tab == "self_test":
        st.subheader("❓ Self-Test Mode")

        if not st.session_state.questions:
            with st.spinner("Generating questions..."):
                questions = MCQGenerator.generate_mcqs_from_file(
                    st.session_state.file_path, max_questions=10
                )
                if questions:
                    st.session_state.questions = questions
                    st.session_state.current_question = 0
                    st.session_state.score = 0
                    st.session_state.answered = False
                else:
                    st.warning("⚠️ No questions could be generated from the uploaded document.")

        if st.session_state.questions:
            QuizEngine.start_quiz_session(st.session_state.questions)
