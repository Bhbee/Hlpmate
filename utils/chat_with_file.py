try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass  # Use built-in sqlite3 on Windows/local
    
import traceback
import streamlit as st
from utils.prepare_vectordb import PrepareVectorDB
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from utils.load_config import LoadConfig

CONFIG = LoadConfig()


#function 1
@st.cache_resource(show_spinner=False)
def get_qa_chain(file_path: str):
    try:
        # Step 1: Prepare vector store (if not already persisted)
        processor = PrepareVectorDB(
            data_directory=[file_path],
            persist_directory=CONFIG.custom_persist_directory,
            openai_api_key=CONFIG.openai_api_key,
            chunk_size=CONFIG.chunk_size,
            chunk_overlap=CONFIG.chunk_overlap
        )
        processor.prepare_and_save_vectordb()

        # Step 2: Load vector store and retriever
        vectordb = Chroma(
            persist_directory=str(CONFIG.custom_persist_directory),
            embedding_function=OpenAIEmbeddings(openai_api_key=CONFIG.openai_api_key)
        )
        retriever = vectordb.as_retriever(search_kwargs={"k": CONFIG.k})

        # Step 3: Setup QA chain
        llm = ChatOpenAI(
            model_name=CONFIG.llm_engine,
            temperature=CONFIG.temperature,
            openai_api_key=CONFIG.openai_api_key
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False
        )

        return qa_chain

    except Exception as e:
        st.error("❌ Failed to initialize QA system.")
        traceback.print_exc()
        return None


#function 2
def chat_with_file(file_path: str):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    #Load or Get Cached QA Chain
    qa_chain = get_qa_chain(file_path)
    if not qa_chain:
        return

    #First-time bot greeting
    if not st.session_state.chat_history:
        st.markdown('''
            <div class="chat-row bot">
                <div class="chat-bubble bot-msg"><b>Helpy:</b> Hey, let me help you have a conversation with your file. Ask me anything!</div>
            </div>
        ''', unsafe_allow_html=True)

    for user_msg, bot_msg in st.session_state.chat_history:
        st.markdown(f'''
            <div class="chat-row user">
                <div class="chat-bubble user-msg"><b>You:</b> {user_msg}</div>
            </div>
            <div class="chat-row bot">
                <div class="chat-bubble bot-msg"><b>Helpy:</b> {bot_msg}</div>
            </div>
        ''', unsafe_allow_html=True)

    #Chat input
    user_input = st.chat_input("Ask something about your uploaded file...")

    if user_input:
        st.markdown(f'''
            <div class="chat-row user">
                <div class="chat-bubble user-msg"><b>You:</b> {user_input}</div>
            </div>
        ''', unsafe_allow_html=True)

        with st.spinner("Thinking..."):
            try:
                response = qa_chain.run(user_input)
                st.session_state.chat_history.append((user_input, response))
            except Exception:
                st.error("❌ Failed to get a response from the model.")
                traceback.print_exc()
                return

        st.markdown(f'''
            <div class="chat-row bot">
                <div class="chat-bubble bot-msg"><b>Helpy:</b> {response}</div>
            </div>
        ''', unsafe_allow_html=True)

        # Scroll to latest message
        st.markdown("""
            <script>
                const chatEnd = document.getElementById("scroll-anchor");
                if (chatEnd) {
                    chatEnd.scrollIntoView({ behavior: "smooth" });
                }
            </script>
        """, unsafe_allow_html=True)

# function to Reset All Session State
def reset_app_state():
    for key in [
        "file_path", "file_text", "questions", "current_question", "score",
        "answered", "score_history", "active_tab", "chat_history"
    ]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()
