import pytest
from unittest.mock import MagicMock, patch
from utils import chat_with_file as chat_module


@pytest.fixture
def mock_config():
    return {
        "custom_persist_directory": "mock_dir",
        "openai_api_key": "test-key",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "llm_engine": "gpt-4",
        "temperature": 0.3,
        "k": 3
    }


@patch("utils.chat_with_file.PrepareVectorDB")
@patch("utils.chat_with_file.Chroma")
@patch("utils.chat_with_file.OpenAIEmbeddings")
@patch("utils.chat_with_file.ChatOpenAI")
@patch("utils.chat_with_file.RetrievalQA")
@patch("utils.chat_with_file.st")
def test_get_qa_chain_success(mock_st, mock_RetrievalQA, mock_ChatOpenAI,
                              mock_Embeddings, mock_Chroma, mock_PrepareVectorDB):
    # Setup
    mock_retriever = MagicMock()
    mock_vectordb = MagicMock()
    mock_vectordb.as_retriever.return_value = mock_retriever
    mock_Chroma.return_value = mock_vectordb

    mock_llm = MagicMock()
    mock_ChatOpenAI.return_value = mock_llm

    mock_qa = MagicMock()
    mock_RetrievalQA.from_chain_type.return_value = mock_qa

    # Act
    qa_chain = chat_module.get_qa_chain("fake_file.txt")

    # Assert
    mock_PrepareVectorDB.return_value.prepare_and_save_vectordb.assert_called_once()
    mock_Chroma.assert_called_once()
    mock_RetrievalQA.from_chain_type.assert_called_once()
    assert qa_chain == mock_qa


@patch("utils.chat_with_file.PrepareVectorDB")
@patch("utils.chat_with_file.st")
def test_get_qa_chain_failure(mock_st, mock_PrepareVectorDB):
    mock_PrepareVectorDB.side_effect = Exception("DB error")

    chain = chat_module.get_qa_chain("bad_file.txt")
    assert chain is None
    mock_st.error.assert_called_once_with("❌ Failed to initialize QA system.")


@patch("utils.chat_with_file.get_qa_chain")
@patch("utils.chat_with_file.st")
def test_chat_with_file_shows_greeting(mock_st, mock_get_chain):
    mock_qa_chain = MagicMock()
    mock_qa_chain.run.return_value = "Sample answer"
    mock_get_chain.return_value = mock_qa_chain

    # Fake empty chat history
    mock_st.session_state = {"chat_history": []}
    mock_st.chat_input.return_value = None

    chat_module.chat_with_file("test.txt")
    assert mock_st.markdown.call_args_list[0][0][0].startswith("<div class=\"chat-row bot\">")


@patch("utils.chat_with_file.get_qa_chain")
@patch("utils.chat_with_file.st")
def test_chat_with_file_user_query(mock_st, mock_get_chain):
    mock_qa_chain = MagicMock()
    mock_qa_chain.run.return_value = "Answer from LLM"
    mock_get_chain.return_value = mock_qa_chain

    mock_st.session_state = {"chat_history": []}
    mock_st.chat_input.return_value = "What is this file about?"
    mock_st.spinner = MagicMock()

    chat_module.chat_with_file("sample.pdf")
    assert ("What is this file about?", "Answer from LLM") in mock_st.session_state["chat_history"]


@patch("utils.chat_with_file.st")
def test_reset_app_state(mock_st):
    mock_st.session_state = {
        "file_path": "path",
        "file_text": "text",
        "questions": [],
        "current_question": 0,
        "score": 0,
        "answered": True,
        "score_history": [],
        "active_tab": "chat",
        "chat_history": [],
    }

    chat_module.reset_app_state()
    for key in [
        "file_path", "file_text", "questions", "current_question", "score",
        "answered", "score_history", "active_tab", "chat_history"
    ]:
        assert key not in mock_st.session_state
