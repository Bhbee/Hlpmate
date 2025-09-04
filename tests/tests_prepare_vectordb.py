import os
import pytest
from unittest.mock import patch, MagicMock
from utils.prepare_vectordb import PrepareVectorDB


FAKE_FILE = "sample.pdf"
FAKE_DB_PATH = "./tmp_chroma_db"
API_KEY = "sk-test"
SAMPLE_DOC = MagicMock()
SAMPLE_CHUNKS = [MagicMock() for _ in range(3)]


# === 1. Successful vector DB preparation ===
@patch("utils.prepare_vectordb.PyPDFLoader")
@patch("utils.prepare_vectordb.OpenAIEmbeddings")
@patch("utils.prepare_vectordb.Chroma")
@patch("utils.prepare_vectordb.RecursiveCharacterTextSplitter")
def test_prepare_vectordb_success(mock_splitter, mock_chroma, mock_embeddings, mock_loader):
    mock_loader.return_value.load.return_value = [SAMPLE_DOC]
    mock_splitter.return_value.split_documents.return_value = SAMPLE_CHUNKS
    mock_chroma.from_documents.return_value.persist.return_value = None

    prep = PrepareVectorDB(
        data_directory=FAKE_FILE,
        persist_directory=FAKE_DB_PATH,
        openai_api_key=API_KEY
    )

    with patch("os.path.exists", return_value=True):
        prep.prepare_and_save_vectordb()
        mock_loader.assert_called_once()
        mock_chroma.from_documents.assert_called_once()


# === 2. Unsupported file extension ===
def test_unsupported_format():
    prep = PrepareVectorDB(
        data_directory="notes.xyz",
        persist_directory="./db",
        openai_api_key=API_KEY
    )
    with patch("os.path.exists", return_value=True):
        with pytest.raises(ValueError, match="Unsupported file format"):
            prep._load_document()


# === 3. File not found ===
def test_file_not_found():
    prep = PrepareVectorDB(
        data_directory=FAKE_FILE,
        persist_directory=FAKE_DB_PATH,
        openai_api_key=API_KEY
    )
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            prep.prepare_and_save_vectordb()


# === 4. Document loading failure ===
@patch("utils.prepare_vectordb.PyPDFLoader")
def test_load_document_failure(mock_loader):
    mock_loader.return_value.load.side_effect = Exception("Corrupted PDF")

    prep = PrepareVectorDB(
        data_directory=FAKE_FILE,
        persist_directory=FAKE_DB_PATH,
        openai_api_key=API_KEY
    )

    with patch("os.path.exists", return_value=True):
        with pytest.raises(Exception, match="Corrupted PDF"):
            prep.prepare_and_save_vectordb()


# === 5. No chunks after splitting ===
@patch("utils.prepare_vectordb.PyPDFLoader")
@patch("utils.prepare_vectordb.RecursiveCharacterTextSplitter")
def test_no_chunks_extracted(mock_splitter, mock_loader):
    mock_loader.return_value.load.return_value = [SAMPLE_DOC]
    mock_splitter.return_value.split_documents.return_value = []

    prep = PrepareVectorDB(
        data_directory=FAKE_FILE,
        persist_directory=FAKE_DB_PATH,
        openai_api_key=API_KEY
    )

    with patch("os.path.exists", return_value=True):
        with pytest.raises(ValueError, match="no text chunks"):
            prep.prepare_and_save_vectordb()


# === 6. Recovery from corrupt vector store (e.g. "no such column") ===
@patch("utils.prepare_vectordb.PyPDFLoader")
@patch("utils.prepare_vectordb.RecursiveCharacterTextSplitter")
@patch("utils.prepare_vectordb.OpenAIEmbeddings")
@patch("utils.prepare_vectordb.Chroma.from_documents")
@patch("os.path.exists", return_value=True)
@patch("shutil.rmtree")
def test_corrupt_vector_db_retry(
    mock_rmtree, mock_exists, mock_chroma, mock_embeddings,
    mock_splitter, mock_loader
):
    # First call fails with "no such column", second succeeds
    mock_loader.return_value.load.return_value = [SAMPLE_DOC]
    mock_splitter.return_value.split_documents.return_value = SAMPLE_CHUNKS

    # First Chroma.from_documents call fails
    def side_effect(*args, **kwargs):
        if not hasattr(side_effect, "called"):
            side_effect.called = True
            raise Exception("no such column: embeddings")
        else:
            return MagicMock(persist=MagicMock())

    mock_chroma.side_effect = side_effect

    prep = PrepareVectorDB(
        data_directory=FAKE_FILE,
        persist_directory=FAKE_DB_PATH,
        openai_api_key=API_KEY
    )

    prep.prepare_and_save_vectordb()

    assert mock_rmtree.called
    assert mock_chroma.call_count == 2
