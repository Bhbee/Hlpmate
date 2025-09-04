import pytest
from unittest.mock import patch, MagicMock
from utils.upload_file import UploadFile


VALID_FILE = "lecture_notes.pdf"
INVALID_FILE = "archive.zip"
CHAT_STATE = []


# === 1. File Type Validation ===
def test_unsupported_file_extension():
    result, chat = UploadFile.process_uploaded_file("badfile.exe", [], "Upload doc: Process for RAG")
    assert "❌ Unsupported file type" in chat[-1][1]
    assert result == ""


# === 2. RAG Vector DB Creation ===
@patch("utils.upload_file.PrepareVectorDB")
def test_rag_vector_creation(mock_processor):
    mock_instance = mock_processor.return_value
    mock_instance.prepare_and_save_vectordb.return_value = None

    result, chat = UploadFile.process_uploaded_file(VALID_FILE, [], "Upload doc: Process for RAG")
    assert "✅ Vector database created" in chat[-1][1]
    mock_processor.assert_called_once()


# === 3. Summary Generation ===
@patch("utils.upload_file.Summarizer.summarize", return_value="Summary text here.")
def test_generate_summary(mock_summary):
    result, chat = UploadFile.process_uploaded_file(VALID_FILE, [], "Upload doc: Give Full summary")
    assert "Summary text here." in chat[-1][1]
    mock_summary.assert_called_once_with(VALID_FILE)


# === 4. MCQ Generation - Empty Text ===
@patch("utils.upload_file.Summarizer.extract_text_from_file", return_value="")
def test_generate_mcqs_empty_text(mock_extract):
    result, chat = UploadFile.process_uploaded_file(VALID_FILE, [], "Upload doc: Generate MCQs")
    assert "⚠️ Could not extract text from the file" in chat[-1][1]


# === 5. MCQ Generation - No Questions ===
@patch("utils.upload_file.Summarizer.extract_text_from_file", return_value="Some valid content")
@patch("utils.upload_file.MCQGenerator", return_value=[])
def test_generate_mcqs_none(mock_mcq, mock_extract):
    result, chat = UploadFile.process_uploaded_file(VALID_FILE, [], "Upload doc: Generate MCQs")
    assert "❌ MCQ generation failed" in chat[-1][1]


# === 6. MCQ Generation - Valid ===
@patch("utils.upload_file.Summarizer.extract_text_from_file", return_value="Some academic content")
@patch("utils.upload_file.MCQGenerator")
def test_generate_mcqs_success(mock_mcq, mock_extract):
    mock_mcq.return_value = [
        {
            "question": "What is AI?",
            "options": ["Artificial Intelligence", "Animal Instinct", "Alien Intelligence", "Automated Interface"],
            "correct": "Artificial Intelligence",
            "explanation": "AI stands for Artificial Intelligence."
        }
    ]
    result, chat = UploadFile.process_uploaded_file(VALID_FILE, [], "Upload doc: Generate MCQs")
    assert "**Q1." in chat[-1][1]
    assert "✅ Correct Answer" in chat[-1][1]


# === 7. Invalid Dropdown Option ===
def test_invalid_dropdown_action():
    result, chat = UploadFile.process_uploaded_file(VALID_FILE, [], "Something else")
    assert "❗ Please select a valid action" in chat[-1][1]


# === 8. Exception Handling ===
@patch("utils.upload_file.Summarizer.summarize", side_effect=Exception("Boom"))
def test_exception_in_summary(mock_sum):
    result, chat = UploadFile.process_uploaded_file(VALID_FILE, [], "Upload doc: Give Full summary")
    assert "❌ Operation failed: Boom" in chat[-1][1]
