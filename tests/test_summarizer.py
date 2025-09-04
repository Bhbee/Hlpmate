import pytest
from unittest.mock import patch, MagicMock, mock_open
from utils.summarizer import Summarizer


# === 1. Text Extraction Tests ===
@patch("utils.summarizer.PdfReader")
def test_extract_pdf(mock_pdf_reader):
    mock_pdf_reader.return_value.pages = [MagicMock(extract_text=lambda: "Page 1"), MagicMock(extract_text=lambda: "Page 2")]
    result = Summarizer.extract_text_from_file("sample.pdf")
    assert result == "Page 1\nPage 2"


@patch("utils.summarizer.docx2txt.process", return_value="Docx text")
def test_extract_docx(mock_process):
    result = Summarizer.extract_text_from_file("file.docx")
    assert result == "Docx text"


@patch("utils.summarizer.pptx.Presentation")
def test_extract_pptx(mock_presentation):
    mock_slide = MagicMock()
    mock_slide.shapes = [MagicMock(text="Title"), MagicMock(text="Content")]
    mock_presentation.return_value.slides = [mock_slide]
    result = Summarizer.extract_text_from_file("slides.pptx")
    assert "Title" in result and "Content" in result


def test_extract_txt(monkeypatch):
    mock_data = "Hello this is a test txt file."
    monkeypatch.setattr("builtins.open", mock_open(read_data=mock_data))
    result = Summarizer.extract_text_from_file("notes.txt")
    assert result == mock_data


@patch("utils.summarizer.pd.read_excel")
def test_extract_xlsx(mock_read_excel):
    df = MagicMock()
    df.astype.return_value.apply.return_value.str.cat.return_value = "Row1 Row2 Row3"
    mock_read_excel.return_value = df
    result = Summarizer.extract_text_from_file("data.xlsx")
    assert "Row1" in result


# === 2. Document Type Detection ===
@pytest.mark.parametrize("text, expected", [
    ("Curriculum Vitae\nLinkedIn.com", "cv"),
    ("Abstract\nIntroduction\nMethods", "academic"),
    ("Dear Guest,\nYou are invited...", "invitation"),
    ("This is a monthly newsletter", "newsletter"),
    ("Invoice\nAmount Due", "invoice"),
    ("Quick note", "short_note"),
    ("This is a generic file with random data", "generic"),
])
def test_detect_type(text, expected):
    assert Summarizer.detect_type(text) == expected


# === 3. Emphasis Formatting ===
def test_emphasize_keywords():
    text = "This is a summary. The definition is important for exams."
    emphasized = Summarizer.emphasize_keywords(text)
    assert "**definition**" in emphasized
    assert "**important**" in emphasized
    assert "**exam**" in emphasized


# === 4. GPT Summarization ===
@patch("utils.summarizer.client.chat.completions.create")
def test_gpt_summarize_success(mock_gpt):
    mock_gpt.return_value.choices = [MagicMock(message=MagicMock(content="Here is your summary."))]
    result = Summarizer.gpt_summarize("Summarize this.")
    assert result == "Here is your summary."


@patch("utils.summarizer.client.chat.completions.create", side_effect=Exception("API error"))
def test_gpt_summarize_failure(mock_gpt):
    result = Summarizer.gpt_summarize("Bad input")
    assert result.startswith("❌ GPT summarization failed")


# === 5. End-to-End Summarize File ===
@patch("utils.summarizer.Summarizer.extract_text_from_file", return_value="Abstract\nIntroduction\nMethods content")
@patch("utils.summarizer.Summarizer.gpt_summarize", return_value="Summary of chunk")
@patch("utils.summarizer.count_num_tokens", return_value=80)
def test_summarize_file_success(mock_token, mock_gpt, mock_extract):
    summary = Summarizer.summarize_file("academic.pdf")
    assert "**summary**" in summary or "Summary of chunk" in summary


@patch("utils.summarizer.Summarizer.extract_text_from_file", return_value="❌ Error reading file")
def test_summarize_file_error(mock_extract):
    summary = Summarizer.summarize_file("broken.pdf")
    assert summary.startswith("❌")
