import pytest
from unittest.mock import patch, MagicMock
from utils import generate_mcqs


@pytest.fixture
def mock_text():
    return """Artificial Intelligence is a branch of computer science that aims to create machines that can think and learn like humans. 
It includes fields such as machine learning, natural language processing, robotics, and computer vision. 
AI is used in healthcare, finance, transportation, and many other sectors."""


@pytest.fixture
def mock_gpt_output():
    return """
Q: What is the primary goal of Artificial Intelligence?
A. To create machines that operate manually
B. To replace all human jobs
C. To create machines that can think and learn like humans
D. To build hardware only
Answer: C

Q: Which of the following is NOT a field of AI?
A. Natural Language Processing
B. Machine Learning
C. Quantum Computing
D. Robotics
Answer: C
"""


@patch("utils.generate_mcqs.Summarizer.extract_text_from_file")
def test_extract_text(mock_extract_text_from_file):
    mock_extract_text_from_file.return_value = "Sample extracted content"
    result = generate_mcqs.MCQGenerator.extract_text("sample.txt")
    assert result == "Sample extracted content"
    mock_extract_text_from_file.assert_called_once_with("sample.txt")


@patch("utils.generate_mcqs.MCQGenerator.gpt_generate_mcqs_cached")
@patch("utils.generate_mcqs.Summarizer.extract_text_from_file")
def test_generate_mcqs_from_file_valid(mock_extract_text, mock_gpt_call, mock_text, mock_gpt_output):
    mock_extract_text.return_value = mock_text
    mock_gpt_call.return_value = mock_gpt_output

    mcqs = generate_mcqs.MCQGenerator.generate_mcqs_from_file("fakefile.pdf", max_questions=5)
    assert len(mcqs) == 2
    assert mcqs[0]["question"].startswith("What is the primary goal")


@patch("utils.generate_mcqs.Summarizer.extract_text_from_file")
def test_generate_mcqs_from_file_too_short(mock_extract_text):
    mock_extract_text.return_value = "Too short"
    result = generate_mcqs.MCQGenerator.generate_mcqs_from_file("tiny.txt")
    assert result == []


def test_parse_mcqs_success(mock_gpt_output):
    parsed = generate_mcqs.MCQGenerator.parse_mcqs(mock_gpt_output)
    assert len(parsed) == 2
    assert parsed[0]["question"].startswith("What is the primary goal")
    assert parsed[0]["correct"].startswith("To create machines")


def test_parse_mcqs_failure():
    malformed_output = """
Q: What is wrong here
A This option has no colon
B. Second option
Answer: B
"""
    result = generate_mcqs.MCQGenerator.parse_mcqs(malformed_output)
    assert result == []


@patch("utils.generate_mcqs.client.chat.completions.create")
def test_gpt_generate_mcqs_cached_success(mock_openai_call):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Q: Sample\nA. One\nB. Two\nC. Three\nD. Four\nAnswer: A"))]
    mock_openai_call.return_value = mock_response

    result = generate_mcqs.MCQGenerator.gpt_generate_mcqs_cached("generate 1 MCQ")
    assert result.startswith("Q:")
    mock_openai_call.assert_called_once()


@patch("utils.generate_mcqs.client.chat.completions.create", side_effect=Exception("API failure"))
def test_gpt_generate_mcqs_cached_failure(mock_openai_call):
    result = generate_mcqs.MCQGenerator.gpt_generate_mcqs_cached("generate 1 MCQ")
    assert result == ""
