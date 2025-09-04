import pytest
from unittest.mock import patch, MagicMock
from HELPY.utils.quiz_engine import QuizEngine


@pytest.fixture
def sample_questions():
    return [
        {
            "question": "What is the capital of France?",
            "options": ["Paris", "London", "Berlin", "Rome"],
            "correct": "Paris",
            "explanation": "Paris is the capital of France."
        },
        {
            "question": "What is 2 + 2?",
            "options": ["3", "4", "5", "6"],
            "correct": "4",
            "explanation": "2 + 2 equals 4."
        }
    ]


@patch("utils.test_engine.st")
def test_start_quiz_first_question(mock_st, sample_questions):
    mock_st.session_state.questions = sample_questions
    mock_st.session_state.current_question = 0
    mock_st.session_state.score = 0
    mock_st.session_state.answered = False

    # Mocks
    mock_st.radio.return_value = "Paris"
    mock_st.button.side_effect = [True, False]  # Submit clicked, Next not clicked

    QuizEngine.start_quiz_session(sample_questions)

    mock_st.markdown.assert_any_call(
        "<div class='quiz-actions'>", unsafe_allow_html=True
    )
    mock_st.success.assert_called_once_with("✅ Correct!")
    assert mock_st.session_state.score == 1


@patch("utils.test_engine.st")
def test_submit_wrong_answer(mock_st, sample_questions):
    mock_st.session_state.questions = sample_questions
    mock_st.session_state.current_question = 0
    mock_st.session_state.score = 0
    mock_st.session_state.answered = False

    mock_st.radio.return_value = "London"
    mock_st.button.side_effect = [True, False]  # Submit clicked

    QuizEngine.start_quiz_session(sample_questions)

    mock_st.error.assert_called_once_with("❌ Incorrect. Correct answer: Paris")
    assert mock_st.session_state.score == 0


@patch("utils.test_engine.st")
def test_next_question_logic(mock_st, sample_questions):
    mock_st.session_state.questions = sample_questions
    mock_st.session_state.current_question = 0
    mock_st.session_state.answered = True

    mock_st.radio.return_value = "Paris"
    mock_st.button.side_effect = [False, True]  # Submit not clicked, Next clicked

    QuizEngine.start_quiz_session(sample_questions)

    assert mock_st.session_state.current_question == 1
    assert mock_st.session_state.answered is False


@patch("utils.test_engine.st")
def test_quiz_completion_logic(mock_st, sample_questions):
    mock_st.session_state.questions = sample_questions
    mock_st.session_state.current_question = 2  # Finished quiz
    mock_st.session_state.score = 1

    mock_button = MagicMock(return_value=False)
    mock_st.button = mock_button

    QuizEngine.show_score()

    mock_st.balloons.assert_called_once()
    mock_st.success.assert_called_once()
    mock_st.metric.assert_any_call("✅ Correct", 1)
    mock_st.metric.assert_any_call("❌ Incorrect", 1)
