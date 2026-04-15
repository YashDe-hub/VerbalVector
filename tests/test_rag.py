"""Tests for RAG answer generation."""
from unittest.mock import patch, MagicMock


def test_generate_rag_answer_returns_answer():
    """generate_rag_answer should return text from Gemini."""
    from src.services.llm import generate_rag_answer

    mock_response = MagicMock()
    mock_response.text = "Based on your presentation, the timeline was discussed in Q3."

    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client_instance), \
         patch("config.GEMINI_API_KEY", "fake-key"), \
         patch("config.GEMINI_LLM_MODEL", "gemini-2.5-flash"):
        result = generate_rag_answer(
            query="What about the timeline?",
            context_chunks=[
                {"text": "We discussed the Q3 timeline.", "source_id": "talk1", "session_label": "Monday standup"},
            ],
        )

    assert result is not None
    assert "answer" in result
    assert result["answer"] == mock_response.text

    call_kwargs = mock_client_instance.models.generate_content.call_args
    assert call_kwargs[1]["config"].temperature == 0.3


def test_generate_rag_answer_no_context():
    """generate_rag_answer with empty context should return a no-content message without calling Gemini."""
    from src.services.llm import generate_rag_answer

    result = generate_rag_answer(
        query="What about the timeline?",
        context_chunks=[],
    )

    assert result is not None
    assert "no relevant" in result["answer"].lower()


def test_generate_rag_answer_gemini_error():
    """generate_rag_answer should return None if Gemini raises."""
    from src.services.llm import generate_rag_answer

    with patch("google.genai.Client", side_effect=Exception("API error")):
        result = generate_rag_answer(
            query="test?",
            context_chunks=[{"text": "some text", "source_id": "x", "session_label": "y"}],
        )

    assert result is None


def test_generate_rag_answer_empty_response():
    """generate_rag_answer should return None if Gemini returns empty text."""
    from src.services.llm import generate_rag_answer

    mock_response = MagicMock()
    mock_response.text = None

    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client_instance), \
         patch("config.GEMINI_API_KEY", "fake-key"), \
         patch("config.GEMINI_LLM_MODEL", "gemini-2.5-flash"):
        result = generate_rag_answer(
            query="test?",
            context_chunks=[{"text": "some text", "source_id": "x", "session_label": "y"}],
        )

    assert result is None
