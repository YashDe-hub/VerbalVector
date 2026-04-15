"""Tests for /api/query and /api/sessions endpoints."""
import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock


@pytest_asyncio.fixture
async def client():
    """Create an async test client with mocked config validation."""
    with patch("config.validate"):
        from api import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


@pytest.mark.asyncio
async def test_query_returns_answer(client):
    """POST /api/query should return an answer from RAG."""
    mock_collection = MagicMock()
    mock_chunks = [
        {"text": "chunk about timelines", "source_id": "talk1", "session_label": "Monday standup"},
    ]

    with (
        patch("api.initialize_vector_store", return_value=mock_collection),
        patch("api.search_transcripts", return_value=mock_chunks),
        patch("api.generate_rag_answer", return_value={"answer": "The timeline is Q3."}),
    ):
        response = await client.post("/api/query", json={"query": "timeline?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The timeline is Q3."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source_id"] == "talk1"


@pytest.mark.asyncio
async def test_query_with_source_id(client):
    """POST /api/query with source_id should pass it to search."""
    mock_collection = MagicMock()
    mock_chunks = [{"text": "scoped chunk", "source_id": "abc123", "session_label": "Talk"}]

    with (
        patch("api.initialize_vector_store", return_value=mock_collection),
        patch("api.search_transcripts", return_value=mock_chunks) as mock_search,
        patch("api.generate_rag_answer", return_value={"answer": "Scoped answer."}),
    ):
        response = await client.post(
            "/api/query",
            json={"query": "test?", "source_id": "abc123"},
        )

    mock_search.assert_called_once()
    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["source_id"] == "abc123"
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_query_no_results(client):
    """POST /api/query with no matching chunks should return no-content message."""
    mock_collection = MagicMock()

    with (
        patch("api.initialize_vector_store", return_value=mock_collection),
        patch("api.search_transcripts", return_value=[]),
    ):
        response = await client.post("/api/query", json={"query": "something obscure"})

    assert response.status_code == 200
    data = response.json()
    assert "no relevant" in data["answer"].lower()
    assert data["sources"] == []


@pytest.mark.asyncio
async def test_sessions_returns_list(client):
    """GET /api/sessions should return session list."""
    mock_collection = MagicMock()
    mock_sessions = [
        {"source_id": "a", "session_label": "Talk", "timestamp": 1000, "chunk_count": 5}
    ]

    with (
        patch("api.initialize_vector_store", return_value=mock_collection),
        patch("api.list_sessions", return_value=mock_sessions),
    ):
        response = await client.get("/api/sessions")

    assert response.status_code == 200
    data = response.json()
    assert len(data["sessions"]) == 1
    assert data["sessions"][0]["source_id"] == "a"
