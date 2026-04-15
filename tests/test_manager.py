"""Tests for vector store manager — filtering and session listing."""
import time
from unittest.mock import MagicMock


def _make_mock_collection():
    """Create a mock ChromaDB collection."""
    collection = MagicMock()
    collection.name = "transcripts"
    return collection


def test_search_transcripts_with_source_filter():
    """search_transcripts with source_id should pass a where filter and return dicts."""
    from src.vector_store.manager import search_transcripts

    collection = _make_mock_collection()
    collection.query.return_value = {
        "documents": [["chunk1", "chunk2"]],
        "metadatas": [[
            {"source": "abc123", "session_label": "Talk 1"},
            {"source": "abc123", "session_label": "Talk 1"},
        ]],
    }

    results = search_transcripts(
        query="test query",
        collection=collection,
        n_results=3,
        source_id="abc123",
    )

    call_kwargs = collection.query.call_args[1]
    assert call_kwargs["where"] == {"source": "abc123"}
    assert len(results) == 2
    assert results[0]["text"] == "chunk1"
    assert results[0]["source_id"] == "abc123"
    assert results[0]["session_label"] == "Talk 1"


def test_search_transcripts_without_source_filter():
    """search_transcripts without source_id should not pass a where filter."""
    from src.vector_store.manager import search_transcripts

    collection = _make_mock_collection()
    collection.query.return_value = {
        "documents": [["chunk1"]],
        "metadatas": [[{"source": "src1", "session_label": "My Talk"}]],
    }

    results = search_transcripts(
        query="test query",
        collection=collection,
        n_results=3,
    )

    call_kwargs = collection.query.call_args[1]
    assert "where" not in call_kwargs
    assert results[0]["text"] == "chunk1"
    assert results[0]["source_id"] == "src1"


def test_store_transcript_includes_session_label():
    """store_transcript should include session_label in metadata."""
    from src.vector_store.manager import store_transcript

    collection = _make_mock_collection()

    store_transcript(
        transcript_text="Hello world. This is a test.",
        source_id="test_source",
        collection=collection,
        session_label="My Presentation",
    )

    call_kwargs = collection.add.call_args[1]
    metadata_list = call_kwargs["metadatas"]
    assert all(m["session_label"] == "My Presentation" for m in metadata_list)
    assert all(m["source"] == "test_source" for m in metadata_list)


def test_list_sessions():
    """list_sessions should return unique sessions with counts."""
    from src.vector_store.manager import list_sessions

    collection = _make_mock_collection()
    ts = time.time()
    collection.get.return_value = {
        "metadatas": [
            {"source": "a", "session_label": "Talk 1", "timestamp": ts},
            {"source": "a", "session_label": "Talk 1", "timestamp": ts},
            {"source": "b", "session_label": "Talk 2", "timestamp": ts + 1},
        ]
    }

    sessions = list_sessions(collection)

    assert len(sessions) == 2
    session_a = next(s for s in sessions if s["source_id"] == "a")
    assert session_a["chunk_count"] == 2
    assert session_a["session_label"] == "Talk 1"
