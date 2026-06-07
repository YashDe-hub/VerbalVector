"""Tests for result-file helpers and GET /api/sessions/{id}/result."""
import json
import logging
import pytest
import pytest_asyncio
from unittest.mock import patch


@pytest_asyncio.fixture
async def client(tmp_path, monkeypatch):
    """Async test client with a temp ANALYSIS_OUTPUT_FOLDER."""
    with patch("config.validate"):
        import api
        monkeypatch.setattr(api, "ANALYSIS_OUTPUT_FOLDER", str(tmp_path))
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=api.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


def _write_results(folder, session_id, *, transcript=True, features=True, feedback=True, corrupt=False):
    if transcript:
        text = "not json{" if corrupt else json.dumps({"text": "hi", "utterances": [], "speakers": [0]})
        (folder / f"{session_id}_transcript.json").write_text(text, encoding="utf-8")
    if features:
        (folder / f"{session_id}_features.json").write_text(json.dumps({"words_per_minute": 100}), encoding="utf-8")
    if feedback:
        (folder / f"{session_id}_feedback.txt").write_text("Good job.", encoding="utf-8")


def test_read_analysis_results_all_present(tmp_path, monkeypatch):
    import api
    monkeypatch.setattr(api, "ANALYSIS_OUTPUT_FOLDER", str(tmp_path))
    sid = "a" * 32
    _write_results(tmp_path, sid)
    t, f, fb = api.result_file_paths(sid)
    out = api.read_analysis_results(t, f, fb)
    assert out == {
        "transcript": {"text": "hi", "utterances": [], "speakers": [0]},
        "features": {"words_per_minute": 100},
        "feedback": "Good job.",
    }


def test_read_analysis_results_missing_returns_none(tmp_path, monkeypatch):
    import api
    monkeypatch.setattr(api, "ANALYSIS_OUTPUT_FOLDER", str(tmp_path))
    sid = "b" * 32
    _write_results(tmp_path, sid, feedback=False)
    t, f, fb = api.result_file_paths(sid)
    assert api.read_analysis_results(t, f, fb) is None
