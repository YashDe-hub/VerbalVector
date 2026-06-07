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


@pytest.mark.asyncio
async def test_result_ready_returns_200(client, tmp_path):
    sid = "c" * 32
    _write_results(tmp_path, sid)
    resp = await client.get(f"/api/sessions/{sid}/result")
    assert resp.status_code == 200
    body = resp.json()
    assert body["feedback"] == "Good job."
    assert body["transcript"]["text"] == "hi"
    assert body["features"]["words_per_minute"] == 100


@pytest.mark.asyncio
async def test_result_pending_returns_202_without_error_log(client, tmp_path, caplog):
    sid = "d" * 32
    _write_results(tmp_path, sid, feedback=False)  # partial → pending
    with caplog.at_level(logging.ERROR):
        resp = await client.get(f"/api/sessions/{sid}/result")
    assert resp.status_code == 202
    assert resp.json() == {"status": "pending"}
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR], "202 must not log ERROR"


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["short", "g" * 32, "A" * 32])
async def test_result_invalid_id_returns_400(client, bad):
    resp = await client.get(f"/api/sessions/{bad}/result")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_result_path_traversal_is_rejected(client):
    # A traversal payload contains slashes, so it can never resolve to the
    # single {session_id} path segment — the request lands on no route (404)
    # rather than serving an out-of-bounds file. Either way it is safely
    # rejected and never reaches read_analysis_results.
    resp = await client.get("/api/sessions/../../etc/passwd/result")
    assert resp.status_code in (400, 404)


@pytest.mark.asyncio
async def test_result_corrupt_file_returns_500(client, tmp_path):
    sid = "e" * 32
    _write_results(tmp_path, sid, corrupt=True)  # all present but transcript is bad JSON
    resp = await client.get(f"/api/sessions/{sid}/result")
    assert resp.status_code == 500
