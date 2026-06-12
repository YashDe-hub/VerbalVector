"""Tests for the /api/enroll endpoints. speaker_id internals are mocked."""
import io
import numpy as np
import pytest
import pytest_asyncio
from unittest.mock import patch


@pytest_asyncio.fixture
async def client():
    with patch("config.validate"):
        import api
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=api.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


def _wav_upload(name="enroll.wav"):
    return {"file": (name, io.BytesIO(b"RIFF....WAVEfmt fake"), "audio/wav")}


@pytest.mark.asyncio
async def test_enroll_happy_path(client):
    meta = {"created_at": "2026-06-12T00:00:00+00:00", "duration_seconds": 31.0, "model": "m"}
    with (
        patch("api.speaker_id.compute_embedding", return_value=np.ones(3, dtype=np.float32)),
        patch("api.speaker_id.save_profile", return_value=meta) as save,
        patch("api.librosa.get_duration", return_value=31.0),
    ):
        resp = await client.post("/api/enroll", files=_wav_upload())
    assert resp.status_code == 200
    assert resp.json()["duration_seconds"] == 31.0
    save.assert_called_once()


@pytest.mark.asyncio
async def test_enroll_too_short_returns_400(client):
    with patch("api.librosa.get_duration", return_value=3.0):
        resp = await client.post("/api/enroll", files=_wav_upload())
    assert resp.status_code == 400
    assert "least" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_enroll_bad_extension_returns_400(client):
    resp = await client.post(
        "/api/enroll", files={"file": ("x.txt", io.BytesIO(b"hi"), "text/plain")}
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_enroll_embedding_failure_returns_500(client):
    with (
        patch("api.librosa.get_duration", return_value=31.0),
        patch("api.speaker_id.compute_embedding", return_value=None),
    ):
        resp = await client.post("/api/enroll", files=_wav_upload())
    assert resp.status_code == 500


@pytest.mark.asyncio
async def test_get_enroll_status(client):
    meta = {"created_at": "x", "duration_seconds": 31.0, "model": "m"}
    with patch("api.speaker_id.get_profile_meta", return_value=meta):
        resp = await client.get("/api/enroll")
    assert resp.status_code == 200
    assert resp.json() == meta


@pytest.mark.asyncio
async def test_get_enroll_404_when_absent(client):
    with patch("api.speaker_id.get_profile_meta", return_value=None):
        resp = await client.get("/api/enroll")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_enroll(client):
    with patch("api.speaker_id.delete_profile", return_value=True):
        resp = await client.delete("/api/enroll")
    assert resp.status_code == 204
    with patch("api.speaker_id.delete_profile", return_value=False):
        resp = await client.delete("/api/enroll")
    assert resp.status_code == 404
