"""Tests for the speaker_id service. The SpeechBrain model is always mocked."""
import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.services import speaker_id


@pytest.fixture
def profile_dir(tmp_path, monkeypatch):
    """Point the profile store at a temp dir."""
    monkeypatch.setattr(speaker_id, "_profile_dir", lambda: tmp_path)
    return tmp_path


def _norm(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


# ---------------- profile store ----------------

def test_save_and_load_profile_roundtrip(profile_dir):
    emb = _norm([1.0, 2.0, 3.0])
    meta = speaker_id.save_profile(emb, duration_seconds=31.5)
    assert meta["duration_seconds"] == 31.5
    assert (profile_dir / "embedding.npy").exists()
    assert (profile_dir / "profile.json").exists()
    loaded = speaker_id.load_profile()
    assert np.allclose(loaded, emb)


def test_load_profile_returns_none_when_absent(profile_dir):
    assert speaker_id.load_profile() is None
    assert speaker_id.get_profile_meta() is None


def test_delete_profile(profile_dir):
    speaker_id.save_profile(_norm([1.0, 0.0]), duration_seconds=12.0)
    assert speaker_id.delete_profile() is True
    assert speaker_id.load_profile() is None
    assert speaker_id.delete_profile() is False  # already gone


def test_load_profile_corrupt_returns_none(profile_dir):
    (profile_dir / "embedding.npy").write_text("not a numpy file")
    assert speaker_id.load_profile() is None  # never raises


# ---------------- matching ----------------

def _utt(speaker, start, end, text="hi"):
    return {"speaker": speaker, "text": text, "start": start, "end": end, "confidence": 0.9}


def test_match_user_picks_highest_similarity():
    profile = _norm([1.0, 0.0, 0.0])
    fake_embeddings = {0: _norm([0.0, 1.0, 0.0]), 1: _norm([0.9, 0.1, 0.0])}
    utts = [_utt(0, 0.0, 5.0), _utt(1, 5.0, 9.0)]
    with patch.object(speaker_id, "_embed_speaker", side_effect=lambda a, s, spk: fake_embeddings[spk]):
        result = speaker_id.match_user("x.wav", utts, profile)
    assert result["user_speaker"] == 1
    assert result["confidence"] > 0.9
    assert result["low_confidence"] is False


def test_match_user_low_confidence_below_threshold():
    profile = _norm([1.0, 0.0, 0.0])
    # best similarity ~0.1 < threshold 0.30
    with patch.object(speaker_id, "_embed_speaker", return_value=_norm([0.1, 1.0, 0.0])):
        result = speaker_id.match_user("x.wav", [_utt(0, 0.0, 5.0)], profile)
    assert result["user_speaker"] == 0
    assert result["low_confidence"] is True


def test_match_user_returns_none_when_no_embeddings():
    with patch.object(speaker_id, "_embed_speaker", return_value=None):
        assert speaker_id.match_user("x.wav", [_utt(0, 0.0, 5.0)], _norm([1.0, 0.0])) is None


def test_match_user_returns_none_on_exception():
    with patch.object(speaker_id, "_embed_speaker", side_effect=RuntimeError("boom")):
        assert speaker_id.match_user("x.wav", [_utt(0, 0.0, 5.0)], _norm([1.0, 0.0])) is None


def test_match_user_caps_segment_seconds():
    """Segments passed to embedding are capped at ~30s per speaker."""
    captured = {}

    def fake_embed(audio_path, segments, spk):
        captured[spk] = segments
        return _norm([1.0, 0.0])

    utts = [_utt(0, float(i * 10), float(i * 10 + 10)) for i in range(10)]  # 100s total
    with patch.object(speaker_id, "_embed_speaker", side_effect=fake_embed):
        speaker_id.match_user("x.wav", utts, _norm([1.0, 0.0]))
    total = sum(e - s for s, e in captured[0])
    assert total == pytest.approx(speaker_id.MAX_MATCH_SECONDS)


# ---------------- embedding path ----------------

def test_compute_embedding_too_little_audio_returns_none():
    # less than _MIN_EMBED_SECONDS (0.5s * 16000 = 8000 samples) → None
    with patch.object(speaker_id, "_load_audio", return_value=np.zeros(100, dtype=np.float32)):
        assert speaker_id.compute_embedding("x.wav") is None


def test_compute_embedding_zero_norm_returns_none():
    import torch
    fake_clf = MagicMock()
    fake_clf.encode_batch.return_value = torch.zeros(1, 1, 8)
    with (
        patch.object(speaker_id, "_load_audio", return_value=np.ones(16000, dtype=np.float32)),
        patch.object(speaker_id, "_get_classifier", return_value=fake_clf),
    ):
        assert speaker_id.compute_embedding("x.wav") is None


def test_compute_embedding_returns_l2_normalized_vector():
    import torch
    fake_clf = MagicMock()
    fake_clf.encode_batch.return_value = torch.tensor([[[3.0, 4.0]]])  # raw norm = 5
    with (
        patch.object(speaker_id, "_load_audio", return_value=np.ones(16000, dtype=np.float32)),
        patch.object(speaker_id, "_get_classifier", return_value=fake_clf),
    ):
        emb = speaker_id.compute_embedding("x.wav")
    assert emb is not None
    assert np.isclose(np.linalg.norm(emb), 1.0)


def test_compute_embedding_returns_none_on_failure():
    with patch.object(speaker_id, "_load_audio", side_effect=RuntimeError("boom")):
        assert speaker_id.compute_embedding("x.wav") is None


def test_export_segments_wav_none_when_no_audio():
    with patch.object(speaker_id, "_load_audio", return_value=None):
        assert speaker_id.export_segments_wav("x.wav", [(0.0, 1.0)], "out.wav") is None
