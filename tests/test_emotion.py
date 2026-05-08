"""Tests for Hume emotion analysis."""
import pytest
from unittest.mock import patch, MagicMock


def _details(status):
    d = MagicMock()
    d.state.status = status
    return d


@pytest.fixture
def fake_audio(tmp_path):
    p = tmp_path / "recording.wav"
    p.write_bytes(b"fake")
    return str(p)


def test_analyze_calls_hume_with_new_signature(fake_audio):
    """Must call start_inference_job_from_local_file with file=[path] and json=InferenceBaseRequest.

    This is the regression-guard for the Hume SDK API change. If the signature
    drifts back to filepath= / models=, this test will catch it.
    """
    from src.services.emotion import analyze
    from hume.expression_measurement.batch.types import InferenceBaseRequest

    mock_client = MagicMock()
    batch = mock_client.expression_measurement.batch
    batch.start_inference_job_from_local_file.return_value = "job_abc"
    batch.get_job_details.return_value = _details("COMPLETED")
    batch.get_job_predictions.return_value = []  # empty → returns None, but the Hume call still happened

    with patch("hume.HumeClient", return_value=mock_client), \
         patch("config.HUME_API_KEY", "fake-key"):
        analyze(fake_audio)

    call = batch.start_inference_job_from_local_file.call_args
    assert call.kwargs["file"] == [fake_audio]
    assert isinstance(call.kwargs["json"], InferenceBaseRequest)
    # Old signature must not be present
    assert "filepath" not in call.kwargs
    assert "models" not in call.kwargs


def test_analyze_missing_api_key(fake_audio):
    from src.services.emotion import analyze

    with patch("config.HUME_API_KEY", ""):
        result = analyze(fake_audio)

    assert result is None


def test_analyze_missing_audio_file():
    from src.services.emotion import analyze

    with patch("config.HUME_API_KEY", "fake-key"):
        result = analyze("/nonexistent/file.wav")

    assert result is None


def test_analyze_failed_state(fake_audio):
    """Hume reports FAILED → return None, do not fetch predictions."""
    from src.services.emotion import analyze

    mock_client = MagicMock()
    batch = mock_client.expression_measurement.batch
    batch.start_inference_job_from_local_file.return_value = "job_abc"
    batch.get_job_details.return_value = _details("FAILED")

    with patch("hume.HumeClient", return_value=mock_client), \
         patch("config.HUME_API_KEY", "fake-key"), \
         patch("src.services.emotion.time.sleep"):
        result = analyze(fake_audio)

    assert result is None
    batch.get_job_predictions.assert_not_called()


def test_analyze_timeout(fake_audio):
    """Job stuck IN_PROGRESS — polling exits on timeout, returns None."""
    from src.services.emotion import analyze

    mock_client = MagicMock()
    batch = mock_client.expression_measurement.batch
    batch.start_inference_job_from_local_file.return_value = "job_abc"
    batch.get_job_details.return_value = _details("IN_PROGRESS")

    with patch("hume.HumeClient", return_value=mock_client), \
         patch("config.HUME_API_KEY", "fake-key"), \
         patch("src.services.emotion.time.sleep"), \
         patch("src.services.emotion._JOB_TIMEOUT_SECONDS", 6), \
         patch("src.services.emotion._POLL_INTERVAL_SECONDS", 3):
        result = analyze(fake_audio)

    assert result is None


def test_analyze_hume_raises(fake_audio):
    """Optional service: if Hume client raises, swallow and return None."""
    from src.services.emotion import analyze

    with patch("hume.HumeClient", side_effect=Exception("network error")), \
         patch("config.HUME_API_KEY", "fake-key"):
        result = analyze(fake_audio)

    assert result is None  # must NOT propagate — emotion is non-fatal


def test_analyze_empty_predictions(fake_audio):
    """Hume returns empty predictions — None."""
    from src.services.emotion import analyze

    mock_client = MagicMock()
    batch = mock_client.expression_measurement.batch
    batch.start_inference_job_from_local_file.return_value = "job_abc"
    batch.get_job_details.return_value = _details("COMPLETED")
    batch.get_job_predictions.return_value = []

    with patch("hume.HumeClient", return_value=mock_client), \
         patch("config.HUME_API_KEY", "fake-key"):
        result = analyze(fake_audio)

    assert result is None
