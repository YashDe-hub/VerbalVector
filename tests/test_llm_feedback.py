"""Tests for generate_feedback (Gemini audio LLM feedback path)."""
import pytest
from unittest.mock import patch, MagicMock


def _make_audio_file(state):
    f = MagicMock()
    f.state = state
    f.name = "files/test123"
    return f


def _make_response(text="## Overall Assessment\n\nGreat job."):
    r = MagicMock()
    r.text = text
    return r


def _make_client(upload_file, get_files=None, response=None):
    client = MagicMock()
    client.files.upload.return_value = upload_file
    if get_files is not None:
        client.files.get.side_effect = get_files
    client.models.generate_content.return_value = response or _make_response()
    return client


@pytest.fixture
def fake_audio(tmp_path):
    p = tmp_path / "recording.wav"
    p.write_bytes(b"RIFF....fake")
    return str(p)


@pytest.fixture
def features():
    return {
        "words_per_minute": 145.0,
        "pitch_std": 32.5,
        "volume_std": 0.020,
        "num_pauses": 12,
        "filler_word_rate": 3.5,
    }


def test_generate_feedback_active_first_poll(fake_audio, features):
    """ACTIVE on upload → no polling, generate_content called."""
    from google.genai import types
    from src.services.llm import generate_feedback

    audio_file = _make_audio_file(types.FileState.ACTIVE)
    client = _make_client(upload_file=audio_file)

    with patch("google.genai.Client", return_value=client), \
         patch("config.GEMINI_API_KEY", "fake-key"), \
         patch("config.GEMINI_LLM_MODEL", "gemini-2.5-flash"):
        result = generate_feedback(fake_audio, "Hello world", features)

    assert result is not None
    client.files.get.assert_not_called()
    client.models.generate_content.assert_called_once()


def test_generate_feedback_eventual_active(fake_audio, features):
    """PROCESSING → PROCESSING → ACTIVE — polls twice then proceeds."""
    from google.genai import types
    from src.services.llm import generate_feedback

    initial = _make_audio_file(types.FileState.PROCESSING)
    poll1 = _make_audio_file(types.FileState.PROCESSING)
    poll2 = _make_audio_file(types.FileState.ACTIVE)
    client = _make_client(upload_file=initial, get_files=[poll1, poll2])

    with patch("google.genai.Client", return_value=client), \
         patch("config.GEMINI_API_KEY", "fake-key"), \
         patch("config.GEMINI_LLM_MODEL", "gemini-2.5-flash"), \
         patch("src.services.llm.time.sleep"):
        result = generate_feedback(fake_audio, "Hello", features)

    assert result is not None
    assert client.files.get.call_count == 2
    client.models.generate_content.assert_called_once()


def test_generate_feedback_failed_state(fake_audio, features):
    """Terminal FAILED state → None, generate_content never called."""
    from google.genai import types
    from src.services.llm import generate_feedback

    initial = _make_audio_file(types.FileState.PROCESSING)
    failed = _make_audio_file(types.FileState.FAILED)
    client = _make_client(upload_file=initial, get_files=[failed])

    with patch("google.genai.Client", return_value=client), \
         patch("config.GEMINI_API_KEY", "fake-key"), \
         patch("config.GEMINI_LLM_MODEL", "gemini-2.5-flash"), \
         patch("src.services.llm.time.sleep"):
        result = generate_feedback(fake_audio, "Hello", features)

    assert result is None
    client.models.generate_content.assert_not_called()


def test_generate_feedback_timeout(fake_audio, features):
    """Stuck in PROCESSING — polling loop is bounded, returns None."""
    from google.genai import types
    from src.services.llm import generate_feedback

    initial = _make_audio_file(types.FileState.PROCESSING)
    stuck = _make_audio_file(types.FileState.PROCESSING)

    client = MagicMock()
    client.files.upload.return_value = initial
    client.files.get.return_value = stuck

    with patch("google.genai.Client", return_value=client), \
         patch("config.GEMINI_API_KEY", "fake-key"), \
         patch("config.GEMINI_LLM_MODEL", "gemini-2.5-flash"), \
         patch("src.services.llm.time.sleep"), \
         patch("src.services.llm._GEMINI_FILE_TIMEOUT_SECONDS", 2), \
         patch("src.services.llm._GEMINI_FILE_POLL_INTERVAL_SECONDS", 1):
        result = generate_feedback(fake_audio, "Hello", features)

    assert result is None
    client.models.generate_content.assert_not_called()


def test_generate_feedback_webm_video_mime_rewritten(fake_audio, features):
    """video/webm (MediaRecorder audio-only) → audio/webm."""
    from google.genai import types
    from src.services.llm import generate_feedback

    audio_file = _make_audio_file(types.FileState.ACTIVE)
    client = _make_client(upload_file=audio_file)

    with patch("google.genai.Client", return_value=client), \
         patch("config.GEMINI_API_KEY", "fake-key"), \
         patch("config.GEMINI_LLM_MODEL", "gemini-2.5-flash"), \
         patch("src.services.llm.mimetypes.guess_type", return_value=("video/webm", None)):
        generate_feedback(fake_audio, "Hello", features)

    upload_kwargs = client.files.upload.call_args.kwargs
    assert upload_kwargs["config"].mime_type == "audio/webm"


def test_generate_feedback_ogg_video_mime_rewritten(fake_audio, features):
    """video/ogg (MediaRecorder audio-only) → audio/ogg."""
    from google.genai import types
    from src.services.llm import generate_feedback

    audio_file = _make_audio_file(types.FileState.ACTIVE)
    client = _make_client(upload_file=audio_file)

    with patch("google.genai.Client", return_value=client), \
         patch("config.GEMINI_API_KEY", "fake-key"), \
         patch("config.GEMINI_LLM_MODEL", "gemini-2.5-flash"), \
         patch("src.services.llm.mimetypes.guess_type", return_value=("video/ogg", None)):
        generate_feedback(fake_audio, "Hello", features)

    upload_kwargs = client.files.upload.call_args.kwargs
    assert upload_kwargs["config"].mime_type == "audio/ogg"


def test_generate_feedback_real_video_mp4_NOT_rewritten(fake_audio, features):
    """video/mp4 is a real video container — must NOT be rewritten to audio/mp4."""
    from google.genai import types
    from src.services.llm import generate_feedback

    audio_file = _make_audio_file(types.FileState.ACTIVE)
    client = _make_client(upload_file=audio_file)

    with patch("google.genai.Client", return_value=client), \
         patch("config.GEMINI_API_KEY", "fake-key"), \
         patch("config.GEMINI_LLM_MODEL", "gemini-2.5-flash"), \
         patch("src.services.llm.mimetypes.guess_type", return_value=("video/mp4", None)):
        generate_feedback(fake_audio, "Hello", features)

    upload_kwargs = client.files.upload.call_args.kwargs
    assert upload_kwargs["config"].mime_type == "video/mp4"


def test_generate_feedback_audio_mime_passthrough(fake_audio, features):
    """audio/wav passes through unchanged."""
    from google.genai import types
    from src.services.llm import generate_feedback

    audio_file = _make_audio_file(types.FileState.ACTIVE)
    client = _make_client(upload_file=audio_file)

    with patch("google.genai.Client", return_value=client), \
         patch("config.GEMINI_API_KEY", "fake-key"), \
         patch("config.GEMINI_LLM_MODEL", "gemini-2.5-flash"), \
         patch("src.services.llm.mimetypes.guess_type", return_value=("audio/wav", None)):
        generate_feedback(fake_audio, "Hello", features)

    upload_kwargs = client.files.upload.call_args.kwargs
    assert upload_kwargs["config"].mime_type == "audio/wav"


def test_generate_feedback_unknown_mime(fake_audio, features):
    """Unknown extension → guess_type returns (None, None) → mime_type stays None."""
    from google.genai import types
    from src.services.llm import generate_feedback

    audio_file = _make_audio_file(types.FileState.ACTIVE)
    client = _make_client(upload_file=audio_file)

    with patch("google.genai.Client", return_value=client), \
         patch("config.GEMINI_API_KEY", "fake-key"), \
         patch("config.GEMINI_LLM_MODEL", "gemini-2.5-flash"), \
         patch("src.services.llm.mimetypes.guess_type", return_value=(None, None)):
        generate_feedback(fake_audio, "Hello", features)

    upload_kwargs = client.files.upload.call_args.kwargs
    assert upload_kwargs["config"].mime_type is None


def test_generate_feedback_missing_api_key(fake_audio, features):
    from src.services.llm import generate_feedback

    with patch("config.GEMINI_API_KEY", ""):
        result = generate_feedback(fake_audio, "Hello", features)

    assert result is None


def test_generate_feedback_missing_audio_file(features):
    from src.services.llm import generate_feedback

    with patch("config.GEMINI_API_KEY", "fake-key"):
        result = generate_feedback("/nonexistent/file.wav", "Hello", features)

    assert result is None


def test_generate_feedback_gemini_raises(fake_audio, features):
    from src.services.llm import generate_feedback

    with patch("google.genai.Client", side_effect=Exception("API error")), \
         patch("config.GEMINI_API_KEY", "fake-key"):
        result = generate_feedback(fake_audio, "Hello", features)

    assert result is None


def test_generate_feedback_empty_response(fake_audio, features):
    from google.genai import types
    from src.services.llm import generate_feedback

    audio_file = _make_audio_file(types.FileState.ACTIVE)
    client = _make_client(upload_file=audio_file, response=_make_response(text=None))

    with patch("google.genai.Client", return_value=client), \
         patch("config.GEMINI_API_KEY", "fake-key"), \
         patch("config.GEMINI_LLM_MODEL", "gemini-2.5-flash"):
        result = generate_feedback(fake_audio, "Hello", features)

    assert result is None


def test_generate_feedback_deletes_file_when_generate_raises(fake_audio, features):
    """try/finally guarantee: file cleanup runs even when generate_content raises."""
    from google.genai import types
    from src.services.llm import generate_feedback

    audio_file = _make_audio_file(types.FileState.ACTIVE)
    client = MagicMock()
    client.files.upload.return_value = audio_file
    client.models.generate_content.side_effect = Exception("Gemini outage")

    with patch("google.genai.Client", return_value=client), \
         patch("config.GEMINI_API_KEY", "fake-key"), \
         patch("config.GEMINI_LLM_MODEL", "gemini-2.5-flash"):
        result = generate_feedback(fake_audio, "Hello", features)

    assert result is None
    client.files.delete.assert_called_once_with(name=audio_file.name)


def test_generate_feedback_deletes_file_on_failed_state(fake_audio, features):
    """try/finally guarantee: cleanup runs when polling lands in non-ACTIVE state."""
    from google.genai import types
    from src.services.llm import generate_feedback

    initial = _make_audio_file(types.FileState.PROCESSING)
    failed = _make_audio_file(types.FileState.FAILED)
    client = _make_client(upload_file=initial, get_files=[failed])

    with patch("google.genai.Client", return_value=client), \
         patch("config.GEMINI_API_KEY", "fake-key"), \
         patch("config.GEMINI_LLM_MODEL", "gemini-2.5-flash"), \
         patch("src.services.llm.time.sleep"):
        result = generate_feedback(fake_audio, "Hello", features)

    assert result is None
    client.files.delete.assert_called_once_with(name=failed.name)


def test_generate_feedback_includes_speaker_section_when_multi_speaker(fake_audio, features):
    """When utterances has multiple speakers, the Gemini prompt must include a speaker breakdown."""
    from google.genai import types
    from src.services.llm import generate_feedback

    captured_prompt = {}
    audio_file = _make_audio_file(types.FileState.ACTIVE)

    def capture_generate(**kwargs):
        captured_prompt["text"] = kwargs["contents"][1]
        return _make_response()

    client = MagicMock()
    client.files.upload.return_value = audio_file
    client.models.generate_content.side_effect = capture_generate

    utterances = [
        {"speaker": 0, "text": "Hello.", "start": 0.0, "end": 0.5, "confidence": 0.98},
        {"speaker": 1, "text": "Hi.", "start": 0.6, "end": 0.8, "confidence": 0.97},
    ]

    with patch("google.genai.Client", return_value=client), \
         patch("config.GEMINI_API_KEY", "fake-key"), \
         patch("config.GEMINI_LLM_MODEL", "gemini-2.5-flash"):
        result = generate_feedback(fake_audio, "Hello. Hi.", features, utterances=utterances)

    assert result is not None
    prompt = captured_prompt["text"]
    assert "Speaker 0" in prompt or "speaker 0" in prompt.lower()
    assert "Speaker 1" in prompt or "speaker 1" in prompt.lower()
    assert "Hello." in prompt
    assert "Hi." in prompt


def test_generate_feedback_omits_speaker_section_when_single_speaker(fake_audio, features):
    """With only one speaker (or no utterances), the prompt should not have a multi-speaker section."""
    from google.genai import types
    from src.services.llm import generate_feedback

    captured_prompt = {}
    audio_file = _make_audio_file(types.FileState.ACTIVE)

    def capture_generate(**kwargs):
        captured_prompt["text"] = kwargs["contents"][1]
        return _make_response()

    client = MagicMock()
    client.files.upload.return_value = audio_file
    client.models.generate_content.side_effect = capture_generate

    utterances = [
        {"speaker": 0, "text": "Hello.", "start": 0.0, "end": 0.5, "confidence": 0.98},
        {"speaker": 0, "text": "How are you?", "start": 0.6, "end": 1.2, "confidence": 0.97},
    ]

    with patch("google.genai.Client", return_value=client), \
         patch("config.GEMINI_API_KEY", "fake-key"), \
         patch("config.GEMINI_LLM_MODEL", "gemini-2.5-flash"):
        generate_feedback(fake_audio, "Hello. How are you?", features, utterances=utterances)

    prompt = captured_prompt["text"]
    assert "Conversation Mode" not in prompt
    assert "Multi-Speaker" not in prompt


def test_generate_feedback_accepts_none_utterances(fake_audio, features):
    """Backward-compat: callers that don't pass utterances still work."""
    from google.genai import types
    from src.services.llm import generate_feedback

    audio_file = _make_audio_file(types.FileState.ACTIVE)
    client = _make_client(upload_file=audio_file)

    with patch("google.genai.Client", return_value=client), \
         patch("config.GEMINI_API_KEY", "fake-key"), \
         patch("config.GEMINI_LLM_MODEL", "gemini-2.5-flash"):
        result = generate_feedback(fake_audio, "Hello.", features)

    assert result is not None


def test_build_prompt_wearer_mode_coaches_only_user_speaker():
    from src.services.llm import _build_prompt
    utterances = [
        {"speaker": 0, "text": "How was the demo?", "start": 0.0, "end": 1.5, "confidence": 0.9},
        {"speaker": 1, "text": "It went well, we shipped it.", "start": 1.6, "end": 3.9, "confidence": 0.9},
    ]
    prompt = _build_prompt(
        transcript="How was the demo? It went well, we shipped it.",
        features={"words_per_minute": 140.0},
        emotion_scores=None,
        utterances=utterances,
        user_speaker=1,
    )
    # Coach-only instruction present and targeted
    assert "Speaker 1" in prompt
    assert "ONLY Speaker 1" in prompt or "only Speaker 1" in prompt
    # Full conversation kept as context (other speaker's words included)
    assert "How was the demo?" in prompt
    # Features clarified as user-only
    assert "only from Speaker 1" in prompt
    # Generic multi-speaker coaching instruction must NOT appear in wearer mode
    assert "address each speaker by their ID" not in prompt


def test_build_prompt_without_user_speaker_keeps_generic_behavior():
    from src.services.llm import _build_prompt
    utterances = [
        {"speaker": 0, "text": "A.", "start": 0.0, "end": 1.0, "confidence": 0.9},
        {"speaker": 1, "text": "B.", "start": 1.1, "end": 2.0, "confidence": 0.9},
    ]
    prompt = _build_prompt("A. B.", {"words_per_minute": 120.0}, None, utterances)
    assert "address each speaker by their ID" in prompt
