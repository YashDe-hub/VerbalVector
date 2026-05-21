"""Tests for Deepgram STT service.

STT is pipeline-fatal per CLAUDE.md — every failure path must return None
so the API endpoint can 500 the request.
"""
import pytest
from unittest.mock import patch, MagicMock


def _make_word(word, start, end, confidence, punctuated_word=None):
    w = MagicMock()
    w.word = word
    w.start = start
    w.end = end
    w.confidence = confidence
    w.speaker = None  # default: not diarized; tests that need a speaker set it explicitly
    if punctuated_word is not None:
        w.punctuated_word = punctuated_word
    else:
        # Mirror the SDK shape — punctuated_word is normally present
        w.punctuated_word = word
    return w


def _make_response(transcript, words=None, detected_language="en"):
    """Build a mock response matching the real Deepgram shape."""
    alternative = MagicMock()
    alternative.transcript = transcript
    alternative.words = words

    channel = MagicMock()
    channel.alternatives = [alternative]
    channel.detected_language = detected_language

    response = MagicMock()
    response.results.channels = [channel]
    return response


def _make_client(response=None, raises=None):
    client = MagicMock()
    transcribe_file = client.listen.prerecorded.v.return_value.transcribe_file
    if raises is not None:
        transcribe_file.side_effect = raises
    else:
        transcribe_file.return_value = response
    return client


@pytest.fixture
def fake_audio(tmp_path):
    p = tmp_path / "recording.wav"
    p.write_bytes(b"RIFF....fake")
    return str(p)


def test_transcribe_happy_path(fake_audio):
    """Successful transcription returns text, language, and word-level segments."""
    from src.services.stt import transcribe

    words = [
        _make_word("hello", 0.0, 0.5, 0.99, punctuated_word="Hello"),
        _make_word("world", 0.5, 1.0, 0.97, punctuated_word="world."),
    ]
    response = _make_response(transcript="Hello world.", words=words)
    client = _make_client(response=response)

    with patch("deepgram.DeepgramClient", return_value=client), \
         patch("config.DEEPGRAM_API_KEY", "fake-key"), \
         patch("config.DEEPGRAM_STT_MODEL", "nova-3"):
        result = transcribe(fake_audio)

    assert result is not None
    assert result["text"] == "Hello world."
    assert result["language"] == "en"
    assert len(result["segments"]) == 2
    assert result["segments"][0] == {
        "word": "Hello",
        "start": 0.0,
        "end": 0.5,
        "confidence": 0.99,
        "speaker": None,
    }
    assert result["segments"][1]["word"] == "world."


def test_transcribe_uses_punctuated_word_when_present(fake_audio):
    """Segments should prefer punctuated_word over raw word when SDK provides it."""
    from src.services.stt import transcribe

    word = _make_word("test", 0.0, 0.3, 0.95, punctuated_word="Test,")
    response = _make_response(transcript="Test", words=[word])
    client = _make_client(response=response)

    with patch("deepgram.DeepgramClient", return_value=client), \
         patch("config.DEEPGRAM_API_KEY", "fake-key"):
        result = transcribe(fake_audio)

    assert result["segments"][0]["word"] == "Test,"


def test_transcribe_handles_missing_punctuated_word(fake_audio):
    """If punctuated_word attribute is absent, fall back to raw word."""
    from src.services.stt import transcribe

    word = MagicMock(spec=["word", "start", "end", "confidence"])
    word.word = "raw"
    word.start = 0.0
    word.end = 0.2
    word.confidence = 0.9
    response = _make_response(transcript="raw", words=[word])
    client = _make_client(response=response)

    with patch("deepgram.DeepgramClient", return_value=client), \
         patch("config.DEEPGRAM_API_KEY", "fake-key"):
        result = transcribe(fake_audio)

    assert result["segments"][0]["word"] == "raw"


def test_transcribe_passes_configured_model_to_deepgram(fake_audio):
    """The DEEPGRAM_STT_MODEL config value must reach PrerecordedOptions."""
    from src.services.stt import transcribe

    response = _make_response(transcript="ok", words=[])
    client = _make_client(response=response)

    captured = {}

    def capture_options(buffer_payload, options):
        captured["model"] = options.model
        return response

    client.listen.prerecorded.v.return_value.transcribe_file.side_effect = capture_options

    with patch("deepgram.DeepgramClient", return_value=client), \
         patch("config.DEEPGRAM_API_KEY", "fake-key"), \
         patch("config.DEEPGRAM_STT_MODEL", "nova-2"):
        transcribe(fake_audio)

    assert captured["model"] == "nova-2"


def test_transcribe_falls_back_to_en_when_no_detected_language(fake_audio):
    """If channel.detected_language is missing or None, default to 'en'."""
    from src.services.stt import transcribe

    response = _make_response(transcript="hi", words=[], detected_language=None)
    client = _make_client(response=response)

    with patch("deepgram.DeepgramClient", return_value=client), \
         patch("config.DEEPGRAM_API_KEY", "fake-key"):
        result = transcribe(fake_audio)

    assert result["language"] == "en"


def test_transcribe_missing_api_key(fake_audio):
    """Missing key returns None without instantiating the client."""
    from src.services.stt import transcribe

    with patch("config.DEEPGRAM_API_KEY", ""), \
         patch("deepgram.DeepgramClient") as mock_client:
        result = transcribe(fake_audio)

    assert result is None
    mock_client.assert_not_called()


def test_transcribe_missing_audio_file():
    """Missing file returns None without calling Deepgram."""
    from src.services.stt import transcribe

    with patch("config.DEEPGRAM_API_KEY", "fake-key"), \
         patch("deepgram.DeepgramClient") as mock_client:
        result = transcribe("/nonexistent/file.wav")

    assert result is None
    mock_client.assert_not_called()


def test_transcribe_empty_transcript(fake_audio):
    """Deepgram returns no spoken content → None (pipeline-fatal: nothing to feed downstream)."""
    from src.services.stt import transcribe

    response = _make_response(transcript="", words=[])
    client = _make_client(response=response)

    with patch("deepgram.DeepgramClient", return_value=client), \
         patch("config.DEEPGRAM_API_KEY", "fake-key"):
        result = transcribe(fake_audio)

    assert result is None


def test_transcribe_deepgram_raises(fake_audio):
    """Deepgram exception caught, returns None — pipeline-fatal contract."""
    from src.services.stt import transcribe

    client = _make_client(raises=Exception("network error"))

    with patch("deepgram.DeepgramClient", return_value=client), \
         patch("config.DEEPGRAM_API_KEY", "fake-key"):
        result = transcribe(fake_audio)

    assert result is None


def test_transcribe_handles_words_none(fake_audio):
    """Some Deepgram responses have words=None — segments should be []."""
    from src.services.stt import transcribe

    response = _make_response(transcript="hello", words=None)
    client = _make_client(response=response)

    with patch("deepgram.DeepgramClient", return_value=client), \
         patch("config.DEEPGRAM_API_KEY", "fake-key"):
        result = transcribe(fake_audio)

    assert result is not None
    assert result["text"] == "hello"
    assert result["segments"] == []


def test_transcribe_extracts_utterances_with_speakers(fake_audio):
    """Diarization output: response.results.utterances → top-level utterances list with speaker IDs."""
    from src.services.stt import transcribe

    words = [
        _make_word("hello", 0.0, 0.5, 0.99, punctuated_word="Hello"),
        _make_word("there", 0.5, 1.0, 0.97, punctuated_word="there."),
    ]
    response = _make_response(transcript="Hello there. How are you?", words=words)
    utt1 = MagicMock()
    utt1.speaker = 0
    utt1.transcript = "Hello there."
    utt1.start = 0.0
    utt1.end = 1.0
    utt1.confidence = 0.98
    utt2 = MagicMock()
    utt2.speaker = 1
    utt2.transcript = "How are you?"
    utt2.start = 1.1
    utt2.end = 2.0
    utt2.confidence = 0.97
    response.results.utterances = [utt1, utt2]
    client = _make_client(response=response)

    with patch("deepgram.DeepgramClient", return_value=client), \
         patch("config.DEEPGRAM_API_KEY", "fake-key"):
        result = transcribe(fake_audio)

    assert result is not None
    assert "utterances" in result
    assert result["utterances"] == [
        {"speaker": 0, "text": "Hello there.", "start": 0.0, "end": 1.0, "confidence": 0.98},
        {"speaker": 1, "text": "How are you?", "start": 1.1, "end": 2.0, "confidence": 0.97},
    ]
    assert result["speakers"] == [0, 1]


def test_transcribe_handles_response_without_utterances(fake_audio):
    """Deepgram occasionally returns no utterances list (very short clip). Result should still be valid."""
    from src.services.stt import transcribe

    words = [_make_word("hi", 0.0, 0.2, 0.95, punctuated_word="Hi.")]
    response = _make_response(transcript="Hi.", words=words)
    response.results.utterances = []  # empty list — same as missing
    client = _make_client(response=response)

    with patch("deepgram.DeepgramClient", return_value=client), \
         patch("config.DEEPGRAM_API_KEY", "fake-key"):
        result = transcribe(fake_audio)

    assert result is not None
    assert result["utterances"] == []
    assert result["speakers"] == []


def test_transcribe_segments_include_speaker_when_diarized(fake_audio):
    """Word-level segments should include speaker for each word when diarization is on."""
    from src.services.stt import transcribe

    word_a = _make_word("hello", 0.0, 0.5, 0.99, punctuated_word="Hello")
    word_a.speaker = 0
    word_b = _make_word("hi", 0.6, 0.8, 0.97, punctuated_word="Hi")
    word_b.speaker = 1
    response = _make_response(transcript="Hello Hi", words=[word_a, word_b])
    response.results.utterances = []
    client = _make_client(response=response)

    with patch("deepgram.DeepgramClient", return_value=client), \
         patch("config.DEEPGRAM_API_KEY", "fake-key"):
        result = transcribe(fake_audio)

    assert result["segments"][0]["speaker"] == 0
    assert result["segments"][1]["speaker"] == 1


def test_transcribe_passes_diarize_true_to_deepgram(fake_audio):
    """Options must have diarize=True so Deepgram does the work."""
    from src.services.stt import transcribe

    response = _make_response(transcript="ok", words=[])
    response.results.utterances = []
    client = _make_client(response=response)

    captured = {}
    def capture_options(buffer_payload, options):
        captured["diarize"] = getattr(options, "diarize", None)
        return response

    client.listen.prerecorded.v.return_value.transcribe_file.side_effect = capture_options

    with patch("deepgram.DeepgramClient", return_value=client), \
         patch("config.DEEPGRAM_API_KEY", "fake-key"):
        transcribe(fake_audio)

    assert captured["diarize"] is True
