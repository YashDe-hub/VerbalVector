"""Tests for StreamingTranscriber (Deepgram live wrapper).

NOTE: The spec used client.listen.asynclive.v("1"), but SDK v3.11.0 marks
asynclive as deprecated since v3.4.0. The canonical async path is
client.listen.asyncwebsocket.v("1"). All patches here reflect that choice.
"""
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.mark.asyncio
async def test_start_opens_deepgram_connection_with_correct_options():
    """start() must open a live Deepgram connection with model + 16kHz linear16 mono options."""
    from src.services.streaming_stt import StreamingTranscriber

    received = []
    async def on_transcript(text, is_final):
        received.append((text, is_final))

    mock_connection = MagicMock()
    mock_connection.start = AsyncMock(return_value=True)
    mock_connection.send = AsyncMock()
    mock_connection.finish = AsyncMock()
    mock_connection.on = MagicMock()

    mock_client = MagicMock()
    mock_client.listen.asyncwebsocket.v.return_value = mock_connection

    with patch("deepgram.DeepgramClient", return_value=mock_client), \
         patch("config.DEEPGRAM_API_KEY", "fake-key"), \
         patch("config.DEEPGRAM_STT_MODEL", "nova-3"), \
         patch("config.DEEPGRAM_STT_SAMPLE_RATE", 16000):
        transcriber = StreamingTranscriber(on_transcript=on_transcript)
        await transcriber.start()

    mock_client.listen.asyncwebsocket.v.assert_called_once_with("1")
    mock_connection.start.assert_called_once()
    options = mock_connection.start.call_args.args[0]
    assert options.model == "nova-3"
    assert options.encoding == "linear16"
    assert options.sample_rate == 16000
    assert options.channels == 1
    assert options.interim_results is True


@pytest.mark.asyncio
async def test_send_audio_forwards_chunks_to_deepgram():
    from src.services.streaming_stt import StreamingTranscriber

    async def on_transcript(text, is_final):
        pass

    mock_connection = MagicMock()
    mock_connection.start = AsyncMock(return_value=True)
    mock_connection.send = AsyncMock()
    mock_connection.finish = AsyncMock()
    mock_connection.on = MagicMock()

    mock_client = MagicMock()
    mock_client.listen.asyncwebsocket.v.return_value = mock_connection

    with patch("deepgram.DeepgramClient", return_value=mock_client), \
         patch("config.DEEPGRAM_API_KEY", "fake-key"):
        transcriber = StreamingTranscriber(on_transcript=on_transcript)
        await transcriber.start()
        await transcriber.send_audio(b"\x01\x02\x03\x04")
        await transcriber.send_audio(b"\x05\x06\x07\x08")

    assert mock_connection.send.await_count == 2
    mock_connection.send.assert_any_await(b"\x01\x02\x03\x04")
    mock_connection.send.assert_any_await(b"\x05\x06\x07\x08")


@pytest.mark.asyncio
async def test_finish_closes_deepgram():
    from src.services.streaming_stt import StreamingTranscriber

    async def on_transcript(text, is_final):
        pass

    mock_connection = MagicMock()
    mock_connection.start = AsyncMock(return_value=True)
    mock_connection.send = AsyncMock()
    mock_connection.finish = AsyncMock()
    mock_connection.on = MagicMock()

    mock_client = MagicMock()
    mock_client.listen.asyncwebsocket.v.return_value = mock_connection

    with patch("deepgram.DeepgramClient", return_value=mock_client), \
         patch("config.DEEPGRAM_API_KEY", "fake-key"):
        transcriber = StreamingTranscriber(on_transcript=on_transcript)
        await transcriber.start()
        await transcriber.finish()

    mock_connection.finish.assert_awaited_once()


@pytest.mark.asyncio
async def test_transcript_events_invoke_callback():
    """When Deepgram emits a transcript event, the registered callback should be invoked with (text, is_final)."""
    from src.services.streaming_stt import StreamingTranscriber
    from deepgram import LiveTranscriptionEvents

    received = []
    async def on_transcript(text, is_final):
        received.append((text, is_final))

    handlers = {}
    mock_connection = MagicMock()
    mock_connection.start = AsyncMock(return_value=True)
    mock_connection.send = AsyncMock()
    mock_connection.finish = AsyncMock()
    def capture_on(event, handler):
        handlers[event] = handler
    mock_connection.on = MagicMock(side_effect=capture_on)

    mock_client = MagicMock()
    mock_client.listen.asyncwebsocket.v.return_value = mock_connection

    with patch("deepgram.DeepgramClient", return_value=mock_client), \
         patch("config.DEEPGRAM_API_KEY", "fake-key"):
        transcriber = StreamingTranscriber(on_transcript=on_transcript)
        await transcriber.start()

    handler = handlers[LiveTranscriptionEvents.Transcript]
    result = MagicMock()
    result.channel.alternatives = [MagicMock(transcript="hello world")]
    result.is_final = True
    await handler(mock_connection, result)

    assert received == [("hello world", True)]


@pytest.mark.asyncio
async def test_missing_api_key_raises():
    from src.services.streaming_stt import StreamingTranscriber, StreamingSttError

    async def on_transcript(text, is_final):
        pass

    with patch("config.DEEPGRAM_API_KEY", ""):
        transcriber = StreamingTranscriber(on_transcript=on_transcript)
        with pytest.raises(StreamingSttError):
            await transcriber.start()


@pytest.mark.asyncio
async def test_start_failure_raises_streaming_stt_error():
    """If Deepgram.start() returns False, wrap in StreamingSttError."""
    from src.services.streaming_stt import StreamingTranscriber, StreamingSttError

    async def on_transcript(text, is_final):
        pass

    mock_connection = MagicMock()
    mock_connection.start = AsyncMock(return_value=False)
    mock_connection.on = MagicMock()

    mock_client = MagicMock()
    mock_client.listen.asyncwebsocket.v.return_value = mock_connection

    with patch("deepgram.DeepgramClient", return_value=mock_client), \
         patch("config.DEEPGRAM_API_KEY", "fake-key"):
        transcriber = StreamingTranscriber(on_transcript=on_transcript)
        with pytest.raises(StreamingSttError):
            await transcriber.start()


@pytest.mark.asyncio
async def test_double_start_raises():
    """Calling start() twice must raise — second call would otherwise leak the first connection."""
    from src.services.streaming_stt import StreamingTranscriber, StreamingSttError

    async def on_transcript(text, is_final):
        pass

    mock_connection = MagicMock()
    mock_connection.start = AsyncMock(return_value=True)
    mock_connection.on = MagicMock()

    mock_client = MagicMock()
    # Adjust if your SDK path is asynclive instead of asyncwebsocket
    mock_client.listen.asyncwebsocket.v.return_value = mock_connection

    with patch("deepgram.DeepgramClient", return_value=mock_client), \
         patch("config.DEEPGRAM_API_KEY", "fake-key"):
        transcriber = StreamingTranscriber(on_transcript=on_transcript)
        await transcriber.start()
        with pytest.raises(StreamingSttError):
            await transcriber.start()
