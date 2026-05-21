"""Tests for the /api/stream WebSocket endpoint."""
import json
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from api import app
    return TestClient(app)


def _patches(*, transcriber=None, assembler=None, pipeline=None):
    """Standard mock stack for stream tests."""
    p_transcriber = patch(
        "api.StreamingTranscriber",
        return_value=transcriber or _default_transcriber(),
    )
    p_assembler = patch(
        "api.AudioAssembler",
        return_value=assembler or _default_assembler(),
    )
    p_pipeline = patch(
        "api.run_analysis_pipeline",
        return_value=pipeline or _default_pipeline_result(),
    )
    return p_transcriber, p_assembler, p_pipeline


def _default_transcriber():
    m = MagicMock()
    m.start = AsyncMock()
    m.send_audio = AsyncMock()
    m.finish = AsyncMock()
    m.last_error = None
    return m


def _default_assembler():
    m = MagicMock()
    m.write_chunk = MagicMock()
    m.close = MagicMock(return_value=MagicMock())
    return m


def _default_pipeline_result():
    return {
        "transcript_path": "/fake/transcript.json",
        "features_path": "/fake/features.json",
        "feedback_path": "/fake/feedback.txt",
    }


def test_session_started_message_sent_on_connect(client):
    """On connect, server must send session_started, session_id."""
    p_t, p_a, p_p = _patches()
    with p_t, p_a, p_p, \
         patch("api.read_file", return_value={"text": "hi"}):
        with client.websocket_connect("/api/stream") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "session_started"
            assert "session_id" in msg
            assert len(msg["session_id"]) > 0
            ws.send_text(json.dumps({"type": "init"}))
            ws.send_text(json.dumps({"type": "end"}))
            while True:
                m = ws.receive_json()
                if m["type"] in ("session_end", "error"):
                    break


def test_init_message_sets_session_label(client):
    """Init's session_label must reach the pipeline."""
    p_t, p_a, p_p = _patches()
    captured_call = {}
    def fake_pipeline(**kwargs):
        captured_call.update(kwargs)
        return _default_pipeline_result()
    with p_t, p_a, patch("api.run_analysis_pipeline", side_effect=fake_pipeline), \
         patch("api.read_file", return_value="content"):
        with client.websocket_connect("/api/stream") as ws:
            ws.receive_json()  # session_started
            ws.send_text(json.dumps({"type": "init", "session_label": "My Live Talk"}))
            ws.send_text(json.dumps({"type": "end"}))
            while True:
                msg = ws.receive_json()
                if msg["type"] in ("session_end", "error"):
                    break
    assert captured_call.get("session_label") == "My Live Talk"


def test_binary_chunks_forwarded_to_transcriber_and_assembler(client):
    """Binary frames must hit BOTH send_audio AND write_chunk."""
    transcriber = _default_transcriber()
    assembler = _default_assembler()
    with patch("api.StreamingTranscriber", return_value=transcriber), \
         patch("api.AudioAssembler", return_value=assembler), \
         patch("api.run_analysis_pipeline", return_value=_default_pipeline_result()), \
         patch("api.read_file", return_value="x"):
        with client.websocket_connect("/api/stream") as ws:
            ws.receive_json()
            ws.send_text(json.dumps({"type": "init"}))
            ws.send_bytes(b"\x01\x02\x03\x04")
            ws.send_bytes(b"\x05\x06")
            ws.send_text(json.dumps({"type": "end"}))
            while True:
                msg = ws.receive_json()
                if msg["type"] in ("session_end", "error"):
                    break
    assert transcriber.send_audio.await_count == 2
    assert assembler.write_chunk.call_count == 2


def test_session_end_includes_analysis_result(client):
    """At session end, the analysis pipeline result must be sent in session_end."""
    p_t, p_a, p_p = _patches()
    with p_t, p_a, p_p, \
         patch("api.read_file") as mock_read:
        mock_read.side_effect = [
            {"text": "transcript text"},
            {"feature_a": 1},
            "feedback markdown",
        ]
        with client.websocket_connect("/api/stream") as ws:
            ws.receive_json()
            ws.send_text(json.dumps({"type": "init"}))
            ws.send_text(json.dumps({"type": "end"}))
            final = ws.receive_json()
    assert final["type"] == "session_end"
    assert final["transcript"] == {"text": "transcript text"}
    assert final["features"] == {"feature_a": 1}
    assert final["feedback"] == "feedback markdown"


def test_analysis_pipeline_failure_sends_error(client):
    """If pipeline returns None, server sends fatal error and closes."""
    p_t, p_a, _ = _patches()
    with p_t, p_a, patch("api.run_analysis_pipeline", return_value=None):
        with client.websocket_connect("/api/stream") as ws:
            ws.receive_json()
            ws.send_text(json.dumps({"type": "init"}))
            ws.send_text(json.dumps({"type": "end"}))
            msg = ws.receive_json()
    assert msg["type"] == "error"
    assert msg.get("fatal") is True


def test_transcriber_start_failure_sends_error(client):
    """If Deepgram fails to start, server sends fatal error."""
    from src.services.streaming_stt import StreamingSttError
    failing_transcriber = _default_transcriber()
    failing_transcriber.start = AsyncMock(side_effect=StreamingSttError("Deepgram offline"))
    with patch("api.StreamingTranscriber", return_value=failing_transcriber), \
         patch("api.AudioAssembler", return_value=_default_assembler()):
        with client.websocket_connect("/api/stream") as ws:
            ws.receive_json()  # session_started
            ws.send_text(json.dumps({"type": "init"}))
            msg = ws.receive_json()
    assert msg["type"] == "error"
    assert msg.get("fatal") is True


def test_transcript_callback_forwards_to_client_via_queue(client):
    """Invoking the on_transcript callback (which puts on the queue) results in a transcript message on the WS.

    Because the WS endpoint decouples Deepgram's callback from the WS send via an
    asyncio.Queue + forwarder task (both running on the WS handler's loop), this
    test schedules the callback onto that loop via run_coroutine_threadsafe.
    """
    import asyncio
    captured = {}

    class _CapturingTranscriber:
        def __init__(self, on_transcript):
            captured["cb"] = on_transcript
            captured["loop"] = asyncio.get_event_loop()
            self.last_error = None
        async def start(self):
            pass
        async def send_audio(self, chunk):
            pass
        async def finish(self):
            pass

    with patch("api.StreamingTranscriber", side_effect=_CapturingTranscriber), \
         patch("api.AudioAssembler", return_value=_default_assembler()), \
         patch("api.run_analysis_pipeline", return_value=_default_pipeline_result()), \
         patch("api.read_file", return_value="x"):
        with client.websocket_connect("/api/stream") as ws:
            ws.receive_json()  # session_started
            ws.send_text(json.dumps({"type": "init"}))

            fut = asyncio.run_coroutine_threadsafe(
                captured["cb"]("hello", False, 0), captured["loop"]
            )
            fut.result(timeout=2)
            fut = asyncio.run_coroutine_threadsafe(
                captured["cb"]("hello world", True, 1), captured["loop"]
            )
            fut.result(timeout=2)

            msg1 = ws.receive_json()
            assert msg1 == {"type": "transcript", "text": "hello", "is_final": False, "speaker": 0}
            msg2 = ws.receive_json()
            assert msg2 == {"type": "transcript", "text": "hello world", "is_final": True, "speaker": 1}

            ws.send_text(json.dumps({"type": "end"}))
            while True:
                m = ws.receive_json()
                if m["type"] in ("session_end", "error"):
                    break


def test_transcript_with_no_speaker_serializes_as_null(client):
    """Interim frames with no words have speaker=None — must serialize as null in JSON."""
    import asyncio
    captured = {}

    class _CapturingTranscriber:
        def __init__(self, on_transcript):
            captured["cb"] = on_transcript
            captured["loop"] = asyncio.get_event_loop()
            self.last_error = None
        async def start(self):
            pass
        async def send_audio(self, chunk):
            pass
        async def finish(self):
            pass

    with patch("api.StreamingTranscriber", side_effect=_CapturingTranscriber), \
         patch("api.AudioAssembler", return_value=_default_assembler()), \
         patch("api.run_analysis_pipeline", return_value=_default_pipeline_result()), \
         patch("api.read_file", return_value="x"):
        with client.websocket_connect("/api/stream") as ws:
            ws.receive_json()  # session_started
            ws.send_text(json.dumps({"type": "init"}))

            fut = asyncio.run_coroutine_threadsafe(
                captured["cb"]("partial text", False, None), captured["loop"]
            )
            fut.result(timeout=2)
            msg = ws.receive_json()
            assert msg == {"type": "transcript", "text": "partial text", "is_final": False, "speaker": None}
            ws.send_text(json.dumps({"type": "end"}))
            while True:
                m = ws.receive_json()
                if m["type"] in ("session_end", "error"):
                    break


def test_init_timeout_closes_connection_with_error(client):
    """If client never sends init, server times out and closes with fatal error."""
    p_t, p_a, p_p = _patches()
    with p_t, p_a, p_p, \
         patch("api.INIT_TIMEOUT_SECONDS", 0.5):  # shrink for fast test
        with client.websocket_connect("/api/stream") as ws:
            ws.receive_json()  # session_started
            msg = ws.receive_json()
    assert msg["type"] == "error"
    assert msg.get("fatal") is True
    assert "timeout" in msg["message"].lower()


def test_duplicate_init_is_non_fatal(client):
    """Sending init twice should produce a non-fatal error on the second one but keep the session alive."""
    p_t, p_a, p_p = _patches()
    with p_t, p_a, p_p, \
         patch("api.read_file", return_value="x"):
        with client.websocket_connect("/api/stream") as ws:
            ws.receive_json()  # session_started
            ws.send_text(json.dumps({"type": "init"}))
            ws.send_text(json.dumps({"type": "init"}))  # duplicate

            saw_dup_error = False
            ws.send_text(json.dumps({"type": "end"}))
            while True:
                m = ws.receive_json()
                if m["type"] == "error" and not m.get("fatal", False):
                    saw_dup_error = True
                if m["type"] in ("session_end",) or (m["type"] == "error" and m.get("fatal")):
                    break
            assert saw_dup_error


def test_deepgram_error_surfaces_as_fatal_error(client):
    """If Deepgram emits an error event during the session, the WS handler must send a fatal error at session end."""

    class _ErrorTranscriber:
        last_error = "Auth failed"
        def __init__(self, on_transcript):
            self.on_transcript = on_transcript
        async def start(self):
            pass
        async def send_audio(self, chunk):
            pass
        async def finish(self):
            pass

    with patch("api.StreamingTranscriber", _ErrorTranscriber), \
         patch("api.AudioAssembler", return_value=_default_assembler()), \
         patch("api.run_analysis_pipeline", return_value=_default_pipeline_result()), \
         patch("api.read_file", return_value="x"):
        with client.websocket_connect("/api/stream") as ws:
            ws.receive_json()  # session_started
            ws.send_text(json.dumps({"type": "init"}))
            ws.send_text(json.dumps({"type": "end"}))
            msg = ws.receive_json()
    assert msg["type"] == "error"
    assert msg.get("fatal") is True
    assert "auth failed" in msg["message"].lower() or "deepgram" in msg["message"].lower()


def test_send_audio_failure_mid_session_sends_fatal_error(client):
    """If transcriber.send_audio raises mid-stream, the WS must send a fatal error and close."""
    failing_transcriber = _default_transcriber()
    failing_transcriber.send_audio = AsyncMock(side_effect=Exception("Deepgram socket died"))
    with patch("api.StreamingTranscriber", return_value=failing_transcriber), \
         patch("api.AudioAssembler", return_value=_default_assembler()), \
         patch("api.run_analysis_pipeline", return_value=_default_pipeline_result()), \
         patch("api.read_file", return_value="x"):
        with client.websocket_connect("/api/stream") as ws:
            ws.receive_json()  # session_started
            ws.send_text(json.dumps({"type": "init"}))
            ws.send_bytes(b"\x01\x02\x03\x04")
            msg = ws.receive_json()
    assert msg["type"] == "error"
    assert msg.get("fatal") is True


def test_streaming_stt_last_error_starts_none():
    """The StreamingTranscriber last_error property starts as None and reflects _on_dg_error calls."""
    from src.services.streaming_stt import StreamingTranscriber

    async def on_transcript(text, is_final):
        pass

    t = StreamingTranscriber(on_transcript=on_transcript)
    assert t.last_error is None
