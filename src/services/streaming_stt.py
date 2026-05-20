"""
Streaming STT service using Deepgram Nova-3 (live API / WebSocket).

Used by the /api/stream WebSocket endpoint for real-time transcription
during live recording sessions. The batch service in stt.py is unchanged
and remains the path for record-then-upload flows.

SDK NOTE (verified against deepgram-sdk v3.11.0):
  client.listen.asyncwebsocket.v("1") is the canonical async path.
  client.listen.asynclive is a deprecated alias (deprecated since v3.4.0,
  removal planned for v4.0.0). If a future SDK upgrade renames asyncwebsocket,
  look for client.listen attrs using:
    [a for a in dir(client.listen) if not a.startswith('_')]
"""

import logging
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

TranscriptCallback = Callable[[str, bool], Awaitable[None]]


class StreamingSttError(Exception):
    """Raised on any unrecoverable streaming STT error (config, connection, etc.)."""


class StreamingTranscriber:
    """
    Manages an async Deepgram live connection.

    Lifecycle: start() -> send_audio() x N -> finish().
    Transcript events invoke the on_transcript callback with (text, is_final).
    """

    def __init__(self, on_transcript: TranscriptCallback) -> None:
        self._on_transcript = on_transcript
        self._connection = None

    async def start(self) -> None:
        try:
            from deepgram import (
                DeepgramClient,
                LiveOptions,
                LiveTranscriptionEvents,
            )
        except ImportError as e:
            raise StreamingSttError("deepgram-sdk is not installed.") from e

        from config import (
            DEEPGRAM_API_KEY,
            DEEPGRAM_STT_MODEL,
            DEEPGRAM_STT_SAMPLE_RATE,
        )

        if not DEEPGRAM_API_KEY:
            raise StreamingSttError("DEEPGRAM_API_KEY is not set.")

        client = DeepgramClient(DEEPGRAM_API_KEY)
        # Canonical async WebSocket path as of deepgram-sdk v3.11.0.
        # asynclive is a deprecated alias — do not use.
        connection = client.listen.asyncwebsocket.v("1")

        async def _on_dg_transcript(connection_arg, result):
            try:
                alternative = result.channel.alternatives[0]
                text = alternative.transcript
                if not text:
                    return  # ignore empty interim frames
                is_final = bool(result.is_final)
                await self._on_transcript(text, is_final)
            except Exception as e:
                logger.error(f"[StreamingSTT] Transcript handler error: {e}", exc_info=True)

        async def _on_dg_error(connection_arg, error):
            logger.error(f"[StreamingSTT] Deepgram error: {error}")

        connection.on(LiveTranscriptionEvents.Transcript, _on_dg_transcript)
        connection.on(LiveTranscriptionEvents.Error, _on_dg_error)

        options = LiveOptions(
            model=DEEPGRAM_STT_MODEL,
            language="en",
            encoding="linear16",
            sample_rate=DEEPGRAM_STT_SAMPLE_RATE,
            channels=1,
            interim_results=True,
            smart_format=True,
            filler_words=True,
            punctuate=True,
        )

        started = await connection.start(options)
        if not started:
            raise StreamingSttError("Deepgram live connection failed to start.")

        self._connection = connection
        logger.info(
            f"[StreamingSTT] Live connection started "
            f"(model={DEEPGRAM_STT_MODEL}, sr={DEEPGRAM_STT_SAMPLE_RATE})"
        )

    async def send_audio(self, chunk: bytes) -> None:
        if self._connection is None:
            raise StreamingSttError("Cannot send audio before start().")
        await self._connection.send(chunk)

    async def finish(self) -> None:
        if self._connection is None:
            return
        await self._connection.finish()
        logger.info("[StreamingSTT] Live connection closed.")
        self._connection = None
