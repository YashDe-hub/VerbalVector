"""
PCM-chunks-to-WAV file assembler.

Used by /api/stream to incrementally write raw 16-bit linear PCM chunks
arriving over the WebSocket into a single WAV file on disk. At session
end the WAV is handed to the existing batch analysis pipeline, so the
streamed session produces the same transcript/features/feedback as a
record-then-upload session.
"""

import logging
import wave
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


class AudioAssemblerError(Exception):
    """Raised on unrecoverable assembler errors (e.g., write after close)."""


class AudioAssembler:
    """
    Append raw PCM frames to a WAV file, finalize on close.

    Defaults match Deepgram's preferred live encoding: 16kHz mono 16-bit
    little-endian PCM. The WS protocol pins these — overriding is for tests.
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
    ) -> None:
        self._output_path = Path(output_path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._bytes_written = 0
        self._closed = False

        self._wave = wave.open(str(self._output_path), "wb")
        self._wave.setnchannels(channels)
        self._wave.setsampwidth(sample_width)
        self._wave.setframerate(sample_rate)

    def write_chunk(self, chunk: bytes) -> None:
        if self._closed:
            raise AudioAssemblerError("Cannot write to closed assembler.")
        if not chunk:
            return
        self._wave.writeframes(chunk)
        self._bytes_written += len(chunk)

    def close(self) -> Path:
        if not self._closed:
            self._wave.close()
            self._closed = True
            logger.info(
                f"[Assembler] Closed {self._output_path} ({self._bytes_written} bytes)"
            )
        return self._output_path

    @property
    def bytes_written(self) -> int:
        return self._bytes_written
