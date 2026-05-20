"""Tests for AudioAssembler (PCM chunks -> WAV file)."""
import wave
from pathlib import Path
import pytest


def test_creates_valid_wav_file(tmp_path):
    """Closing produces a valid WAV file readable by stdlib wave module."""
    from src.services.audio_assembler import AudioAssembler

    out = tmp_path / "session.wav"
    asm = AudioAssembler(out)
    asm.write_chunk(b"\x00\x01" * 8)
    asm.write_chunk(b"\x02\x03" * 8)
    finalized = asm.close()

    assert finalized == out
    assert finalized.exists()

    with wave.open(str(finalized), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 16000
        assert w.getnframes() == 16  # 32 bytes / 2 bytes per frame


def test_bytes_written_tracks_input(tmp_path):
    from src.services.audio_assembler import AudioAssembler

    asm = AudioAssembler(tmp_path / "s.wav")
    asm.write_chunk(b"\x00\x01\x02\x03")
    assert asm.bytes_written == 4
    asm.write_chunk(b"\x04\x05")
    assert asm.bytes_written == 6
    asm.close()


def test_empty_chunk_is_noop(tmp_path):
    """write_chunk(b'') should not bump bytes_written or produce frames."""
    from src.services.audio_assembler import AudioAssembler

    asm = AudioAssembler(tmp_path / "s.wav")
    asm.write_chunk(b"")
    assert asm.bytes_written == 0
    asm.close()
    with wave.open(str(tmp_path / "s.wav"), "rb") as w:
        assert w.getnframes() == 0


def test_zero_chunks_produces_empty_but_valid_wav(tmp_path):
    """Closing with no chunks should still produce a valid (empty) WAV header."""
    from src.services.audio_assembler import AudioAssembler

    asm = AudioAssembler(tmp_path / "empty.wav")
    finalized = asm.close()

    with wave.open(str(finalized), "rb") as w:
        assert w.getnframes() == 0
        assert w.getframerate() == 16000


def test_custom_sample_rate(tmp_path):
    from src.services.audio_assembler import AudioAssembler

    asm = AudioAssembler(tmp_path / "s.wav", sample_rate=8000)
    asm.write_chunk(b"\x00" * 16)
    asm.close()

    with wave.open(str(tmp_path / "s.wav"), "rb") as w:
        assert w.getframerate() == 8000


def test_write_after_close_raises(tmp_path):
    from src.services.audio_assembler import AudioAssembler, AudioAssemblerError

    asm = AudioAssembler(tmp_path / "s.wav")
    asm.close()
    with pytest.raises(AudioAssemblerError):
        asm.write_chunk(b"\x00\x01")


def test_double_close_is_idempotent(tmp_path):
    """Calling close() twice should not raise (defensive — the WS handler may close in a finally block)."""
    from src.services.audio_assembler import AudioAssembler

    asm = AudioAssembler(tmp_path / "s.wav")
    asm.write_chunk(b"\x00\x01")
    path1 = asm.close()
    path2 = asm.close()
    assert path1 == path2


def test_parent_directory_created_if_missing(tmp_path):
    """The assembler should create parent directories so the WS handler can pass a deep path."""
    from src.services.audio_assembler import AudioAssembler

    out = tmp_path / "sessions" / "2026" / "session.wav"
    asm = AudioAssembler(out)
    asm.write_chunk(b"\x00" * 8)
    asm.close()
    assert out.exists()
