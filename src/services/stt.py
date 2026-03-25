"""
Speech-to-Text service using Deepgram Nova-3 (batch / pre-recorded).

Returns a result dict with the same shape as the old Whisper output so
that nothing downstream needs to change:
    {
        "text":     str,   # full transcript
        "language": str,   # detected language code (e.g. "en")
        "segments": list,  # word-level dicts: {word, start, end, confidence}
    }
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def transcribe(audio_path: str) -> Optional[Dict[str, Any]]:
    """
    Transcribe a local audio file using Deepgram Nova-3.

    Args:
        audio_path: Path to the audio file (any format Deepgram supports).

    Returns:
        Dict with keys 'text', 'language', 'segments', or None on error.
    """
    # Import here so the module can be imported even before the SDK is installed
    try:
        from deepgram import DeepgramClient, PrerecordedOptions, FileSource
    except ImportError:
        logger.error("deepgram-sdk is not installed. Run: pip install deepgram-sdk")
        return None

    try:
        from config import DEEPGRAM_API_KEY, DEEPGRAM_STT_MODEL
    except ImportError:
        import os
        DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
        DEEPGRAM_STT_MODEL = os.environ.get("DEEPGRAM_STT_MODEL", "nova-3")

    if not DEEPGRAM_API_KEY:
        logger.error("DEEPGRAM_API_KEY is not set.")
        return None

    audio_path = str(audio_path)
    if not Path(audio_path).is_file():
        logger.error(f"Audio file not found: {audio_path}")
        return None

    logger.info(f"[STT] Transcribing with Deepgram {DEEPGRAM_STT_MODEL}: {audio_path}")

    try:
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)

        with open(audio_path, "rb") as f:
            buffer_data = f.read()

        payload: FileSource = {"buffer": buffer_data}

        options = PrerecordedOptions(
            model=DEEPGRAM_STT_MODEL,
            language="en",
            smart_format=True,
            filler_words=True,   # detects "um", "uh" etc. natively
            utterances=True,     # sentence-level segments
            punctuate=True,
        )

        response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)

        # Extract transcript
        channel = response.results.channels[0]
        alternative = channel.alternatives[0]
        transcript_text = alternative.transcript

        if not transcript_text:
            logger.warning("[STT] Deepgram returned an empty transcript.")
            return None

        # Build word-level segments
        segments = []
        for w in (alternative.words or []):
            segments.append({
                "word": getattr(w, "punctuated_word", w.word),
                "start": w.start,
                "end": w.end,
                "confidence": w.confidence,
            })

        # Deepgram returns detected_language on the channel
        language = getattr(channel, "detected_language", "en") or "en"

        logger.info(
            f"[STT] Transcription complete. "
            f"Language: {language}, Words: {len(segments)}, "
            f"Snippet: {transcript_text[:80]}..."
        )

        return {
            "text": transcript_text,
            "language": language,
            "segments": segments,
        }

    except Exception as e:
        logger.error(f"[STT] Deepgram transcription failed: {e}", exc_info=True)
        return None
