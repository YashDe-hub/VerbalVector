"""
Vocal emotion analysis using Hume AI Expression Measurement API (batch).

Submits the audio file, waits for the job to complete, then returns a flat
dict of emotion label → averaged score (0–1 float) across all time windows.

This supplements Librosa's hard acoustic numbers with dimensions Librosa
cannot capture: nervousness, confidence, energy, valence, etc.

Returns None on any error so the pipeline can continue without it.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

_JOB_TIMEOUT_SECONDS = 120
_POLL_INTERVAL_SECONDS = 3


def analyze(audio_path: str) -> Optional[Dict[str, float]]:
    """
    Submit audio to Hume Expression Measurement and return averaged vocal
    emotion scores across the full recording.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Dict mapping emotion label → float score (0–1), or None on error.
        Example: {"Confidence": 0.72, "Nervousness": 0.31, "Joy": 0.15, ...}
    """
    try:
        from hume import HumeClient
        from hume.expression_measurement.batch import Models, Prosody
    except ImportError:
        logger.error("hume SDK is not installed. Run: pip install hume")
        return None

    try:
        from config import HUME_API_KEY
    except ImportError:
        import os
        HUME_API_KEY = os.environ.get("HUME_API_KEY", "")

    if not HUME_API_KEY:
        logger.error("HUME_API_KEY is not set.")
        return None

    audio_path = str(audio_path)
    if not Path(audio_path).is_file():
        logger.error(f"[Emotion] Audio file not found: {audio_path}")
        return None

    logger.info(f"[Emotion] Submitting to Hume Expression Measurement: {audio_path}")

    try:
        client = HumeClient(api_key=HUME_API_KEY)

        job_id = client.expression_measurement.batch.start_inference_job_from_local_file(
            filepath=[audio_path],
            models=Models(prosody=Prosody()),
        )
        logger.info(f"[Emotion] Job submitted: {job_id}. Waiting for completion...")

        # Poll until complete or timeout
        elapsed = 0
        state = None
        while elapsed < _JOB_TIMEOUT_SECONDS:
            details = client.expression_measurement.batch.get_job_details(id=job_id)
            state = details.state.status
            if state in ("COMPLETED", "FAILED"):
                break
            time.sleep(_POLL_INTERVAL_SECONDS)
            elapsed += _POLL_INTERVAL_SECONDS

        if state != "COMPLETED":
            logger.error(f"[Emotion] Hume job did not complete in time. State: {state}")
            return None

        predictions = client.expression_measurement.batch.get_job_predictions(id=job_id)
        return _aggregate_emotion_scores(predictions)

    except Exception as e:
        logger.error(f"[Emotion] Hume analysis failed: {e}", exc_info=True)
        return None


def _aggregate_emotion_scores(predictions) -> Optional[Dict[str, float]]:
    """
    Average emotion scores across all prosody time-windows in the prediction.
    """
    if not predictions:
        logger.warning("[Emotion] Hume returned empty predictions.")
        return None

    emotion_totals: Dict[str, float] = {}
    emotion_counts: Dict[str, int] = {}

    try:
        for source_prediction in predictions:
            results = source_prediction.results
            if not results:
                continue
            for pred in results.predictions:
                prosody = pred.models.prosody
                if not prosody:
                    continue
                for group in (prosody.grouped_predictions or []):
                    for window in (group.predictions or []):
                        for emotion in (window.emotions or []):
                            name = emotion.name
                            score = emotion.score
                            if name:
                                emotion_totals[name] = emotion_totals.get(name, 0.0) + score
                                emotion_counts[name] = emotion_counts.get(name, 0) + 1

        if not emotion_totals:
            logger.warning("[Emotion] No emotion data found in Hume predictions.")
            return None

        averaged = {
            name: round(emotion_totals[name] / emotion_counts[name], 4)
            for name in emotion_totals
        }

        top5 = sorted(averaged.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"[Emotion] Top emotions: {top5}")

        return averaged

    except Exception as e:
        logger.error(f"[Emotion] Failed to parse Hume predictions: {e}", exc_info=True)
        return None
