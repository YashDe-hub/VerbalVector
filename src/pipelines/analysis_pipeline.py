"""
Orchestrates the full pipeline:
  STT (Deepgram) → Features (Librosa + NLTK) + Emotion (Hume) → LLM Feedback (Gemini)

Threading:
  - Analysis thread:      feature extraction + emotion analysis + LLM feedback
  - Vector storage thread: stores transcript in ChromaDB (optional, non-fatal)
"""

import logging
import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional

# Feature extraction
from src.features.audio_features import process_presentation
from src.features.text_features import TextFeatureExtractor
from src.features.feature_combiner import FeatureCombiner

# External API services
from src.services import stt, emotion, llm, speaker_id

# Vector store
from src.vector_store.manager import initialize_vector_store, store_transcript

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Analysis thread
# ---------------------------------------------------------------------------

def _perform_analysis(
    audio_path_str: str,
    transcript_text: str,
    utterances: list[dict] | None,
    output_dir_path: Path,
    results_dict: dict,
    user_speaker: int | None = None,
) -> None:
    """
    Runs in a background thread:
      1. Extracts Librosa + text features via FeatureCombiner
      2. Runs Hume vocal emotion analysis (non-fatal if it fails)
      3. Generates LLM feedback via Gemini (audio + features + emotions)
      4. Saves features JSON and feedback TXT to output_dir_path
    """
    logger.info(f"[Thread Analysis] Starting for {audio_path_str}")
    features_path = None
    feedback_path = None
    combined_features = {}

    try:
        base_name = Path(audio_path_str).stem
        features_out = output_dir_path / f"{base_name}_features.json"
        feedback_out = output_dir_path / f"{base_name}_feedback.txt"

        # Wearer-focused mode: compute features on the user's speech only.
        # The LLM still receives the FULL audio + transcript as context.
        features_audio_path = audio_path_str
        features_text = transcript_text
        if user_speaker is not None and utterances:
            user_utts = [u for u in utterances if u.get("speaker") == user_speaker]
            if user_utts:
                features_text = " ".join(u["text"] for u in user_utts)
                user_wav = speaker_id.export_segments_wav(
                    audio_path_str,
                    [(u["start"], u["end"]) for u in user_utts],
                    str(output_dir_path / f"{base_name}_user.wav"),
                )
                if user_wav:
                    features_audio_path = user_wav
                else:
                    logger.warning(
                        "[Thread Analysis] User-only WAV export failed — "
                        "audio features fall back to the full recording."
                    )

        # 1. Feature extraction (Librosa + NLTK)
        try:
            combiner = FeatureCombiner()
            combined_features = combiner.combine_features(
                audio_path=features_audio_path,
                transcript_text=features_text,
            )
            if not combined_features:
                logger.error("[Thread Analysis] FeatureCombiner returned empty dict.")
                results_dict["analysis_error"] = True
                return
            logger.info("[Thread Analysis] Features extracted.")
        except Exception as e:
            logger.error(f"[Thread Analysis] Feature extraction failed: {e}", exc_info=True)
            results_dict["analysis_error"] = True
            return

        # 2. Save features JSON
        try:
            with open(features_out, "w", encoding="utf-8") as f:
                json.dump(combined_features, f, indent=2, ensure_ascii=False, default=str)
            features_path = str(features_out)
            logger.info(f"[Thread Analysis] Features saved: {features_path}")
        except Exception as e:
            logger.error(f"[Thread Analysis] Could not save features: {e}", exc_info=True)
            results_dict["analysis_error"] = True
            return

        # 3. Vocal emotion analysis via Hume (non-fatal)
        emotion_scores = None
        try:
            emotion_scores = emotion.analyze(audio_path_str)
            if emotion_scores:
                logger.info(f"[Thread Analysis] Emotion scores received ({len(emotion_scores)} emotions).")
            else:
                logger.warning("[Thread Analysis] Hume returned no scores — continuing without emotion data.")
        except Exception as e:
            logger.warning(f"[Thread Analysis] Hume analysis failed (non-fatal): {e}")

        # 4. LLM feedback via Gemini (audio + features + emotions)
        feedback_text = llm.generate_feedback(
            audio_path=audio_path_str,
            transcript=transcript_text,
            features=combined_features,
            emotion_scores=emotion_scores,
            utterances=utterances,
            user_speaker=user_speaker,
        )

        if feedback_text:
            try:
                with open(feedback_out, "w", encoding="utf-8") as f:
                    f.write(feedback_text)
                feedback_path = str(feedback_out)
                logger.info(f"[Thread Analysis] Feedback saved: {feedback_path}")
            except Exception as e:
                logger.error(f"[Thread Analysis] Could not save feedback: {e}", exc_info=True)
        else:
            logger.error("[Thread Analysis] LLM returned no feedback.")

        # 5. Store results
        results_dict["features_path"] = features_path
        results_dict["feedback_path"] = feedback_path
        if not results_dict.get("analysis_error"):
            results_dict["analysis_error"] = not (features_path and feedback_path)

        logger.info(f"[Thread Analysis] Done. Error: {results_dict.get('analysis_error')}")

    except Exception as e:
        logger.error(f"[Thread Analysis] Uncaught error: {e}", exc_info=True)
        results_dict["features_path"] = None
        results_dict["feedback_path"] = None
        results_dict["analysis_error"] = True


# ---------------------------------------------------------------------------
# Vector storage thread
# ---------------------------------------------------------------------------

def _perform_vector_storage(
    transcript_text: str,
    source_id: str,
    collection,
    session_label: str = "",
    utterances: list[dict] | None = None,
) -> None:
    """Stores transcript chunks in ChromaDB. Runs in a background thread."""
    logger.info(f"[Thread VectorStore] Starting for source_id: {source_id}")
    try:
        success = store_transcript(
            transcript_text,
            source_id,
            collection,
            session_label=session_label,
            utterances=utterances,
        )
        if success:
            logger.info(f"[Thread VectorStore] Stored transcript for {source_id}.")
        else:
            logger.error(f"[Thread VectorStore] store_transcript returned False for {source_id}.")
    except Exception as e:
        logger.error(f"[Thread VectorStore] Error for {source_id}: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_analysis_pipeline(
    audio_path: str,
    output_dir: str = "analysis_output",
    source_id: str = "",
    session_label: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Full pipeline: STT → features + emotion (parallel with vector storage) → LLM feedback.

    Returns:
        Dict with keys 'transcript_path', 'features_path', 'feedback_path',
        or None on fatal error.
    """
    logger.info(f"Starting analysis pipeline for: {audio_path}")
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    base_name = Path(audio_path).stem

    final_results: Dict[str, Any] = {
        "transcript_path": None,
        "features_path": None,
        "feedback_path": None,
    }

    # 1. Initialise vector store (non-fatal)
    collection = initialize_vector_store()
    if not collection:
        logger.warning("Vector store initialisation failed — continuing without it.")

    # 2. Speech-to-Text via Deepgram
    stt_result = stt.transcribe(audio_path)
    if not stt_result or not stt_result.get("text"):
        logger.error("STT failed or returned empty transcript.")
        return None

    transcript_text = stt_result["text"]
    utterances = stt_result.get("utterances")  # may be None or []

    # Wearer-focused attribution (optional, non-fatal): which speaker is the user?
    user_speaker: Optional[int] = None
    attribution: Dict[str, Any] = {"enabled": False, "reason": "no_profile"}
    profile = speaker_id.load_profile()
    if profile is not None:
        match = None
        try:
            match = speaker_id.match_user(audio_path, utterances or [], profile)
        except Exception as e:
            logger.warning(f"[SpeakerID] match_user raised (non-fatal): {e}")
        if match:
            attribution = {"enabled": True, **match}
            user_speaker = match["user_speaker"]
            logger.info(
                f"[SpeakerID] User is Speaker {user_speaker} "
                f"(confidence {match['confidence']}, low_confidence={match['low_confidence']})."
            )
        else:
            attribution = {"enabled": False, "reason": "match_failed"}
            logger.warning("[SpeakerID] Could not identify the user — generic analysis.")
    stt_result["speaker_attribution"] = attribution

    # Save transcript JSON
    transcript_path = output_dir_path / f"{base_name}_transcript.json"
    try:
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(stt_result, f, indent=4)
        final_results["transcript_path"] = str(transcript_path)
        logger.info(f"Transcript saved: {transcript_path}")
    except Exception as e:
        logger.error(f"Could not save transcript: {e}", exc_info=True)
        return None

    # 3. Launch parallel threads
    analysis_results: dict = {}

    analysis_thread = threading.Thread(
        target=_perform_analysis,
        args=(audio_path, transcript_text, utterances, output_dir_path, analysis_results, user_speaker),
        daemon=True,
    )
    effective_source_id = source_id or base_name
    vector_thread = threading.Thread(
        target=_perform_vector_storage,
        args=(transcript_text, effective_source_id, collection),
        kwargs={"session_label": session_label, "utterances": utterances},
        daemon=True,
    ) if collection else None

    logger.info("Starting analysis and vector storage threads...")
    analysis_thread.start()
    if vector_thread:
        vector_thread.start()

    analysis_thread.join()
    if vector_thread:
        vector_thread.join()

    logger.info("All threads completed.")

    # 4. Consolidate results
    final_results["features_path"] = analysis_results.get("features_path")
    final_results["feedback_path"] = analysis_results.get("feedback_path")

    if analysis_results.get("analysis_error", True):
        logger.error("Analysis thread reported an error. Check logs above.")

    if not final_results["features_path"]:
        logger.warning("Features path is missing after analysis thread.")

    logger.info(f"Pipeline finished: {final_results}")
    return final_results
