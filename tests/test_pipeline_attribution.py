"""Pipeline branches for wearer-focused analysis. All services mocked."""
import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.pipelines.analysis_pipeline import run_analysis_pipeline

STT_RESULT = {
    "text": "How was the demo? It went well, we shipped it.",
    "language": "en",
    "segments": [],
    "utterances": [
        {"speaker": 0, "text": "How was the demo?", "start": 0.0, "end": 1.5, "confidence": 0.9},
        {"speaker": 1, "text": "It went well, we shipped it.", "start": 1.6, "end": 3.9, "confidence": 0.9},
    ],
    "speakers": [0, 1],
}

MATCH = {"user_speaker": 1, "confidence": 0.81, "low_confidence": False}


def _run(tmp_path, *, profile, match, export_path="user.wav"):
    """Run the pipeline with all externals mocked; returns (results, mocks)."""
    combiner = MagicMock()
    combiner.combine_features.return_value = {"words_per_minute": 100.0}
    with (
        patch("src.pipelines.analysis_pipeline.stt.transcribe", return_value=dict(STT_RESULT)),
        patch("src.pipelines.analysis_pipeline.initialize_vector_store", return_value=None),
        patch("src.pipelines.analysis_pipeline.speaker_id.load_profile", return_value=profile),
        patch("src.pipelines.analysis_pipeline.speaker_id.match_user", return_value=match) as match_mock,
        patch("src.pipelines.analysis_pipeline.speaker_id.export_segments_wav", return_value=export_path) as export_mock,
        patch("src.pipelines.analysis_pipeline.FeatureCombiner", return_value=combiner),
        patch("src.pipelines.analysis_pipeline.emotion.analyze", return_value=None),
        patch("src.pipelines.analysis_pipeline.llm.generate_feedback", return_value="fb") as llm_mock,
    ):
        results = run_analysis_pipeline("audio.wav", output_dir=str(tmp_path))
    return results, {"combiner": combiner, "llm": llm_mock, "match": match_mock, "export": export_mock}


def _read_transcript(results):
    with open(results["transcript_path"], encoding="utf-8") as f:
        return json.load(f)


def test_no_profile_keeps_generic_behavior(tmp_path):
    results, mocks = _run(tmp_path, profile=None, match=None)
    saved = _read_transcript(results)
    assert saved["speaker_attribution"] == {"enabled": False, "reason": "no_profile"}
    mocks["match"].assert_not_called()
    kwargs = mocks["combiner"].combine_features.call_args.kwargs
    assert kwargs["transcript_text"] == STT_RESULT["text"]
    assert kwargs["audio_path"] == "audio.wav"
    assert mocks["llm"].call_args.kwargs.get("user_speaker") is None


def test_enrolled_match_runs_wearer_focused_analysis(tmp_path):
    results, mocks = _run(tmp_path, profile=np.ones(3), match=dict(MATCH))
    saved = _read_transcript(results)
    assert saved["speaker_attribution"]["enabled"] is True
    assert saved["speaker_attribution"]["user_speaker"] == 1
    kwargs = mocks["combiner"].combine_features.call_args.kwargs
    assert kwargs["transcript_text"] == "It went well, we shipped it."
    assert kwargs["audio_path"] == "user.wav"
    llm_kwargs = mocks["llm"].call_args.kwargs
    assert llm_kwargs["transcript"] == STT_RESULT["text"]
    assert llm_kwargs["user_speaker"] == 1


def test_match_failure_falls_back_to_generic(tmp_path):
    results, mocks = _run(tmp_path, profile=np.ones(3), match=None)
    saved = _read_transcript(results)
    assert saved["speaker_attribution"] == {"enabled": False, "reason": "match_failed"}
    kwargs = mocks["combiner"].combine_features.call_args.kwargs
    assert kwargs["transcript_text"] == STT_RESULT["text"]
    assert kwargs["audio_path"] == "audio.wav"
    assert results["features_path"] is not None


def test_export_failure_keeps_full_audio_but_user_text(tmp_path):
    results, mocks = _run(tmp_path, profile=np.ones(3), match=dict(MATCH), export_path=None)
    kwargs = mocks["combiner"].combine_features.call_args.kwargs
    assert kwargs["audio_path"] == "audio.wav"
    assert kwargs["transcript_text"] == "It went well, we shipped it."


def test_low_confidence_flag_persisted(tmp_path):
    low = {"user_speaker": 0, "confidence": 0.12, "low_confidence": True}
    results, _ = _run(tmp_path, profile=np.ones(3), match=low)
    saved = _read_transcript(results)
    assert saved["speaker_attribution"]["low_confidence"] is True
