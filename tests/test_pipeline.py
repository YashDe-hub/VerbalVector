"""Tests for analysis pipeline metadata passthrough."""
from unittest.mock import patch, MagicMock


def test_pipeline_passes_session_label_to_vector_storage():
    """run_analysis_pipeline should forward session_label to store_transcript."""
    from src.pipelines.analysis_pipeline import run_analysis_pipeline

    mock_collection = MagicMock()
    mock_collection.name = "transcripts"

    with (
        patch("src.pipelines.analysis_pipeline.initialize_vector_store", return_value=mock_collection),
        patch("src.pipelines.analysis_pipeline.stt.transcribe", return_value={"text": "Hello world."}),
        patch("src.pipelines.analysis_pipeline.store_transcript", return_value=True) as mock_store,
        patch("src.pipelines.analysis_pipeline._perform_analysis") as mock_analysis,
    ):
        mock_analysis.side_effect = lambda *args: args[3].update({
            "features_path": "/fake/features.json",
            "feedback_path": "/fake/feedback.txt",
            "analysis_error": False,
        })

        run_analysis_pipeline(
            audio_path="/fake/audio.mp3",
            output_dir="/tmp/test_output",
            source_id="src123",
            session_label="My Talk",
        )

        mock_store.assert_called_once()
        call_kwargs = mock_store.call_args
        assert call_kwargs[1].get("session_label") == "My Talk" or (
            len(call_kwargs[0]) >= 4 and call_kwargs[0][3] == "My Talk"
        )


def test_pipeline_uses_basename_as_default_source_id():
    """When no source_id is given, pipeline should use the audio file basename."""
    from src.pipelines.analysis_pipeline import run_analysis_pipeline

    mock_collection = MagicMock()
    mock_collection.name = "transcripts"

    with (
        patch("src.pipelines.analysis_pipeline.initialize_vector_store", return_value=mock_collection),
        patch("src.pipelines.analysis_pipeline.stt.transcribe", return_value={"text": "Hello world."}),
        patch("src.pipelines.analysis_pipeline.store_transcript", return_value=True) as mock_store,
        patch("src.pipelines.analysis_pipeline._perform_analysis") as mock_analysis,
    ):
        mock_analysis.side_effect = lambda *args: args[3].update({
            "features_path": "/fake/features.json",
            "feedback_path": "/fake/feedback.txt",
            "analysis_error": False,
        })

        run_analysis_pipeline(
            audio_path="/fake/my_recording.mp3",
            output_dir="/tmp/test_output",
        )

        call_args = mock_store.call_args[0]
        assert call_args[1] == "my_recording"
