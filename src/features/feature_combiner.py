"""
Feature combination for VerbalVector.

This module combines audio and text features into a unified feature set for analysis.
"""

import logging
from typing import Dict, Any, Optional

from src.features.audio_features import process_presentation
from src.features.text_features import TextFeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureCombiner:
    """Class for combining audio and text features."""

    def __init__(self):
        """Initialize the feature combiner and text extractor."""
        self.text_extractor = TextFeatureExtractor()

    def combine_features(self, audio_path: str, transcript_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Combine audio and text features from a presentation.

        Args:
            audio_path: Path to the audio file
            transcript_text: The actual transcript text string (Optional).

        Returns:
            Dictionary of combined features
        """
        logger.info(f"Extracting features from audio: {audio_path}")

        # Extract audio features
        audio_features = process_presentation(audio_path)
        if not audio_features:
            logger.error(f"Audio feature extraction failed for {audio_path}. Cannot proceed.")
            return {}

        # Extract text features if transcript is provided
        text_features = {}

        if transcript_text:
            logger.info("Extracting text features from provided transcript text...")
            try:
                text_features.update(self.text_extractor._extract_basic_metrics(transcript_text))
                text_features.update(self.text_extractor._extract_filler_word_metrics(transcript_text))
                text_features.update(self.text_extractor._extract_vocabulary_metrics(transcript_text))
                text_features.update(self.text_extractor._extract_sentence_metrics(transcript_text))
                text_features.update(self.text_extractor._calculate_readability_metrics(transcript_text))
            except Exception as e:
                logger.error(f"Failed to extract text features from provided text: {e}", exc_info=True)
        else:
            logger.warning("No transcript text provided, text/speech features will be skipped.")

        # Combine features
        combined_features = {}
        combined_features.update(audio_features)
        combined_features.update(text_features)

        # Add derived features
        try:
            derived_features = self._derive_features_inline(combined_features, transcript_text)
            combined_features.update(derived_features)
            logger.info(f"Added derived features: {list(derived_features.keys())}")
        except Exception as e:
            logger.error(f"Failed to calculate derived features: {e}", exc_info=True)

        return combined_features

    def _derive_features_inline(self, features: Dict[str, Any], transcript_text: Optional[str]) -> Dict[str, Any]:
        """Calculate derived features directly using the combined feature set."""
        derived = {}

        # Calculate speech rate in words per minute
        duration_seconds = features.get('duration', 0)
        word_count = features.get('word_count', 0)
        if duration_seconds > 0 and word_count > 0:
            wpm = word_count / (duration_seconds / 60.0)
            derived['words_per_minute'] = float(wpm)
        else:
            derived['words_per_minute'] = 0.0

        # Calculate vocal expressiveness score
        pitch_std = features.get('pitch_std', 0.0)
        volume_std = features.get('volume_std', 0.0)
        normalized_pitch_var = min(1.0, pitch_std / 50.0) if pitch_std > 0 else 0.0
        normalized_volume_var = min(1.0, volume_std / 0.1) if volume_std > 0 else 0.0
        expressiveness = (normalized_pitch_var + normalized_volume_var) / 2.0
        derived['vocal_expressiveness'] = float(expressiveness)

        # Calculate speech clarity score
        wpm = derived.get('words_per_minute', 0)
        rate_factor = 1.0 - min(1.0, abs(wpm - 150) / 40.0) if wpm > 0 else 0.0
        complex_word_perc = features.get('complex_word_percentage', 0.0)
        complexity_factor = 1.0 - min(1.0, complex_word_perc / 20.0)
        filler_rate = features.get('filler_word_rate', 0.0)  # 0–100 percentage
        filler_factor = 1.0 - min(1.0, filler_rate / 30.0)  # 30% filler rate = fully penalised
        clarity = (rate_factor * 0.4 + complexity_factor * 0.3 + filler_factor * 0.3)
        derived['speech_clarity'] = float(clarity)

        return derived
