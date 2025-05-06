"""
Feature combination for VerbalVector.

This module combines audio and text features into a unified feature set for analysis.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import ollama

from src.features.audio_features import process_presentation
from src.features.text_features import TextFeatureExtractor
# from src.features.speech_quality import analyze_speech_quality # Removed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureCombiner:
    """Class for combining audio and text features."""
    
    def __init__(self):
        """Initialize the feature combiner and text extractor."""
        self.text_extractor = TextFeatureExtractor()
    
    def combine_features(self, audio_path: str, transcript_text: Optional[str] = None, ollama_client: Optional[ollama.Client] = None, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Combine audio and text features from a presentation.
        Speech quality analysis (pronunciation, grammar) is skipped.
        
        Args:
            audio_path: Path to the audio file
            transcript_text: The actual transcript text string (Optional).
            ollama_client: Not used in this version, kept for potential future use.
            model_name: Not used in this version, kept for potential future use.
            
        Returns:
            Dictionary of combined features
        """
        logger.info(f"Extracting features from audio: {audio_path}")
        
        # Extract audio features
        audio_features = process_presentation(audio_path)
        if not audio_features:
            logger.error(f"Audio feature extraction failed for {audio_path}. Cannot proceed.")
            return {} # Return empty or raise error if audio features are essential
        
        # Extract text features if transcript is provided
        text_features = {}
        # speech_quality_features = {} # Removed
        
        if transcript_text:
            # if not ollama_client or not model_name: # Removed Ollama dependency for features
            #      logger.error("Ollama client and model_name are required when transcript_text is provided for speech quality analysis.")
            #      # Decide how to handle: return partial features or raise error?
            #      # Returning partial for now
            # else:
            logger.info("Extracting text features from provided transcript text...")
            try:
                # Extract basic text features (assuming flat keys like 'word_count')
                text_features.update(self.text_extractor._extract_basic_metrics(transcript_text))
                text_features.update(self.text_extractor._extract_filler_word_metrics(transcript_text))
                text_features.update(self.text_extractor._extract_vocabulary_metrics(transcript_text))
                text_features.update(self.text_extractor._extract_sentence_metrics(transcript_text))
                text_features.update(self.text_extractor._calculate_readability_metrics(transcript_text))
                
                # Analyze speech quality (pronunciation and grammar) - REMOVED
                # logger.info("Analyzing speech quality (grammar only)...")
                # sq_result = analyze_speech_quality(transcript_text, ollama_client, model_name)
                # if sq_result: 
                #      # Expecting {'pronunciation': {...}, 'grammar': {...}} or just {'grammar': {...}}
                #      # Store only grammar directly to preserve nesting in the final output
                #      # if 'pronunciation' in sq_result: # REMOVED
                #      #     speech_quality_features['speech_quality_pronunciation'] = sq_result['pronunciation'] # REMOVED
                #      if 'grammar' in sq_result:
                #          speech_quality_features['speech_quality_grammar'] = sq_result['grammar']
                #      else:
                #          logger.warning("Grammar analysis did not return 'grammar' key in results.")
                # else:
                #     logger.warning("Speech quality analysis (grammar) returned None or empty.")

            except Exception as e:
                logger.error(f"Failed to extract text features from provided text: {e}", exc_info=True)
                # Keep text_features potentially partially filled
        else:
            logger.warning("No transcript text provided, text/speech features will be skipped.")

        # Combine features directly without prefixes to match example output
        # Start with audio, then update with text.
        combined_features = {}
        combined_features.update(audio_features)
        combined_features.update(text_features)
        # combined_features.update(speech_quality_features) # Removed
        
        # Add metadata (optional, check if present in example)
        # combined_features['metadata'] = { ... }

        # --- Add derived features back? ---
        # The prompt expects some derived features like words_per_minute, vocal_expressiveness etc.
        # These were previously calculated in _derive_cross_modal_features but removed.
        # Let's add them back here using the available base features. Requires careful key matching.
        try:
            derived_features = self._derive_features_inline(combined_features, transcript_text)
            combined_features.update(derived_features)
            logger.info(f"Added derived features: {list(derived_features.keys())}")
        except Exception as e:
            logger.error(f"Failed to calculate derived features: {e}", exc_info=True)
        # --- End derived features addition ---

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
            
        # Fillers per minute (using filler rate instead of count/duration)
        # This might need adjustment based on exact base features available
        # filler_rate = features.get('filler_rate_percentage', 0.0) / 100.0
        # derived['fillers_per_minute'] = filler_rate * derived.get('words_per_minute', 0.0) # Approximation

        # Calculate vocal expressiveness score
        pitch_std = features.get('pitch_std', 0.0)
        volume_std = features.get('volume_std', 0.0)
        # Normalize based on typical ranges (these are estimates)
        normalized_pitch_var = min(1.0, pitch_std / 50.0) if pitch_std > 0 else 0.0 
        normalized_volume_var = min(1.0, volume_std / 0.1) if volume_std > 0 else 0.0 
        expressiveness = (normalized_pitch_var + normalized_volume_var) / 2.0
        derived['vocal_expressiveness'] = float(expressiveness)
        
        # Calculate speech clarity score (simplified example)
        wpm = derived.get('words_per_minute', 0)
        # Target WPM range 130-170, ideal center 150
        rate_factor = 1.0 - min(1.0, abs(wpm - 150) / 40.0) if wpm > 0 else 0.0 
        complex_word_perc = features.get('complex_word_percentage', 0.0) # Assuming this exists from text features
        complexity_factor = 1.0 - min(1.0, complex_word_perc / 20.0) # Penalize high complexity
        filler_rate = features.get('filler_rate_percentage', 0.0)
        filler_factor = 1.0 - min(1.0, filler_rate / 10.0) # Penalize high filler rate (>10%)
        clarity = (rate_factor * 0.4 + complexity_factor * 0.3 + filler_factor * 0.3) # Weighted average
        derived['speech_clarity'] = float(clarity)

        # Grammar score (simplified from count) - REMOVED
        # grammar_issues_count = features.get('speech_quality_grammar', {}).get('total_issues', 0)
        # if word_count > 0:
        #      grammar_score = max(0.0, 1.0 - (grammar_issues_count / word_count) * 5) # Penalize heavily
        # else:
        #      grammar_score = 1.0 if grammar_issues_count == 0 else 0.0
        # derived['grammar_score'] = float(grammar_score)

        # Note: Other derived features like comprehension/engagement are harder to estimate reliably
        # without more sophisticated models or specific indicators. Adding placeholder values or omitting.
        # derived['estimated_comprehension'] = 0.0
        # derived['estimated_engagement'] = 0.0

        return derived

    def _merge_feature_dicts(
        self, 
        audio_features: Dict[str, Any], 
        text_features: Dict[str, Any],
        speech_quality_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge audio, text, and speech quality feature dictionaries with prefixes.
        
        Args:
            audio_features: Dictionary of audio features
            text_features: Dictionary of text features
            speech_quality_features: Dictionary of speech quality features
            
        Returns:
            Merged dictionary
        """
        merged = {}
        
        # Add audio features with prefix
        for key, value in audio_features.items():
            merged[f"audio_{key}"] = value
            
        # Add text features with prefix
        for key, value in text_features.items():
            merged[f"text_{key}"] = value
            
        # Add speech quality features with prefix
        for key, value in speech_quality_features.items():
            merged[f"speech_quality_{key}"] = value
            
        return merged
    
    def _derive_cross_modal_features(
        self, 
        audio_features: Dict[str, Any], 
        text_features: Dict[str, Any],
        speech_quality_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Derive new features that combine information from audio, text, and speech quality.
        
        Args:
            audio_features: Dictionary of audio features
            text_features: Dictionary of text features
            speech_quality_features: Dictionary of speech quality features
            
        Returns:
            Dictionary of derived features
        """
        derived = {}
        
        # Calculate speech rate in words per minute
        duration_minutes = audio_features.get('duration', 0) / 60.0
        word_count = text_features.get('word_count', 0)
        if duration_minutes > 0 and word_count > 0:
            wpm = word_count / duration_minutes
            derived['words_per_minute'] = float(wpm)
        else:
            derived['words_per_minute'] = 0.0
        
        # Calculate filler word frequency (fillers per minute)
        total_fillers = text_features.get('total_filler_words', 0)
        if duration_minutes > 0:
            fillers_per_minute = total_fillers / duration_minutes
            derived['fillers_per_minute'] = float(fillers_per_minute)
        else:
            derived['fillers_per_minute'] = 0.0
        
        # Calculate pause-to-speech ratio
        total_pause_duration = audio_features.get('total_pause_duration', 0)
        speech_duration = audio_features.get('duration', 0) - total_pause_duration
        if speech_duration > 0:
            pause_to_speech_ratio = total_pause_duration / speech_duration
            derived['pause_to_speech_ratio'] = float(pause_to_speech_ratio)
        else:
            derived['pause_to_speech_ratio'] = 0.0
        
        # Calculate vocal expressiveness score
        pitch_std = audio_features.get('pitch_std', 0.0)
        volume_std = audio_features.get('volume_std', 0.0)
        normalized_pitch_var = min(1.0, pitch_std / 50.0)
        normalized_volume_var = min(1.0, volume_std / 0.1)
        expressiveness = (normalized_pitch_var + normalized_volume_var) / 2.0
        derived['vocal_expressiveness'] = float(expressiveness)
        
        # Calculate speech clarity score
        wpm = derived.get('words_per_minute', 0)
        rate_factor = 1.0 - min(1.0, abs(wpm - 155) / 50.0) if wpm > 0 else 0.0
        complex_word_perc = text_features.get('complex_word_percentage', 100.0)
        complexity_factor = 1.0 - min(1.0, complex_word_perc / 25.0)
        clarity = (rate_factor + complexity_factor) / 2.0
        derived['speech_clarity'] = float(clarity)
        
        # Calculate overall speech quality score
        pronunciation_score = 1.0 - min(1.0, speech_quality_features.get('pronunciation', {}).get('words_with_issues', 0) / max(1, speech_quality_features.get('pronunciation', {}).get('total_words_checked', 1)))
        grammar_score = 1.0 - min(1.0, speech_quality_features.get('grammar', {}).get('total_issues', 0) / max(1, word_count))
        derived['speech_quality_score'] = float((pronunciation_score + grammar_score) / 2.0)
        
        return derived


def _extract_text_from_json(transcript_json_path: Path) -> Optional[str]:
    """Loads transcript JSON and extracts the concatenated text."""
    try:
        with open(transcript_json_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)

        # transcript_data is expected to be a list of dicts, each with a 'text' key
        if isinstance(transcript_data, list) and all('text' in item for item in transcript_data):
            full_text = ' '.join([item['text'] for item in transcript_data])
            return full_text
        else:
            logger.warning(f"Transcript JSON format not as expected in {transcript_json_path}. Expected list of dicts with 'text' key.")
            return None
    except FileNotFoundError:
        logger.warning(f"Transcript JSON file not found: {transcript_json_path}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in transcript file: {transcript_json_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading transcript JSON {transcript_json_path}: {e}", exc_info=True)
        return None


def process_presentation_data(
    audio_path: str, 
    transcript_json_path: Optional[str] = None, # Path to JSON transcript
    output_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Process presentation data and extract combined features.
    Uses transcript JSON if available, otherwise proceeds with audio-only features.

    Args:
        audio_path: Path to audio file.
        transcript_json_path: Optional path to the downloaded transcript JSON file.
        output_path: Optional path to save features as JSON.

    Returns:
        Dictionary of combined features, or None if audio processing fails.
    """
    # Ensure audio path exists
    if not Path(audio_path).is_file():
        logger.error(f"Audio file not found: {audio_path}")
        return None

    combiner = FeatureCombiner()
    transcript_text = None

    # Attempt to load transcript text from JSON if path is provided
    if transcript_json_path:
        transcript_text = _extract_text_from_json(Path(transcript_json_path))
        if not transcript_text:
            logger.warning(f"Could not load text from {transcript_json_path}, proceeding without text features.")
    else:
        logger.info("No transcript path provided, text features will not be extracted.")

    # Combine features (handles None for transcript_text)
    try:
        features = combiner.combine_features(audio_path, transcript_text)
    except Exception as e:
        logger.error(f"Failed during feature combination for {audio_path}: {e}", exc_info=True)
        return None # Cannot proceed if combination fails (likely audio error)

    # Save to file if output path is provided
    if output_path:
        output_p = Path(output_path)
        output_p.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(output_p, 'w', encoding='utf-8') as f:
                json.dump(features, f, indent=2)
            logger.info(f"Saved combined features to {output_p}")
        except IOError as e:
            logger.error(f"Failed to save features to {output_p}: {e}")

    return features


# Example usage needs update or removal as it assumes .txt transcript
# if __name__ == "__main__":
#     ... 