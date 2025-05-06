"""
Orchestrates the full STT -> Features -> LLM analysis pipeline, with parallel
vector storage.
"""

import logging
import os
import json
import ollama # For LLM call
from pathlib import Path
from typing import Dict, Any, Optional
import threading # Added for parallelism
import time # Added for potential source ID

# Import necessary feature extraction components
from src.features.audio_features import process_presentation # For direct audio features
from src.features.speech_quality import analyze_speech_quality # For pronunciation/grammar
from src.features.text_features import TextFeatureExtractor # Import Text Feature Extractor
# We'll need an STT component here - let's assume Whisper for now
import whisper # Import Whisper

# Import vector store functions
from src.vector_store.manager import initialize_vector_store, store_transcript

# Import FeatureCombiner
from src.features.feature_combiner import FeatureCombiner

logger = logging.getLogger(__name__)

# Global variable for the loaded Whisper model (lazy loading)
whisper_model = None

def run_stt(audio_path: str) -> Optional[Dict[str, Any]]:
    """
    Performs Speech-to-Text using Whisper.
    
    Args:
        audio_path: Path to the audio file.
        
    Returns:
        Whisper transcription result dictionary, or None if error.
    """
    global whisper_model
    logger.info(f"Running STT on: {audio_path}")
    
    # Lazy load model if not already loaded
    if whisper_model is None:
        try:
            model_size = "base.en" # Or "small.en", "medium.en" etc.
            logger.info(f"Loading Whisper STT model ({model_size})...")
            whisper_model = whisper.load_model(model_size) 
            logger.info("Whisper model loaded.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
            return None
            
    try:
        # Transcribe the audio
        logger.info("Starting transcription...")
        # Using fp16=False might be necessary on MPS but can be slower
        # Check device availability for optimal performance later if needed
        result = whisper_model.transcribe(audio_path, fp16=False) 
        logger.info("Transcription complete.")
        # Log detected language and a snippet
        lang = result.get('language', 'unknown')
        text_snippet = result.get('text', '')[:100] + "..."
        logger.info(f"Detected language: {lang}")
        logger.info(f"Transcript snippet: {text_snippet}")
        return result
    except Exception as e:
        logger.error(f"Error during Whisper transcription: {e}", exc_info=True)
        return None

def generate_final_feedback(transcript: str, features: Dict[str, Any], model_name: str, ollama_client: ollama.Client) -> Optional[str]:
    """
    Generates final feedback as a markdown string by querying an LLM using the 
    transcript and a JSON representation of the features.
    
    Args:
        transcript: The text transcript from STT.
        features: Combined dictionary of audio and text features.
        model_name: Name of the Ollama model to use.
        ollama_client: The Ollama client instance.
        
    Returns:
        Markdown formatted feedback string, or None if error.
    """
    logger.info(f"Generating final detailed feedback using LLM: {model_name}")
    
    # 1. Prepare features JSON string for the prompt
    try:
        features_json_string = json.dumps(features, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error(f"Error serializing features to JSON: {e}", exc_info=True)
        return None

    # 2. Define the REINFORCED Markdown-based prompt
    # Pre-format key features to safely embed in the f-string prompt
    wpm_val = features.get('words_per_minute', 'N/A')
    wpm_str = f"{wpm_val:.1f} wpm" if isinstance(wpm_val, float) else str(wpm_val)
    
    pitch_std_val = features.get('pitch_std', 'N/A')
    pitch_std_str = f"{pitch_std_val:.1f} Hz std" if isinstance(pitch_std_val, float) else str(pitch_std_val)
    
    volume_std_val = features.get('volume_std', 'N/A')
    volume_std_str = f"{volume_std_val:.3f} std" if isinstance(volume_std_val, float) else str(volume_std_val)
    
    num_pauses_val = features.get('num_pauses', 'N/A')
    num_pauses_str = str(num_pauses_val)
    
    filler_rate_val = features.get('filler_word_rate', 'N/A')
    filler_rate_str = f"{filler_rate_val:.1f}%" if isinstance(filler_rate_val, float) else str(filler_rate_val)
    
    transcript_snippet = transcript[:75].replace("\n", " ").replace("'", "`") + "..."
    
    prompt = f"""You are an expert communication coach tasked with analyzing the provided presentation materials.

**IMPORTANT:** Your feedback MUST be based **specifically** on the following transcript and calculated features. Do not provide generic advice. Reference specific examples from the transcript or values from the feature JSON to justify every point you make.

**Presentation Transcript:**
```text
{transcript}
```

**Calculated Features:**
```json
{features_json_string}
```

**Your Task:**
Generate high-value, insightful feedback following the Markdown structure below precisely.

**Output Format:**

## Overall Assessment
(Provide a concise summary of the presentation's effectiveness, highlighting key strengths and the most critical areas for improvement **based solely on the materials provided above**.)

## Key Scores (1-10)
(Provide scores for the following aspects. For each score, provide clear reasoning **referencing specific feature values from the JSON block above or direct quotes/examples from the transcript above**.)
*   **Clarity:** <score_1_10> (Reasoning: Must reference features like WPM, pauses, lexical diversity, filler words, speech clarity, OR transcript examples.)
*   **Engagement:** <score_1_10> (Reasoning: Must reference features like pitch/volume variation, WPM, OR transcript examples.)
*   **Pacing:** <score_1_10> (Reasoning: **Must specifically reference calculated WPM [{wpm_str}] and pause data [{num_pauses_str} pauses] from the features JSON.**)
*   **Vocal Variety:** <score_1_10> (Reasoning: **Must specifically reference pitch variation [{pitch_std_str}] and volume variation [{volume_std_str}] from the features JSON.**)

## Strengths
(List 2-3 specific strengths of the presentation. **Justify each point with specific evidence** from the **transcript** or **features JSON provided above**.)
*   Strength 1: ... (Evidence: e.g., "The speaker effectively used pauses [{num_pauses_str} total] for emphasis" or "The opening, '{transcript_snippet}', immediately grabbed attention.")
*   Strength 2: ... (Evidence: ...)

## Areas for Improvement
(List 2-3 specific, actionable areas for improvement. Explain the issue, **citing specific evidence** from the **transcript** or **features JSON provided above** (e.g., filler rate is {filler_rate_str}), and provide concrete suggestions.)
*   Area 1: ... (Issue: ..., Evidence: ..., Suggestion: ...)
*   Area 2: ... (Issue: ..., Evidence: ..., Suggestion: ...)

## Content & Structure Analysis
(**Based ONLY on the provided transcript text above**, first summarize the core topic/message in 1-2 sentences. Then, analyze the message clarity, logical flow, use of transitions, organization, and the effectiveness of the introduction and conclusion. **Do not ask for the transcript content - analyze the one provided.** Provide specific examples from the transcript text to support your analysis.)

"""
        
    # 3. Query the LLM (Removed format='json')
    logger.info(f"Sending request to Ollama model: {model_name}")
    try:
        response = ollama_client.generate(
            model=model_name,
            prompt=prompt,
            # format='json', # REMOVED - Expecting markdown text now
            options={'temperature': 0.6} # Slightly higher temp for more natural text
        )
        logger.info("Received response from Ollama.")
        
        # 4. Return the response string directly
        if response and 'response' in response:
            feedback_text = response['response'].strip()
            logger.info("Returning feedback text from LLM.")
            return feedback_text # Return the string
        else:
            logger.error(f"Unexpected response structure from Ollama: {response}")
            return None

    except Exception as e:
        logger.error(f"Error during Ollama API call: {e}", exc_info=True)
        return None # Corrected indentation

# --- Target Function for Analysis Thread --- 
def _perform_analysis(audio_path_str: str, transcript_text: str, model_name: str, output_dir_path: Path, results_dict: dict):
    """Uses FeatureCombiner to get features, then generates feedback."""
    logger.info(f"[Thread Analysis] Starting for {audio_path_str}")
    features_path = None
    feedback_path = None
    ollama_client = None # Initialize client variable
    combined_features = {} # Initialize features variable
    
    try:
        base_name = Path(audio_path_str).stem
        current_features_path = output_dir_path / f"{base_name}_features.json"
        current_feedback_path = output_dir_path / f"{base_name}_feedback.txt" # Changed extension to .txt

        # --- 1. Initialize Ollama Client (needed by Feedback generator) --- 
        try:
            ollama_client = ollama.Client()
            logger.info("[Thread Analysis] Initialized Ollama client.")
        except Exception as ollama_e:
            logger.error(f"[Thread Analysis] Failed to initialize Ollama client: {ollama_e}", exc_info=True)
            results_dict['analysis_error'] = True 
            return 

        # --- 2. Use FeatureCombiner --- 
        try:
            combiner = FeatureCombiner()
            # Pass ollama_client=None, model_name=None since combiner no longer uses them
            combined_features = combiner.combine_features(
                audio_path=audio_path_str,
                transcript_text=transcript_text,
                ollama_client=None, 
                model_name=None
            )
            
            if not combined_features: 
                 logger.error("[Thread Analysis] FeatureCombiner returned empty dictionary, likely an audio processing error.")
                 results_dict['analysis_error'] = True
            else:
                logger.info("[Thread Analysis] Features successfully combined by FeatureCombiner.")

        except Exception as combine_e:
             logger.error(f"[Thread Analysis] Error calling FeatureCombiner: {combine_e}", exc_info=True)
             results_dict['analysis_error'] = True
        
        # --- 3. Save Combined Features (only if successfully generated) --- 
        if combined_features and not results_dict.get('analysis_error'): 
            try:
                # Use default=str for saving in case of non-standard types
                with open(current_features_path, 'w', encoding='utf-8') as f:
                    json.dump(combined_features, f, indent=2, ensure_ascii=False, default=str)
                features_path = str(current_features_path)
                logger.info(f"[Thread Analysis] Combined features saved to: {features_path}")
            except Exception as save_e:
                 logger.error(f"[Thread Analysis] Error saving combined features JSON: {save_e}", exc_info=True)
                 results_dict['analysis_error'] = True
                 features_path = None 
        else:
             logger.warning("[Thread Analysis] Skipping feature save due to combination error or empty result.")

        # --- 4. Generate Feedback (only if features were generated) --- 
        if combined_features and ollama_client and not results_dict.get('analysis_error'): 
             feedback_text = generate_final_feedback(transcript_text, combined_features, model_name, ollama_client) # Now returns string
             if feedback_text:
                 try:
                     # Save feedback as text file
                     with open(current_feedback_path, 'w', encoding='utf-8') as f:
                         f.write(feedback_text) 
                     feedback_path = str(current_feedback_path)
                     logger.info(f"[Thread Analysis] Feedback saved to: {feedback_path}")
                 except Exception as save_f_e:
                      logger.error(f"[Thread Analysis] Error saving feedback text file: {save_f_e}", exc_info=True)
                      feedback_path = None # Feedback generated but couldn't save
             else:
                 logger.error("[Thread Analysis] Failed to generate LLM feedback.")
                 # feedback_path remains None
        else:
            logger.warning("[Thread Analysis] Skipping feedback generation due to missing features, client error, or previous error.")
            # feedback_path remains None

        # --- 5. Store results --- 
        results_dict['features_path'] = features_path
        results_dict['feedback_path'] = feedback_path # Path to .txt file now
        # Update error flag based on overall success 
        if not results_dict.get('analysis_error'):
             results_dict['analysis_error'] = not (features_path and feedback_path) 
        
        logger.info(f"[Thread Analysis] Finished. Error status: {results_dict.get('analysis_error')}")

    except Exception as e:
        logger.error(f"[Thread Analysis] Uncaught error in _perform_analysis: {e}", exc_info=True)
        results_dict['features_path'] = None
        results_dict['feedback_path'] = None
        results_dict['analysis_error'] = True

# --- Target Function for Vector Storage Thread --- 
def _perform_vector_storage(transcript_text: str, source_id: str, collection):
     """Encapsulates vector storage call for threading."""
     logger.info(f"[Thread VectorStore] Starting for source_id: {source_id}")
     try:
         success = store_transcript(transcript_text, source_id, collection)
         if success:
             logger.info(f"[Thread VectorStore] Successfully stored transcript for {source_id}.")
         else:
             logger.error(f"[Thread VectorStore] store_transcript function returned False for {source_id}.")
         # We might store the success status if needed, but for now just log
     except Exception as e:
         logger.error(f"[Thread VectorStore] Error during vector storage for {source_id}: {e}", exc_info=True)

# --- Main Pipeline Function (Refactored for Parallelism) ---
def run_analysis_pipeline(audio_path: str, model_name: str, output_dir: str = "analysis_output") -> Optional[Dict[str, Any]]:
    """
    Runs STT, then performs feature/feedback analysis and vector storage in parallel.
    Returns paths to transcript JSON, features JSON, and feedback TXT.
    """
    logger.info(f"Starting analysis pipeline for: {audio_path}")
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    base_name = Path(audio_path).stem

    # Result dictionary to hold paths
    final_results = {
        'transcript_path': None,
        'features_path': None,
        'feedback_path': None # Will point to .txt file
    }

    # 1. Initialize Vector Store Collection (do this early)
    collection = initialize_vector_store()
    if not collection:
        logger.error("Failed to initialize vector store collection. Cannot proceed with vector storage.")
        # Decide if pipeline should fail entirely or proceed without vector storage
        # For now, let's allow it to proceed but vector storage thread won't run effectively.
        # return None # Option: Fail early

    # 2. Run STT
    stt_result = run_stt(audio_path)
    if not stt_result or 'text' not in stt_result:
        logger.error("STT failed or returned invalid result.")
        return None
    transcript_text = stt_result['text']
    
    # Save transcript JSON
    transcript_path = output_dir_path / f"{base_name}_transcript.json"
    try:
        with open(transcript_path, 'w') as f:
            json.dump(stt_result, f, indent=4)
        final_results['transcript_path'] = str(transcript_path)
        logger.info(f"Transcript saved to: {transcript_path}")
    except Exception as e:
        logger.error(f"Failed to save transcript JSON: {e}", exc_info=True)
        return None # Fail if we can't even save the transcript

    # 3. Prepare for Parallel Tasks
    source_id = base_name # Use filename stem as unique ID for this transcript
    analysis_thread_results = {} # Dictionary to receive results from analysis thread

    # Create threads
    analysis_thread = threading.Thread(
        target=_perform_analysis,
        args=(audio_path, transcript_text, model_name, output_dir_path, analysis_thread_results)
    )
    vector_store_thread = threading.Thread(
        target=_perform_vector_storage,
        args=(transcript_text, source_id, collection) # Pass the initialized collection
    )

    # 4. Start and Join Threads
    logger.info("Starting parallel analysis and vector storage threads...")
    analysis_thread.start()
    if collection: # Only start vector thread if collection was initialized
        vector_store_thread.start()
    else:
        logger.warning("Skipping vector storage thread due to initialization failure.")

    analysis_thread.join() # Wait for analysis thread to finish
    logger.info("Analysis thread finished.")
    if collection:
        vector_store_thread.join() # Wait for vector thread to finish
        logger.info("Vector storage thread finished.")

    logger.info("Both threads completed.")

    # 5. Consolidate Results
    final_results['features_path'] = analysis_thread_results.get('features_path')
    final_results['feedback_path'] = analysis_thread_results.get('feedback_path') # Now gets .txt path

    # Check if analysis thread reported an error
    if analysis_thread_results.get('analysis_error', True): # Default to error if key missing
         logger.error("Analysis thread reported an error. Check logs above.")
         # Decide if partial results are acceptable or return None
         # Returning partial results for now, API layer might decide to error out
         # return None 

    # Check if essential results are missing (even if no explicit error was logged)
    if not final_results['features_path']:
         logger.warning("Features path is missing after analysis thread completion.")
         # Feedback path might also be missing if feedback generation failed

    logger.info(f"Pipeline finished. Returning results: {final_results}")
    return final_results

# Example Usage (can be called from main.py's 'infer' command)
if __name__ == '__main__':
    # Create a dummy audio file path for testing
    # You'll need an actual audio file (e.g., sample.wav) in the root or specify a full path
    test_audio = "sample.wav" # Replace with your audio file
    ollama_model = "gemma2:9b" # Or your preferred model

    if not Path(test_audio).exists():
         logger.error(f"Test audio file not found: {test_audio}")
         logger.error("Please place a sample audio file or update the path in the script.")
    else:
        logger.info("Running example analysis pipeline...")
        feedback = run_analysis_pipeline(test_audio, ollama_model)
        
        if feedback:
            print("\n--- Pipeline Completed --- Final Feedback:")
            print(json.dumps(feedback, indent=2))
        else:
            print("\n--- Pipeline Failed ---") 
            print("\n--- Pipeline Failed ---") 