#!/usr/bin/env python3
"""
VerbalVector - Model Distillation for Presentation Feedback

This script demonstrates the full workflow of the VerbalVector project using local models:
1. Download audio & transcripts from YouTube (optional)
2. Feature extraction from audio recordings (using downloaded transcripts if available)
3. Teacher model (Local Gemma 2 9B via Ollama) feedback generation
4. Student model (Local Gemma 2 2B) training through distillation
5. Student model evaluation against the local teacher
6. Inference using the distilled student model
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
import ollama
import time
import tempfile

try:
    import sounddevice as sd
    import soundfile as sf
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    logger.debug("sounddevice library not found. Live recording will be unavailable.")
except Exception as e: # Catch other potential sounddevice import errors (e.g., PortAudio issues)
    SOUNDDEVICE_AVAILABLE = False
    logger.warning(f"Failed to import or initialize sounddevice: {e}. Live recording disabled.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables (if any, though not for API keys now)
# load_dotenv() # Commented out or removed

# Import VerbalVector modules with Defaults
try:
    from src.data.data_collector import download_batch_youtube
    from src.features.feature_combiner import process_presentation_data
    # from src.models.teacher_model import batch_generate_feedback, DEFAULT_TEACHER_MODEL # REMOVED
    # from src.models.student_model import train_student_model, StudentModel, DEFAULT_STUDENT_MODEL # REMOVED
    # from src.evaluation.model_evaluation import evaluate_model # REMOVED
    # Import the new pipeline function
    from src.pipelines.analysis_pipeline import run_analysis_pipeline, generate_final_feedback
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Please make sure you've installed all requirements and are running from the project root.")
    sys.exit(1)

# Import vector store functions for search command
try:
    from src.vector_store.manager import initialize_vector_store, search_transcripts
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False
    logger.warning("Vector store manager not found. Search functionality will be unavailable.")


def check_ollama_connection(model_name: str):
    """Check connection to Ollama and if the model exists."""
    try:
        client = ollama.Client()
        client.list() # Test connection
        logger.info("Successfully connected to local Ollama server.")

        # Check if the required model is pulled (More robust parsing)
        ollama_list_response = client.list()
        models_list = ollama_list_response.get('models', []) # Default to empty list if 'models' key missing
        # Safely get 'name' if item is a dict and has the key, filter out None/empty names
        logger.debug(f"Raw Ollama list() response: {ollama_list_response}") # DEBUG: Log raw response
        logger.debug(f"Parsed 'models' list: {models_list}") # DEBUG: Log parsed list
        # Correctly access the model name using attribute access (m.model)
        available_models = [m.model for m in models_list if hasattr(m, 'model') and m.model]

        if not available_models:
            logger.warning("Could not retrieve any available models from Ollama response.")
            # Optionally log the raw response for debugging:
            # logger.debug(f"Raw Ollama list response: {ollama_list_response}") # Already added above
            return False

        # Check for base model name (e.g., gemma2:9b from gemma2:9b-instruct-q4_0)
        base_model_name = model_name.split(':')[0]
        if not any(m.startswith(base_model_name) for m in available_models):
            logger.warning(f"Model '{model_name}' (or variant) not found in Ollama.")
            logger.warning(f"Please run: ollama pull {model_name}")
            return False
        logger.info(f"Ollama model '{model_name}' seems available.")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Ollama or list models: {e}")
        logger.error("Please ensure the Ollama application is running locally.")
        return False


def setup_directories():
    """Set up project directories if they don't exist."""
    directories = [
        "data/raw",
        "data/processed",
        "data/feedback",
        "models/distilled_feedback",
        "evaluation_results" # Added for evaluation output
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        # logger.info(f"Ensured directory exists: {directory}") # Less verbose
    logger.info("Project directories ensured.")
    return True


def download_data(args):
    """Download audio and transcripts from YouTube URLs."""
    if not args.urls:
        logger.error("No YouTube URLs provided. Use the --urls argument.")
        return False

    urls_list = [url.strip() for url in args.urls.split(',') if url.strip()]
    if not urls_list:
        logger.error("No valid URLs found after splitting and stripping the --urls argument.")
        return False

    logger.info(f"Starting download for {len(urls_list)} URLs into '{args.output_dir}'")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    try:
        results = download_batch_youtube(urls=urls_list, output_dir=args.output_dir)
        # Log summary
        success_count = sum(1 for res in results.values() if res.get('audio') or res.get('transcript'))
        logger.info(f"Download process finished. Check '{args.output_dir}' for files. Successfully processed {success_count}/{len(urls_list)} URLs (some might have only audio or transcript).")
        return True # Indicate command executed, success depends on individual downloads
    except Exception as e:
        logger.error(f"An error occurred during batch download: {e}", exc_info=True)
        return False


def extract_features(args):
    """Extract features from audio files, using transcripts if available."""
    input_dir = Path(args.input_dir) # Changed arg name for clarity
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        return False

    # Find all potential audio files
    try:
        all_files = os.listdir(input_dir)
        audio_files = [f for f in all_files if f.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac', '.mp4'))]
        if not audio_files:
            logger.warning(f"No supported audio files found in {input_dir}")
            return False
    except OSError as e:
        logger.error(f"Error reading input directory {input_dir}: {e}")
        return False

    logger.info(f"Found {len(audio_files)} potential audio files in '{input_dir}'. Extracting features to '{output_dir}'...")
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    error_count = 0
    skipped_count = 0

    for i, audio_filename in enumerate(audio_files):
        audio_path = input_dir / audio_filename
        base_name = audio_path.stem
        # Expected transcript name based on data_collector output
        transcript_json_path = input_dir / f"{base_name}_transcript.json"
        output_features_path = output_dir / f"{base_name}_features.json"

        logger.info(f"Processing ({i+1}/{len(audio_files)}): {audio_filename}")

        # Optional: Skip if features already exist
        if output_features_path.exists():
             logger.info(f"Feature file {output_features_path.name} already exists. Skipping.")
             skipped_count += 1
             continue

        # Check if transcript exists
        transcript_path_to_pass = None
        if transcript_json_path.is_file():
            logger.info(f"  Found transcript: {transcript_json_path.name}")
            transcript_path_to_pass = str(transcript_json_path)
        else:
            logger.warning(f"  Transcript file not found: {transcript_json_path.name}. Proceeding with audio features only.")

        try:
            # Call the updated process_presentation_data function
            features = process_presentation_data(
                audio_path=str(audio_path),
                transcript_json_path=transcript_path_to_pass,
                output_path=str(output_features_path)
            )

            if features:
                success_count += 1
            else:
                # Error logged within process_presentation_data
                error_count += 1

        except Exception as e:
            logger.error(f"Unhandled error processing {audio_filename}: {e}", exc_info=True)
            error_count += 1

    logger.info(f"Feature extraction complete. Success: {success_count}, Errors: {error_count}, Skipped: {skipped_count}")
    return error_count == 0


def record_audio(duration_seconds: int = 0, samplerate: int = 44100) -> str:
    """Records audio from the default microphone.

    Args:
        duration_seconds: Recording duration in seconds. If 0 or less, records until Enter is pressed.
        samplerate: The sample rate for the recording.

    Returns:
        The path to the temporary WAV file containing the recording.
        Returns None if recording fails or sounddevice is unavailable.
    """
    if not SOUNDDEVICE_AVAILABLE:
        logger.error("sounddevice library is not available. Cannot perform live recording.")
        return None

    try:
        # Create a temporary file to store the recording
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_filename = temp_file.name
        temp_file.close() # Close the file handle immediately, soundfile will reopen
        logger.info(f"Temporary recording file: {temp_filename}")

        if duration_seconds > 0:
            logger.info(f"Starting recording for {duration_seconds} seconds...")
            recording = sd.rec(int(duration_seconds * samplerate), samplerate=samplerate, channels=1, dtype='float32')
            sd.wait() # Wait until recording is finished
            logger.info("Recording finished.")
        else:
            logger.info("Starting recording... Press Enter to stop.")
            # Use InputStream for indefinite recording until stopped
            with sf.SoundFile(temp_filename, mode='wb', samplerate=samplerate, channels=1) as file:
                with sd.InputStream(samplerate=samplerate, channels=1, callback=lambda indata, frames, time, status: file.write(indata)):
                    input("  (Press Enter to stop recording)\n") # Wait for user input
            logger.info("Recording stopped.")
            # If stopped manually, the file is already written, return the name
            return temp_filename

        # Save the recording (only if duration > 0)
        sf.write(temp_filename, recording, samplerate)
        logger.info(f"Recording saved to temporary file: {temp_filename}")
        return temp_filename

    except Exception as e:
        logger.error(f"Error during audio recording: {e}", exc_info=True)
        if 'temp_filename' in locals() and os.path.exists(temp_filename):
            os.remove(temp_filename) # Clean up incomplete file
        return None


def run_inference(args):
    """Runs the analysis pipeline on audio or existing features/transcript."""
    feedback = None
    start_time = time.time()
    temp_audio_path = None # To store path of live recording

    try:
        # Option 0: Check Ollama connection first, applies to all modes
        if not check_ollama_connection(args.model_name):
            logger.error(f"Cannot run inference without connection to Ollama model '{args.model_name}'.")
            return False

        # Determine the audio source
        audio_path_to_process = None
        if args.live:
            logger.info("Attempting live recording...")
            temp_audio_path = record_audio(duration_seconds=args.duration)
            if not temp_audio_path:
                logger.error("Live recording failed. Cannot proceed.")
                return False
            audio_path_to_process = temp_audio_path
            base_name = f"live_recording_{int(time.time())}" # Create a unique base name

        elif args.audio_path:
            logger.info(f"Using provided audio file: {args.audio_path}")
            if not Path(args.audio_path).is_file():
                logger.error(f"Audio file not found: {args.audio_path}")
                return False
            audio_path_to_process = args.audio_path
            base_name = Path(audio_path_to_process).stem

        elif args.feature_path:
            # This mode doesn't use audio directly, handle separately
            logger.info(f"Generating feedback from existing feature file: {args.feature_path}")
            if not args.transcript_path:
                logger.error("If using --feature-path, you must also provide --transcript-path.")
                return False
            
            feature_p = Path(args.feature_path)
            transcript_p = Path(args.transcript_path)

            if not feature_p.is_file():
                logger.error(f"Feature file not found: {args.feature_path}")
                return False
            if not transcript_p.is_file():
                logger.error(f"Transcript file not found: {args.transcript_path}")
                return False
                
            try:
                # Load features
                with open(feature_p, 'r', encoding='utf-8') as f:
                    features = json.load(f)
                # Load transcript text
                with open(transcript_p, 'r', encoding='utf-8') as f:
                    # Assuming transcript JSON has a 'text' field like Whisper output
                    transcript_data = json.load(f)
                    transcript = transcript_data.get('text')
                    if not transcript:
                         logger.warning(f"Could not find 'text' key in transcript file: {transcript_p}. Using raw content.")
                         # Attempt to read raw content if 'text' key is missing
                         transcript_p.seek(0)
                         transcript = transcript_p.read()
                
                # We need an ollama client here too
                ollama_client = ollama.Client()
                feedback = generate_final_feedback(transcript, features, args.model_name, ollama_client)
            except Exception as e:
                logger.error(f"Error generating feedback from features/transcript: {e}", exc_info=True)
                return False

            # If feedback generated from features, print and exit successfully
            if feedback:
                end_time = time.time()
                logger.info(f"Inference complete in {end_time - start_time:.2f} seconds.")
                print("\n--- Presentation Feedback ---")
                # Print formatted feedback (adjust as needed based on final structure)
                try:
                    scores = feedback.get("scores", {})
                    overall = scores.get("overall", {}).get("score", "N/A")
                    print(f"Overall Score: {overall}/10")
                    # Add more detailed printing if desired
                    # print(json.dumps(feedback, indent=2)) # For full JSON
                except Exception as print_e:
                     logger.error(f"Error formatting feedback for printing: {print_e}")
                     print(json.dumps(feedback, indent=2)) # Fallback to raw JSON
                print("    ----------------------------------------------")
                return True
            else:
                 logger.error("Feedback generation failed.")
                 return False
        
        # If we are processing audio (live or file)
        if audio_path_to_process:
            logger.info(f"Running full analysis pipeline for: {audio_path_to_process}")
            try:
                # Specify output directory for intermediate files from pipeline
                output_dir = f"analysis_output/{base_name}"
                feedback = run_analysis_pipeline(
                    audio_path=audio_path_to_process,
                    model_name=args.model_name,
                    output_dir=output_dir
                )
            except Exception as e:
                logger.error(f"An error occurred during analysis pipeline execution: {e}", exc_info=True)
                return False
        else:
             # This case should not be reached if args are mutually exclusive
             logger.error("No valid input mode specified (--live, --audio-path, or --feature-path).")
             return False

        # If feedback generated from pipeline
        if feedback:
            end_time = time.time()
            logger.info(f"Inference complete in {end_time - start_time:.2f} seconds.")
            print("\n--- Presentation Feedback ---")
            # Print formatted feedback (adjust as needed based on final structure)
            try:
                scores = feedback.get("scores", {})
                overall = scores.get("overall", {}).get("score", "N/A")
                print(f"Overall Score: {overall}/10")
                # Add more detailed printing if desired
                # print(json.dumps(feedback, indent=2)) # For full JSON
            except Exception as print_e:
                logger.error(f"Error formatting feedback for printing: {print_e}")
                print(json.dumps(feedback, indent=2)) # Fallback to raw JSON
            print("    ----------------------------------------------")
            return True
        else:
            logger.error("Feedback generation failed.")
            return False
            
    finally:
        # Clean up temporary recording file if it exists
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                logger.info(f"Cleaned up temporary file: {temp_audio_path}")
            except OSError as e:
                logger.error(f"Error removing temporary file {temp_audio_path}: {e}")


# --- Search Command Function ---
def run_search(args):
    """Handles the search command: retrieves context and generates LLM answer."""
    if not VECTOR_STORE_AVAILABLE:
        logger.error("Vector store components are not available. Cannot perform search.")
        return False
        
    logger.info(f"Received search query: '{args.query}'")
    start_time = time.time()
    
    try:
        # 1. Initialize Vector Store
        logger.info("Initializing vector store for search...")
        collection = initialize_vector_store()
        if not collection:
            logger.error("Failed to initialize vector store collection. Cannot search.")
            return False
            
        # 2. Retrieve Relevant Chunks
        retrieved_chunks = search_transcripts(
            query=args.query,
            collection=collection,
            n_results=args.n_results
        )
        
        if not retrieved_chunks:
            logger.warning("No relevant information found in stored transcripts for your query.")
            print("\nSorry, I couldn't find relevant information in the stored transcripts to answer that query.")
            return True # Command executed, but no results found
            
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for context.")
        context_string = "\n---\n".join(retrieved_chunks)
        # logger.debug(f"Context string:\n---\n{context_string}\n---") # Optional: Log full context
        
        # 3. Check Ollama Connection for Generator Model
        logger.info(f"Checking Ollama connection for generator model: {args.model_name}")
        if not check_ollama_connection(args.model_name):
             logger.error(f"Cannot generate answer without connection to Ollama model '{args.model_name}'.")
             return False
             
        # 4. Construct Augmented Prompt
        prompt = f"""Based solely on the following context provided from presentation transcripts, please answer the user's question. Focus only on information present in the context. If the context does not contain the answer, clearly state that the context does not provide sufficient information.

Context:
---
{context_string}
---

User Question: {args.query}

Answer:"""
        # logger.debug(f"LLM Prompt:\n{prompt}") # Optional: Log prompt

        # 5. Call Ollama LLM for Generation
        logger.info(f"Sending query and context to LLM ({args.model_name}) for answer generation...")
        client = ollama.Client()
        response = client.chat(
            model=args.model_name,
            messages=[{'role': 'user', 'content': prompt}]
            # stream=False # Default is false, ensure we get the full response
        )
        answer = response.get('message', {}).get('content', '').strip()
        
        end_time = time.time()
        logger.info(f"Search and answer generation complete in {end_time - start_time:.2f} seconds.")
        
        # 6. Display Answer
        print("\n--- Answer --- ")
        if answer:
            print(answer)
        else:
            print("(LLM did not provide an answer.)")
        print("---------------")
        
        return True
        
    except Exception as e:
        logger.error(f"An error occurred during the search process: {e}", exc_info=True)
        return False


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="VerbalVector - Local Model Distillation for Presentation Feedback",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run', required=True)

    # Setup command
    parser_setup = subparsers.add_parser('setup', help='Set up project directories')
    # No arguments needed for setup
    parser_setup.set_defaults(func=lambda args: setup_directories()) # Added func call

    # Download command NEW
    download_parser = subparsers.add_parser('download', help='Download audio & transcripts from YouTube URLs')
    download_parser.add_argument('--urls', type=str, required=True,
                               help='Comma-separated list of YouTube video URLs to download')
    download_parser.add_argument('--output-dir', type=str, default='data/raw',
                               help='Directory to save downloaded audio and transcript files')
    download_parser.set_defaults(func=download_data)

    # Extract features command
    extract_parser = subparsers.add_parser('extract', help='Extract features from audio/transcript pairs')
    extract_parser.add_argument('--input-dir', type=str, default='data/raw', # Changed from audio-dir
                               help='Directory containing audio and transcript JSON files')
    extract_parser.add_argument('--output-dir', type=str, default='data/processed',
                               help='Directory to save feature files')
    extract_parser.set_defaults(func=extract_features)

    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run the full analysis pipeline on an audio file or feature file.')
    # Group for mutually exclusive inputs
    input_group_infer = infer_parser.add_mutually_exclusive_group(required=True)
    input_group_infer.add_argument('--audio-path', help='Path to the audio file to analyze')
    input_group_infer.add_argument('--feature-path', help='Path to pre-computed feature JSON file')
    input_group_infer.add_argument('--live', action='store_true', help='Record audio live from microphone')

    infer_parser.add_argument('--transcript-path', help='Path to transcript JSON file (required if using --feature-path)')
    infer_parser.add_argument('--duration', type=int, default=0, help='Duration for live recording in seconds (0 for manual stop via Enter)')
    infer_parser.add_argument('--model-name', type=str, default="gemma2:9b", help='LLM model name for analysis (e.g., Ollama model).')
    # infer_parser.add_argument('--output-path', type=str, default=None, help='Optional path/directory to save output feedback JSON file') # Remove? Output is now in analysis_output
    infer_parser.set_defaults(func=run_inference)

    # --- Search Command (NEW) ---
    search_parser = subparsers.add_parser('search', help='Search stored transcripts and ask the LLM to answer based on context.')
    search_parser.add_argument('-q', '--query', type=str, required=True, help='Your question about the presentation content.')
    search_parser.add_argument('-n', '--n-results', type=int, default=5, help='Number of relevant transcript chunks to retrieve for context.')
    search_parser.add_argument('--model-name', type=str, default="gemma2:9b", help='LLM model name to generate the answer (e.g., Ollama model).')
    search_parser.set_defaults(func=run_search)

    # Parse arguments
    args = parser.parse_args()

    # Run the appropriate command
    logger.info(f"Running command: {args.command}")
    success = False
    if args.command == 'setup':
        success = args.func(args)
    elif args.command == 'download':
        success = args.func(args)
    elif args.command == 'extract':
        success = args.func(args)
    elif args.command == 'infer':
        # Check for valid input for infer command (live, audio, or feature path)
        if not args.live and not args.audio_path and not args.feature_path:
             logger.error("For the 'infer' command, you must provide one of: --live, --audio-path, or --feature-path.")
             infer_parser.print_help() # Show help specific to infer
             sys.exit(1)
             
        success = args.func(args) # Calls run_inference
    elif args.command == 'search': # Add handler for search
         if not VECTOR_STORE_AVAILABLE:
              logger.error("Search command requires vector store components. Please check installation.")
              sys.exit(1)
         success = args.func(args) # Calls run_search
    else:
        parser.print_help()
        sys.exit(1)

    if success:
        logger.info(f"Command '{args.command}' completed successfully.")
        sys.exit(0)
    else:
        logger.error(f"Command '{args.command}' failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main() 