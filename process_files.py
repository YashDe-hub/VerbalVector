import os
from pathlib import Path
import logging
import sys
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure the src directory is in the Python path
# This allows importing from src.*
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

try:
    from src.pipelines.analysis_pipeline import run_analysis_pipeline
except ImportError as e:
    logger.error(f"Failed to import analysis pipeline: {e}. Make sure the script is run from the project root and the src directory exists.")
    sys.exit(1)

# --- Configuration ---
# Directory where the raw audio files are located
RAW_DATA_DIR = project_root / "data" / "raw"

# Directory where the analysis output (transcript, features, feedback JSONs) will be saved
OUTPUT_DIR = project_root / "analysis_output"

# Name of the Ollama model to use for feedback generation
OLLAMA_MODEL_NAME = "gemma2:9b"
# --- End Configuration ---

def main(filename: str):
    logger.info(f"--- Starting processing for file: {filename} ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

    audio_path = RAW_DATA_DIR / filename

    if not audio_path.is_file():
        logger.error(f"Audio file not found: {audio_path}. Exiting.")
        sys.exit(1)

    try:
        # Run the full pipeline for the current audio file
        result_paths = run_analysis_pipeline(
            audio_path=str(audio_path),
            model_name=OLLAMA_MODEL_NAME,
            output_dir=str(OUTPUT_DIR)
        )

        if result_paths:
            logger.info(f"Successfully processed {filename}.")
            logger.info(f"  Transcript: {result_paths.get('transcript_path')}")
            logger.info(f"  Features:   {result_paths.get('features_path')}")
            logger.info(f"  Feedback:   {result_paths.get('feedback_path')}")
            logger.info(f"  Vector DB source_id: {result_paths.get('source_id')}")
        else:
            logger.error(f"Pipeline returned no results for {filename}. It might have failed internally. Check logs.")

    except Exception as e:
        logger.error(f"An unexpected error occurred while processing {filename}: {e}", exc_info=True)

    logger.info(f"--- Finished processing file: {filename} ---")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a single audio file through the analysis pipeline.")
    parser.add_argument("-f", "--file", required=True, help="Filename of the audio file in the data/raw directory to process.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call main function with the provided filename
    main(filename=args.file) 