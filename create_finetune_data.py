import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
ANALYSIS_OUTPUT_DIR = Path("./analysis_output")
FINETUNE_OUTPUT_FILE = Path("./finetune_dataset.jsonl")

# Instruction for the fine-tuning model
# Adjust this if you want to guide the fine-tuned model differently
INSTRUCTION = "Analyze the provided presentation transcript and features. Generate structured feedback highlighting strengths and areas for improvement regarding content, delivery, clarity, and engagement, referencing specific evidence from the inputs."
# --- End Configuration ---

def create_dataset_entry(transcript_text: str, features_data: dict, feedback_text: str) -> dict:
    """Creates a single data entry dictionary for the fine-tuning dataset."""
    try:
        # Serialize features data back to a JSON string for the input
        features_json_string = json.dumps(features_data, indent=2, ensure_ascii=False)
        
        input_text = f"""Transcript:
```text
{transcript_text}
```

Features:
```json
{features_json_string}
```"""
        
        return {
            "instruction": INSTRUCTION,
            "input": input_text,
            "output": feedback_text.strip() # Use the raw feedback text as output
        }
    except Exception as e:
        logger.error(f"Error creating dataset entry: {e}", exc_info=True)
        return None

def main():
    logger.info(f"Starting fine-tuning dataset creation.")
    logger.info(f"Looking for analysis files in: {ANALYSIS_OUTPUT_DIR.resolve()}")
    
    processed_count = 0
    skipped_count = 0
    
    # Clear the output file if it exists, or create it
    with open(FINETUNE_OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        pass # Just creating/clearing the file

    # Find all feature files as a starting point
    feature_files = list(ANALYSIS_OUTPUT_DIR.glob("*_features.json"))
    if not feature_files:
         logger.warning(f"No *_features.json files found in {ANALYSIS_OUTPUT_DIR}. Cannot create dataset.")
         return

    logger.info(f"Found {len(feature_files)} potential samples based on feature files.")

    for features_path in feature_files:
        base_name = features_path.name.replace("_features.json", "")
        logger.info(f"Processing sample: {base_name}")
        
        transcript_path = ANALYSIS_OUTPUT_DIR / f"{base_name}_transcript.json"
        feedback_path = ANALYSIS_OUTPUT_DIR / f"{base_name}_feedback.txt"
        
        # Check if all corresponding files exist
        if not transcript_path.is_file():
            logger.warning(f"Missing transcript file: {transcript_path}. Skipping sample '{base_name}'.")
            skipped_count += 1
            continue
        if not feedback_path.is_file():
            logger.warning(f"Missing feedback file: {feedback_path}. Skipping sample '{base_name}'.")
            skipped_count += 1
            continue
            
        # Read content
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
                # Assuming Whisper format with a top-level 'text' key
                transcript_text = transcript_data.get('text')
                if transcript_text is None:
                     logger.error(f"'text' key not found in {transcript_path}. Skipping sample '{base_name}'.")
                     skipped_count += 1
                     continue
                     
            with open(features_path, 'r', encoding='utf-8') as f:
                features_data = json.load(f)
                
            with open(feedback_path, 'r', encoding='utf-8') as f:
                feedback_text = f.read()
                
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON in file related to '{base_name}': {e}. Skipping sample.")
            skipped_count += 1
            continue
        except IOError as e:
             logger.error(f"Error reading file related to '{base_name}': {e}. Skipping sample.")
             skipped_count += 1
             continue
        except Exception as e:
             logger.error(f"Unexpected error processing files for '{base_name}': {e}", exc_info=True)
             skipped_count += 1
             continue

        # Create dataset entry dictionary
        entry = create_dataset_entry(transcript_text, features_data, feedback_text)
        
        if entry:
            # Append entry as a JSON line to the output file
            try:
                with open(FINETUNE_OUTPUT_FILE, 'a', encoding='utf-8') as outfile:
                    json.dump(entry, outfile, ensure_ascii=False)
                    outfile.write('\n') # Write each JSON object on a new line
                processed_count += 1
                logger.info(f"Successfully added sample '{base_name}' to dataset.")
            except IOError as e:
                 logger.error(f"Error writing entry for '{base_name}' to {FINETUNE_OUTPUT_FILE}: {e}")
                 # Optionally handle partial writes or stop? For now, just log.
                 skipped_count += 1
        else:
             logger.error(f"Failed to create dataset entry for '{base_name}'. Skipping.")
             skipped_count += 1

    logger.info("--- Dataset Creation Summary ---")
    logger.info(f"Output file: {FINETUNE_OUTPUT_FILE.resolve()}")
    logger.info(f"Entries successfully processed: {processed_count}")
    logger.info(f"Entries skipped due to errors/missing files: {skipped_count}")
    logger.info("---------------------------------")

if __name__ == "__main__":
    main() 