from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json # Added for loading result JSONs
import logging # Added for logging

# Import the pipeline function
from src.pipelines.analysis_pipeline import run_analysis_pipeline

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Setup logging for Flask app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the directory to store uploaded files and generated JSONs
# These might be superseded by paths used within the pipeline, adjust if needed
UPLOAD_FOLDER = 'data/uploads'
ANALYSIS_OUTPUT_FOLDER = 'analysis_output' # Directory used by the pipeline
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANALYSIS_OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return "VerbalVector Backend is running!"

# Function to safely read JSON content
def read_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Result JSON not found: {file_path}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None

# Function to safely read plain text content
def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Result text file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading text file {file_path}: {e}")
        return None

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logger.warning("Upload attempt with no file part")
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        logger.warning("Upload attempt with no selected file")
        return jsonify({'error': 'No selected file'}), 400

    # Ensure filename is safe (optional but recommended)
    # filename = secure_filename(file.filename)
    filename = file.filename # Using original for now
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    logger.info(f"Receiving file: {filename}")
    try:
        file.save(filepath)
        logger.info(f"File saved to: {filepath}")
    except Exception as e:
        logger.error(f"Error saving file {filepath}: {e}", exc_info=True)
        return jsonify({'error': 'Could not save uploaded file.'}), 500

    # --- Call the Analysis Pipeline --- 
    # Define the LLM model to use (Ensure this is available in Ollama)
    llm_model_name = "verbalvector-llama3-lora:latest" # Or choose another model like gemma2:latest, gemma:2b
    logger.info(f"Starting analysis pipeline for {filepath} using model {llm_model_name}")

    try:
        # The pipeline function expects the output directory where it will save its results
        analysis_results = run_analysis_pipeline(
            audio_path=filepath,
            model_name=llm_model_name,
            output_dir=ANALYSIS_OUTPUT_FOLDER
        )

        if analysis_results is None:
            logger.error("Analysis pipeline returned None, indicating an error.")
            return jsonify({'error': 'Analysis failed. Check backend logs for details.'}), 500

        logger.info(f"Analysis pipeline completed. Result dictionary: {analysis_results}")

        # --- Read the content of the result files --- 
        transcript_path = analysis_results.get('transcript_path') # Get path first
        features_path = analysis_results.get('features_path')
        feedback_path = analysis_results.get('feedback_path')
        logger.info(f"Attempting to read transcript from: {transcript_path}") # Log paths
        logger.info(f"Attempting to read features from: {features_path}")
        logger.info(f"Attempting to read feedback from: {feedback_path}")

        transcript_content = read_json_file(transcript_path)
        features_content = read_json_file(features_path)
        # Use the new text reader for the feedback file
        feedback_content = read_text_file(feedback_path)

        # Check if any file reading failed
        if transcript_content is None or features_content is None or feedback_content is None:
             # Be more specific about which file failed
             failed_files = []
             if transcript_content is None: failed_files.append(transcript_path)
             if features_content is None: failed_files.append(features_path)
             if feedback_content is None: failed_files.append(feedback_path)
             logger.error(f"Failed to read one or more result files: {failed_files}")
             return jsonify({'error': f'Analysis completed, but failed to read result files: {failed_files}'}), 500

        logger.info("Successfully read transcript, features JSON, and feedback text content.")
        # Return the actual content of the analysis
        return jsonify({
            'message': f'File {filename} processed successfully.',
            'transcript': transcript_content,
            'features': features_content,
            'feedback': feedback_content
        }), 200

    except Exception as e:
        logger.error(f"An unexpected error occurred during analysis for {filepath}: {e}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred during analysis.'}), 500

# Placeholder for transcript query endpoint
@app.route('/api/query', methods=['POST'])
def query_transcript():
    data = request.get_json()
    query = data.get('query')
    # TODO: Add logic to load the relevant transcript/vectorDB
    # TODO: Perform RAG query using the user's query
    # TODO: Return the LLM response

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    # Placeholder response
    return jsonify({
        'query': query,
        'answer': 'This is a placeholder answer from the RAG system.'
    }), 200


if __name__ == '__main__':
    # Note: host='0.0.0.0' makes the server accessible from your network,
    # which is useful for testing from your phone or another device.
    # Remove it if you only want to access it from your local machine.
    # debug=True enables auto-reloading and provides detailed error pages.
    # DO NOT use debug=True in a production environment.
    app.run(host='0.0.0.0', port=5002, debug=True) # Using port 5002 to avoid conflicts 