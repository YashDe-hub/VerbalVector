import asyncio
import json
import logging
import os

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config  # loads .env and exposes API keys / paths
from src.pipelines.analysis_pipeline import run_analysis_pipeline

# Validate that all required API keys are present before starting
config.validate()

app = FastAPI(title="VerbalVector API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = str(config.UPLOAD_DIR)
ANALYSIS_OUTPUT_FOLDER = str(config.OUTPUT_DIR)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANALYSIS_OUTPUT_FOLDER, exist_ok=True)


class QueryRequest(BaseModel):
    query: str


def read_file(file_path: str | None, parser=None):
    if not file_path:
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            return parser(content) if parser else content
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None


@app.get("/")
async def home():
    return {"status": "VerbalVector Backend is running!"}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    logger.info(f"Receiving file: {file.filename}")

    try:
        with open(filepath, "wb") as f:
            while chunk := await file.read(8192):
                f.write(chunk)
        logger.info(f"File saved to: {filepath}")
    except Exception as e:
        logger.error(f"Error saving file {filepath}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not save uploaded file.")

    logger.info(f"Starting analysis pipeline for {filepath}")

    try:
        # Run the blocking pipeline in a thread pool so the event loop stays free
        analysis_results = await asyncio.to_thread(
            run_analysis_pipeline,
            audio_path=filepath,
            output_dir=ANALYSIS_OUTPUT_FOLDER,
        )

        if analysis_results is None:
            logger.error("Analysis pipeline returned None.")
            raise HTTPException(status_code=500, detail="Analysis failed. Check backend logs.")

        logger.info(f"Analysis pipeline completed: {analysis_results}")

        transcript_path = analysis_results.get("transcript_path")
        features_path = analysis_results.get("features_path")
        feedback_path = analysis_results.get("feedback_path")

        logger.info(f"Reading results: {transcript_path}, {features_path}, {feedback_path}")

        transcript_content = read_file(transcript_path, json.loads)
        features_content = read_file(features_path, json.loads)
        feedback_content = read_file(feedback_path)

        failed = [
            p for p, c in [
                (transcript_path, transcript_content),
                (features_path, features_content),
                (feedback_path, feedback_content),
            ]
            if c is None
        ]
        if failed:
            logger.error(f"Failed to read result files: {failed}")
            raise HTTPException(
                status_code=500,
                detail=f"Analysis completed but failed to read result files: {failed}",
            )

        logger.info("Successfully read all result files.")
        return {
            "message": f"File {file.filename} processed successfully.",
            "transcript": transcript_content,
            "features": features_content,
            "feedback": feedback_content,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during analysis for {filepath}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during analysis.")


@app.post("/api/query")
async def query_transcript(body: QueryRequest):
    # TODO: load relevant transcript/vectorDB
    # TODO: perform RAG query using body.query
    # TODO: return LLM response
    return {
        "query": body.query,
        "answer": "This is a placeholder answer from the RAG system.",
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=5002, reload=True)
