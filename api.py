import asyncio
import json
import logging
import os
import uuid

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config  # loads .env and exposes API keys / paths
from src.pipelines.analysis_pipeline import run_analysis_pipeline

config.validate()

app = FastAPI(title="VerbalVector API", version="1.0.0")

ALLOWED_ORIGINS = config.CORS_ALLOWED_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = str(config.UPLOAD_DIR)
ANALYSIS_OUTPUT_FOLDER = str(config.OUTPUT_DIR)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}
MAX_FILE_SIZE_BYTES = config.MAX_FILE_SIZE_BYTES


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)


def read_file(file_path: str | None, parser=None):
    if not file_path:
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            return parser(content) if parser else content
    except FileNotFoundError:
        logger.error("Result file not found (id=%s)", os.path.basename(file_path))
        return None
    except Exception as e:
        logger.error("Error reading result file (id=%s): %s", os.path.basename(file_path), e)
        return None


@app.get("/")
async def home():
    return {"status": "VerbalVector Backend is running!"}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}")

    # Sanitize: strip any path components, prefix with UUID to prevent collisions
    safe_name = f"{uuid.uuid4().hex}_{os.path.basename(file.filename)}"
    filepath = os.path.join(UPLOAD_FOLDER, safe_name)
    logger.info("Receiving upload (id=%s)", safe_name[:8])

    try:
        total_bytes = 0
        with open(filepath, "wb") as f:
            while chunk := await file.read(8192):
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_SIZE_BYTES // 1024 // 1024} MB limit")
                f.write(chunk)
        logger.info("File saved (id=%s, size=%d bytes)", safe_name[:8], total_bytes)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error saving upload (id=%s): %s", safe_name[:8], e, exc_info=True)
        raise HTTPException(status_code=500, detail="Could not save uploaded file.")

    logger.info("Starting analysis pipeline (id=%s)", safe_name[:8])

    try:
        # Run the blocking pipeline in a thread pool so the event loop stays free
        analysis_results = await asyncio.to_thread(
            run_analysis_pipeline,
            audio_path=filepath,
            output_dir=ANALYSIS_OUTPUT_FOLDER,
        )

        if analysis_results is None:
            logger.error("Analysis pipeline returned None (id=%s)", safe_name[:8])
            raise HTTPException(status_code=500, detail="Analysis failed. Check backend logs.")

        transcript_path = analysis_results.get("transcript_path")
        features_path = analysis_results.get("features_path")
        feedback_path = analysis_results.get("feedback_path")

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
            logger.error("Failed to read result files (id=%s): %d files", safe_name[:8], len(failed))
            raise HTTPException(
                status_code=500,
                detail="Analysis completed but failed to read result files.",
            )

        logger.info("Analysis complete (id=%s)", safe_name[:8])
        return {
            "message": f"File '{file.filename}' processed successfully.",
            "transcript": transcript_content,
            "features": features_content,
            "feedback": feedback_content,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error (id=%s): %s", safe_name[:8], e, exc_info=True)
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
    uvicorn.run("api:app", host=config.API_HOST, port=config.API_PORT, reload=config.API_RELOAD)
