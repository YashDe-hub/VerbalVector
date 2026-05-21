import asyncio
import json
import logging
import os
import uuid

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config  # loads .env and exposes API keys / paths
from src.pipelines.analysis_pipeline import run_analysis_pipeline
from src.vector_store.manager import initialize_vector_store, search_transcripts, list_sessions
from src.services.llm import generate_rag_answer, NO_RELEVANT_CONTENT
from src.services.streaming_stt import StreamingTranscriber, StreamingSttError
from src.services.audio_assembler import AudioAssembler

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
STREAM_AUDIO_FOLDER = str(config.STREAM_AUDIO_DIR)
INIT_TIMEOUT_SECONDS = 10.0

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}
MAX_FILE_SIZE_BYTES = config.MAX_FILE_SIZE_BYTES


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    source_id: str | None = Field(None, description="Scope query to a specific session")
    n_results: int = Field(5, ge=1, le=20, description="Number of context chunks to retrieve")


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
async def upload_file(
    file: UploadFile = File(...),
    session_label: str = Form(""),
):
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
            source_id=safe_name,
            session_label=session_label or f"Upload {safe_name[:8]}",
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
    collection = initialize_vector_store()
    if not collection:
        raise HTTPException(status_code=503, detail="Vector store unavailable.")

    chunks = search_transcripts(
        query=body.query,
        collection=collection,
        n_results=body.n_results,
        source_id=body.source_id,
    )

    if not chunks:
        return {
            "query": body.query,
            "answer": NO_RELEVANT_CONTENT,
            "sources": [],
        }

    result = await asyncio.to_thread(generate_rag_answer, body.query, chunks)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to generate answer.")

    return {
        "query": body.query,
        "answer": result["answer"],
        "sources": chunks,
    }


@app.get("/api/sessions")
async def get_sessions():
    collection = initialize_vector_store()
    if not collection:
        raise HTTPException(status_code=503, detail="Vector store unavailable.")

    sessions = list_sessions(collection)
    return {"sessions": sessions}


@app.websocket("/api/stream")
async def stream_audio(websocket: WebSocket) -> None:
    """
    Bidirectional WebSocket for live recording sessions.

    Protocol — see docs/superpowers/plans/2026-05-19-phase3c-backend-streaming.md.

    Architecture note: Deepgram transcript events arrive via a callback
    registered on the live connection. We decouple that from the WS send
    path via an asyncio.Queue + forwarder task — that way the wiring is
    deterministic regardless of which event loop Deepgram fires the
    callback on, and the forwarder cancels cleanly on shutdown.
    """
    await websocket.accept()
    session_id = uuid.uuid4().hex
    short_id = session_id[:8]
    logger.info("Live session opened (id=%s)", short_id)

    await websocket.send_json({"type": "session_started", "session_id": session_id})

    audio_path = os.path.join(STREAM_AUDIO_FOLDER, f"{session_id}.wav")
    assembler: AudioAssembler | None = None
    transcriber: StreamingTranscriber | None = None
    forwarder_task: asyncio.Task | None = None
    transcript_queue: asyncio.Queue = asyncio.Queue()
    session_label: str = ""

    async def on_transcript(text: str, is_final: bool, speaker: int | None) -> None:
        await transcript_queue.put((text, is_final, speaker))

    async def forward_transcripts() -> None:
        try:
            while True:
                text, is_final, speaker = await transcript_queue.get()
                try:
                    await websocket.send_json(
                        {"type": "transcript", "text": text, "is_final": is_final, "speaker": speaker}
                    )
                except Exception as e:
                    logger.warning("Failed to forward transcript (id=%s): %s — forwarder exiting", short_id, e)
                    return  # WS is dead; let the main loop discover the disconnect
        except asyncio.CancelledError:
            return

    try:
        assembler = AudioAssembler(audio_path)
        transcriber = StreamingTranscriber(on_transcript=on_transcript)

        # Wait for init message with a bounded timeout so abandoned
        # connections don't hold resources indefinitely.
        try:
            init_message = await asyncio.wait_for(
                websocket.receive(), timeout=INIT_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            await websocket.send_json(
                {"type": "error", "message": "init timeout", "fatal": True}
            )
            return

        if init_message.get("type") == "websocket.disconnect":
            logger.info("Client disconnected before init (id=%s)", short_id)
            return

        init_text = init_message.get("text")
        if init_text is None:
            await websocket.send_json(
                {"type": "error", "message": "First frame must be a JSON init message.", "fatal": True}
            )
            return

        try:
            init_payload = json.loads(init_text)
        except json.JSONDecodeError:
            await websocket.send_json(
                {"type": "error", "message": "Malformed init JSON", "fatal": True}
            )
            return

        if init_payload.get("type") != "init":
            await websocket.send_json(
                {"type": "error", "message": "First message must have type='init'.", "fatal": True}
            )
            return

        session_label = init_payload.get("session_label", "") or f"Live {short_id}"

        try:
            await transcriber.start()
        except StreamingSttError as e:
            logger.error("STT start failed (id=%s): %s", short_id, e)
            await websocket.send_json(
                {"type": "error", "message": str(e), "fatal": True}
            )
            return

        forwarder_task = asyncio.create_task(forward_transcripts())

        # Main message loop — binary audio chunks + control messages.
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                logger.info("Client disconnected mid-session (id=%s)", short_id)
                return

            if "text" in message and message["text"] is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {"type": "error", "message": "Malformed JSON", "fatal": False}
                    )
                    continue

                msg_type = payload.get("type")
                if msg_type == "init":
                    await websocket.send_json(
                        {"type": "error", "message": "Session already initialized.", "fatal": False}
                    )
                elif msg_type == "end":
                    break
                else:
                    await websocket.send_json(
                        {"type": "error", "message": f"Unknown message type: {msg_type}", "fatal": False}
                    )

            elif "bytes" in message and message["bytes"] is not None:
                chunk: bytes = message["bytes"]
                try:
                    assembler.write_chunk(chunk)
                    await transcriber.send_audio(chunk)
                except Exception as e:
                    logger.error("Chunk forward failed (id=%s): %s", short_id, e, exc_info=True)
                    await websocket.send_json(
                        {"type": "error", "message": "Audio chunk failed", "fatal": True}
                    )
                    return

        # Check if Deepgram emitted a fatal error during the session
        if transcriber is not None and transcriber.last_error:
            logger.error("Deepgram error during session (id=%s): %s", short_id, transcriber.last_error)
            await websocket.send_json(
                {"type": "error", "message": f"Deepgram error: {transcriber.last_error}", "fatal": True}
            )
            return

        # End of recording — drain Deepgram, finalize WAV, run analysis.
        try:
            if transcriber is not None:
                await transcriber.finish()
        except Exception as e:
            logger.warning("Transcriber finish error (id=%s): %s", short_id, e)

        final_audio_path = assembler.close()
        logger.info("Live session audio finalized (id=%s, path=%s)", short_id, final_audio_path)

        analysis_results = await asyncio.to_thread(
            run_analysis_pipeline,
            audio_path=str(final_audio_path),
            output_dir=ANALYSIS_OUTPUT_FOLDER,
            source_id=session_id,
            session_label=session_label,
        )

        if analysis_results is None:
            await websocket.send_json(
                {"type": "error", "message": "Analysis pipeline failed.", "fatal": True}
            )
            return

        transcript_content = read_file(analysis_results.get("transcript_path"), json.loads)
        features_content = read_file(analysis_results.get("features_path"), json.loads)
        feedback_content = read_file(analysis_results.get("feedback_path"))

        if transcript_content is None or features_content is None or feedback_content is None:
            await websocket.send_json(
                {"type": "error", "message": "Analysis completed but result files unreadable.", "fatal": True}
            )
            return

        await websocket.send_json(
            {
                "type": "session_end",
                "transcript": transcript_content,
                "features": features_content,
                "feedback": feedback_content,
            }
        )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected (id=%s)", short_id)
    except Exception as e:
        logger.error("Unexpected stream error (id=%s): %s", short_id, e, exc_info=True)
        try:
            await websocket.send_json(
                {"type": "error", "message": "Internal server error.", "fatal": True}
            )
        except Exception:
            pass
    finally:
        if forwarder_task is not None:
            forwarder_task.cancel()
            await asyncio.gather(forwarder_task, return_exceptions=True)
        try:
            if assembler is not None:
                assembler.close()
        except Exception:
            pass
        try:
            if transcriber is not None:
                await transcriber.finish()
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run("api:app", host=config.API_HOST, port=config.API_PORT, reload=config.API_RELOAD)
