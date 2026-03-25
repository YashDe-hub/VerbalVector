# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VerbalVector is a web application that provides comprehensive feedback on verbal communication skills. The system analyses audio recordings through a pipeline involving speech-to-text, acoustic feature extraction, vocal emotion analysis, and LLM-generated feedback.

**Note:** The README.md references the old stack (Flask, Whisper, Ollama). The current stack uses FastAPI, Deepgram, Gemini, and Hume. Trust this file over README.md.

## Environment Setup

```bash
# 1. Copy and fill in API keys
cp .env.example .env

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

Required keys in `.env`:
- `DEEPGRAM_API_KEY` — speech-to-text (Nova-3)
- `GEMINI_API_KEY` — LLM feedback (Gemini 2.5 Flash, native audio)
- `HUME_API_KEY` — vocal emotion analysis (Expression Measurement)

## Development Commands

### Backend (FastAPI)
```bash
python api.py
# Runs on http://localhost:5002
# API docs available at http://localhost:5002/docs
```

### Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev      # http://localhost:5173
npm run build
npm run lint
```

### Tests
No test suite exists yet. The `tests/` directory is empty. Use pytest if adding tests:
```bash
pytest tests/
```

## Pipeline Architecture

```
Audio file (upload or record)
  │
  ├── src/services/stt.py         → Deepgram Nova-3: transcript + word timestamps
  ├── src/features/audio_features.py → Librosa: pitch, volume, WPM, pauses
  ├── src/features/text_features.py  → NLTK: filler words, readability, vocabulary
  ├── src/services/emotion.py     → Hume: vocal emotion scores (non-fatal if fails)
  │
  └── src/services/llm.py         → Gemini 2.5 Flash (receives raw audio + all features)
                                     → Markdown feedback string
```

Threading: `analysis_pipeline.py` runs feature extraction and vector storage in parallel threads. Vector storage (ChromaDB) is optional — pipeline continues if it fails.

## Key Design Principles

- **`config.py` is the single source of truth** for all API keys and model names. Never hardcode these.
- **`src/services/`** isolates every external API behind a thin wrapper. Swap providers by editing one file.
- **Pipeline shape is stable**: `analysis_pipeline.py` calls services by interface, not by provider. Adding a new provider means editing only the relevant service file.
- **`/api/query` is a stub** — the RAG endpoint returns placeholder text. Vector store + retrieval logic is partially implemented in `src/vector_store/manager.py`.

## Common Issues

- Missing `.env` → `config.validate()` raises `EnvironmentError` with clear message
- Audio must be a supported format (mp3, wav, m4a, flac, etc.)
- Frontend expects backend on port 5002 with CORS enabled
- Vector storage failure is non-fatal — pipeline logs a warning and continues
