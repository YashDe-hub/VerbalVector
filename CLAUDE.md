# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VerbalVector is a web application that provides comprehensive feedback on verbal communication skills. The system analyses audio recordings through a pipeline involving speech-to-text, acoustic feature extraction, vocal emotion analysis, and LLM-generated feedback.

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

### Backend (Flask)
```bash
python api.py
# Runs on http://localhost:5002
```

### Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev      # http://localhost:5173
npm run build
npm run lint
```

## Repository Structure

```
VerbalVector/
├── api.py                        # Flask entry point — /api/upload, /api/query
├── main.py                       # CLI entry point (setup, download, extract, infer, search)
├── config.py                     # Loads .env, exposes API keys + model names + paths
├── requirements.txt
│
├── src/
│   ├── pipelines/
│   │   └── analysis_pipeline.py  # Orchestrates STT → features → LLM (with threading)
│   ├── features/
│   │   ├── audio_features.py     # Librosa: pitch, volume, WPM, pauses
│   │   ├── text_features.py      # NLTK: filler words, readability, vocabulary
│   │   ├── feature_combiner.py   # Merges audio + text features
│   │   └── speech_quality.py     # Pronunciation / grammar analysis
│   ├── services/                 # External API clients (one file per provider)
│   │   ├── stt.py                # Deepgram Nova-3 (batch + streaming)
│   │   ├── llm.py                # Gemini 2.5 Flash with native audio input
│   │   └── emotion.py            # Hume Expression Measurement
│   ├── data/
│   │   └── data_collector.py     # YouTube audio/transcript downloader
│   └── vector_store/
│       └── manager.py            # ChromaDB storage and RAG search
│
├── frontend/                     # React 19 + TypeScript + Vite + Material-UI
│   └── src/
│       ├── components/
│       │   ├── VerbalVector.tsx   # Audio upload / recording interface
│       │   └── ResultsDisplay.tsx # Feedback display
│       └── ...
│
├── scripts/                      # One-off utility scripts (not part of the app)
│   ├── create_finetune_data.py   # Generates fine-tuning JSONL dataset
│   └── process_files.py          # Batch-process audio files through the pipeline
│
├── data/                         # Data files (mostly gitignored)
│   ├── raw/                      # Input audio (gitignored)
│   ├── processed/                # Feature JSONs (gitignored)
│   ├── uploads/                  # Flask upload staging (gitignored)
│   └── finetune_dataset.jsonl    # LoRA fine-tuning dataset
│
├── docs/                         # Project documentation and presentations
│   ├── Final_presentation.pdf
│   └── VV_metrics.pdf
│
├── notebooks/                    # Jupyter notebooks
├── tests/                        # Test suite (pytest)
│
├── .env.example                  # Required keys template — copy to .env
├── .gitignore
├── README.md
└── CLAUDE.md
```

## Pipeline Architecture

```
Audio file (upload or record)
  │
  ├── src/services/stt.py         → Deepgram Nova-3: transcript + word timestamps
  ├── src/features/audio_features.py → Librosa: pitch, volume, WPM, pauses (hard numbers)
  ├── src/features/text_features.py  → NLTK: filler words, readability, vocabulary
  ├── src/services/emotion.py     → Hume: vocal emotion scores (nervousness, confidence…)
  │
  └── src/services/llm.py         → Gemini 2.5 Flash (receives audio + all features)
                                     → Markdown feedback string
```

Threading: feature extraction and vector storage run in parallel threads.
Vector storage (ChromaDB) is optional — pipeline continues if it fails.

## Key Design Principles

- **`config.py` is the single source of truth** for all API keys and model names. Never hardcode these.
- **`src/services/`** isolates every external API behind a thin wrapper. Swap providers by editing one file.
- **Pipeline shape is stable**: `analysis_pipeline.py` calls services by interface, not by provider. Adding a new provider means editing only the relevant service file.
- **No secrets in git**: `.env` is gitignored; `.env.example` documents the required keys.

## Common Issues

- Missing `.env` → `config.validate()` raises `EnvironmentError` with clear message
- Audio must be a supported format (mp3, wav, m4a, flac, etc.)
- Frontend expects backend on port 5002 with CORS enabled
- Vector storage failure is non-fatal — pipeline logs a warning and continues
