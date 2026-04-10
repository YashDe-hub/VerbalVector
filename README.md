# VerbalVector: AI-Powered Communication Analysis

VerbalVector is a web application that provides comprehensive feedback on verbal communication skills. Users upload or record audio, and the system analyses it through a pipeline of speech-to-text, feature extraction, vocal emotion analysis, and LLM-generated feedback.

## Current Stack

| Layer | Provider | Model |
|-------|----------|-------|
| STT | Deepgram | Nova-3 |
| LLM | Google Gemini | 2.5 Flash (native audio) |
| Emotion | Hume | Expression Measurement |
| Vector store | ChromaDB | Sentence Transformers |
| Backend | FastAPI + Uvicorn | — |
| Frontend | React + Vite | — |

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js and npm (for frontend)
- FFmpeg (`brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Ubuntu)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YashDe-hub/VerbalVector.git
cd VerbalVector

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env  # then fill in your API keys
```

Required keys in `.env`:
- `DEEPGRAM_API_KEY` — speech-to-text
- `GEMINI_API_KEY` — LLM feedback
- `HUME_API_KEY` — vocal emotion analysis

### Running

```bash
# Backend (http://localhost:5002, API docs at /docs)
python api.py

# Frontend (http://localhost:5173)
cd frontend && npm install && npm run dev
```

## Pipeline Architecture

```
Audio file (upload or record)
  │
  ├── src/services/stt.py            → Deepgram Nova-3: transcript + word timestamps
  ├── src/features/audio_features.py → Librosa: pitch, volume, WPM, pauses
  ├── src/features/text_features.py  → NLTK: filler words, readability, vocabulary
  ├── src/services/emotion.py        → Hume: vocal emotion scores (non-fatal)
  │
  └── src/services/llm.py            → Gemini 2.5 Flash (raw audio + all features)
                                       → Markdown feedback string
```

Feature extraction and vector storage run in parallel threads via `analysis_pipeline.py`. Vector storage (ChromaDB) is optional — the pipeline continues if it fails.

## Known Limitations

- `/api/query` is a stub — RAG retrieval is not yet implemented

## Common Issues

- Missing `.env` → `config.validate()` raises `EnvironmentError` with a clear message listing missing keys
- Audio must be a supported format (mp3, wav, m4a, flac, etc.)
- Frontend expects backend on port 5002 with CORS enabled

## License

MIT License
