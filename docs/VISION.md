# VerbalVector Vision

## What We're Building

A wearable speech analysis and conversation note-taking system powered by Meta Ray-Bans. VerbalVector captures spoken audio — presentations, conversations, meetings — and provides structured feedback on communication skills while storing searchable transcripts for later retrieval.

## The End State

You put on your Ray-Bans, have a conversation or give a presentation, and VerbalVector:

1. **Transcribes** everything in real-time (Deepgram Nova-3 streaming STT)
2. **Analyzes** your speech — pitch, pace, filler words, vocabulary, emotional tone (librosa, NLTK, Hume)
3. **Generates feedback** on how you communicated (Gemini 2.5 Flash with raw audio + features)
4. **Stores the transcript** as searchable chunks in a vector database (ChromaDB + SentenceTransformers)
5. **Answers questions** about past conversations via RAG — "What did we discuss in Monday's meeting?" or "How was my pacing in yesterday's presentation?"

## How We Get There

### Phase 0 + Phase 1 (Complete)

Foundation work. Migrated from Flask/Whisper/Ollama to FastAPI/Deepgram/Gemini/Hume. Centralized config, removed dead code, set up tooling.

### Phase 2A: RAG Pipeline

Wire up the note-taking system. Every uploaded recording gets its transcript stored in ChromaDB. The `/api/query` endpoint retrieves relevant transcript chunks and uses Gemini to answer questions — scoped to a single session or across all stored transcripts. A `/api/sessions` endpoint lists past recordings.

### Phase 2B: Browser Live Recording

Add a Record button to the React frontend using the MediaRecorder API. Captures audio from the browser mic, sends it through the same upload + analysis pipeline. Record-now, analyze-after workflow. No backend changes needed.

### Phase 3: Meta Ray-Bans Integration

Connect the Ray-Bans as an audio source via Meta's SDK. If the glasses expose a standard Bluetooth mic, Phase 2B's recording flow works as-is. If the SDK provides a streaming API, build a WebSocket endpoint (`/api/stream`) that accepts chunked audio and forwards it to Deepgram's streaming STT for real-time transcription. Add conversation mode (multi-speaker) alongside the existing presentation mode.

## Core Principles

- **Pipeline shape is stable.** New input sources (browser mic, Ray-Bans) feed into the same analysis pipeline. Adding a source never means rewriting the pipeline.
- **Config is centralized.** Every API key, model name, and tunable lives in `config.py` with env-var overrides.
- **Services are swappable.** Each external API sits behind a thin wrapper in `src/services/`. Swap Deepgram for Whisper or Gemini for GPT by editing one file.
- **Storage failures are non-fatal.** If ChromaDB is down, the pipeline still returns feedback — it just doesn't store the transcript.
- **Build incrementally.** Each phase delivers working, demoable functionality on its own.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| STT | Deepgram Nova-3 (batch now, streaming in Phase 3) |
| Audio features | librosa (pitch, volume, WPM, pauses) |
| Text features | NLTK (filler words, readability, vocabulary) |
| Emotion | Hume Expression Measurement (non-fatal) |
| LLM | Gemini 2.5 Flash (feedback generation + RAG answers) |
| Vector store | ChromaDB + SentenceTransformers (all-MiniLM-L6-v2) |
| Backend | FastAPI + uvicorn |
| Frontend | React + Vite |
| Wearable | Meta Ray-Bans (Phase 3) |
