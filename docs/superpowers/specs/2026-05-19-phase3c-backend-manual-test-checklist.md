# Phase 3c.1 Backend Streaming — Manual Test Checklist

> **Purpose:** Verify the /api/stream WebSocket endpoint end-to-end before the frontend hook (Plan B) consumes it. Automated tests cover the unit/integration boundary; these checks verify the live Deepgram path and pipeline wiring.

## Prerequisites

- Backend running: `python api.py` on port 5002
- `wscat` or `websocat` installed (`npm i -g wscat` / `cargo install websocat`)
- Valid DEEPGRAM_API_KEY + GEMINI_API_KEY + HUME_API_KEY in `.env`
- A short PCM file at 16kHz mono 16-bit you can stream (5–10 seconds is fine — use `ffmpeg -i sample.wav -ac 1 -ar 16000 -f s16le sample.pcm` from any speech sample)

## Smoke (mock-mode reachable)

- [ ] Server starts cleanly (`python api.py`). No errors on import of the new modules.
- [ ] `wscat -c ws://localhost:5002/api/stream` connects.
- [ ] Server immediately sends `{"type": "session_started", "session_id": "..."}`.
- [ ] Sending malformed JSON yields `{"type": "error", "message": "Malformed JSON", "fatal": false}` (non-fatal — connection stays open).
- [ ] Sending audio before `init` yields a fatal error and the connection closes.

## Live Deepgram path

- [ ] Send `{"type": "init", "session_label": "Test"}`.
- [ ] Pipe the test PCM file as binary to the WS:
  - With websocat: `cat sample.pcm | websocat --binary ws://localhost:5002/api/stream` (after sending init)
  - With wscat: trickier — write a quick Python client (a 10-line snippet using `websockets` library is fine).
- [ ] Interim transcripts arrive within ~500ms of the first chunks (`{"type":"transcript","text":"...","is_final":false}`).
- [ ] Final transcripts arrive at sentence boundaries (`"is_final":true`).
- [ ] Sending `{"type":"end"}` triggers analysis. Within ~30s the server sends `{"type":"session_end","transcript":...,"features":...,"feedback":...}` then closes.
- [ ] The assembled WAV file at `data/streams/<session_id>.wav` exists, is valid (open in a player), and matches the original audio.
- [ ] The feedback content is reasonable — Gemini received the audio file and the transcript and produced coaching output.

## Failure paths

- [ ] Kill the server mid-session — client sees connection drop, no zombie file in `data/streams` (or it exists but is recoverable). No backend stack traces about leaked Deepgram connections.
- [ ] Set DEEPGRAM_API_KEY to empty in env, restart, connect, send `init`. Should get `{"type":"error","fatal":true}` and clean close.
- [ ] Disconnect WebSocket without sending `end`. Server logs "Client disconnected mid-session", no orphaned Deepgram connection (verify via Deepgram dashboard if you want to be thorough).

## Rollback criteria

If observed, revert the PR:

- [ ] /api/upload (the existing batch endpoint) regresses in any way.
- [ ] `data/streams/` fills the disk unboundedly (no per-session cleanup, but acceptable for v1 — flag for future eviction policy).
- [ ] Deepgram connections leak (verify via dashboard).
- [ ] The analysis pipeline takes materially longer for streamed sessions than for batch sessions of equivalent audio length.
