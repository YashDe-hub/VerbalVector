# Phase 3d.1 Backend Diarization — Manual Test Checklist

## Prerequisites

- Backend running with valid DEEPGRAM_API_KEY + GEMINI_API_KEY + HUME_API_KEY
- A short multi-speaker audio file (15–30s, two people talking). Easy way to make one: record yourself + a friend, or download a public conversation clip
- `wscat` or `websocat` for live mode verification

## Batch flow (smoke)

- [ ] POST a single-speaker audio file to /api/upload. Response.transcript has utterances list with all speakers === 0 and speakers === [0].
- [ ] POST a multi-speaker audio file. Response.transcript.utterances has multiple speaker IDs.
- [ ] response.transcript.segments has 'speaker' on each word.
- [ ] response.feedback (Gemini output) mentions "Speaker 0" / "Speaker 1" (verify the multi-speaker prompt section landed). Single-speaker recording does NOT have "Conversation Mode" in feedback text.

## Vector storage

- [ ] Query ChromaDB or use /api/query and verify that chunks from a multi-speaker upload have speaker metadata.
- [ ] /api/query results show transcript chunks (verify nothing in the response shape regressed for the consumer).

## Live mode (Plan B / 3d.2 will add UI; for now test via wscat)

- [ ] Connect to /api/stream, send init, pipe a multi-speaker PCM file.
- [ ] transcript messages include "speaker" field. Interim frames may have speaker=null, final frames have an integer.
- [ ] session_end's transcript.utterances has all speakers represented.

## Rollback criteria

If observed, revert:

- [ ] /api/upload regresses (single-speaker transcripts no longer work the same way).
- [ ] /api/stream regresses (single-speaker live sessions break).
- [ ] feedback quality for single-speaker recordings drops noticeably vs pre-diarization.
- [ ] ChromaDB storage breaks for transcripts that don't have utterances (the fallback path).
