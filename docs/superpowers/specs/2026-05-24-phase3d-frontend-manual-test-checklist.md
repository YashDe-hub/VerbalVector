# Phase 3d.2 Frontend Diarization Rendering — Manual Test Checklist

## Prerequisites

- Backend + frontend running (per Phase 3c.2 checklist)
- Multi-speaker audio file ready (15-30s, two distinct voices)
- Chrome recommended for live mode (Phase 3c.2 caveats apply)

## Live mode (single speaker)

- [ ] Switch to Live, record yourself solo for ~15s.
- [ ] LiveTranscript shows transcript WITHOUT "Speaker 0:" prefixes — looks identical to pre-diarization UI.
- [ ] Stop → processing → results.
- [ ] ResultsDisplay's Transcript section shows flat text — no "Speaker 0:" prefix.

## Live mode (multi-speaker)

- [ ] Switch to Live, record a conversation with two people for ~15s.
- [ ] LiveTranscript shows interim text in lighter color WITHOUT a speaker label (this is intentional).
- [ ] As each segment finalizes, it appears on its own line with "Speaker 0:" or "Speaker 1:" prefix.
- [ ] Known limitation: interim speaker IDs may shift before finalization. The UI doesn't expose this — interim has no label — but if you're watching the WS frames you'll see the speaker field flip. Expected.
- [ ] Stop → processing → results.
- [ ] ResultsDisplay's Transcript section shows speaker-labeled list with proper attribution.

## Batch mode (multi-speaker upload)

- [ ] Upload a multi-speaker audio file in Batch mode.
- [ ] ResultsDisplay shows speaker-labeled transcript.
- [ ] Feedback (Gemini output) mentions speakers — the Phase 3d.1 prompt section landed.

## Batch mode (single-speaker upload — regression check)

- [ ] Upload a single-speaker recording in Batch mode.
- [ ] Transcript renders as flat text (no labels).
- [ ] Feedback does NOT mention "Conversation Mode" / "Multi-Speaker".

## Rollback criteria

If observed, revert:

- [ ] Single-speaker recordings (batch or live) show "Speaker 0:" prefixes — strip-when-1 logic is broken.
- [ ] Pre-diarization saved transcripts (legacy data — no utterances field) crash ResultsDisplay.
- [ ] LiveTranscript shows the interim text WITH a speaker label (we explicitly don't label interim).
- [ ] Switching between Batch and Live mode breaks the transcript display.
