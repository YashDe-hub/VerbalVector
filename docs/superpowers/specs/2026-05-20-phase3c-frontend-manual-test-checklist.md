# Phase 3c.2 Frontend Live Mode — Manual Test Checklist

## Prerequisites

- Backend running: `python api.py` on port 5002 with valid DEEPGRAM_API_KEY + GEMINI_API_KEY + HUME_API_KEY in `.env`
- Frontend running: `cd frontend && npm run dev` on port 5173
- Chrome 80+ recommended (AudioWorklet support); Firefox 76+ also fine
- Working microphone

## Smoke (mode toggle + batch still works)

- [ ] Open `http://localhost:5173`. The Analyze view loads.
- [ ] Two radio buttons visible: "Batch (record + upload)" (selected) and "Live".
- [ ] In Batch mode (default): the existing Record + Upload buttons + Microphone dropdown are visible. Behaviour unchanged from Phase 3a.
- [ ] Switch to Live: the UI layout adjusts; Record still visible.
- [ ] Switch back to Batch: layout returns.

## Live mode happy path

- [ ] Switch to Live. Click Record. Grant mic permission if prompted.
- [ ] Within ~1s, the UI moves to the recording stage. A LiveTranscript pane shows "Connecting…" then "Listening…" then real transcripts as you speak.
- [ ] Interim transcripts appear in a lighter grey color; finals appear in normal color.
- [ ] After ~10–20 seconds of speech, click Stop Recording.
- [ ] The UI moves to the processing stage.
- [ ] Within ~30s, the analysis result appears in ResultsDisplay (same UI as batch).
- [ ] Verify the assembled WAV file appears at `data/streams/<session_id>.wav` on the backend.
- [ ] Open the WAV — it should match what you recorded (no audio dropouts, no inverted channels).

## Failure paths

- [ ] Live mode with backend offline: Click Record. UI should show "WebSocket error" or "Connection closed unexpectedly." and return to input.
- [ ] Live mode then kill backend mid-session: client should show error and return to input.
- [ ] Live mode then click Stop *before* any transcript arrives: should still trigger session_end (server will run analysis on the short recording).
- [ ] Live mode click Stop during 'connecting' (before session_started arrives): UI returns to input without spinning.

## Browser compatibility

- [ ] Chrome — should fully work.
- [ ] Firefox — AudioWorklet supported; verify the transcript pane updates correctly.
- [ ] Safari — AudioWorklet supported since 14.5 but historically flaky with Bluetooth audio inputs. Verify with the default mic at minimum.

## Rollback criteria

If observed, revert the PR:

- [ ] Batch mode regresses in any way (the existing flow MUST be unchanged).
- [ ] AudioContext never closes when stop is pressed — memory leak via persistent worklet.
- [ ] Microphone indicator stays on after Stop is pressed (capture not properly torn down).
- [ ] Backend `data/streams/` accumulates WAVs that don't match the recorded audio.
- [ ] Stop button is unclickable in live mode (regression of the fix in commit 8fc0de4).
