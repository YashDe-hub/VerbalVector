# Live Result Poll-Fallback — Design

**Date:** 2026-06-06
**Status:** Approved (pending spec review)
**Branch:** `fix/live-result-poll-fallback`

## Problem

Live-recording analysis results are delivered **only** over the `/api/stream`
WebSocket, and **only after** the full post-recording pipeline finishes
(~60 s: Hume emotion + Gemini feedback + file uploads — `api.py:343-384`).

If the socket closes during that minute — the user navigates to another view
(unmounting `useLiveStream`, which calls `cleanup()` → `ws.close()`), backgrounds
the tab, or hits a network blip — the server computes the result and sends
`session_end` into a dead socket. The result is lost from the UI.

Observed in production logs (session `f8b03df1`):

| Time | Event |
|------|-------|
| 20:32:43 | Stop → `{type:"end"}` → pipeline starts |
| 20:32:48 | `INFO: connection closed` — browser WS drops |
| 20:33:44 | Pipeline `Done. Error: False` (files written) |
| 20:33:44 | `WebSocket disconnected` — `send_json(session_end)` hits closed socket |

The analysis itself is **not lost** — `analysis_output/{id}_transcript.json`,
`_features.json`, `_feedback.txt` are all written to disk and the session is
indexed in the vector store. Only live delivery failed.

**Root cause:** result delivery is coupled to a WebSocket surviving a slow,
~60 s post-recording pipeline.

## Goal

Make live results **durable** — show them after Stop whether or not the socket
survives — with an explicit "analyzing" state and a defined timeout escape hatch.

## Approach (chosen: A)

**Additive read endpoint + dual-path frontend.** Keep the WS `session_end` fast
path; add a poll-on-disk fallback. The pipeline and WS handler are unchanged —
we rely on the verified behavior that the pipeline (running in
`asyncio.to_thread` inside the handler, which Starlette does not cancel on
disconnect) completes and writes all three files even after the client drops.

Rejected alternatives:
- **B — Decouple analysis into a tracked background job** (in-memory
  `session_id → pending|done|failed` store, close socket on `end`). More robust
  but restructures the handler and adds state. YAGNI for this bug; revisit only
  if we later close the socket eagerly on Stop.
- **C — Pull from History/vector store.** The vector store holds transcript
  chunks, not the `features`/`feedback` payload shape; reshaping is awkward.

## Data Flow

```
Stop → ws.send({type:"end"})
       status: 'analyzing'              # new; socket stays open
   ├── fast path: session_end over socket → result → status 'completed'
   └── fallback: socket closes while 'analyzing'
            → poll GET /api/sessions/{id}/result every 2s, ≤ 45 attempts (~90s)
                202 (pending) → keep polling
                200 (ready)   → result → status 'completed'
                exhausted     → status 'error' (Retry / Check History)
```

Both paths converge on the same `UploadResponse` shape.

## Components

### Backend — `api.py`

New route: `GET /api/sessions/{session_id}/result`

- **Validate** `session_id` before any filesystem use: it is interpolated into a
  path, so reject anything outside the result-id convention (32-char lowercase
  hex) → `400`. This is the path-traversal guard.
- **Resolve paths** via a small shared helper, `result_file_paths(session_id)`,
  returning the three `Path`s under `ANALYSIS_OUTPUT_FOLDER`:
  - `{session_id}_transcript.json`
  - `{session_id}_features.json`
  - `{session_id}_feedback.txt`
- **Ready** — all three exist and are readable (reuse existing `read_file`
  helper) → `200 {transcript, features, feedback}` (same shape as `session_end`).
- **Pending** — any file missing/unreadable → `202 {status: "pending"}`.

**Naming-convention assumption:** result files are named from
`Path(audio_path).stem` (`analysis_pipeline.py:54,181`). For live sessions the
streaming WAV is `data/streams/{session_id}.wav`, so the stem equals the
`session_id` passed to the WS handler. The endpoint depends on this equality;
the implementation plan must re-verify the WAV naming in the audio assembler.
(The batch/upload path names files from the uploaded filename's stem, not a
session id — but this endpoint serves live sessions only, so that path is
untouched and out of scope.)

### Frontend — `api.ts`

`getSessionResult(sessionId): Promise<{status:'ready', data:UploadResponse} | {status:'pending'}>`
- `200` → `{status:'ready', data}`; `202` → `{status:'pending'}`.
- Network/5xx errors throw — caller treats a thrown error as one failed attempt
  (counts toward the cap), not an immediate giveup.

### Frontend — `useLiveStream.ts`

- Add `'analyzing'` to `LiveStreamStatus`.
- Add `sessionIdRef` — capture `msg.session_id` in the `session_started` handler
  (currently discarded; required for polling).
- `stop()`: after sending `{type:"end"}`, set status `'analyzing'` (was
  `'stopping'`); keep the socket open.
- `session_end` handler: unchanged (sets result + `'completed'`).
- `ws.onclose`: if status is `'analyzing'`, **do not** flag
  "Connection closed unexpectedly" — start the polling fallback. Other statuses
  keep current behavior.
- Poller: loop `getSessionResult(sessionIdRef.current)` every 2 s, ≤ 45
  attempts. `ready` → set result + `'completed'`; exhausted → `'error'`. Guard
  against the component having unmounted (the existing `cleanup`/ref pattern).

### Frontend — `VerbalVector.tsx` (UI)

- `'analyzing'` → "Analyzing your speech… (~1 min)" spinner state.
- `'completed'` → existing results render (unchanged).
- `'error'` (timeout) → message + **Check History** (navigate to history) and
  **Retry** (return to input / record again) actions.

## Error Handling

- **Pending (`202`)** — normal during the ~60 s window; keep polling.
- **Polling network error** — counts as a failed attempt toward the cap; do not
  abort early (the result may still land).
- **Timeout (cap reached)** — `'error'` with Retry / Check History. Results may
  still appear in History (files on disk + vector store), which the message says.
- **Invalid `session_id`** — `400` from the endpoint; should not occur in normal
  flow since the id comes from the server's `session_started`.

## Testing

### Backend (`pytest`)
- `200` with `{transcript, features, feedback}` when all three files present.
- `202 {status:"pending"}` when any file missing.
- `400` on non-hex / wrong-length `session_id` (path-traversal guard, e.g.
  `../../etc/passwd`).
- Use a temp `ANALYSIS_OUTPUT_FOLDER`; no real pipeline run.

### Frontend (`vitest`)
- Fast path: `session_end` over socket still renders results (regression).
- Fallback: socket closes while `'analyzing'` → poll → `200` → results shown,
  no "connection closed" error.
- `202`-then-`200` sequence → results shown after pending polls.
- Timeout: poller exhausts attempts → `'error'` with Retry / History affordances.
- `sessionIdRef` captured from `session_started`.
- Mock `fetch`/`getSessionResult` and `WebSocket` per existing test patterns
  (`VerbalVector.test.tsx`, `useLiveStream.test.ts`).

## Scope Guard (YAGNI)

- Batch/upload path untouched — it's synchronous HTTP with no delivery bug.
- No background-job store (Approach B) unless we later close the socket eagerly.
- No retry of the analysis pipeline itself — Retry means record again.
