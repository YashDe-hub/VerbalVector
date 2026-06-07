# Live Result Poll-Fallback — Design

**Date:** 2026-06-06
**Status:** Revised after staff-engineer (arch) review — pending spec review
**Branch:** `fix/live-result-poll-fallback`

## Problem

Live-recording analysis results are delivered **only** over the `/api/stream`
WebSocket, and **only after** the full post-recording pipeline finishes
(~60 s: Hume emotion + Gemini feedback + file uploads — `api.py:343-384`).

If the socket closes during that minute — most commonly because the user
navigates to another view — the server computes the result and sends
`session_end` into a dead socket. The result is lost from the UI.

Critically, navigation **unmounts** the live component: `App.tsx:38` renders
`{currentView === "analysis" && <VerbalVector/>}`, so switching to query/history
removes `VerbalVector` from the tree, firing `useLiveStream`'s cleanup
(`useLiveStream.ts:210-214`) → `ws.close()`.

Observed in production logs (session `f8b03df1`):

| Time | Event |
|------|-------|
| 20:32:43 | Stop → `{type:"end"}` → pipeline starts |
| 20:32:48 | `INFO: connection closed` — browser WS drops (user navigated to query) |
| 20:33:23+ | `/api/query` calls — user is on the query view |
| 20:33:44 | Pipeline `Done. Error: False` (files written) |
| 20:33:44 | `WebSocket disconnected` — `send_json(session_end)` hits closed socket |

The analysis itself is **not lost** — `analysis_output/{id}_transcript.json`,
`_features.json`, `_feedback.txt` are all written and the session is indexed in
the vector store. Only live delivery failed.

**Root cause:** result delivery is coupled to a WebSocket surviving a slow,
~60 s pipeline — and the socket dies when the user navigates away.

## Goal

Make live results **durable** — show them after Stop whether or not the socket
(or the live component) survives — with an explicit "analyzing" state and a
defined timeout escape hatch.

## Approach (chosen: A)

**Additive read endpoint + a poll loop that lives above the unmount boundary.**
Keep the WS `session_end` fast path; add a poll-on-disk fallback. The pipeline
and WS handler are unchanged — we rely on the verified behavior that the
pipeline (in `asyncio.to_thread` inside the handler, which Starlette does not
cancel on disconnect) completes and writes all three files even after the
client drops.

**Key correction from arch review:** the poll loop must NOT live inside
`useLiveStream`. That hook is destroyed by the very navigation that causes the
bug, so a poller started in its `onclose` would render into an unmounted
component and do nothing. The analyzing → polling → result lifecycle is lifted
to **`App.tsx`**, which persists across all view switches.

Rejected alternatives:
- **B — Decouple analysis into a tracked background job** (server-side
  `session_id → pending|done|failed` store). More robust but restructures the
  handler and adds state. YAGNI for this bug; revisit only if we later close the
  socket eagerly on Stop.
- **C — Pull from History/vector store.** The vector store holds transcript
  chunks, not the `features`/`feedback` payload shape; reshaping is awkward.

## Data Flow

```
Stop (in useLiveStream)
  → ws.send({type:"end"})
  → onSessionEnding(sessionId)  ──────────────►  App: pendingAnalysis = {id, 'analyzing'}
                                                  App starts poll loop (survives nav)

  ┌── fast path (component still mounted):
  │     session_end over socket → onAnalysisComplete(result)
  │       → App: status 'completed', cancel poll, show results
  │
  └── durable path (App-owned poll, runs regardless of mount/nav):
        GET /api/sessions/{id}/result every 2s, ≤ 45 attempts (~90s)
            202 (pending) → keep polling
            200 (ready)   → App: result, 'completed', show results
            exhausted     → App: status 'error' (Retry / Check History)
```

Both paths funnel into the **same App-level setter**; whichever resolves first
wins, and once `status !== 'analyzing'` the poll effect stops. Both deliver the
same `UploadResponse` shape.

## Components

### Backend — `api.py`

New route: `GET /api/sessions/{session_id}/result`

- **Validate** `session_id` before any filesystem use. `session_id =
  uuid.uuid4().hex` (`api.py:210`), so accept only `^[0-9a-f]{32}$`; anything
  else → `400`. This is the path-traversal guard.
- **Resolve paths** via a new shared helper `result_file_paths(session_id)`
  returning the three paths under `ANALYSIS_OUTPUT_FOLDER`:
  `{id}_transcript.json`, `{id}_features.json`, `{id}_feedback.txt`.
- **Pending check uses `Path.exists()`, not `read_file`.** `read_file` logs an
  `ERROR` on `FileNotFoundError` (`api.py:56-57`); calling it on not-yet-written
  files every 2s would emit ~30 ERROR lines per *healthy* session. So: if any of
  the three paths does not exist → `202 {status:"pending"}` (no read, no log).
- **Ready** — all three exist: read via the shared
  `read_analysis_results(paths)` helper (see below). If all parse → `200
  {transcript, features, feedback}` (same shape as `session_end`). If a file
  exists but is unreadable/corrupt → `500`.

**Refactor (arch should-fix):** extract the "read three files + validate"
logic — currently duplicated in upload (`api.py:121-138`) and the WS handler
(`api.py:367-375`) — into `read_analysis_results(transcript_path,
features_path, feedback_path) -> dict | None`. Use it in all three call sites
(upload, WS handler, new endpoint) so the convention lives in one place.

**Naming-convention assumption (verified):** result files are named from
`Path(audio_path).stem` (`analysis_pipeline.py:54,181`). The live WAV is
`STREAM_AUDIO_FOLDER/{session_id}.wav` (`api.py:216`), so the stem equals the
`session_id`. The endpoint depends on this equality. (The upload path names
files from the uploaded filename's stem, not a session id — but this endpoint
serves live sessions only, so that path is untouched and out of scope.)

### Frontend — `api.ts`

`getSessionResult(sessionId): Promise<{status:'ready', data:UploadResponse} | {status:'pending'}>`
- `200` → `{status:'ready', data}`; `202` → `{status:'pending'}`.
- Network/5xx errors throw — the App poller treats a thrown error as one failed
  attempt (counts toward the cap), not an immediate giveup.

### Frontend — `useLiveStream.ts`

- Add `sessionIdRef` — capture `msg.session_id` in the `session_started`
  handler (currently discarded; needed so `stop()` can report it upward).
- New option callback `onSessionEnding?: (sessionId: string) => void`.
- `stop()`: after sending `{type:"end"}`, call
  `onSessionEnding(sessionIdRef.current)` and set local status `'stopping'`
  (unchanged name). The hook no longer owns the post-stop lifecycle.
- `session_end` handler: unchanged — still sets `result` + `'completed'` so a
  mounted component shows results immediately (fast path). VerbalVector forwards
  this to App via its existing `onAnalysisComplete`.
- `ws.onclose`: if status is `'stopping'`, **do not** flag
  "Connection closed unexpectedly" — App's poller owns recovery now. No polling
  is started here.

### Frontend — `App.tsx` (owns the durable lifecycle)

- New state `pendingAnalysis: { sessionId: string; status: 'analyzing' |
  'error' } | null`.
- `handleSessionEnding(sessionId)` (passed into `VerbalVector` →
  `useLiveStream`): set `pendingAnalysis = { sessionId, status: 'analyzing' }`.
- Poll effect — runs when `pendingAnalysis?.status === 'analyzing'`: call
  `getSessionResult(sessionId)` every 2 s, ≤ 45 attempts. On `ready` →
  `setAnalysisResult(data)`, clear `pendingAnalysis`, behave like
  `handleAnalysisComplete` (auto-show, see below). On exhausted →
  `pendingAnalysis.status = 'error'`. Effect cleans up its timer on unmount/
  status-change. Because `App` never unmounts on view switches, the poll
  survives navigation.
- `handleAnalysisComplete(result)` (exists, fast path): also clears
  `pendingAnalysis` so the poll effect stops.

**Auto-show resolution (decision for review):**
- If `currentView === "analysis"` when the result arrives → navigate to the
  `results` view automatically (true auto-show in place).
- If the user has navigated to query/history → do **not** yank them away
  mid-task. Show a dismissible App-level banner: "✓ Your analysis is ready —
  View results", which routes to `results` on click. This honors "auto-show"
  without hijacking an active task. (Closer to the chosen option than the
  rejected History-link; flagged here so you can object.)

### Frontend — `VerbalVector.tsx` (UI)

- While `pendingAnalysis?.status === 'analyzing'` and on the analysis view →
  show an "Analyzing your speech… (~1 min)" spinner (driven by an
  `analyzing` prop from App, or App renders the spinner itself — implementation
  plan decides the cleanest wiring).
- `'error'` (timeout) → message + **Check History** (navigate to history) and
  **Retry** (return to input / record again) actions.

## Error Handling

- **Pending (`202`)** — normal during the ~60 s window; keep polling. No server
  log noise (existence check, not `read_file`).
- **Polling network error** — counts as a failed attempt toward the cap; do not
  abort early (the result may still land).
- **Partial pipeline success** — `feedback_path` can be legitimately `None` when
  feedback generation fails but transcript + features succeed
  (`analysis_pipeline.py:121` records this as a partial, not a crash). In that
  case the feedback file never appears, so the endpoint stays `202` and the
  client times out after ~90 s → `'error'` + Check History. Acceptable for v1;
  documented so it isn't a surprise. (A future Approach-B status store could
  distinguish "done-but-partial" from "in-progress" and return early.)
- **Timeout (cap reached)** — `'error'` with Retry / Check History. Results may
  still appear in History (files on disk + vector store); the message says so.
- **Invalid `session_id`** — `400`; should not occur since the id comes from the
  server's `session_started`.

## Testing

### Backend (`pytest`)
- `200` with `{transcript, features, feedback}` when all three files present.
- `202 {status:"pending"}` when any file missing — and assert **no ERROR log**
  is emitted (regression guard on the `read_file`-spam fix).
- `400` on non-hex / wrong-length `session_id` (traversal guard, e.g.
  `../../etc/passwd`, uppercase, wrong length).
- `500` when a file exists but is unreadable/corrupt JSON.
- `read_analysis_results` helper: returns dict when all readable, `None` when
  any missing/corrupt; upload + WS handler still behave identically after the
  refactor (existing tests must stay green).
- Use a temp `ANALYSIS_OUTPUT_FOLDER`; no real pipeline run.

### Frontend (`vitest`)
- **Fast path:** `session_end` over socket → `onAnalysisComplete` → results
  shown; `pendingAnalysis` cleared, poll never fires (regression).
- **Durable path (the bug):** simulate navigation — `VerbalVector` unmounts
  after `onSessionEnding`; App's poll still runs → `200` → results shown / banner
  surfaced. This is the core regression test for the arch finding.
- `202`-then-`200` sequence → results shown after pending polls.
- **Timeout:** poller exhausts attempts → `'error'` with Retry / History.
- `sessionIdRef` captured from `session_started`; `onSessionEnding` fired on
  `stop()` with the right id.
- Auto-show branch: on `analysis` view → routes to results; on `query`/`history`
  view → banner shown, no forced navigation.
- Mock `getSessionResult` and `WebSocket` per existing patterns
  (`VerbalVector.test.tsx`, `useLiveStream.test.ts`); fake timers for the poll.

## Scope Guard (YAGNI)

- Batch/upload path untouched — synchronous HTTP, no delivery bug. (Only the
  internal `read_analysis_results` refactor touches it; behavior identical.)
- No server-side background-job store (Approach B) unless we later close the
  socket eagerly on Stop.
- No retry of the analysis pipeline itself — Retry means record again.
- Polling cap (45 × 2 s = 90 s) is tuned to a single observed ~61 s run; if
  Gemini/Hume latency proves variable in practice, revisit the cap (frontend
  module constant).
