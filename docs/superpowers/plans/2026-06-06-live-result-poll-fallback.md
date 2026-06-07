# Live Result Poll-Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Post-Commit Quality Loop (CLAUDE.md):** after each task's commit, run Simplify → Verify (pytest + vitest + lint/types) → code-reviewer (+ silent-failure-hunter for the polling/error tasks, pr-test-analyzer for test tasks). Address Critical/Important before the next task.

**Goal:** Make live-recording analysis results durable — delivered after Stop whether or not the WebSocket (or the live component) survives the ~60 s pipeline.

**Architecture:** Keep the WS `session_end` fast path. Add a backend `GET /api/sessions/{id}/result` that reads the already-written result files, and an **App-level** poll loop (above the view-switch unmount boundary) that fetches the result and shows it even after the user navigates away. Spec: `docs/superpowers/specs/2026-06-06-live-result-poll-fallback-design.md`.

**Tech Stack:** FastAPI + httpx/pytest (backend), React 19 + TypeScript + Vitest + React Testing Library (frontend), axios.

---

## File Structure

- `api.py` — add `SESSION_ID_RE`, `result_file_paths()`, `read_analysis_results()` helpers; refactor upload + WS handler to use the helper; add `GET /api/sessions/{session_id}/result`.
- `tests/test_api_result.py` (new) — endpoint + helper tests.
- `frontend/src/api.ts` — add `getSessionResult()` + `SessionResultPoll` type.
- `frontend/src/api.test.ts` (new) — `getSessionResult` mapping tests.
- `frontend/src/hooks/useLiveStream.ts` — capture `sessionId`, add `onSessionEnding` callback, stop erroring on close during `stopping`.
- `frontend/src/hooks/useLiveStream.test.ts` — new cases.
- `frontend/src/components/VerbalVector.tsx` — accept + forward `onLiveSessionEnding`.
- `frontend/src/components/VerbalVector.test.tsx` — forwarding test.
- `frontend/src/App.tsx` — `pending` state, poll effect, auto-show/banner, retry remount.
- `frontend/src/App.test.tsx` (new) — durable-path, banner, timeout tests.

---

### Task 1: Backend — shared result helpers + refactor call sites

**Files:**
- Modify: `api.py` (add helpers near `read_file` at `api.py:49-61`; refactor upload `api.py:117-146` and WS handler `api.py:367-384`)
- Test: `tests/test_api_result.py` (new)

- [ ] **Step 1: Write failing tests for the helpers**

Create `tests/test_api_result.py`:

```python
"""Tests for result-file helpers and GET /api/sessions/{id}/result."""
import json
import logging
import pytest
import pytest_asyncio
from unittest.mock import patch


@pytest_asyncio.fixture
async def client(tmp_path, monkeypatch):
    """Async test client with a temp ANALYSIS_OUTPUT_FOLDER."""
    with patch("config.validate"):
        import api
        monkeypatch.setattr(api, "ANALYSIS_OUTPUT_FOLDER", str(tmp_path))
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=api.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


def _write_results(folder, session_id, *, transcript=True, features=True, feedback=True, corrupt=False):
    if transcript:
        text = "not json{" if corrupt else json.dumps({"text": "hi", "utterances": [], "speakers": [0]})
        (folder / f"{session_id}_transcript.json").write_text(text, encoding="utf-8")
    if features:
        (folder / f"{session_id}_features.json").write_text(json.dumps({"words_per_minute": 100}), encoding="utf-8")
    if feedback:
        (folder / f"{session_id}_feedback.txt").write_text("Good job.", encoding="utf-8")


def test_read_analysis_results_all_present(tmp_path, monkeypatch):
    import api
    monkeypatch.setattr(api, "ANALYSIS_OUTPUT_FOLDER", str(tmp_path))
    sid = "a" * 32
    _write_results(tmp_path, sid)
    t, f, fb = api.result_file_paths(sid)
    out = api.read_analysis_results(t, f, fb)
    assert out == {
        "transcript": {"text": "hi", "utterances": [], "speakers": [0]},
        "features": {"words_per_minute": 100},
        "feedback": "Good job.",
    }


def test_read_analysis_results_missing_returns_none(tmp_path, monkeypatch):
    import api
    monkeypatch.setattr(api, "ANALYSIS_OUTPUT_FOLDER", str(tmp_path))
    sid = "b" * 32
    _write_results(tmp_path, sid, feedback=False)
    t, f, fb = api.result_file_paths(sid)
    assert api.read_analysis_results(t, f, fb) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_api_result.py -v`
Expected: FAIL — `AttributeError: module 'api' has no attribute 'result_file_paths'`.

- [ ] **Step 3: Add the helpers to `api.py`**

Add `import re` to the top imports (alongside `import os`), and add `from pathlib import Path` and `from fastapi.responses import JSONResponse` (used in Task 2). Then add, immediately after `read_file` (after `api.py:61`):

```python
SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def result_file_paths(session_id: str) -> tuple[str, str, str]:
    """The three analysis-output paths for a session, by naming convention.

    Files are named from the audio stem; for live sessions the WAV is
    {session_id}.wav, so the stem equals session_id (see api.py stream handler).
    """
    base = os.path.join(ANALYSIS_OUTPUT_FOLDER, session_id)
    return (f"{base}_transcript.json", f"{base}_features.json", f"{base}_feedback.txt")


def read_analysis_results(
    transcript_path: str | None,
    features_path: str | None,
    feedback_path: str | None,
) -> dict | None:
    """Read + parse the three result files. Returns the API payload or None
    if any file is missing/unreadable."""
    transcript = read_file(transcript_path, json.loads)
    features = read_file(features_path, json.loads)
    feedback = read_file(feedback_path)
    if transcript is None or features is None or feedback is None:
        return None
    return {"transcript": transcript, "features": features, "feedback": feedback}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_api_result.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Refactor the upload endpoint to use the helper**

In `api.py`, replace the result-reading block at `api.py:117-146` (from `transcript_path = analysis_results.get(...)` through the `return {...}`) with:

```python
        results = read_analysis_results(
            analysis_results.get("transcript_path"),
            analysis_results.get("features_path"),
            analysis_results.get("feedback_path"),
        )
        if results is None:
            logger.error("Failed to read result files (id=%s)", safe_name[:8])
            raise HTTPException(
                status_code=500,
                detail="Analysis completed but failed to read result files.",
            )

        logger.info("Analysis complete (id=%s)", safe_name[:8])
        return {"message": f"File '{file.filename}' processed successfully.", **results}
```

- [ ] **Step 6: Refactor the WS handler to use the helper**

In `api.py`, replace the block at `api.py:367-384` (from `transcript_content = read_file(...)` through the `session_end` `send_json`) with:

```python
        results = read_analysis_results(
            analysis_results.get("transcript_path"),
            analysis_results.get("features_path"),
            analysis_results.get("feedback_path"),
        )
        if results is None:
            await websocket.send_json(
                {"type": "error", "message": "Analysis completed but result files unreadable.", "fatal": True}
            )
            return

        await websocket.send_json({"type": "session_end", **results})
```

- [ ] **Step 7: Run the full backend suite to confirm no regressions**

Run: `python -m pytest tests/ -q`
Expected: PASS (all existing tests, incl. `test_api_stream.py`, still green).

- [ ] **Step 8: Commit**

```bash
git add api.py tests/test_api_result.py
git commit -m "refactor(api): extract read_analysis_results helper; use in upload + stream"
```

---

### Task 2: Backend — `GET /api/sessions/{session_id}/result` endpoint

**Files:**
- Modify: `api.py` (add route after `get_sessions` at `api.py:186-193`)
- Test: `tests/test_api_result.py` (append)

- [ ] **Step 1: Write failing endpoint tests**

Append to `tests/test_api_result.py`:

```python
@pytest.mark.asyncio
async def test_result_ready_returns_200(client, tmp_path):
    sid = "c" * 32
    _write_results(tmp_path, sid)
    resp = await client.get(f"/api/sessions/{sid}/result")
    assert resp.status_code == 200
    body = resp.json()
    assert body["feedback"] == "Good job."
    assert body["transcript"]["text"] == "hi"
    assert body["features"]["words_per_minute"] == 100


@pytest.mark.asyncio
async def test_result_pending_returns_202_without_error_log(client, tmp_path, caplog):
    sid = "d" * 32
    _write_results(tmp_path, sid, feedback=False)  # partial → pending
    with caplog.at_level(logging.ERROR):
        resp = await client.get(f"/api/sessions/{sid}/result")
    assert resp.status_code == 202
    assert resp.json() == {"status": "pending"}
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR], "202 must not log ERROR"


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["short", "g" * 32, "A" * 32, "../../etc/passwd"])
async def test_result_invalid_id_returns_400(client, bad):
    resp = await client.get(f"/api/sessions/{bad}/result")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_result_corrupt_file_returns_500(client, tmp_path):
    sid = "e" * 32
    _write_results(tmp_path, sid, corrupt=True)  # all present but transcript is bad JSON
    resp = await client.get(f"/api/sessions/{sid}/result")
    assert resp.status_code == 500
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_api_result.py -k result_ -v`
Expected: FAIL — 404 (route not defined) for the 200/202/500 cases.

- [ ] **Step 3: Add the endpoint**

In `api.py`, add immediately after the `get_sessions` function (after `api.py:193`):

```python
@app.get("/api/sessions/{session_id}/result")
async def get_session_result(session_id: str):
    """Fetch the completed analysis for a live session by id.

    Durable fallback for live streaming: the result files are written even if
    the WebSocket closed before delivery. Returns 202 while still pending so the
    frontend can poll. The existence check avoids read_file's not-found ERROR
    logging on every poll during the normal pending window.
    """
    if not SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session id.")

    transcript_path, features_path, feedback_path = result_file_paths(session_id)

    if not all(Path(p).exists() for p in (transcript_path, features_path, feedback_path)):
        return JSONResponse(status_code=202, content={"status": "pending"})

    results = read_analysis_results(transcript_path, features_path, feedback_path)
    if results is None:
        raise HTTPException(status_code=500, detail="Result files present but unreadable.")
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_api_result.py -v`
Expected: PASS (all helper + endpoint tests).

- [ ] **Step 5: Commit**

```bash
git add api.py tests/test_api_result.py
git commit -m "feat(api): add GET /api/sessions/{id}/result poll endpoint"
```

---

### Task 3: Frontend — `getSessionResult` in `api.ts`

**Files:**
- Modify: `frontend/src/api.ts` (after `getStreamWsUrl` at `frontend/src/api.ts:82-84`)
- Test: `frontend/src/api.test.ts` (new)

- [ ] **Step 1: Write failing tests**

Create `frontend/src/api.test.ts`:

```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest';

const mockGet = vi.fn();
vi.mock('axios', () => ({
  default: { create: () => ({ get: mockGet, post: vi.fn() }) },
}));

import { getSessionResult } from './api';

describe('getSessionResult', () => {
  beforeEach(() => mockGet.mockReset());

  it('maps 200 to {status:"ready", data}', async () => {
    const payload = { message: 'Live session complete.', transcript: { text: 'hi' }, features: {}, feedback: 'ok' };
    mockGet.mockResolvedValue({ status: 200, data: payload });
    await expect(getSessionResult('a'.repeat(32))).resolves.toEqual({ status: 'ready', data: payload });
  });

  it('maps 202 to {status:"pending"}', async () => {
    mockGet.mockResolvedValue({ status: 202, data: { status: 'pending' } });
    await expect(getSessionResult('b'.repeat(32))).resolves.toEqual({ status: 'pending' });
  });

  it('throws on transport error (treated as a failed attempt by caller)', async () => {
    mockGet.mockRejectedValue(new Error('Network Error'));
    await expect(getSessionResult('c'.repeat(32))).rejects.toThrow('Network Error');
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npx vitest run src/api.test.ts`
Expected: FAIL — `getSessionResult is not a function` / export missing.

- [ ] **Step 3: Implement `getSessionResult`**

Append to `frontend/src/api.ts`:

```typescript
export type SessionResultPoll =
  | { status: 'ready'; data: UploadResponse }
  | { status: 'pending' };

export async function getSessionResult(sessionId: string): Promise<SessionResultPoll> {
  const res = await client.get<UploadResponse | { status: 'pending' }>(
    `/api/sessions/${sessionId}/result`,
    { validateStatus: (s) => s === 200 || s === 202 },
  );
  if (res.status === 202) return { status: 'pending' };
  return { status: 'ready', data: res.data as UploadResponse };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/api.test.ts`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api.ts frontend/src/api.test.ts
git commit -m "feat(frontend): add getSessionResult poll helper"
```

---

### Task 4: Frontend — `useLiveStream` reports sessionId, survives close-during-stopping

**Files:**
- Modify: `frontend/src/hooks/useLiveStream.ts` (`UseLiveStreamOptions` at `:28-31`, `handleServerMessage` session_started at `:65-85`, `stop` at `:183-208`, `ws.onclose` at `:173-180`)
- Test: `frontend/src/hooks/useLiveStream.test.ts`

- [ ] **Step 1: Write failing tests**

Append these to the existing `describe` in `frontend/src/hooks/useLiveStream.test.ts` (reuse the file's existing `MockWebSocket`, `makeMockCapture`, `opts` helpers; mirror their shapes if names differ):

```typescript
it('calls onSessionEnding with the server session id when stopped while recording', async () => {
  const ws = new MockWebSocket('ws://test/api/stream');
  const capture = makeMockCapture();
  const onSessionEnding = vi.fn();
  const { result } = renderHook(() => useLiveStream({ ...opts(ws, capture), onSessionEnding }));
  await act(async () => { await result.current.start(); });
  act(() => ws.fireOpen());
  act(() => ws.fireServerMessage({ type: 'session_started', session_id: 'sess-123' }));
  await waitFor(() => expect(result.current.status).toBe('recording'));
  act(() => result.current.stop());
  expect(onSessionEnding).toHaveBeenCalledWith('sess-123');
});

it('does NOT set error when the socket closes while stopping', async () => {
  const ws = new MockWebSocket('ws://test/api/stream');
  const capture = makeMockCapture();
  const { result } = renderHook(() => useLiveStream(opts(ws, capture)));
  await act(async () => { await result.current.start(); });
  act(() => ws.fireOpen());
  act(() => ws.fireServerMessage({ type: 'session_started', session_id: 'sess-123' }));
  await waitFor(() => expect(result.current.status).toBe('recording'));
  act(() => result.current.stop());            // status → 'stopping', socket still open
  act(() => ws.fireClose());                    // socket drops before session_end
  expect(result.current.status).toBe('stopping');
  expect(result.current.error).toBeNull();
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npx vitest run src/hooks/useLiveStream.test.ts`
Expected: FAIL — `onSessionEnding` undefined / status becomes `'error'` on close.

- [ ] **Step 3: Add `onSessionEnding` to options**

In `frontend/src/hooks/useLiveStream.ts`, extend the options interface (`:28-31`):

```typescript
export interface UseLiveStreamOptions {
  createWebSocket?: (url: string) => WebSocket;
  createCapture?: () => PcmAudioCaptureLike;
  onSessionEnding?: (sessionId: string) => void;
}
```

- [ ] **Step 4: Capture the session id**

Add a ref alongside the others (near `:42`):

```typescript
  const sessionIdRef = useRef<string>('');
```

In the `session_started` case of `handleServerMessage` (at `:65`), as the first line of the block:

```typescript
      case 'session_started': {
        sessionIdRef.current = msg.session_id;
```

- [ ] **Step 5: Report sessionId on stop**

In `stop()`, in the "Normal stop from recording" branch (after the `wsRef.current.send(JSON.stringify({ type: 'end' }))` at `:205-207`), add:

```typescript
    if (sessionIdRef.current) {
      options.onSessionEnding?.(sessionIdRef.current);
    }
```

Also add `options.onSessionEnding` to the `useCallback` dependency array of `stop` (currently `[cleanup]` at `:208`) → `[cleanup, options.onSessionEnding]`.

- [ ] **Step 6: Don't error on close during stopping**

In `ws.onclose` (`:173-180`), add `'stopping'` to the exclusions:

```typescript
    ws.onclose = () => {
      const cur = statusRef.current;
      if (cur !== 'completed' && cur !== 'error' && cur !== 'idle' && cur !== 'stopping') {
        setError('Connection closed unexpectedly.');
        setStatus('error');
      }
      void cleanup();
    };
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/hooks/useLiveStream.test.ts`
Expected: PASS (existing + 2 new).

- [ ] **Step 8: Commit**

```bash
git add frontend/src/hooks/useLiveStream.ts frontend/src/hooks/useLiveStream.test.ts
git commit -m "feat(frontend): useLiveStream reports sessionId, tolerates close-during-stopping"
```

---

### Task 5: Frontend — `VerbalVector` forwards `onLiveSessionEnding`

**Files:**
- Modify: `frontend/src/components/VerbalVector.tsx` (props `:14-17`, hook init `:60`)
- Test: `frontend/src/components/VerbalVector.test.tsx`

- [ ] **Step 1: Write a failing test**

Append to the `VerbalVector live mode` describe in `frontend/src/components/VerbalVector.test.tsx` (reuse the file's `FakeWebSocket`, `setupNavigatorMock`, `MockMediaRecorder`):

```typescript
it('forwards onLiveSessionEnding with the session id when live recording stops', async () => {
  setupNavigatorMock([]);
  (global as unknown as { WebSocket: unknown }).WebSocket = FakeWebSocket;
  const onLiveSessionEnding = vi.fn();
  render(
    <VerbalVector
      onAnalysisComplete={() => {}}
      onNavigate={() => {}}
      onLiveSessionEnding={onLiveSessionEnding}
    />,
  );
  await userEvent.click(screen.getByLabelText(/live/i));
  await userEvent.click(screen.getByRole('button', { name: /record audio/i }));
  await waitFor(() => expect(FakeWebSocket.instances.length).toBe(1));
  const ws = FakeWebSocket.instances[0];
  if (ws.onopen) ws.onopen(new Event('open'));
  if (ws.onmessage) {
    ws.onmessage(new MessageEvent('message', {
      data: JSON.stringify({ type: 'session_started', session_id: 'live-xyz' }),
    }));
  }
  await waitFor(() => expect(screen.getByRole('button', { name: /stop recording/i })).not.toBeDisabled());
  await userEvent.click(screen.getByRole('button', { name: /stop recording/i }));
  expect(onLiveSessionEnding).toHaveBeenCalledWith('live-xyz');
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/components/VerbalVector.test.tsx -t "forwards onLiveSessionEnding"`
Expected: FAIL — prop not wired; spy not called.

- [ ] **Step 3: Add and wire the prop**

In `frontend/src/components/VerbalVector.tsx`, extend the props interface (`:14-17`):

```typescript
interface VerbalVectorProps {
  onAnalysisComplete: (result: AnalysisResult) => void;
  onNavigate: (view: NavView) => void;
  onLiveSessionEnding?: (sessionId: string) => void;
}
```

Add it to the destructured params (`:41`):

```typescript
const VerbalVector: React.FC<VerbalVectorProps> = ({ onAnalysisComplete, onNavigate, onLiveSessionEnding }) => {
```

Pass it into the hook (`:60`):

```typescript
  const live = useLiveStream({ onSessionEnding: onLiveSessionEnding });
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run src/components/VerbalVector.test.tsx`
Expected: PASS (existing + new).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/VerbalVector.tsx frontend/src/components/VerbalVector.test.tsx
git commit -m "feat(frontend): VerbalVector forwards onLiveSessionEnding to App"
```

---

### Task 6: Frontend — App-level durable poller, auto-show + banner + retry

**Files:**
- Modify: `frontend/src/App.tsx` (whole component `:11-52`)
- Test: `frontend/src/App.test.tsx` (new)

- [ ] **Step 1: Write failing tests**

Create `frontend/src/App.test.tsx`. The child views are stubbed so the test drives App's poll logic directly; `VerbalVector` is stubbed to expose a button that fires `onLiveSessionEnding`:

```typescript
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const mockGetSessionResult = vi.fn();
vi.mock('./api', () => ({ getSessionResult: (id: string) => mockGetSessionResult(id) }));

vi.mock('./components/VerbalVector', () => ({
  default: ({ onLiveSessionEnding, onNavigate }: {
    onLiveSessionEnding?: (id: string) => void;
    onNavigate: (v: string) => void;
  }) => (
    <div>
      <button onClick={() => onLiveSessionEnding?.('sess-1')}>fire-ending</button>
      <button onClick={() => onNavigate('query')}>go-query</button>
    </div>
  ),
}));
vi.mock('./components/ResultsDisplay', () => ({
  default: () => <div>RESULTS_VIEW</div>,
}));
vi.mock('./components/QueryInterface', () => ({ default: () => <div>QUERY_VIEW</div> }));
vi.mock('./components/SessionHistory', () => ({ default: () => <div>HISTORY_VIEW</div> }));

import App from './App';

const READY = { message: 'Live session complete.', transcript: { text: 'hi' }, features: {}, feedback: 'ok' };

describe('App live poll fallback', () => {
  beforeEach(() => { mockGetSessionResult.mockReset(); vi.useFakeTimers(); });
  afterEach(() => { vi.runOnlyPendingTimers(); vi.useRealTimers(); });

  it('auto-shows results when ready while still on the analysis view', async () => {
    mockGetSessionResult.mockResolvedValue({ status: 'ready', data: READY });
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<App />);
    await user.click(screen.getByText('fire-ending'));
    await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
    await waitFor(() => expect(screen.getByText('RESULTS_VIEW')).toBeInTheDocument());
  });

  it('shows a "ready" banner (no forced nav) when the user navigated away', async () => {
    mockGetSessionResult.mockResolvedValue({ status: 'ready', data: READY });
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<App />);
    await user.click(screen.getByText('fire-ending'));
    await user.click(screen.getByText('go-query'));            // unmounts stubbed VerbalVector
    await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
    await waitFor(() => expect(screen.getByText(/analysis is ready/i)).toBeInTheDocument());
    expect(screen.getByText('QUERY_VIEW')).toBeInTheDocument();  // not yanked away
    await user.click(screen.getByRole('button', { name: /view results/i }));
    expect(screen.getByText('RESULTS_VIEW')).toBeInTheDocument();
  });

  it('keeps polling through pending then shows results', async () => {
    mockGetSessionResult
      .mockResolvedValueOnce({ status: 'pending' })
      .mockResolvedValueOnce({ status: 'ready', data: READY });
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<App />);
    await user.click(screen.getByText('fire-ending'));
    await act(async () => { await vi.advanceTimersByTimeAsync(2000); });  // pending
    await act(async () => { await vi.advanceTimersByTimeAsync(2000); });  // ready
    await waitFor(() => expect(screen.getByText('RESULTS_VIEW')).toBeInTheDocument());
  });

  it('shows a timeout error banner after the attempt cap', async () => {
    mockGetSessionResult.mockResolvedValue({ status: 'pending' });
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<App />);
    await user.click(screen.getByText('fire-ending'));
    await act(async () => { await vi.advanceTimersByTimeAsync(45 * 2000); });
    await waitFor(() => expect(screen.getByText(/taking longer than expected/i)).toBeInTheDocument());
    expect(screen.getByRole('button', { name: /check history/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npx vitest run src/App.test.tsx`
Expected: FAIL — App doesn't render banners / doesn't poll / doesn't pass `onLiveSessionEnding`.

- [ ] **Step 3: Rewrite `App.tsx` with the durable poller**

Replace the entire contents of `frontend/src/App.tsx` with:

```typescript
import { useState, useEffect, type CSSProperties } from "react";
import VerbalVector, { AnalysisResult } from "./components/VerbalVector";
import ResultsDisplay from "./components/ResultsDisplay";
import QueryInterface from "./components/QueryInterface";
import SessionHistory from "./components/SessionHistory";
import { getSessionResult, type NavView } from "./api";
import "./App.css";

type View = NavView | "results";
type PendingStatus = "analyzing" | "ready" | "error";

const POLL_INTERVAL_MS = 2000;
const POLL_MAX_ATTEMPTS = 45; // ~90s; tuned to observed ~61s pipeline

const bannerBase: CSSProperties = {
  padding: "0.75rem 1rem",
  textAlign: "center",
  fontSize: "0.875rem",
  display: "flex",
  gap: "0.75rem",
  alignItems: "center",
  justifyContent: "center",
};

function App() {
  const [currentView, setCurrentView] = useState<View>("analysis");
  const [analysisData, setAnalysisData] = useState<AnalysisResult | null>(null);
  const [scopedSourceId, setScopedSourceId] = useState<string>("");
  const [pending, setPending] = useState<{ sessionId: string; status: PendingStatus } | null>(null);
  const [analysisKey, setAnalysisKey] = useState(0); // bump to remount VerbalVector on retry

  const handleAnalysisComplete = (result: AnalysisResult) => {
    setAnalysisData(result);
    setPending(null); // fast path delivered — stop any poll
    setCurrentView("results");
  };

  const handleAnalyzeAnother = () => {
    setAnalysisData(null);
    setCurrentView("analysis");
  };

  const handleNavigate = (view: NavView) => {
    setScopedSourceId("");
    setCurrentView(view);
  };

  const handleQuerySession = (sourceId: string) => {
    setScopedSourceId(sourceId);
    setCurrentView("query");
  };

  const handleLiveSessionEnding = (sessionId: string) => {
    setAnalysisData(null);
    setPending({ sessionId, status: "analyzing" });
  };

  // Durable poll loop — lives in App so it survives the view-switch unmount of
  // VerbalVector that drops the WebSocket (the original bug).
  useEffect(() => {
    if (pending?.status !== "analyzing") return;
    const sessionId = pending.sessionId;
    let cancelled = false;
    let attempts = 0;
    let timer: ReturnType<typeof setTimeout>;

    const tick = async () => {
      attempts += 1;
      try {
        const res = await getSessionResult(sessionId);
        if (cancelled) return;
        if (res.status === "ready") {
          setAnalysisData(res.data);
          setPending({ sessionId, status: "ready" });
          return;
        }
      } catch {
        // transport/5xx — counts as a failed attempt, keep polling
      }
      if (cancelled) return;
      if (attempts >= POLL_MAX_ATTEMPTS) {
        setPending({ sessionId, status: "error" });
        return;
      }
      timer = setTimeout(tick, POLL_INTERVAL_MS);
    };

    timer = setTimeout(tick, POLL_INTERVAL_MS);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [pending]);

  // Auto-show in place: if the result is ready while the user is still on the
  // analysis view, route to results. If they navigated away, the banner handles it.
  useEffect(() => {
    if (pending?.status === "ready" && currentView === "analysis") {
      setCurrentView("results");
      setPending(null);
    }
  }, [pending, currentView]);

  const showReadyBanner = pending?.status === "ready" && currentView !== "analysis";

  return (
    <div className="App">
      {pending?.status === "analyzing" && currentView !== "analysis" && (
        <div style={{ ...bannerBase, background: "#eef2ff", color: "#4338ca" }}>
          Analyzing your last recording…
        </div>
      )}
      {showReadyBanner && (
        <div style={{ ...bannerBase, background: "#ecfdf5", color: "#065f46" }}>
          <span>✓ Your analysis is ready</span>
          <button onClick={() => { setPending(null); setCurrentView("results"); }}>
            View results
          </button>
        </div>
      )}
      {pending?.status === "error" && (
        <div style={{ ...bannerBase, background: "#fef2f2", color: "#b91c1c" }}>
          <span>Analysis is taking longer than expected.</span>
          <button onClick={() => { setPending(null); handleNavigate("history"); }}>
            Check History
          </button>
          <button onClick={() => { setPending(null); setAnalysisKey((k) => k + 1); setCurrentView("analysis"); }}>
            Retry
          </button>
        </div>
      )}

      {currentView === "analysis" && (
        <VerbalVector
          key={analysisKey}
          onAnalysisComplete={handleAnalysisComplete}
          onNavigate={handleNavigate}
          onLiveSessionEnding={handleLiveSessionEnding}
        />
      )}
      {currentView === "results" && (
        <ResultsDisplay analysisResult={analysisData} onAnalyzeAnother={handleAnalyzeAnother} onNavigate={handleNavigate} />
      )}
      {currentView === "query" && (
        <QueryInterface key={scopedSourceId} onNavigate={handleNavigate} initialSourceId={scopedSourceId} />
      )}
      {currentView === "history" && (
        <SessionHistory onNavigate={handleNavigate} onQuerySession={handleQuerySession} />
      )}
    </div>
  );
}

export default App;
```

- [ ] **Step 4: Run the App tests to verify they pass**

Run: `cd frontend && npx vitest run src/App.test.tsx`
Expected: PASS (4 passed).

- [ ] **Step 5: Run the full frontend suite + typecheck**

Run: `cd frontend && npx vitest run && npx tsc --noEmit`
Expected: PASS, no type errors.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/App.tsx frontend/src/App.test.tsx
git commit -m "feat(frontend): App-level durable poll for live results (survives nav)"
```

---

## Final Verification (after all tasks)

- [ ] Backend: `python -m pytest tests/ -q` — all green.
- [ ] Frontend: `cd frontend && npx vitest run && npx tsc --noEmit && npx eslint src` — all green.
- [ ] Manual smoke: start a Live session, Stop, immediately navigate to Query; within ~60–90 s a "✓ Your analysis is ready" banner appears; clicking it shows the results. Repeat staying on the analysis view → auto-routes to results.
- [ ] Dispatch `pr-review-toolkit:code-reviewer` + `silent-failure-hunter` (polling/error handling) + `pr-test-analyzer` (new tests). Address Critical/Important.
- [ ] Push branch, open PR with summary + test plan + rollback notes; run `pr-review-toolkit:review-pr`.

## Rollback

All changes are additive. To roll back: revert the frontend `App.tsx` poller commit (live falls back to the prior WS-only behavior) and/or the backend endpoint commit (the route 404s; nothing else depends on it). The `read_analysis_results` refactor (Task 1) is behavior-preserving and can stay.
