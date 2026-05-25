# Phase 3d.2: Frontend Diarization Rendering Implementation Plan

> **Revised after staff-engineer review.** Changes from initial draft: (1) multi-speaker render now handles `speaker === null` correctly — null-speaker segments render WITHOUT the "Speaker N:" label so they don't show "Speaker null:" literally when mixed with real speakers; (2) useLiveStream normalizes `msg.speaker ?? null` defensively at the boundary; (3) ResultsDisplay's conditional render is extracted to a `renderTranscript` helper instead of an IIFE inside JSX.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render speaker labels in the live transcript pane and the batch results transcript. Strip labels when only one speaker was detected so single-speaker recordings look identical to today's UI.

**Architecture:** Frontend-only. Backend (Phase 3d.1, already merged) returns speaker per WS transcript message and `utterances`/`speakers` on the batch transcript dict. This plan consumes those fields:
- `useLiveStream` accumulates finals as `{text, speaker}[]` instead of `string[]`
- `LiveTranscript` renders each final with a "Speaker N:" prefix; interim stays unattributed (lighter color, no speaker label)
- `ResultsDisplay` renders `transcript.utterances` as a speaker-labeled list when present; falls back to flat text otherwise
- "Strip labels when single speaker" lives per-component (one-line logic — no shared helper)

**Tech Stack:** Same as 3c.2 — React 19, TS 5.7, Vitest + RTL.

**Branch:** `phase3d.2/frontend-diarization` off `main`.

**Locked product decisions (confirmed):**
- Finals rendered as a vertical list of "Speaker N: text" segments; interim appended in lighter color (no speaker tag on interim — sidesteps Deepgram's interim-speaker churn).
- No per-speaker colors — just text prefixes.
- Strip labels when 0 or 1 unique speakers in the finals.
- ResultsDisplay shows the same speaker-labeled list when batch result has utterances; falls back to existing flat `<pre>` when not.
- Interim speaker churn is acceptable — the interim has no label anyway, so re-classification on finalization is invisible.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `frontend/src/api.ts` | Modify | Add `speaker: number | null` to ServerMessage transcript; add `utterances?`/`speakers?` to UploadResponse transcript |
| `frontend/src/hooks/useLiveStream.ts` | Modify | Change `finals: string[]` → `finals: {text, speaker}[]`; capture speaker on session_started's transcript handler |
| `frontend/src/hooks/useLiveStream.test.ts` | Modify | Update existing finals assertions to the new shape; add tests covering speaker capture |
| `frontend/src/components/LiveTranscript.tsx` | Modify | Render `Speaker N: text` per final when there are 2+ speakers; flat rendering when 0/1 speakers |
| `frontend/src/components/LiveTranscript.test.tsx` | Modify | Update existing tests for new finals shape; add multi-speaker + strip-when-1-speaker tests |
| `frontend/src/components/ResultsDisplay.tsx` | Modify | When `transcript.utterances` is present + speakers.length > 1, render speaker-labeled list; otherwise existing flat text |
| `frontend/src/components/ResultsDisplay.test.tsx` | Create | New file — no tests today. Cover both multi-speaker and single-speaker render paths. |
| `docs/superpowers/specs/2026-05-24-phase3d-frontend-manual-test-checklist.md` | Create | Manual verification including the interim-churn known limitation |

**Decomposition rationale:**
- `api.ts` types + `useLiveStream` shape change bundled (Task 1) — they're tightly coupled, and the consumers (LiveTranscript, ResultsDisplay) only need the new shape ready.
- `LiveTranscript` + `ResultsDisplay` are independent consumers — separate tasks (2 and 3).
- The strip-when-1-speaker logic is one line per component (`finals.filter(f => f.speaker !== null).map(f => f.speaker)` → unique → check length). No shared helper for v1.

---

## Task 1: api.ts types + useLiveStream finals shape

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/hooks/useLiveStream.ts`
- Modify: `frontend/src/hooks/useLiveStream.test.ts`

### Step 1.1: Update api.ts types

In `frontend/src/api.ts`:

1. Update `UploadResponse.transcript` shape to include the new fields (still backward-compat — they're optional):

```ts
export interface UploadResponse {
  message: string;
  transcript:
    | {
        text: string;
        language?: string;
        segments?: any[];
        utterances?: { speaker: number | null; text: string; start: number; end: number; confidence: number }[];
        speakers?: number[];
      }
    | string;
  features: Record<string, any>;
  feedback: string;
}
```

2. Update the `ServerMessage.transcript` variant to include speaker:

```ts
export type ServerMessage =
  | { type: 'session_started'; session_id: string }
  | { type: 'transcript'; text: string; is_final: boolean; speaker: number | null }
  | { type: 'session_end'; transcript: UploadResponse['transcript']; features: UploadResponse['features']; feedback: string }
  | { type: 'error'; message: string; fatal: boolean };
```

### Step 1.2: Update useLiveStream tests for the new finals shape

In `frontend/src/hooks/useLiveStream.test.ts`, change every place that asserts on `result.current.finals === [...]` from `string[]` to `{text, speaker}[]`. Also update the `fireServerMessage` calls that send a transcript to include `speaker`:

For example, the existing test:

```ts
act(() => ws.fireServerMessage({ type: 'transcript', text: 'hello world', is_final: true }));
expect(result.current.interim).toBe('');
expect(result.current.finals).toEqual(['hello world']);
```

Becomes:

```ts
act(() => ws.fireServerMessage({ type: 'transcript', text: 'hello world', is_final: true, speaker: 0 }));
expect(result.current.interim).toBe('');
expect(result.current.finals).toEqual([{ text: 'hello world', speaker: 0 }]);
```

Find ALL transcript-related assertions and updates. There are 2-3 in `test_forwards_transcripts_to_interim_and_finals_state` plus possibly others.

Also UPDATE the `test_transcript_callback_forwards_to_client_via_queue` style tests in `tests/test_api_stream.py` — wait, that's backend tests, not frontend. Skip.

Add a new test verifying multi-speaker finals accumulate correctly:

```ts
it('accumulates finals with their speaker IDs in order', async () => {
  const ws = new MockWebSocket('ws://test/api/stream');
  const capture = makeMockCapture();
  const { result } = renderHook(() => useLiveStream(opts(ws, capture)));

  await act(async () => { await result.current.start(); });
  act(() => ws.fireOpen());
  act(() => ws.fireServerMessage({ type: 'session_started', session_id: 'x' }));
  await waitFor(() => expect(result.current.status).toBe('recording'));

  act(() => ws.fireServerMessage({ type: 'transcript', text: 'Hello.', is_final: true, speaker: 0 }));
  act(() => ws.fireServerMessage({ type: 'transcript', text: 'Hi there.', is_final: true, speaker: 1 }));
  act(() => ws.fireServerMessage({ type: 'transcript', text: 'How are you?', is_final: true, speaker: 0 }));

  expect(result.current.finals).toEqual([
    { text: 'Hello.', speaker: 0 },
    { text: 'Hi there.', speaker: 1 },
    { text: 'How are you?', speaker: 0 },
  ]);
});

it('accepts transcript messages with speaker=null (interim from Deepgram)', async () => {
  const ws = new MockWebSocket('ws://test/api/stream');
  const capture = makeMockCapture();
  const { result } = renderHook(() => useLiveStream(opts(ws, capture)));

  await act(async () => { await result.current.start(); });
  act(() => ws.fireOpen());
  act(() => ws.fireServerMessage({ type: 'session_started', session_id: 'x' }));
  await waitFor(() => expect(result.current.status).toBe('recording'));

  // Interim with null speaker
  act(() => ws.fireServerMessage({ type: 'transcript', text: 'partial...', is_final: false, speaker: null }));
  expect(result.current.interim).toBe('partial...');

  // Final with null speaker (rare but possible) — should still record as a final with speaker=null
  act(() => ws.fireServerMessage({ type: 'transcript', text: 'finalized.', is_final: true, speaker: null }));
  expect(result.current.finals).toEqual([{ text: 'finalized.', speaker: null }]);
});
```

### Step 1.3: Run tests, confirm fail

```bash
cd /Users/yashdeshmukh/Downloads/Cornell_docs/Spring25/AI_Engineers/VerbalVector/frontend
npm test -- useLiveStream
```

Expected: many tests fail — existing tests because the new `speaker` field on transcript messages now travels through, and the old finals assertion (`expect(...).toEqual(['hello world'])`) won't match `[{text:'hello world', speaker:0}]`. The 2 new tests also fail.

### Step 1.4: Update useLiveStream.ts

In `frontend/src/hooks/useLiveStream.ts`:

1. Change the `finals` state type and the `UseLiveStreamReturn` interface:

```ts
export interface FinalSegment {
  text: string;
  speaker: number | null;
}

export interface UseLiveStreamReturn {
  status: LiveStreamStatus;
  interim: string;
  finals: FinalSegment[];  // CHANGED from string[]
  error: string | null;
  result: UploadResponse | null;
  start: (sessionLabel?: string, deviceId?: string) => Promise<void>;
  stop: () => void;
}
```

And the state:

```ts
const [finals, setFinals] = useState<FinalSegment[]>([]);
```

2. Update the `handleServerMessage` transcript case to push the new shape. Normalize `msg.speaker ?? null` defensively at the boundary so that if a server frame ever lacks the speaker field (test fixtures, stale backend during deploy, protocol drift), we get `null` not `undefined` — keeping the FinalSegment type honest at runtime:

```ts
case 'transcript': {
  const speaker = msg.speaker ?? null;
  if (msg.is_final) {
    setFinals((prev) => [...prev, { text: msg.text, speaker }]);
    setInterim('');
  } else {
    setInterim(msg.text);
    // Ignore speaker for interim — we don't label interim text
  }
  break;
}
```

### Step 1.5: Run tests, confirm pass

```bash
npm test -- useLiveStream
```

Expected: all useLiveStream tests pass with the new shape.

### Step 1.6: Build + lint

```bash
npm run build
npm run lint
```

Expected: build may fail because `LiveTranscript.tsx` still expects `finals: string[]` — Task 2 will fix. If it's an outright build error, you may need to do a stopgap: cast in `VerbalVector.tsx` where `live.finals` is passed to LiveTranscript, OR commit Task 1 with a known-failing build (NOT recommended — better to make Task 2 immediately follow, even if it means leaving Task 1 uncommitted briefly).

**Recommendation:** check if `LiveTranscript` is referenced via `live.finals` anywhere in `VerbalVector.tsx`. If yes, the TypeScript build breaks until Task 2 is done. You have two options:
- (A) Commit Task 1 with a temporary `.map(f => f.text)` adapter at the LiveTranscript call site in VerbalVector.tsx. Task 2 removes the adapter.
- (B) Squash Task 1 + Task 2 into a single commit. Worse for review but avoids the broken intermediate state.

Go with (A). It's slightly ugly but each commit is independently runnable.

### Step 1.7: Adapter in VerbalVector.tsx (if needed)

If `npm run build` fails because of `live.finals` being passed to `LiveTranscript`, find that call site in `VerbalVector.tsx` and temporarily map:

```tsx
<LiveTranscript
  interim={live.interim}
  finals={live.finals.map(f => f.text)}  // TEMP: Task 2 will pass FinalSegment[] directly
  status={live.status}
/>
```

The `// TEMP` comment makes the adapter visible for cleanup.

### Step 1.8: Commit

```bash
git add frontend/src/api.ts frontend/src/hooks/useLiveStream.ts frontend/src/hooks/useLiveStream.test.ts frontend/src/components/VerbalVector.tsx
git commit -m "feat(frontend): accept speaker on transcript messages; finals carry FinalSegment shape

Updates api.ts ServerMessage transcript variant to include speaker
(number | null), and UploadResponse.transcript to include the new
optional utterances + speakers fields from Phase 3d.1.

useLiveStream's finals state changes from string[] to FinalSegment[]
({text, speaker}). Interim stays as plain string — Deepgram's interim
speaker IDs shift and we don't label interim text in the UI.

VerbalVector.tsx has a temporary .map(f => f.text) adapter when
passing finals to LiveTranscript — Task 2 will remove it once
LiveTranscript consumes the new shape directly."
```

No `Co-Authored-By` trailer.

---

## Task 2: LiveTranscript renders speaker labels with strip-when-1 logic

**Files:**
- Modify: `frontend/src/components/LiveTranscript.tsx`
- Modify: `frontend/src/components/LiveTranscript.test.tsx`
- Modify: `frontend/src/components/VerbalVector.tsx` (remove the Task 1 adapter)

### Step 2.1: Update LiveTranscript tests

In `frontend/src/components/LiveTranscript.test.tsx`, update the existing tests' `finals` props from `string[]` to `FinalSegment[]`. For example:

```tsx
// Before
<LiveTranscript interim="" finals={['Hello there.', 'How are you today?']} status="recording" />

// After
<LiveTranscript
  interim=""
  finals={[{ text: 'Hello there.', speaker: 0 }, { text: 'How are you today?', speaker: 0 }]}
  status="recording"
/>
```

(All existing tests use speaker=0 for consistency with single-speaker rendering.)

Add new tests for multi-speaker rendering and strip-when-1-speaker:

```tsx
it('renders Speaker N: prefix when multiple speakers are detected', () => {
  render(
    <LiveTranscript
      interim=""
      finals={[
        { text: 'Hello.', speaker: 0 },
        { text: 'Hi.', speaker: 1 },
      ]}
      status="recording"
    />,
  );
  expect(screen.getByText(/Speaker 0/)).toBeInTheDocument();
  expect(screen.getByText(/Speaker 1/)).toBeInTheDocument();
  expect(screen.getByText(/Hello\./)).toBeInTheDocument();
  expect(screen.getByText(/Hi\./)).toBeInTheDocument();
});

it('does NOT render Speaker N: prefix when only one speaker is detected', () => {
  render(
    <LiveTranscript
      interim=""
      finals={[
        { text: 'Hello.', speaker: 0 },
        { text: 'How are you?', speaker: 0 },
      ]}
      status="recording"
    />,
  );
  expect(screen.queryByText(/Speaker 0/)).not.toBeInTheDocument();
  expect(screen.getByText(/Hello\./)).toBeInTheDocument();
});

it('does NOT render labels when all speakers are null', () => {
  render(
    <LiveTranscript
      interim=""
      finals={[
        { text: 'Hello.', speaker: null },
        { text: 'World.', speaker: null },
      ]}
      status="recording"
    />,
  );
  expect(screen.queryByText(/Speaker/)).not.toBeInTheDocument();
  expect(screen.getByText(/Hello\./)).toBeInTheDocument();
});

it('keeps interim in lighter color even in multi-speaker mode (no speaker tag on interim)', () => {
  render(
    <LiveTranscript
      interim="continuing..."
      finals={[
        { text: 'Hello.', speaker: 0 },
        { text: 'Hi.', speaker: 1 },
      ]}
      status="recording"
    />,
  );
  const interimEl = screen.getByText('continuing...');
  expect(interimEl).toHaveStyle({ color: '#94a3b8' });
});

it('renders null-speaker segments without a label when mixed with real speakers', () => {
  render(
    <LiveTranscript
      interim=""
      finals={[
        { text: 'Hello.', speaker: 0 },
        { text: 'untagged.', speaker: null },
        { text: 'Hi.', speaker: 1 },
      ]}
      status="recording"
    />,
  );
  expect(screen.getByText(/Speaker 0/)).toBeInTheDocument();
  expect(screen.getByText(/Speaker 1/)).toBeInTheDocument();
  // The null-speaker segment must NOT render "Speaker null:" literally
  expect(screen.queryByText(/Speaker null/)).not.toBeInTheDocument();
  expect(screen.getByText(/untagged\./)).toBeInTheDocument();
});
```

### Step 2.2: Run tests, confirm fail

```bash
npm test -- LiveTranscript
```

Expected: existing tests fail because `finals` shape changed; new tests fail because the component doesn't render speaker labels yet.

### Step 2.3: Update LiveTranscript.tsx

Update the props interface to use `FinalSegment[]`:

```tsx
import type { FinalSegment, LiveStreamStatus } from '../hooks/useLiveStream';

interface LiveTranscriptProps {
  interim: string;
  finals: FinalSegment[];
  status: LiveStreamStatus;
}
```

Update the render body:

```tsx
export const LiveTranscript: React.FC<LiveTranscriptProps> = ({ interim, finals, status }) => {
  const hasAnyText = interim.length > 0 || finals.length > 0;

  // Determine whether to show speaker labels: strip when 0 or 1 unique non-null speakers
  const uniqueSpeakers = new Set(
    finals.map((f) => f.speaker).filter((s): s is number => s !== null),
  );
  const showSpeakers = uniqueSpeakers.size > 1;

  return (
    <div style={wrapperStyle} aria-live="polite" role="log">
      {!hasAnyText && (
        <span style={placeholderStyle}>
          {status === 'connecting' ? 'Connecting…' : 'Listening…'}
        </span>
      )}
      {showSpeakers
        ? finals.map((seg, i) => (
            <div key={i} style={{ marginBottom: '0.5rem' }}>
              {/* Null-speaker segments render unlabeled — avoids 'Speaker null:' display */}
              {seg.speaker !== null && <strong>Speaker {seg.speaker}: </strong>}
              <span>{seg.text}</span>
            </div>
          ))
        : finals.length > 0 && <span>{finals.map((f) => f.text).join(' ')} </span>}
      {interim.length > 0 && <span style={interimStyle}>{interim}</span>}
    </div>
  );
};
```

Note the rendering pattern: when speakers should be shown, each final is its own `<div>`; otherwise they're concatenated like before (so single-speaker recordings look identical to today).

### Step 2.4: Remove the Task 1 adapter in VerbalVector.tsx

Find the `<LiveTranscript ... />` JSX in `VerbalVector.tsx` and revert the temporary adapter:

```tsx
<LiveTranscript
  interim={live.interim}
  finals={live.finals}  // was: live.finals.map(f => f.text) — Task 2 cleanup
  status={live.status}
/>
```

### Step 2.5: Run tests, confirm pass

```bash
npm test -- LiveTranscript
```

Expected: all LiveTranscript tests pass.

### Step 2.6: Full suite + build + lint

```bash
npm test
npm run build
npm run lint
```

Expected: full suite passes, build clean.

### Step 2.7: Commit

```bash
git add frontend/src/components/LiveTranscript.tsx frontend/src/components/LiveTranscript.test.tsx frontend/src/components/VerbalVector.tsx
git commit -m "feat(frontend): render speaker labels in LiveTranscript with strip-when-1 fallback

LiveTranscript now accepts finals: FinalSegment[] instead of string[].
When 2+ unique speakers are present (excluding null), renders each
final as 'Speaker N: text' on its own line. When 0 or 1 unique
speakers, falls back to the existing concatenated rendering — single-
speaker recordings look identical to today.

Interim text remains unattributed (no speaker label) per the locked
product decision — sidesteps Deepgram's interim-speaker churn.

Removes the temporary .map(f => f.text) adapter in VerbalVector.tsx
from Task 1."
```

No `Co-Authored-By` trailer.

---

## Task 3: ResultsDisplay renders utterance-grouped transcript

**Files:**
- Modify: `frontend/src/components/ResultsDisplay.tsx`
- Create: `frontend/src/components/ResultsDisplay.test.tsx`

ResultsDisplay currently renders `transcript.text` as a flat `<pre>`. When `transcript.utterances` is present AND speakers.length > 1, render the speaker-labeled list. Otherwise, fall back to the existing `<pre>`.

### Step 3.1: Write failing tests

Create `frontend/src/components/ResultsDisplay.test.tsx`:

```tsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import ResultsDisplay from './ResultsDisplay';

const baseFeatures = {
  speech_clarity: 0.8,
  vocal_expressiveness: 0.7,
  words_per_minute: 150,
  unique_word_count: 200,
  total_filler_words: 5,
  mean_sentence_length: 12,
};

describe('ResultsDisplay transcript rendering', () => {
  it('renders speaker-labeled utterances when transcript has 2+ speakers', () => {
    const analysisResult = {
      message: 'Live session complete.',
      transcript: {
        text: 'Hello. Hi there.',
        utterances: [
          { speaker: 0, text: 'Hello.', start: 0.0, end: 0.5, confidence: 0.98 },
          { speaker: 1, text: 'Hi there.', start: 0.6, end: 1.2, confidence: 0.97 },
        ],
        speakers: [0, 1],
      },
      features: baseFeatures,
      feedback: 'Great job.',
    };
    render(
      <ResultsDisplay
        analysisResult={analysisResult}
        onAnalyzeAnother={() => {}}
        onNavigate={() => {}}
      />,
    );
    expect(screen.getByText(/Speaker 0/)).toBeInTheDocument();
    expect(screen.getByText(/Speaker 1/)).toBeInTheDocument();
    expect(screen.getByText(/Hello\./)).toBeInTheDocument();
    expect(screen.getByText(/Hi there\./)).toBeInTheDocument();
  });

  it('renders flat transcript when only one speaker', () => {
    const analysisResult = {
      message: 'Upload complete.',
      transcript: {
        text: 'Hello. How are you?',
        utterances: [
          { speaker: 0, text: 'Hello.', start: 0.0, end: 0.5, confidence: 0.98 },
          { speaker: 0, text: 'How are you?', start: 0.6, end: 1.5, confidence: 0.97 },
        ],
        speakers: [0],
      },
      features: baseFeatures,
      feedback: 'Good.',
    };
    render(
      <ResultsDisplay
        analysisResult={analysisResult}
        onAnalyzeAnother={() => {}}
        onNavigate={() => {}}
      />,
    );
    expect(screen.queryByText(/Speaker 0:/)).not.toBeInTheDocument();
    expect(screen.getByText('Hello. How are you?')).toBeInTheDocument();
  });

  it('renders flat transcript when utterances field is absent (legacy shape)', () => {
    const analysisResult = {
      message: 'Upload complete.',
      transcript: {
        text: 'Some flat transcript.',
        // no utterances, no speakers
      },
      features: baseFeatures,
      feedback: 'OK.',
    };
    render(
      <ResultsDisplay
        analysisResult={analysisResult}
        onAnalyzeAnother={() => {}}
        onNavigate={() => {}}
      />,
    );
    expect(screen.getByText('Some flat transcript.')).toBeInTheDocument();
    expect(screen.queryByText(/Speaker/)).not.toBeInTheDocument();
  });

  it('renders the existing flat text when transcript is a string (very-old shape)', () => {
    const analysisResult = {
      message: 'Upload complete.',
      transcript: 'String transcript fallback.',
      features: baseFeatures,
      feedback: 'OK.',
    };
    render(
      <ResultsDisplay
        analysisResult={analysisResult}
        onAnalyzeAnother={() => {}}
        onNavigate={() => {}}
      />,
    );
    expect(screen.getByText('String transcript fallback.')).toBeInTheDocument();
  });

  it('renders null-speaker utterances without a label when mixed with real speakers', () => {
    const analysisResult = {
      message: 'Upload complete.',
      transcript: {
        text: 'Hello. untagged. Hi.',
        utterances: [
          { speaker: 0, text: 'Hello.', start: 0.0, end: 0.5, confidence: 0.98 },
          { speaker: null, text: 'untagged.', start: 0.6, end: 1.0, confidence: 0.95 },
          { speaker: 1, text: 'Hi.', start: 1.1, end: 1.5, confidence: 0.97 },
        ],
        speakers: [0, 1],
      },
      features: baseFeatures,
      feedback: 'OK.',
    };
    render(
      <ResultsDisplay
        analysisResult={analysisResult}
        onAnalyzeAnother={() => {}}
        onNavigate={() => {}}
      />,
    );
    expect(screen.getByText(/Speaker 0/)).toBeInTheDocument();
    expect(screen.getByText(/Speaker 1/)).toBeInTheDocument();
    // The null-speaker utterance must NOT render "Speaker null:" literally
    expect(screen.queryByText(/Speaker null/)).not.toBeInTheDocument();
    expect(screen.getByText(/untagged\./)).toBeInTheDocument();
  });
});
```

### Step 3.2: Run tests, confirm fail

```bash
npm test -- ResultsDisplay
```

Expected: 4 tests fail because the multi-speaker rendering doesn't exist yet.

### Step 3.3: Update ResultsDisplay.tsx

Extract a `renderTranscript` helper function (place it above the `ResultsDisplay` component, after the style constants):

```tsx
import type { UploadResponse } from '../api';

function renderTranscript(transcript: UploadResponse['transcript']): React.ReactElement {
  // Legacy string shape
  if (typeof transcript === 'string') {
    return <pre style={transcriptBoxStyle}>{transcript}</pre>;
  }
  if (!transcript) {
    return <pre style={transcriptBoxStyle}>Transcript not available.</pre>;
  }

  const { utterances, speakers } = transcript;

  // Multi-speaker: render speaker-labeled list
  if (utterances && utterances.length > 0 && speakers && speakers.length > 1) {
    return (
      <div style={transcriptBoxStyle}>
        {utterances.map((u, i) => (
          <div key={i} style={{ marginBottom: '0.5rem' }}>
            {/* Null-speaker utterances render unlabeled — avoids 'Speaker null:' display */}
            {u.speaker !== null && <strong>Speaker {u.speaker}: </strong>}
            <span>{u.text}</span>
          </div>
        ))}
      </div>
    );
  }

  // Fall back to flat text (single speaker or no utterances)
  return <pre style={transcriptBoxStyle}>{transcript.text || 'Transcript not available.'}</pre>;
}
```

Then replace the existing Transcript Section JSX (lines ~211-219) with:

```tsx
{/* --- Transcript Section --- */}
<div style={sectionStyle}>
    <h3 style={sectionTitleStyle}>Transcript</h3>
    {renderTranscript(analysisResult.transcript)}
</div>
```

The `UploadResponse` import from `../api` may already exist (it's used in the existing imports for `AnalysisResult`) — verify; if not, add it.

### Step 3.4: Run tests, confirm pass

```bash
npm test -- ResultsDisplay
```

Expected: all 4 tests pass.

### Step 3.5: Full suite + build + lint

```bash
npm test
npm run build
npm run lint
```

Expected: full suite passes, build clean.

### Step 3.6: Commit

```bash
git add frontend/src/components/ResultsDisplay.tsx frontend/src/components/ResultsDisplay.test.tsx
git commit -m "feat(frontend): render speaker-labeled transcript in ResultsDisplay when multi-speaker

When the analysis result's transcript has utterances + 2+ speakers,
ResultsDisplay renders each utterance as 'Speaker N: text' on its own
line. Single-speaker (or no-utterances) recordings fall back to the
existing flat <pre> rendering — no visual change for pre-diarization
data and for single-speaker live/batch sessions.

Also handles the very-old string-shape transcript for backward
compatibility (the production code's type union still includes it).

Adds a new test file covering all four render paths: multi-speaker,
single-speaker (with utterances), no utterances, string transcript."
```

No `Co-Authored-By` trailer.

---

## Task 4: Manual test checklist + push + open PR

**Files:**
- Create: `docs/superpowers/specs/2026-05-24-phase3d-frontend-manual-test-checklist.md`

### Step 4.1: Create the checklist

```markdown
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
```

### Step 4.2: Commit

```bash
git add docs/superpowers/specs/2026-05-24-phase3d-frontend-manual-test-checklist.md
git commit -m "docs: add Phase 3d.2 frontend diarization manual test checklist

Covers single-speaker (strip labels), multi-speaker live + batch
paths, and regression checks for legacy data and the no-label-on-
interim invariant."
```

### Step 4.3: Push and open PR

```bash
git -c url.https://github.com/.insteadOf=git@github.com: -c credential.helper='!gh auth git-credential' push -u origin phase3d.2/frontend-diarization

gh pr create --base main --head phase3d.2/frontend-diarization --title "feat(phase3d.2): frontend renders speaker labels" --body "$(cat <<'EOF'
## Summary
Phase 3d.2 — frontend rendering of the speaker info shipped in Phase 3d.1. Single-speaker recordings look identical to today; multi-speaker recordings get 'Speaker N: text' labels per segment.

### What ships
- \`useLiveStream\` finals shape: \`string[]\` → \`{text, speaker: number | null}[]\`
- \`LiveTranscript\`: renders speaker labels when 2+ speakers, flat when 0/1
- \`ResultsDisplay\`: same logic for batch results
- \`api.ts\` types: \`ServerMessage.transcript\` adds \`speaker\`; \`UploadResponse.transcript\` adds optional \`utterances\` + \`speakers\`
- New: \`ResultsDisplay.test.tsx\` (no tests there before)

### Design notes
- Interim text has NO speaker label — sidesteps Deepgram's interim-speaker churn. Locked product decision.
- Strip-when-1-speaker logic lives per-component (one-line check). No shared helper for v1.
- Backward compat: legacy \`transcript\` shape (string, or object without utterances) still renders.

### Out of scope
- Per-speaker colors / avatars
- Speaker-scoped search filter on /api/query (backend metadata is there, UI not built)
- Editable speaker names ('Yash' instead of 'Speaker 0')

### Test plan
- [x] Vitest: full suite passes
- [x] \`npm run build\` and \`npm run lint\` clean
- [ ] Manual checklist (\`docs/superpowers/specs/2026-05-24-phase3d-frontend-manual-test-checklist.md\`)
EOF
)"
```

### Step 4.4: Report PR URL

## Self-Review Checklist (plan author)

- [ ] Every product decision locked at the top is reflected in implementation.
- [ ] Type names consistent: `FinalSegment` is one type, exported from `useLiveStream`.
- [ ] Backward-compat: single-speaker + legacy transcript shape verified by tests.
- [ ] Manual checklist includes regression checks.
- [ ] No placeholders.

## Out of Scope (separate plans needed if pursued)

- **Per-speaker analytics UI** — counts of words / time per speaker; chart of speaker balance.
- **Speaker-scoped search** in /api/query — backend metadata is in ChromaDB; UI would add a filter dropdown.
- **Editable speaker names** — UI to rename "Speaker 0" → "Yash" and persist.
- **Per-speaker colors** — accessibility-aware color palette per speaker ID.
