import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, act, fireEvent } from '@testing-library/react';

// The live audio pipeline is irrelevant to this test — stub it out.
vi.mock('./audio/pcmAudioCapture', () => ({
  PcmAudioCapture: class {
    async start() { /* noop */ }
    setHandler() { /* noop */ }
    async stop() { /* noop */ }
  },
}));

// Partial-mock the api module: keep getStreamWsUrl (used by useLiveStream) real,
// override ONLY getSessionResult so we control the poll result.
const mockGetSessionResult = vi.fn();
vi.mock('./api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./api')>();
  return { ...actual, getSessionResult: (id: string) => mockGetSessionResult(id) };
});

// Stub the sibling views so navigating away doesn't trigger their data fetches.
vi.mock('./components/ResultsDisplay', () => ({ default: () => <div>RESULTS_VIEW</div> }));
vi.mock('./components/QueryInterface', () => ({ default: () => <div>QUERY_VIEW</div> }));
vi.mock('./components/SessionHistory', () => ({ default: () => <div>HISTORY_VIEW</div> }));

import App from './App';

// ---------------------------------------------------------------------------
// Minimal harness (mirrors VerbalVector.test.tsx) to drive the real live flow
// ---------------------------------------------------------------------------
class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  static reset() { FakeWebSocket.instances = []; }
  url: string;
  readyState = 1;
  sent: (string | ArrayBuffer)[] = [];
  onopen: ((ev: Event) => void) | null = null;
  onmessage: ((ev: MessageEvent) => void) | null = null;
  onclose: ((ev: CloseEvent) => void) | null = null;
  onerror: ((ev: Event) => void) | null = null;
  constructor(url: string) { this.url = url; FakeWebSocket.instances.push(this); }
  send(d: string | ArrayBuffer) { this.sent.push(d); }
  close() { this.readyState = 3; this.onclose?.(new CloseEvent('close')); }
}

const originalMediaDevices = Object.getOwnPropertyDescriptor(global.navigator, 'mediaDevices');
const originalWebSocket = (global as unknown as { WebSocket: typeof WebSocket }).WebSocket;

function installNavigator() {
  Object.defineProperty(global.navigator, 'mediaDevices', {
    configurable: true,
    value: {
      getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [{ stop: vi.fn() }] }),
      enumerateDevices: vi.fn().mockResolvedValue([]),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    },
  });
}

const READY = { message: 'Live session complete.', transcript: { text: 'hi' }, features: {}, feedback: 'ok' };
const flush = () => act(async () => { /* let pending microtasks settle */ });
const tick = (ms: number) => act(async () => { await vi.advanceTimersByTimeAsync(ms); });

describe('App live result survives navigation (real VerbalVector)', () => {
  beforeEach(() => {
    mockGetSessionResult.mockReset();
    FakeWebSocket.reset();
    installNavigator();
    (global as unknown as { WebSocket: unknown }).WebSocket = FakeWebSocket;
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.runOnlyPendingTimers();
    vi.useRealTimers();
    if (originalMediaDevices) Object.defineProperty(global.navigator, 'mediaDevices', originalMediaDevices);
    (global as unknown as { WebSocket: unknown }).WebSocket = originalWebSocket;
    FakeWebSocket.reset();
  });

  // This is the literal regression the PR fixes: the user stops a live session,
  // navigates away (which UNMOUNTS the real VerbalVector and closes the WS), and
  // the App-level poll must still deliver the result. If the poll were ever moved
  // back into VerbalVector, this test fails.
  it('delivers the result via banner after the live view unmounts on navigation', async () => {
    mockGetSessionResult.mockResolvedValue({ status: 'ready', data: READY });
    render(<App />);

    // Switch to Live mode and start recording.
    await act(async () => { fireEvent.click(screen.getByLabelText(/live/i)); });
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: /record audio/i })); });
    await flush(); // let live.start() resolve and open the WebSocket

    expect(FakeWebSocket.instances.length).toBe(1);
    const ws = FakeWebSocket.instances[0];

    // Drive the session to 'recording'.
    await act(async () => {
      ws.onopen?.(new Event('open'));
      ws.onmessage?.(new MessageEvent('message', {
        data: JSON.stringify({ type: 'session_started', session_id: 'live-1' }),
      }));
    });

    // Stop — fires onLiveSessionEnding('live-1') → App arms the durable poll.
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: /stop recording/i })); });

    // Navigate away via the real NavHeader → VerbalVector unmounts, WS closes.
    await act(async () => { fireEvent.click(screen.getByText('Ask')); });
    expect(screen.getByText('QUERY_VIEW')).toBeInTheDocument();

    // The poll lives in App, so it survives the unmount and delivers the result.
    await tick(2000);
    expect(screen.getByText(/analysis is ready/i)).toBeInTheDocument();
    expect(screen.getByText('QUERY_VIEW')).toBeInTheDocument(); // not yanked away

    // The banner routes to the (stubbed) results view.
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: /view results/i })); });
    expect(screen.getByText('RESULTS_VIEW')).toBeInTheDocument();
  });
});
