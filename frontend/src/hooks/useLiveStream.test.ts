import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useLiveStream, type PcmAudioCaptureLike } from './useLiveStream';

class MockWebSocket {
  static instances: MockWebSocket[] = [];
  url: string;
  readyState: number = 0;
  onopen: ((ev: Event) => void) | null = null;
  onmessage: ((ev: MessageEvent) => void) | null = null;
  onclose: ((ev: CloseEvent) => void) | null = null;
  onerror: ((ev: Event) => void) | null = null;
  sent: (string | ArrayBuffer)[] = [];

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  send(data: string | ArrayBuffer): void {
    this.sent.push(data);
  }

  close(): void {
    this.readyState = 3;
    this.onclose?.(new CloseEvent('close'));
  }

  fireOpen() {
    this.readyState = 1;
    this.onopen?.(new Event('open'));
  }
  fireServerMessage(msg: unknown) {
    this.onmessage?.(new MessageEvent('message', { data: JSON.stringify(msg) }));
  }
}

interface CaptureMockHandle {
  instance: PcmAudioCaptureLike;
  handler: () => ((chunk: ArrayBuffer) => void) | null;
  startMock: ReturnType<typeof vi.fn>;
  stopMock: ReturnType<typeof vi.fn>;
  setHandlerMock: ReturnType<typeof vi.fn>;
}

function makeMockCapture(startBehavior?: () => Promise<void>): CaptureMockHandle {
  let handler: ((chunk: ArrayBuffer) => void) | null = null;
  const startMock = vi.fn(startBehavior ?? (async () => {}));
  const stopMock = vi.fn(async () => { handler = null; });
  const setHandlerMock = vi.fn((h: ((chunk: ArrayBuffer) => void) | null) => {
    handler = h;
  });
  return {
    instance: {
      start: startMock as unknown as PcmAudioCaptureLike['start'],
      stop: stopMock as unknown as PcmAudioCaptureLike['stop'],
      setHandler: setHandlerMock as unknown as PcmAudioCaptureLike['setHandler'],
    },
    handler: () => handler,
    startMock,
    stopMock,
    setHandlerMock,
  };
}

function opts(ws: MockWebSocket, capture: CaptureMockHandle) {
  return {
    createWebSocket: () => ws as unknown as WebSocket,
    createCapture: () => capture.instance,
  };
}

describe('useLiveStream', () => {
  beforeEach(() => {
    MockWebSocket.instances = [];
  });

  it('starts in idle state with empty transcripts and no result', () => {
    const { result } = renderHook(() => useLiveStream());
    expect(result.current.status).toBe('idle');
    expect(result.current.interim).toBe('');
    expect(result.current.finals).toEqual([]);
    expect(result.current.error).toBeNull();
    expect(result.current.result).toBeNull();
  });

  it('starts capture BEFORE opening the WebSocket', async () => {
    const ws = new MockWebSocket('ws://test/api/stream');
    const capture = makeMockCapture();
    const { result } = renderHook(() => useLiveStream(opts(ws, capture)));

    await act(async () => {
      await result.current.start('My Talk');
    });

    expect(capture.startMock).toHaveBeenCalled();
    expect(MockWebSocket.instances.length).toBe(1);
  });

  it('does NOT open a WebSocket if capture.start() rejects', async () => {
    const createWsSpy = vi.fn((url: string) => new MockWebSocket(url) as unknown as WebSocket);
    const capture = makeMockCapture(async () => {
      throw new Error('Permission denied');
    });
    const { result } = renderHook(() =>
      useLiveStream({ createWebSocket: createWsSpy, createCapture: () => capture.instance }),
    );

    await act(async () => { await result.current.start(); });

    await waitFor(() => expect(result.current.status).toBe('error'));
    expect(result.current.error).toContain('Permission denied');
    // The hook must NOT have called the WebSocket factory
    expect(createWsSpy).not.toHaveBeenCalled();
  });

  it('sends init and attaches the capture handler on session_started', async () => {
    const ws = new MockWebSocket('ws://test/api/stream');
    const capture = makeMockCapture();
    const { result } = renderHook(() => useLiveStream(opts(ws, capture)));

    await act(async () => { await result.current.start('My Talk'); });
    act(() => ws.fireOpen());
    act(() => ws.fireServerMessage({ type: 'session_started', session_id: 'abc123' }));

    await waitFor(() => expect(result.current.status).toBe('recording'));

    const initMsg = ws.sent.find((m) => typeof m === 'string' && m.includes('init')) as string | undefined;
    expect(initMsg).toBeDefined();
    expect(JSON.parse(initMsg!)).toEqual({ type: 'init', session_label: 'My Talk' });
    expect(capture.setHandlerMock).toHaveBeenCalledWith(expect.any(Function));
  });

  it('forwards transcripts to interim and finals state', async () => {
    const ws = new MockWebSocket('ws://test/api/stream');
    const capture = makeMockCapture();
    const { result } = renderHook(() => useLiveStream(opts(ws, capture)));

    await act(async () => { await result.current.start(); });
    act(() => ws.fireOpen());
    act(() => ws.fireServerMessage({ type: 'session_started', session_id: 'x' }));
    await waitFor(() => expect(result.current.status).toBe('recording'));

    act(() => ws.fireServerMessage({ type: 'transcript', text: 'hello', is_final: false, speaker: 0 }));
    expect(result.current.interim).toBe('hello');
    expect(result.current.finals).toEqual([]);

    act(() => ws.fireServerMessage({ type: 'transcript', text: 'hello world', is_final: true, speaker: 0 }));
    expect(result.current.interim).toBe('');
    expect(result.current.finals).toEqual([{ text: 'hello world', speaker: 0 }]);
  });

  it('forwards captured PCM chunks as WebSocket binary frames', async () => {
    const ws = new MockWebSocket('ws://test/api/stream');
    const capture = makeMockCapture();
    const { result } = renderHook(() => useLiveStream(opts(ws, capture)));

    await act(async () => { await result.current.start(); });
    act(() => ws.fireOpen());
    act(() => ws.fireServerMessage({ type: 'session_started', session_id: 'x' }));
    await waitFor(() => expect(result.current.status).toBe('recording'));

    const chunk = new ArrayBuffer(16);
    capture.handler()?.(chunk);

    const binaryFrames = ws.sent.filter((m) => m instanceof ArrayBuffer);
    expect(binaryFrames).toContain(chunk);
  });

  it('stop() sends end and transitions to stopping', async () => {
    const ws = new MockWebSocket('ws://test/api/stream');
    const capture = makeMockCapture();
    const { result } = renderHook(() => useLiveStream(opts(ws, capture)));

    await act(async () => { await result.current.start(); });
    act(() => ws.fireOpen());
    act(() => ws.fireServerMessage({ type: 'session_started', session_id: 'x' }));
    await waitFor(() => expect(result.current.status).toBe('recording'));

    await act(async () => { result.current.stop(); });

    expect(result.current.status).toBe('stopping');
    expect(capture.stopMock).toHaveBeenCalled();
    const endMsg = ws.sent.find((m) => typeof m === 'string' && m.includes('end')) as string | undefined;
    expect(endMsg).toBeDefined();
    expect(JSON.parse(endMsg!)).toEqual({ type: 'end' });
  });

  it('stop() during connecting cancels cleanly without sending end', async () => {
    const ws = new MockWebSocket('ws://test/api/stream');
    const capture = makeMockCapture();
    const { result } = renderHook(() => useLiveStream(opts(ws, capture)));

    await act(async () => { await result.current.start(); });
    // Don't fire session_started — we're stuck in 'connecting'
    expect(result.current.status).toBe('connecting');

    await act(async () => { result.current.stop(); });

    // Should NOT have sent end (server hasn't completed handshake)
    const endMsg = ws.sent.find((m) => typeof m === 'string' && m.includes('"end"'));
    expect(endMsg).toBeUndefined();
    // Should have cleaned up capture and returned to idle
    await waitFor(() => expect(result.current.status).toBe('idle'));
    expect(capture.stopMock).toHaveBeenCalled();
  });

  it('session_end transitions to completed and populates result', async () => {
    const ws = new MockWebSocket('ws://test/api/stream');
    const capture = makeMockCapture();
    const { result } = renderHook(() => useLiveStream(opts(ws, capture)));

    await act(async () => { await result.current.start(); });
    act(() => ws.fireOpen());
    act(() => ws.fireServerMessage({ type: 'session_started', session_id: 'x' }));
    await waitFor(() => expect(result.current.status).toBe('recording'));

    act(() =>
      ws.fireServerMessage({
        type: 'session_end',
        transcript: { text: 'hello world' },
        features: { wpm: 120 },
        feedback: '## Great job',
      }),
    );

    await waitFor(() => expect(result.current.status).toBe('completed'));
    expect(result.current.result).toEqual({
      message: expect.any(String),
      transcript: { text: 'hello world' },
      features: { wpm: 120 },
      feedback: '## Great job',
    });
  });

  it('error message with fatal=true transitions to error state', async () => {
    const ws = new MockWebSocket('ws://test/api/stream');
    const capture = makeMockCapture();
    const { result } = renderHook(() => useLiveStream(opts(ws, capture)));

    await act(async () => { await result.current.start(); });
    act(() => ws.fireOpen());
    act(() => ws.fireServerMessage({ type: 'session_started', session_id: 'x' }));
    await waitFor(() => expect(result.current.status).toBe('recording'));

    act(() => ws.fireServerMessage({ type: 'error', message: 'Deepgram offline', fatal: true }));

    await waitFor(() => expect(result.current.status).toBe('error'));
    expect(result.current.error).toContain('Deepgram offline');
    expect(capture.stopMock).toHaveBeenCalled();
  });

  it('WebSocket close before session_end transitions to error', async () => {
    const ws = new MockWebSocket('ws://test/api/stream');
    const capture = makeMockCapture();
    const { result } = renderHook(() => useLiveStream(opts(ws, capture)));

    await act(async () => { await result.current.start(); });
    act(() => ws.fireOpen());
    act(() => ws.fireServerMessage({ type: 'session_started', session_id: 'x' }));
    await waitFor(() => expect(result.current.status).toBe('recording'));

    act(() => ws.close());

    await waitFor(() => expect(result.current.status).toBe('error'));
    expect(result.current.error).toBeDefined();
  });

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

    act(() => ws.fireServerMessage({ type: 'transcript', text: 'partial...', is_final: false, speaker: null }));
    expect(result.current.interim).toBe('partial...');

    act(() => ws.fireServerMessage({ type: 'transcript', text: 'finalized.', is_final: true, speaker: null }));
    expect(result.current.finals).toEqual([{ text: 'finalized.', speaker: null }]);
  });

  // Pins the `msg.speaker ?? null` boundary normalization: if a server frame
  // ever lacks the `speaker` field entirely (TS lies, server regresses), the
  // hook must still store `null` so downstream rendering doesn't choke on
  // `undefined`.
  it('normalizes a transcript frame with no speaker field to speaker=null', async () => {
    const ws = new MockWebSocket('ws://test/api/stream');
    const capture = makeMockCapture();
    const { result } = renderHook(() => useLiveStream(opts(ws, capture)));

    await act(async () => { await result.current.start(); });
    act(() => ws.fireOpen());
    act(() => ws.fireServerMessage({ type: 'session_started', session_id: 'x' }));
    await waitFor(() => expect(result.current.status).toBe('recording'));

    // Intentionally omit `speaker` — fireServerMessage JSON.stringifies, so
    // the field is absent on the wire (not just undefined in JS).
    act(() => ws.fireServerMessage({ type: 'transcript', text: 'no speaker.', is_final: true }));

    expect(result.current.finals).toEqual([{ text: 'no speaker.', speaker: null }]);
  });

  it('restarting from completed runs cleanup on the previous session first', async () => {
    const ws1 = new MockWebSocket('ws://test/api/stream');
    const capture1 = makeMockCapture();
    const { result, rerender } = renderHook(
      (overrides: ReturnType<typeof opts>) => useLiveStream(overrides),
      { initialProps: opts(ws1, capture1) },
    );

    await act(async () => { await result.current.start(); });
    act(() => ws1.fireOpen());
    act(() => ws1.fireServerMessage({ type: 'session_started', session_id: 'x' }));
    await waitFor(() => expect(result.current.status).toBe('recording'));
    act(() =>
      ws1.fireServerMessage({
        type: 'session_end',
        transcript: { text: 'one' },
        features: {},
        feedback: 'feedback one',
      }),
    );
    await waitFor(() => expect(result.current.status).toBe('completed'));

    const ws2 = new MockWebSocket('ws://test/api/stream');
    const capture2 = makeMockCapture();
    rerender(opts(ws2, capture2));

    await act(async () => { await result.current.start(); });

    expect(capture1.stopMock).toHaveBeenCalled();
    expect(capture2.startMock).toHaveBeenCalled();
  });

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
    act(() => ws.close());                        // socket drops before session_end
    expect(result.current.status).toBe('stopping');
    expect(result.current.error).toBeNull();
  });
});
