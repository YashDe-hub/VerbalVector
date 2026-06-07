import { vi } from 'vitest';
vi.mock('../audio/pcmAudioCapture', () => ({
  PcmAudioCapture: class {
    async start() { /* noop */ }
    setHandler() { /* noop */ }
    async stop() { /* noop */ }
  },
}));

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, cleanup } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import VerbalVector from './VerbalVector';

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

function makeStreamMock() {
  return {
    getTracks: () => [{ stop: vi.fn() }],
  } as unknown as MediaStream;
}

function makeDeviceInfo(partial: Partial<MediaDeviceInfo>): MediaDeviceInfo {
  return {
    deviceId: '',
    groupId: '',
    kind: 'audioinput',
    label: '',
    toJSON: () => ({}),
    ...partial,
  } as MediaDeviceInfo;
}

// Single MockMediaRecorder definition shared by every test.
class MockMediaRecorder {
  static isTypeSupported() { return true; }
  ondataavailable = () => {};
  onstop = () => {};
  mimeType = 'audio/webm';
  state = 'recording';
  start() {}
  stop() { this.onstop(); }
}

// Capture the original navigator.mediaDevices descriptor once at module load
// time so afterEach can restore it precisely.
const originalMediaDevicesDescriptor = Object.getOwnPropertyDescriptor(
  global.navigator,
  'mediaDevices',
);

// Capture original WebSocket at module load so afterEach can restore.
const originalWebSocket = (global as unknown as { WebSocket: typeof WebSocket }).WebSocket;

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
  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }
  send(d: string | ArrayBuffer) { this.sent.push(d); }
  close() {}
}

/**
 * Installs navigator.mediaDevices with the given device list and a fresh
 * getUserMedia mock.  Returns the getUserMedia mock so callers can assert on it.
 */
function setupNavigatorMock(devices: MediaDeviceInfo[]): ReturnType<typeof vi.fn> {
  const getUserMedia = vi.fn().mockResolvedValue(makeStreamMock());
  const enumerateDevices = vi.fn().mockResolvedValue(devices);

  Object.defineProperty(global.navigator, 'mediaDevices', {
    configurable: true,
    value: {
      getUserMedia,
      enumerateDevices,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    },
  });

  return getUserMedia;
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

describe('VerbalVector device selection integration', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    // Install the shared MediaRecorder mock before each test.
    (global as unknown as { MediaRecorder: typeof MockMediaRecorder }).MediaRecorder =
      MockMediaRecorder;
  });

  afterEach(() => {
    // Explicitly unmount all rendered components first so React's effect
    // cleanups run while navigator.mediaDevices is still the mock (the hook's
    // cleanup closure calls removeEventListener on it).
    cleanup();

    // Now it is safe to remove the globals — React is done with them.
    delete (global as unknown as Record<string, unknown>).MediaRecorder;

    if (originalMediaDevicesDescriptor) {
      Object.defineProperty(global.navigator, 'mediaDevices', originalMediaDevicesDescriptor);
    } else {
      delete (navigator as unknown as Record<string, unknown>).mediaDevices;
    }

    (global as unknown as { WebSocket: unknown }).WebSocket = originalWebSocket;
    FakeWebSocket.reset();
  });

  it('passes the selected deviceId as an exact constraint to getUserMedia', async () => {
    const getUserMedia = setupNavigatorMock([
      makeDeviceInfo({ deviceId: 'builtin', kind: 'audioinput', label: 'Built-in' }),
      makeDeviceInfo({ deviceId: 'raybans', kind: 'audioinput', label: 'Ray-Ban Meta' }),
    ]);

    render(
      <VerbalVector
        onAnalysisComplete={() => {}}
        onNavigate={() => {}}
      />,
    );

    await waitFor(() => {
      expect(screen.getByRole('option', { name: 'Ray-Ban Meta' })).toBeInTheDocument();
    });

    await userEvent.selectOptions(screen.getByRole('combobox'), 'raybans');
    await userEvent.click(screen.getByRole('button', { name: /record audio/i }));

    await waitFor(() => expect(getUserMedia).toHaveBeenCalled());

    expect(getUserMedia).toHaveBeenLastCalledWith({
      audio: { deviceId: { exact: 'raybans' } },
    });
  });

  it('passes audio:true (no constraint) when system default is selected', async () => {
    const getUserMedia = setupNavigatorMock([
      makeDeviceInfo({ deviceId: 'builtin', kind: 'audioinput', label: 'Built-in' }),
    ]);

    render(
      <VerbalVector
        onAnalysisComplete={() => {}}
        onNavigate={() => {}}
      />,
    );

    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole('button', { name: /record audio/i }));

    await waitFor(() => expect(getUserMedia).toHaveBeenCalled());

    expect(getUserMedia).toHaveBeenLastCalledWith({ audio: true });
  });

  it('surfaces a permission-denied message when getUserMedia rejects with NotAllowedError', async () => {
    const getUserMedia = vi.fn().mockRejectedValue(
      new DOMException('Permission denied', 'NotAllowedError'),
    );
    setupNavigatorMock([]);
    // Override the getUserMedia mock with a rejecting one
    Object.defineProperty(global.navigator, 'mediaDevices', {
      configurable: true,
      value: {
        getUserMedia,
        enumerateDevices: vi.fn().mockResolvedValue([]),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      },
    });

    render(
      <VerbalVector
        onAnalysisComplete={() => {}}
        onNavigate={() => {}}
      />,
    );

    await userEvent.click(screen.getByRole('button', { name: /record audio/i }));

    // Expect the permission-denied error message text — match the existing copy
    await waitFor(() => {
      expect(screen.getByText(/microphone access denied/i)).toBeInTheDocument();
    });
  });
});

describe('VerbalVector live mode', () => {
  beforeEach(() => {
    FakeWebSocket.reset();
    (global as unknown as { MediaRecorder: typeof MockMediaRecorder }).MediaRecorder =
      MockMediaRecorder;
  });

  afterEach(() => {
    cleanup();
    delete (global as unknown as Record<string, unknown>).MediaRecorder;
    if (originalMediaDevicesDescriptor) {
      Object.defineProperty(global.navigator, 'mediaDevices', originalMediaDevicesDescriptor);
    } else {
      delete (navigator as unknown as Record<string, unknown>).mediaDevices;
    }
    (global as unknown as { WebSocket: unknown }).WebSocket = originalWebSocket;
    FakeWebSocket.reset();
  });

  it('shows a mode toggle in the input stage with Batch selected by default and can switch to Live', async () => {
    setupNavigatorMock([]);
    render(
      <VerbalVector
        onAnalysisComplete={() => {}}
        onNavigate={() => {}}
      />,
    );

    const batchRadio = screen.getByLabelText(/batch/i) as HTMLInputElement;
    const liveRadio = screen.getByLabelText(/live/i) as HTMLInputElement;
    expect(batchRadio.checked).toBe(true);
    expect(liveRadio.checked).toBe(false);

    await userEvent.click(liveRadio);

    expect(liveRadio.checked).toBe(true);
    expect(batchRadio.checked).toBe(false);
  });

  it('switching to Live mode opens a WebSocket to /api/stream when Record is clicked', async () => {
    setupNavigatorMock([]);
    (global as unknown as { WebSocket: unknown }).WebSocket = FakeWebSocket;

    render(
      <VerbalVector
        onAnalysisComplete={() => {}}
        onNavigate={() => {}}
      />,
    );

    await userEvent.click(screen.getByLabelText(/live/i));
    await userEvent.click(screen.getByRole('button', { name: /record audio/i }));

    await waitFor(() => expect(FakeWebSocket.instances.length).toBe(1));
    expect(FakeWebSocket.instances[0].url).toContain('/api/stream');
  });

  // End-to-end wiring check: ensures transcript messages flow from the
  // useLiveStream hook into LiveTranscript with the speaker field intact.
  // Catches regressions where the prop is renamed or the FinalSegment shape
  // is unwired between hook and component.
  it('renders speaker-labeled finals in LiveTranscript when multi-speaker transcripts arrive', async () => {
    setupNavigatorMock([]);
    (global as unknown as { WebSocket: unknown }).WebSocket = FakeWebSocket;

    render(
      <VerbalVector
        onAnalysisComplete={() => {}}
        onNavigate={() => {}}
      />,
    );

    await userEvent.click(screen.getByLabelText(/live/i));
    await userEvent.click(screen.getByRole('button', { name: /record audio/i }));

    await waitFor(() => expect(FakeWebSocket.instances.length).toBe(1));
    const ws = FakeWebSocket.instances[0];

    if (ws.onopen) ws.onopen(new Event('open'));
    if (ws.onmessage) {
      ws.onmessage(new MessageEvent('message', {
        data: JSON.stringify({ type: 'session_started', session_id: 'test' }),
      }));
      ws.onmessage(new MessageEvent('message', {
        data: JSON.stringify({ type: 'transcript', text: 'Hello.', is_final: true, speaker: 0 }),
      }));
      ws.onmessage(new MessageEvent('message', {
        data: JSON.stringify({ type: 'transcript', text: 'Hi there.', is_final: true, speaker: 1 }),
      }));
    }

    await waitFor(() => {
      expect(screen.getByText(/Speaker 0/)).toBeInTheDocument();
      expect(screen.getByText(/Speaker 1/)).toBeInTheDocument();
    });
    expect(screen.getByText('Hello.')).toBeInTheDocument();
    expect(screen.getByText('Hi there.')).toBeInTheDocument();
  });

  it('enables the Stop Recording button once live status reaches recording', async () => {
    setupNavigatorMock([]);
    (global as unknown as { WebSocket: unknown }).WebSocket = FakeWebSocket;

    render(
      <VerbalVector
        onAnalysisComplete={() => {}}
        onNavigate={() => {}}
      />,
    );

    await userEvent.click(screen.getByLabelText(/live/i));
    await userEvent.click(screen.getByRole('button', { name: /record audio/i }));

    await waitFor(() => expect(FakeWebSocket.instances.length).toBe(1));
    const ws = FakeWebSocket.instances[0];

    // Drive the WS lifecycle: open + session_started → hook flips to 'recording'
    if (ws.onopen) ws.onopen(new Event('open'));
    if (ws.onmessage) {
      ws.onmessage(new MessageEvent('message', {
        data: JSON.stringify({ type: 'session_started', session_id: 'test' }),
      }));
    }

    // Stop button should now be enabled (covers the 8fc0de4 fix)
    await waitFor(() => {
      const stopBtn = screen.getByRole('button', { name: /stop recording/i });
      expect(stopBtn).not.toBeDisabled();
    });
  });

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
});
