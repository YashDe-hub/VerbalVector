import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
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
});
