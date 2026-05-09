import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import VerbalVector from './VerbalVector';

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

describe('VerbalVector device selection integration', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('passes the selected deviceId as an exact constraint to getUserMedia', async () => {
    const getUserMedia = vi.fn().mockResolvedValue(makeStreamMock());
    const enumerateDevices = vi.fn().mockResolvedValue([
      makeDeviceInfo({ deviceId: 'builtin', kind: 'audioinput', label: 'Built-in' }),
      makeDeviceInfo({ deviceId: 'raybans', kind: 'audioinput', label: 'Ray-Ban Meta' }),
    ]);

    Object.defineProperty(global.navigator, 'mediaDevices', {
      configurable: true,
      value: {
        getUserMedia,
        enumerateDevices,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      },
    });

    class MockMediaRecorder {
      static isTypeSupported() { return true; }
      ondataavailable = () => {};
      onstop = () => {};
      mimeType = 'audio/webm';
      state = 'recording';
      start() {}
      stop() { this.onstop(); }
    }
    (global as unknown as { MediaRecorder: typeof MockMediaRecorder }).MediaRecorder =
      MockMediaRecorder;

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

    expect(getUserMedia).toHaveBeenCalledWith({
      audio: { deviceId: { exact: 'raybans' } },
    });
  });

  it('passes audio:true (no constraint) when system default is selected', async () => {
    const getUserMedia = vi.fn().mockResolvedValue(makeStreamMock());
    const enumerateDevices = vi.fn().mockResolvedValue([
      makeDeviceInfo({ deviceId: 'builtin', kind: 'audioinput', label: 'Built-in' }),
    ]);

    Object.defineProperty(global.navigator, 'mediaDevices', {
      configurable: true,
      value: {
        getUserMedia,
        enumerateDevices,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      },
    });

    class MockMediaRecorder {
      static isTypeSupported() { return true; }
      ondataavailable = () => {};
      onstop = () => {};
      mimeType = 'audio/webm';
      state = 'recording';
      start() {}
      stop() { this.onstop(); }
    }
    (global as unknown as { MediaRecorder: typeof MockMediaRecorder }).MediaRecorder =
      MockMediaRecorder;

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

    expect(getUserMedia).toHaveBeenCalledWith({ audio: true });
  });
});
