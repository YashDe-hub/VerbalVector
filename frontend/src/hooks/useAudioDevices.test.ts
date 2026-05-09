import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useAudioDevices } from './useAudioDevices';

type EnumerateDevicesFn = () => Promise<MediaDeviceInfo[]>;

function mockMediaDevices(
  enumerate: EnumerateDevicesFn,
  options: { addEventListener?: ReturnType<typeof vi.fn>; removeEventListener?: ReturnType<typeof vi.fn> } = {},
) {
  Object.defineProperty(global.navigator, 'mediaDevices', {
    configurable: true,
    value: {
      enumerateDevices: enumerate,
      addEventListener: options.addEventListener ?? vi.fn(),
      removeEventListener: options.removeEventListener ?? vi.fn(),
    },
  });
}

function makeDevice(partial: Partial<MediaDeviceInfo>): MediaDeviceInfo {
  return {
    deviceId: '',
    groupId: '',
    kind: 'audioinput',
    label: '',
    toJSON: () => ({}),
    ...partial,
  } as MediaDeviceInfo;
}

describe('useAudioDevices', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('lists only audioinput devices', async () => {
    mockMediaDevices(async () => [
      makeDevice({ deviceId: 'mic1', kind: 'audioinput', label: 'Built-in Mic' }),
      makeDevice({ deviceId: 'cam1', kind: 'videoinput', label: 'Camera' }),
      makeDevice({ deviceId: 'spk1', kind: 'audiooutput', label: 'Speakers' }),
    ]);
    const { result } = renderHook(() => useAudioDevices());
    await waitFor(() => expect(result.current.devices.length).toBe(1));
    expect(result.current.devices[0].deviceId).toBe('mic1');
  });

  it('reports hasLabels=false when all labels are empty, true otherwise', async () => {
    mockMediaDevices(async () => [
      makeDevice({ deviceId: 'mic1', kind: 'audioinput', label: '' }),
    ]);
    const { result, rerender } = renderHook(() => useAudioDevices());
    await waitFor(() => expect(result.current.devices.length).toBe(1));
    expect(result.current.hasLabels).toBe(false);

    mockMediaDevices(async () => [
      makeDevice({ deviceId: 'mic1', kind: 'audioinput', label: 'Built-in Mic' }),
    ]);
    await act(async () => { await result.current.refresh(); });
    rerender();
    expect(result.current.hasLabels).toBe(true);
  });

  it('updates selectedDeviceId when setSelectedDeviceId is called', async () => {
    mockMediaDevices(async () => [
      makeDevice({ deviceId: 'mic1', kind: 'audioinput', label: 'Built-in Mic' }),
    ]);
    const { result } = renderHook(() => useAudioDevices());
    await waitFor(() => expect(result.current.devices.length).toBe(1));
    act(() => result.current.setSelectedDeviceId('mic1'));
    expect(result.current.selectedDeviceId).toBe('mic1');
  });

  it('refresh() re-enumerates the device list', async () => {
    let call = 0;
    mockMediaDevices(async () => {
      call += 1;
      return call === 1
        ? [makeDevice({ deviceId: 'mic1', kind: 'audioinput', label: '' })]
        : [
            makeDevice({ deviceId: 'mic1', kind: 'audioinput', label: 'Built-in' }),
            makeDevice({ deviceId: 'mic2', kind: 'audioinput', label: 'Ray-Ban Meta' }),
          ];
    });
    const { result } = renderHook(() => useAudioDevices());
    await waitFor(() => expect(result.current.devices.length).toBe(1));
    await act(async () => { await result.current.refresh(); });
    expect(result.current.devices.length).toBe(2);
  });

  it('resets selectedDeviceId when the selected device disappears on refresh', async () => {
    let call = 0;
    mockMediaDevices(async () => {
      call += 1;
      return call === 1
        ? [
            makeDevice({ deviceId: 'mic1', kind: 'audioinput', label: 'Built-in' }),
            makeDevice({ deviceId: 'raybans', kind: 'audioinput', label: 'Ray-Ban Meta' }),
          ]
        : [makeDevice({ deviceId: 'mic1', kind: 'audioinput', label: 'Built-in' })];
    });
    const { result } = renderHook(() => useAudioDevices());
    await waitFor(() => expect(result.current.devices.length).toBe(2));

    act(() => result.current.setSelectedDeviceId('raybans'));
    expect(result.current.selectedDeviceId).toBe('raybans');

    await act(async () => { await result.current.refresh(); });

    // raybans is no longer in the device list — selection should reset to ''
    expect(result.current.selectedDeviceId).toBe('');
  });

  it('keeps selectedDeviceId on refresh if the device is still present', async () => {
    mockMediaDevices(async () => [
      makeDevice({ deviceId: 'mic1', kind: 'audioinput', label: 'Built-in' }),
    ]);
    const { result } = renderHook(() => useAudioDevices());
    await waitFor(() => expect(result.current.devices.length).toBe(1));

    act(() => result.current.setSelectedDeviceId('mic1'));
    await act(async () => { await result.current.refresh(); });

    expect(result.current.selectedDeviceId).toBe('mic1');
  });

  it('sets error and empty devices when enumerateDevices rejects', async () => {
    mockMediaDevices(async () => { throw new Error('not allowed'); });
    const { result } = renderHook(() => useAudioDevices());
    await waitFor(() => expect(result.current.error).not.toBeNull());
    expect(result.current.devices).toEqual([]);
  });

  it('reports error and empty devices when navigator.mediaDevices is undefined', async () => {
    Object.defineProperty(global.navigator, 'mediaDevices', {
      configurable: true,
      value: undefined,
    });
    const { result } = renderHook(() => useAudioDevices());
    await waitFor(() => expect(result.current.error).not.toBeNull());
    expect(result.current.devices).toEqual([]);
  });

  it('subscribes to devicechange events and unsubscribes on unmount', async () => {
    const addEventListener = vi.fn();
    const removeEventListener = vi.fn();
    mockMediaDevices(async () => [], { addEventListener, removeEventListener });
    const { unmount } = renderHook(() => useAudioDevices());
    await waitFor(() => expect(addEventListener).toHaveBeenCalledWith('devicechange', expect.any(Function)));
    unmount();
    expect(removeEventListener).toHaveBeenCalledWith('devicechange', expect.any(Function));
  });
});
