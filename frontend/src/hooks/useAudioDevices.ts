import { useCallback, useEffect, useState } from 'react';

export type AudioDevice = {
  deviceId: string;
  label: string;
};

export interface UseAudioDevicesReturn {
  devices: AudioDevice[];
  selectedDeviceId: string;
  setSelectedDeviceId: (id: string) => void;
  refresh: () => Promise<void>;
  hasLabels: boolean;
  error: string | null;
}

export function useAudioDevices(): UseAudioDevicesReturn {
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) {
      setError('Audio device enumeration is not supported in this browser.');
      setDevices([]);
      return;
    }
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      const audio = all
        .filter((d) => d.kind === 'audioinput')
        .map((d) => ({ deviceId: d.deviceId, label: d.label }));
      setDevices(audio);
      setSelectedDeviceId((current) =>
        current && !audio.some((d) => d.deviceId === current) ? '' : current,
      );
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to enumerate audio devices');
      setDevices([]);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Listen for device hot-plug events (Bluetooth connect/disconnect, USB plug, etc.)
  useEffect(() => {
    if (!navigator.mediaDevices?.addEventListener) return;
    const handler = () => { void refresh(); };
    navigator.mediaDevices.addEventListener('devicechange', handler);
    return () => navigator.mediaDevices.removeEventListener('devicechange', handler);
  }, [refresh]);

  const hasLabels = devices.some((d) => d.label !== '');

  return { devices, selectedDeviceId, setSelectedDeviceId, refresh, hasLabels, error };
}
