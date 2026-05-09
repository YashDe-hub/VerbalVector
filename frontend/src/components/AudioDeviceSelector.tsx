import React from 'react';
import type { AudioDevice } from '../hooks/useAudioDevices';

interface AudioDeviceSelectorProps {
  devices: AudioDevice[];
  selectedDeviceId: string;
  onChange: (deviceId: string) => void;
  hasLabels: boolean;
  disabled?: boolean;
}

const wrapperStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: '0.25rem',
  width: '100%',
  marginBottom: '1rem',
};

const labelStyle: React.CSSProperties = {
  fontSize: '0.75rem',
  color: '#475569',
  fontWeight: 500,
};

const selectStyle: React.CSSProperties = {
  padding: '0.5rem 0.75rem',
  borderRadius: '0.375rem',
  border: '1px solid #e2e8f0',
  fontSize: '0.875rem',
  backgroundColor: 'white',
};

const hintStyle: React.CSSProperties = {
  fontSize: '0.75rem',
  color: '#94a3b8',
  marginTop: '0.125rem',
};

function deviceLabel(device: AudioDevice): string {
  if (device.label) return device.label;
  const short = device.deviceId.slice(0, 8) || 'unknown';
  return `Microphone (${short})`;
}

export const AudioDeviceSelector: React.FC<AudioDeviceSelectorProps> = ({
  devices,
  selectedDeviceId,
  onChange,
  hasLabels,
  disabled = false,
}) => {
  return (
    <div style={wrapperStyle}>
      <label htmlFor="audio-device-select" style={labelStyle}>
        Microphone
      </label>
      <select
        id="audio-device-select"
        value={selectedDeviceId}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        style={selectStyle}
      >
        <option value="">System default</option>
        {devices.map((d) => (
          <option key={d.deviceId} value={d.deviceId}>
            {deviceLabel(d)}
          </option>
        ))}
      </select>
      {!hasLabels && (
        <span style={hintStyle}>
          Allow microphone access (start a recording once) to see device names.
        </span>
      )}
    </div>
  );
};
