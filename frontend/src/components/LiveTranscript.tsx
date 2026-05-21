import React from 'react';
import type { LiveStreamStatus } from '../hooks/useLiveStream';

interface LiveTranscriptProps {
  interim: string;
  finals: string[];
  status: LiveStreamStatus;
}

const wrapperStyle: React.CSSProperties = {
  width: '100%',
  minHeight: '8rem',
  padding: '1rem 1.25rem',
  borderRadius: '0.5rem',
  border: '1px solid #e2e8f0',
  backgroundColor: '#f8fafc',
  color: '#1e293b',
  fontSize: '0.9375rem',
  lineHeight: 1.5,
  overflowY: 'auto',
  whiteSpace: 'pre-wrap',
};

const placeholderStyle: React.CSSProperties = {
  color: '#94a3b8',
  fontStyle: 'italic',
};

const interimStyle: React.CSSProperties = {
  color: '#94a3b8',
};

export const LiveTranscript: React.FC<LiveTranscriptProps> = ({ interim, finals, status }) => {
  const hasAnyText = interim.length > 0 || finals.length > 0;

  return (
    <div style={wrapperStyle} aria-live="polite" role="log">
      {!hasAnyText && (
        <span style={placeholderStyle}>
          {status === 'connecting' ? 'Connecting…' : 'Listening…'}
        </span>
      )}
      {finals.length > 0 && <span>{finals.join(' ')} </span>}
      {interim.length > 0 && <span style={interimStyle}>{interim}</span>}
    </div>
  );
};
