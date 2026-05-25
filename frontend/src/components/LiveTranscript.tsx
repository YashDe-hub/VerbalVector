import React from 'react';
import type { FinalSegment, LiveStreamStatus } from '../hooks/useLiveStream';

interface LiveTranscriptProps {
  interim: string;
  finals: FinalSegment[];
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

  // Strip "Speaker N:" labels when 0 or 1 unique non-null speakers — keeps the
  // single-speaker UI identical to today's pre-diarization rendering.
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
