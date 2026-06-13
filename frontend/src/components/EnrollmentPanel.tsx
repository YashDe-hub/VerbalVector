import { useState, useRef, useEffect, type CSSProperties } from 'react';
import { ChevronRight, ChevronDown, Mic, Square, Trash2 } from 'lucide-react';
import { getEnrollment, enrollVoice, deleteEnrollment, type EnrollmentStatus } from '../api';

const boxStyle: CSSProperties = {
  marginTop: '0.5rem',
  padding: '0.75rem 1rem',
  backgroundColor: '#f8fafc',
  border: '1px solid #e2e8f0',
  borderRadius: '0.5rem',
  color: '#475569',
  lineHeight: 1.5,
  textAlign: 'left',
};

const buttonStyle: CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: '0.375rem',
  padding: '0.375rem 0.75rem',
  borderRadius: '0.375rem',
  border: '1px solid #c7d2fe',
  background: '#eef2ff',
  color: '#4338ca',
  cursor: 'pointer',
  fontSize: '0.8125rem',
  marginRight: '0.5rem',
};

/**
 * One-time voice enrollment for wearer-focused analysis. Collapsible panel:
 * fetches status lazily on first expand, records ~30s via MediaRecorder,
 * uploads to /api/enroll.
 */
export function EnrollmentPanel() {
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState<EnrollmentStatus | null | 'loading'>('loading');
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const fetchedRef = useRef(false);

  useEffect(() => {
    if (!open || fetchedRef.current) return;
    fetchedRef.current = true;
    getEnrollment()
      .then(setStatus)
      .catch(() => { setStatus(null); setError('Could not reach the enrollment service.'); });
  }, [open]);

  useEffect(() => () => { streamRef.current?.getTracks().forEach((t) => t.stop()); }, []);

  const startRecording = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];
      const rec = new MediaRecorder(stream);
      recorderRef.current = rec;
      rec.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      rec.onstop = () => { void upload(); };
      rec.start(1000);
      setRecording(true);
    } catch {
      setError('Microphone access denied or unavailable.');
    }
  };

  const stopRecording = () => {
    recorderRef.current?.stop();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    setRecording(false);
  };

  const upload = async () => {
    setBusy(true);
    try {
      const mime = recorderRef.current?.mimeType || 'audio/webm';
      const blob = new Blob(chunksRef.current, { type: mime });
      const updated = await enrollVoice(blob);
      setStatus(updated);
      setError(null);
    } catch (err) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(detail || 'Enrollment upload failed.');
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    setBusy(true);
    try {
      await deleteEnrollment();
      setStatus(null);
      setError(null);
    } catch {
      setError('Could not delete the profile.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{ width: '100%', fontSize: '0.875rem', marginBottom: '1.5rem' }}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', background: 'none', border: 'none', cursor: 'pointer', color: '#4f46e5', padding: 0, fontSize: '0.875rem' }}
      >
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        Enroll my voice (get feedback about you, not the whole room)
      </button>
      {open && (
        <div style={boxStyle}>
          {status === 'loading' && <p style={{ margin: 0 }}>Checking enrollment…</p>}
          {status !== 'loading' && status !== null && (
            <>
              <p style={{ margin: 0 }}>
                ✓ <strong>Voice enrolled</strong> ({status.duration_seconds.toFixed(0)}s sample
                , {new Date(status.created_at).toLocaleDateString()}). Conversations will be analyzed
                for <em>your</em> performance only.
              </p>
              <div style={{ marginTop: '0.5rem' }}>
                {!recording ? (
                  <button style={buttonStyle} disabled={busy} onClick={startRecording}>
                    <Mic size={14} /> Re-record
                  </button>
                ) : (
                  <button style={buttonStyle} disabled={busy} onClick={stopRecording}>
                    <Square size={14} /> Stop &amp; save
                  </button>
                )}
                <button style={buttonStyle} disabled={busy || recording} onClick={remove}>
                  <Trash2 size={14} /> Delete
                </button>
              </div>
            </>
          )}
          {status === null && (
            <>
              <p style={{ margin: 0 }}>
                <strong>Not enrolled.</strong> Record ~30 seconds of normal speech once; after that,
                conversation analyses focus on you and treat everyone else as context.
              </p>
              <div style={{ marginTop: '0.5rem' }}>
                {!recording ? (
                  <button style={buttonStyle} disabled={busy} onClick={startRecording}>
                    <Mic size={14} /> Record enrollment
                  </button>
                ) : (
                  <button style={buttonStyle} disabled={busy} onClick={stopRecording}>
                    <Square size={14} /> Stop &amp; save
                  </button>
                )}
              </div>
            </>
          )}
          {error && <p style={{ margin: '0.5rem 0 0', color: '#b91c1c' }}>{error}</p>}
        </div>
      )}
    </div>
  );
}
