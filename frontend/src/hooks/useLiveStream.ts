import { useCallback, useEffect, useRef, useState } from 'react';
import { getStreamWsUrl, type ServerMessage, type UploadResponse } from '../api';
import { PcmAudioCapture } from '../audio/pcmAudioCapture';

export type LiveStreamStatus = 'idle' | 'connecting' | 'recording' | 'stopping' | 'completed' | 'error';

export interface FinalSegment {
  text: string;
  speaker: number | null;
}

export interface PcmAudioCaptureLike {
  start(options: { deviceId?: string }): Promise<void>;
  setHandler(handler: ((chunk: ArrayBuffer) => void) | null): void;
  stop(): Promise<void>;
}

export interface UseLiveStreamReturn {
  status: LiveStreamStatus;
  interim: string;
  finals: FinalSegment[];
  error: string | null;
  result: UploadResponse | null;
  start: (sessionLabel?: string, deviceId?: string) => Promise<void>;
  stop: () => void;
}

export interface UseLiveStreamOptions {
  createWebSocket?: (url: string) => WebSocket;
  createCapture?: () => PcmAudioCaptureLike;
  onSessionEnding?: (sessionId: string) => void;
}

export function useLiveStream(options: UseLiveStreamOptions = {}): UseLiveStreamReturn {
  const [status, setStatus] = useState<LiveStreamStatus>('idle');
  const [interim, setInterim] = useState<string>('');
  const [finals, setFinals] = useState<FinalSegment[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<UploadResponse | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const captureRef = useRef<PcmAudioCaptureLike | null>(null);
  const pendingLabelRef = useRef<string>('');
  const sessionIdRef = useRef<string>('');
  // Status ref so ws.onclose (which fires outside React's render cycle) can
  // make a decision based on current status without triggering side effects
  // from inside a setState updater.
  const statusRef = useRef<LiveStreamStatus>('idle');

  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  const cleanup = useCallback(async () => {
    if (captureRef.current) {
      try { await captureRef.current.stop(); } catch { /* ignore */ }
      captureRef.current = null;
    }
    if (wsRef.current) {
      try { wsRef.current.close(); } catch { /* ignore */ }
      wsRef.current = null;
    }
  }, []);

  const handleServerMessage = useCallback(async (msg: ServerMessage) => {
    switch (msg.type) {
      case 'session_started': {
        sessionIdRef.current = msg.session_id;
        const initPayload = JSON.stringify({ type: 'init', session_label: pendingLabelRef.current });
        wsRef.current?.send(initPayload);
        const capture = captureRef.current;
        if (capture) {
          capture.setHandler((chunk) => {
            const ws = wsRef.current;
            if (!ws || ws.readyState !== WebSocket.OPEN) return;
            try {
              ws.send(chunk);
            } catch (err) {
              const msg = err instanceof Error ? err.message : 'Connection lost.';
              setError(msg);
              setStatus('error');
              void cleanup();
            }
          });
        }
        setStatus('recording');
        break;
      }
      case 'transcript': {
        // Normalize undefined → null at the boundary so FinalSegment stays honest at runtime
        // even if a server frame ever lacks the speaker field.
        const speaker = msg.speaker ?? null;
        if (msg.is_final) {
          setFinals((prev) => [...prev, { text: msg.text, speaker }]);
          setInterim('');
        } else {
          setInterim(msg.text);
          // Ignore speaker for interim — we don't label interim text in the UI
        }
        break;
      }
      case 'session_end': {
        setResult({
          message: 'Live session complete.',
          transcript: msg.transcript,
          features: msg.features,
          feedback: msg.feedback,
        });
        setStatus('completed');
        await cleanup();
        break;
      }
      case 'error': {
        setError(msg.message);
        if (msg.fatal) {
          setStatus('error');
          await cleanup();
        }
        break;
      }
    }
  }, [cleanup]);

  const start = useCallback(async (sessionLabel: string = '', deviceId?: string) => {
    const current = statusRef.current;
    if (current !== 'idle' && current !== 'completed' && current !== 'error') {
      return;
    }

    await cleanup();

    setStatus('connecting');
    setInterim('');
    setFinals([]);
    setError(null);
    setResult(null);
    pendingLabelRef.current = sessionLabel;

    const createWs = options.createWebSocket ?? ((url: string) => new WebSocket(url));
    const createCapture = options.createCapture ?? (() => new PcmAudioCapture());

    // 1. Acquire mic + audio pipeline FIRST. If this throws, never open WS.
    const capture = createCapture();
    try {
      await capture.start({ deviceId });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Could not access microphone.';
      setError(msg);
      setStatus('error');
      return;
    }
    captureRef.current = capture;

    // 2. Open WebSocket. Handler will be wired up after session_started.
    const ws = createWs(getStreamWsUrl());
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const raw = typeof event.data === 'string' ? event.data : '';
      let parsed: ServerMessage;
      try {
        parsed = JSON.parse(raw) as ServerMessage;
      } catch {
        setError('Received unrecognized message from server.');
        setStatus('error');
        void cleanup();
        return;
      }
      void handleServerMessage(parsed);
    };
    ws.onerror = () => {
      setError('WebSocket error');
      setStatus('error');
      void cleanup();
    };
    ws.onclose = () => {
      const cur = statusRef.current;
      if (cur !== 'completed' && cur !== 'error' && cur !== 'idle' && cur !== 'stopping') {
        setError('Connection closed unexpectedly.');
        setStatus('error');
      }
      void cleanup();
    };
  }, [options.createWebSocket, options.createCapture, handleServerMessage, cleanup]);

  const stop = useCallback(() => {
    // Allow stop from 'recording' (normal) AND 'connecting' (user cancels before session_started)
    const current = statusRef.current;
    if (current !== 'recording' && current !== 'connecting') return;

    if (current === 'connecting') {
      // No server session started yet — just tear down locally.
      // Eagerly update statusRef so that the ws.onclose handler (which fires
      // synchronously in the mock, and before the useEffect syncs statusRef)
      // sees 'idle' and does NOT treat the close as an unexpected error.
      statusRef.current = 'idle';
      setStatus('idle');
      void cleanup();
      return;
    }

    // Normal stop from recording
    setStatus('stopping');
    if (captureRef.current) {
      void captureRef.current.stop();
      captureRef.current = null;
    }
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'end' }));
    }
    if (sessionIdRef.current) {
      options.onSessionEnding?.(sessionIdRef.current);
    }
  }, [cleanup, options.onSessionEnding]);

  useEffect(() => {
    return () => {
      void cleanup();
    };
  }, [cleanup]);

  return { status, interim, finals, error, result, start, stop };
}
