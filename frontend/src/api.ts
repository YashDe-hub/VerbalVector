import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:5002";

const client = axios.create({ baseURL: API_BASE });

export interface UploadResponse {
  message?: string;
  transcript:
    | {
        text: string;
        language?: string;
        segments?: any[];
        utterances?: { speaker: number | null; text: string; start: number; end: number; confidence: number }[];
        speakers?: number[];
      }
    | string;
  features: Record<string, any>;
  feedback: string;
}

export interface QueryResponse {
  query: string;
  answer: string;
  sources: { text: string; source_id: string; session_label: string }[];
}

export interface Session {
  source_id: string;
  session_label: string;
  timestamp: number;
  chunk_count: number;
}

export interface SessionsResponse {
  sessions: Session[];
}

export type NavView = "analysis" | "query" | "history";

export async function uploadAudio(
  file: File,
  sessionLabel: string = "",
  onUploadProgress?: (pct: number) => void,
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (sessionLabel) formData.append("session_label", sessionLabel);

  const res = await client.post<UploadResponse>("/api/upload", formData, {
    onUploadProgress: (e) => {
      if (e.total && onUploadProgress) onUploadProgress(Math.round((e.loaded / e.total) * 100));
    },
  });
  return res.data;
}

export async function queryTranscripts(
  query: string,
  sourceId?: string,
  nResults: number = 5,
): Promise<QueryResponse> {
  const res = await client.post<QueryResponse>("/api/query", {
    query,
    source_id: sourceId || undefined,
    n_results: nResults,
  });
  return res.data;
}

export async function getSessions(): Promise<SessionsResponse> {
  const res = await client.get<SessionsResponse>("/api/sessions");
  return res.data;
}

export type ServerMessage =
  | { type: 'session_started'; session_id: string }
  | { type: 'transcript'; text: string; is_final: boolean; speaker: number | null }
  | { type: 'session_end'; transcript: UploadResponse['transcript']; features: UploadResponse['features']; feedback: string }
  | { type: 'error'; message: string; fatal: boolean };

export function getStreamWsUrl(): string {
  return API_BASE.replace(/^http/, 'ws') + '/api/stream';
}

export type SessionResultPoll =
  | { status: 'ready'; data: UploadResponse }
  | { status: 'pending' }
  | { status: 'failed'; permanent: boolean; detail?: string };

export async function getSessionResult(sessionId: string): Promise<SessionResultPoll> {
  try {
    const res = await client.get<UploadResponse | { status: 'pending' }>(
      `/api/sessions/${sessionId}/result`,
      { validateStatus: (s) => s === 200 || s === 202 },
    );
    if (res.status === 202) return { status: 'pending' };
    return { status: 'ready', data: res.data as UploadResponse };
  } catch (err) {
    const httpStatus = axios.isAxiosError(err) ? err.response?.status : undefined;
    const data = axios.isAxiosError(err) ? err.response?.data : undefined;
    const detail = data && typeof data === 'object' ? (data as { detail?: string }).detail : undefined;
    // Any 4xx (bad id, auth, not-found, unprocessable) can never recover → permanent.
    // 5xx / network / unknown → transient; the caller counts the attempt and keeps polling.
    const permanent = typeof httpStatus === 'number' && httpStatus >= 400 && httpStatus < 500;
    return { status: 'failed', permanent, detail };
  }
}
