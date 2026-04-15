import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:5002";

const client = axios.create({ baseURL: API_BASE });

export interface UploadResponse {
  message: string;
  transcript: { text: string; language?: string; segments?: any[] } | string;
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
