# Phase 2B: Query UI & Session History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add frontend components for querying stored transcripts and browsing session history. Users can ask questions about past recordings and scope queries to specific sessions. Also add a session label input to the upload flow.

**Architecture:** React frontend talks to Phase 2A's backend endpoints: `POST /api/query` for RAG Q&A, `GET /api/sessions` for session listing. App.tsx adds navigation between analysis, query, and history views.

**Tech Stack:** React 19, TypeScript, MUI (already installed), axios, Vite

**Depends on:** Phase 2A backend endpoints must exist. This plan can be developed in parallel using mocked API responses, then connected once Phase 2A lands.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `frontend/src/App.tsx` | Modify | Add view routing (analysis, query, history), manage navigation state |
| `frontend/src/components/VerbalVector.tsx` | Modify | Add session label input to upload flow |
| `frontend/src/components/QueryInterface.tsx` | Create | Search bar + results display for transcript Q&A |
| `frontend/src/components/SessionHistory.tsx` | Create | List of past sessions with metadata, click to scope queries |
| `frontend/src/components/NavHeader.tsx` | Create | Shared header with working navigation links |
| `frontend/src/api.ts` | Create | Centralized API client (query, sessions, upload) |

---

### Task 1: Create centralized API client

**Files:**
- Create: `frontend/src/api.ts`

Currently `VerbalVector.tsx` calls axios directly with a hardcoded URL. Centralizing API calls makes it easy to add the new query and sessions endpoints, and avoids scattering `http://localhost:5002` across components.

- [ ] **Step 1: Create api.ts**

Create `frontend/src/api.ts`:

```typescript
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

export async function uploadAudio(
  file: File,
  sessionLabel: string = "",
  onUploadProgress?: (pct: number) => void,
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (sessionLabel) formData.append("session_label", sessionLabel);

  const res = await client.post<UploadResponse>("/api/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
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
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd frontend && npx tsc --noEmit src/api.ts`

- [ ] **Step 3: Commit**

```bash
git add frontend/src/api.ts
git commit -m "feat(frontend): add centralized API client

Typed wrappers for /api/upload, /api/query, /api/sessions.
Uses VITE_API_URL env var with localhost:5002 fallback."
```

---

### Task 2: Extract shared NavHeader component

**Files:**
- Create: `frontend/src/components/NavHeader.tsx`

Both `VerbalVector.tsx` and `ResultsDisplay.tsx` duplicate the header markup. Extract it into a shared component that also supports actual navigation between views.

- [ ] **Step 1: Create NavHeader.tsx**

Create `frontend/src/components/NavHeader.tsx`:

```typescript
import React from "react";

type View = "analysis" | "query" | "history";

interface NavHeaderProps {
  activeView: View;
  onNavigate: (view: View) => void;
}

const NavHeader: React.FC<NavHeaderProps> = ({ activeView, onNavigate }) => {
  const navItems: { label: string; view: View }[] = [
    { label: "Analyze", view: "analysis" },
    { label: "Ask", view: "query" },
    { label: "History", view: "history" },
  ];

  return (
    <header
      style={{
        padding: "1.5rem 2rem",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        borderBottom: "1px solid #e2e8f0",
        position: "sticky",
        top: 0,
        backgroundColor: "white",
        zIndex: 10,
        width: "100%",
        boxSizing: "border-box",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
        <div
          style={{
            width: "2rem",
            height: "2rem",
            borderRadius: "9999px",
            backgroundColor: "#6366f1",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <span style={{ color: "#fff", fontWeight: 600, fontSize: "1.125rem" }}>V</span>
        </div>
        <h1 style={{ fontSize: "1.5rem", fontWeight: 300, letterSpacing: "0.025em", color: "#334155" }}>
          <span style={{ fontWeight: 500 }}>Verbal</span> Vector
        </h1>
      </div>
      <nav style={{ display: "flex", gap: "1.5rem", fontSize: "0.875rem", fontWeight: 500 }}>
        {navItems.map((item) => (
          <a
            key={item.view}
            href="#"
            onClick={(e) => {
              e.preventDefault();
              onNavigate(item.view);
            }}
            style={{
              color: activeView === item.view ? "#6366f1" : "#475569",
              textDecoration: "none",
              borderBottom: activeView === item.view ? "2px solid #6366f1" : "2px solid transparent",
              paddingBottom: "0.25rem",
              transition: "color 0.2s ease-in-out",
            }}
          >
            {item.label}
          </a>
        ))}
      </nav>
    </header>
  );
};

export default NavHeader;
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/NavHeader.tsx
git commit -m "feat(frontend): add shared NavHeader with view navigation

Replaces duplicated header in VerbalVector.tsx and ResultsDisplay.tsx.
Active view is highlighted. Supports analysis, query, and history views."
```

---

### Task 3: Add session label input to upload flow

**Files:**
- Modify: `frontend/src/components/VerbalVector.tsx`

Add an optional text input where users can name their session before uploading or recording. Pass it through the upload API call.

- [ ] **Step 1: Add session label state and input**

In `VerbalVector.tsx`, add state:

```typescript
const [sessionLabel, setSessionLabel] = useState('');
```

In the `stage === 'input'` section, add a text input above the Record/Upload buttons:

```tsx
<input
  type="text"
  placeholder="Session name (optional)"
  value={sessionLabel}
  onChange={(e) => setSessionLabel(e.target.value)}
  style={{
    width: "100%",
    padding: "0.75rem 1rem",
    borderRadius: "0.5rem",
    border: "1px solid #e2e8f0",
    fontSize: "0.875rem",
    marginBottom: "1.5rem",
    outline: "none",
  }}
/>
```

- [ ] **Step 2: Update API call to include session_label**

In the `processFile` function (inside the `useEffect` for processing), update the FormData:

```typescript
const formData = new FormData();
formData.append('file', audioFile);
if (sessionLabel) formData.append('session_label', sessionLabel);
```

- [ ] **Step 3: Remove duplicated header from VerbalVector.tsx**

Replace the inline `<header>` JSX with the shared `NavHeader` component. Add import:

```typescript
import NavHeader from './NavHeader';
```

Replace the `<header style={headerStyle}>...</header>` block with:

```tsx
<NavHeader activeView="analysis" onNavigate={onNavigate} />
```

Add `onNavigate` to `VerbalVectorProps`:

```typescript
interface VerbalVectorProps {
  onAnalysisComplete: (result: AnalysisResult) => void;
  onNavigate: (view: "analysis" | "query" | "history") => void;
}
```

Remove all the header-related style constants (`headerStyle`, `logoContainerStyle`, `logoCircleStyle`, `logoTextStyle`, `titleStyle`, `titleSpanStyle`, `navStyle`, `navLinkStyle`, `navLinkHoverStyle`).

- [ ] **Step 4: Remove duplicated header from ResultsDisplay.tsx**

Same treatment — import `NavHeader`, replace inline header, add `onNavigate` prop.

- [ ] **Step 5: Verify it builds**

Run: `cd frontend && npm run build`

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/VerbalVector.tsx frontend/src/components/ResultsDisplay.tsx
git commit -m "feat(frontend): add session label input and use shared NavHeader

Users can name sessions before uploading. session_label is sent as a form
field alongside the audio file. Both VerbalVector and ResultsDisplay now
use the shared NavHeader component."
```

---

### Task 4: Create QueryInterface component

**Files:**
- Create: `frontend/src/components/QueryInterface.tsx`

A search interface for asking questions about stored transcripts. Text input for the query, optional session scope dropdown, and a results area showing the answer with source citations.

- [ ] **Step 1: Create QueryInterface.tsx**

Create `frontend/src/components/QueryInterface.tsx`:

```typescript
import React, { useState, useEffect } from "react";
import { Search, Loader2, AlertCircle, Filter } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { queryTranscripts, getSessions, type QueryResponse, type Session } from "../api";
import NavHeader from "./NavHeader";

interface QueryInterfaceProps {
  onNavigate: (view: "analysis" | "query" | "history") => void;
}

const QueryInterface: React.FC<QueryInterfaceProps> = ({ onNavigate }) => {
  const [query, setQuery] = useState("");
  const [sourceId, setSourceId] = useState<string>("");
  const [sessions, setSessions] = useState<Session[]>([]);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getSessions()
      .then((res) => setSessions(res.sessions))
      .catch(() => {});
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await queryTranscripts(query, sourceId || undefined);
      setResult(res);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to get answer. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", minHeight: "100vh", width: "100%" }}>
      <NavHeader activeView="query" onNavigate={onNavigate} />

      <main style={{ flexGrow: 1, padding: "2rem", maxWidth: "48rem", margin: "0 auto", width: "100%" }}>
        <h2 style={{ fontSize: "1.875rem", fontWeight: 300, marginBottom: "0.5rem", textAlign: "center" }}>
          Ask your transcripts
        </h2>
        <p style={{ color: "#64748b", fontSize: "0.875rem", textAlign: "center", marginBottom: "2rem" }}>
          Search across all your stored recordings or scope to a specific session.
        </p>

        <form onSubmit={handleSubmit} style={{ marginBottom: "2rem" }}>
          <div style={{ display: "flex", gap: "0.5rem", marginBottom: "0.75rem" }}>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="What did I say about the project timeline?"
              style={{
                flex: 1,
                padding: "0.75rem 1rem",
                borderRadius: "0.5rem",
                border: "1px solid #e2e8f0",
                fontSize: "0.95rem",
                outline: "none",
              }}
            />
            <button
              type="submit"
              disabled={loading || !query.trim()}
              style={{
                backgroundColor: loading ? "#a5b4fc" : "#6366f1",
                color: "white",
                padding: "0.75rem 1.25rem",
                borderRadius: "0.5rem",
                border: "none",
                cursor: loading ? "not-allowed" : "pointer",
                display: "flex",
                alignItems: "center",
                gap: "0.5rem",
              }}
            >
              {loading ? <Loader2 size={18} className="animate-spin" /> : <Search size={18} />}
              Ask
            </button>
          </div>

          {sessions.length > 0 && (
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <Filter size={14} style={{ color: "#64748b" }} />
              <select
                value={sourceId}
                onChange={(e) => setSourceId(e.target.value)}
                style={{
                  padding: "0.5rem 0.75rem",
                  borderRadius: "0.375rem",
                  border: "1px solid #e2e8f0",
                  fontSize: "0.8rem",
                  color: "#475569",
                  backgroundColor: "white",
                }}
              >
                <option value="">All sessions</option>
                {sessions.map((s) => (
                  <option key={s.source_id} value={s.source_id}>
                    {s.session_label || s.source_id.slice(0, 12)}
                  </option>
                ))}
              </select>
            </div>
          )}
        </form>

        {error && (
          <div
            style={{
              padding: "1rem",
              backgroundColor: "#fee2e2",
              border: "1px solid #fecaca",
              color: "#b91c1c",
              borderRadius: "0.5rem",
              display: "flex",
              alignItems: "center",
              gap: "0.5rem",
              marginBottom: "1.5rem",
            }}
          >
            <AlertCircle size={18} />
            <span style={{ fontSize: "0.875rem" }}>{error}</span>
          </div>
        )}

        {result && (
          <div style={{ backgroundColor: "#f8fafc", borderRadius: "0.5rem", border: "1px solid #e2e8f0", padding: "1.5rem" }}>
            <div style={{ fontSize: "0.95rem", lineHeight: 1.7, marginBottom: "1rem" }} className="prose">
              <ReactMarkdown>{result.answer}</ReactMarkdown>
            </div>

            {result.sources.length > 0 && (
              <div style={{ borderTop: "1px solid #e2e8f0", paddingTop: "1rem" }}>
                <h4 style={{ fontSize: "0.75rem", color: "#64748b", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "0.5rem" }}>
                  Sources
                </h4>
                {result.sources.map((src, i) => (
                  <div
                    key={i}
                    style={{
                      fontSize: "0.8rem",
                      color: "#475569",
                      padding: "0.5rem 0.75rem",
                      backgroundColor: "white",
                      borderRadius: "0.375rem",
                      border: "1px solid #e2e8f0",
                      marginBottom: "0.5rem",
                    }}
                  >
                    <span style={{ fontWeight: 500 }}>{src.session_label || src.source_id.slice(0, 12)}</span>
                    {src.text && (
                      <p style={{ color: "#64748b", marginTop: "0.25rem", fontStyle: "italic" }}>
                        "{src.text.slice(0, 150)}..."
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>

      <footer style={{ padding: "1rem 2rem", borderTop: "1px solid #e2e8f0", textAlign: "center", fontSize: "0.75rem", color: "#94a3b8", marginTop: "3rem" }}>
        <p>&copy; {new Date().getFullYear()} Verbal Vector – Advanced Speech Analytics</p>
      </footer>
    </div>
  );
};

export default QueryInterface;
```

- [ ] **Step 2: Verify it compiles**

Run: `cd frontend && npx tsc --noEmit`

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/QueryInterface.tsx
git commit -m "feat(frontend): add QueryInterface component for transcript Q&A

Search bar with optional session scoping dropdown. Shows RAG answer
with source citations. Loads sessions on mount for the filter dropdown."
```

---

### Task 5: Create SessionHistory component

**Files:**
- Create: `frontend/src/components/SessionHistory.tsx`

A list view showing all stored sessions with metadata (label, timestamp, chunk count). Clicking a session navigates to the query view scoped to that session.

- [ ] **Step 1: Create SessionHistory.tsx**

Create `frontend/src/components/SessionHistory.tsx`:

```typescript
import React, { useState, useEffect } from "react";
import { Clock, FileText, Loader2, AlertCircle, Database } from "lucide-react";
import { getSessions, type Session } from "../api";
import NavHeader from "./NavHeader";

interface SessionHistoryProps {
  onNavigate: (view: "analysis" | "query" | "history") => void;
  onQuerySession: (sourceId: string) => void;
}

const SessionHistory: React.FC<SessionHistoryProps> = ({ onNavigate, onQuerySession }) => {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    getSessions()
      .then((res) => setSessions(res.sessions))
      .catch((err) => setError(err.response?.data?.detail || "Failed to load sessions."))
      .finally(() => setLoading(false));
  }, []);

  const formatDate = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", minHeight: "100vh", width: "100%" }}>
      <NavHeader activeView="history" onNavigate={onNavigate} />

      <main style={{ flexGrow: 1, padding: "2rem", maxWidth: "48rem", margin: "0 auto", width: "100%" }}>
        <h2 style={{ fontSize: "1.875rem", fontWeight: 300, marginBottom: "0.5rem", textAlign: "center" }}>
          Session History
        </h2>
        <p style={{ color: "#64748b", fontSize: "0.875rem", textAlign: "center", marginBottom: "2rem" }}>
          Browse your past recordings. Click a session to ask questions about it.
        </p>

        {loading && (
          <div style={{ textAlign: "center", padding: "3rem" }}>
            <Loader2 size={32} className="animate-spin" style={{ color: "#6366f1" }} />
          </div>
        )}

        {error && (
          <div
            style={{
              padding: "1rem",
              backgroundColor: "#fee2e2",
              border: "1px solid #fecaca",
              color: "#b91c1c",
              borderRadius: "0.5rem",
              display: "flex",
              alignItems: "center",
              gap: "0.5rem",
            }}
          >
            <AlertCircle size={18} />
            <span style={{ fontSize: "0.875rem" }}>{error}</span>
          </div>
        )}

        {!loading && !error && sessions.length === 0 && (
          <div style={{ textAlign: "center", padding: "3rem", color: "#94a3b8" }}>
            <Database size={48} style={{ marginBottom: "1rem", opacity: 0.5 }} />
            <p>No sessions stored yet. Upload or record audio to get started.</p>
          </div>
        )}

        {!loading && sessions.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
            {sessions.map((session) => (
              <div
                key={session.source_id}
                onClick={() => onQuerySession(session.source_id)}
                style={{
                  padding: "1.25rem",
                  backgroundColor: "#f8fafc",
                  borderRadius: "0.5rem",
                  border: "1px solid #e2e8f0",
                  cursor: "pointer",
                  transition: "border-color 0.2s ease-in-out, box-shadow 0.2s ease-in-out",
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.borderColor = "#6366f1";
                  e.currentTarget.style.boxShadow = "0 1px 3px rgba(99, 102, 241, 0.1)";
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.borderColor = "#e2e8f0";
                  e.currentTarget.style.boxShadow = "none";
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <h3 style={{ fontWeight: 500, color: "#334155", fontSize: "1rem" }}>
                    {session.session_label || session.source_id.slice(0, 20)}
                  </h3>
                  <span
                    style={{
                      fontSize: "0.75rem",
                      backgroundColor: "#e0e7ff",
                      color: "#4338ca",
                      padding: "0.125rem 0.5rem",
                      borderRadius: "9999px",
                    }}
                  >
                    {session.chunk_count} chunks
                  </span>
                </div>
                <div style={{ display: "flex", gap: "1rem", marginTop: "0.5rem", fontSize: "0.8rem", color: "#64748b" }}>
                  <span style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
                    <Clock size={12} /> {formatDate(session.timestamp)}
                  </span>
                  <span style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
                    <FileText size={12} /> {session.source_id.slice(0, 12)}...
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>

      <footer style={{ padding: "1rem 2rem", borderTop: "1px solid #e2e8f0", textAlign: "center", fontSize: "0.75rem", color: "#94a3b8", marginTop: "3rem" }}>
        <p>&copy; {new Date().getFullYear()} Verbal Vector – Advanced Speech Analytics</p>
      </footer>
    </div>
  );
};

export default SessionHistory;
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/SessionHistory.tsx
git commit -m "feat(frontend): add SessionHistory component

Lists stored sessions with labels, timestamps, and chunk counts.
Click a session to navigate to the query view scoped to that session."
```

---

### Task 6: Wire up App.tsx with view routing

**Files:**
- Modify: `frontend/src/App.tsx`

Replace the current two-state toggle (`input` vs `results`) with a view router that supports analysis, query, history, and results views.

- [ ] **Step 1: Update App.tsx**

Replace the contents of `App.tsx` with:

```typescript
import { useState } from "react";
import VerbalVector, { AnalysisResult } from "./components/VerbalVector";
import ResultsDisplay from "./components/ResultsDisplay";
import QueryInterface from "./components/QueryInterface";
import SessionHistory from "./components/SessionHistory";
import "./App.css";

type View = "analysis" | "results" | "query" | "history";

function App() {
  const [currentView, setCurrentView] = useState<View>("analysis");
  const [analysisData, setAnalysisData] = useState<AnalysisResult | null>(null);
  const [scopedSourceId, setScopedSourceId] = useState<string>("");

  const handleAnalysisComplete = (result: AnalysisResult) => {
    setAnalysisData(result);
    setCurrentView("results");
  };

  const handleAnalyzeAnother = () => {
    setAnalysisData(null);
    setCurrentView("analysis");
  };

  const handleNavigate = (view: "analysis" | "query" | "history") => {
    setScopedSourceId("");
    setCurrentView(view);
  };

  const handleQuerySession = (sourceId: string) => {
    setScopedSourceId(sourceId);
    setCurrentView("query");
  };

  return (
    <div className="App">
      {currentView === "analysis" && (
        <VerbalVector onAnalysisComplete={handleAnalysisComplete} onNavigate={handleNavigate} />
      )}
      {currentView === "results" && (
        <ResultsDisplay analysisResult={analysisData} onAnalyzeAnother={handleAnalyzeAnother} onNavigate={handleNavigate} />
      )}
      {currentView === "query" && (
        <QueryInterface onNavigate={handleNavigate} initialSourceId={scopedSourceId} />
      )}
      {currentView === "history" && (
        <SessionHistory onNavigate={handleNavigate} onQuerySession={handleQuerySession} />
      )}
    </div>
  );
}

export default App;
```

Note: `QueryInterface` needs an `initialSourceId` prop added — update the component to accept it and use it to pre-populate the source filter dropdown.

- [ ] **Step 2: Add initialSourceId to QueryInterface**

In `QueryInterface.tsx`, update the props interface:

```typescript
interface QueryInterfaceProps {
  onNavigate: (view: "analysis" | "query" | "history") => void;
  initialSourceId?: string;
}
```

Update the component to use it:

```typescript
const QueryInterface: React.FC<QueryInterfaceProps> = ({ onNavigate, initialSourceId = "" }) => {
  const [sourceId, setSourceId] = useState<string>(initialSourceId);
```

- [ ] **Step 3: Add onNavigate prop to ResultsDisplay**

Update `ResultsDisplay.tsx` props:

```typescript
interface ResultsDisplayProps {
  analysisResult: AnalysisResult | null;
  onAnalyzeAnother: () => void;
  onNavigate: (view: "analysis" | "query" | "history") => void;
}
```

Replace the inline header with `NavHeader`:

```tsx
<NavHeader activeView="analysis" onNavigate={onNavigate} />
```

- [ ] **Step 4: Build and verify**

Run: `cd frontend && npm run build`

Expected: Build succeeds with no errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/App.tsx frontend/src/components/QueryInterface.tsx frontend/src/components/ResultsDisplay.tsx
git commit -m "feat(frontend): wire up view routing in App.tsx

App manages navigation between analysis, results, query, and history views.
Session history click pre-scopes the query interface to that session."
```

---

## Summary

| Task | Files | What |
|------|-------|------|
| 1 | `api.ts` | Centralized typed API client |
| 2 | `NavHeader.tsx` | Shared navigation header |
| 3 | `VerbalVector.tsx`, `ResultsDisplay.tsx` | Session label input, use NavHeader |
| 4 | `QueryInterface.tsx` | Transcript Q&A search interface |
| 5 | `SessionHistory.tsx` | Session listing with click-to-query |
| 6 | `App.tsx`, `QueryInterface.tsx`, `ResultsDisplay.tsx` | View routing and wiring |

Tasks 1, 2, 4, and 5 are independent (create new files). Task 3 depends on Task 2 (uses NavHeader). Task 6 depends on all (wires everything together).
