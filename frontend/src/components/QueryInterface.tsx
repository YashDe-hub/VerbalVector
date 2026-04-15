import React, { useState, useEffect } from "react";
import { Search, Loader2, AlertCircle, Filter } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { queryTranscripts, getSessions, type QueryResponse, type Session, type NavView } from "../api";
import NavHeader from "./NavHeader";

interface QueryInterfaceProps {
  onNavigate: (view: NavView) => void;
  initialSourceId?: string;
}

const QueryInterface: React.FC<QueryInterfaceProps> = ({ onNavigate, initialSourceId = "" }) => {
  const [query, setQuery] = useState("");
  const [sourceId, setSourceId] = useState<string>(initialSourceId);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [sessionLoadError, setSessionLoadError] = useState(false);

  useEffect(() => {
    getSessions()
      .then((res) => setSessions(res.sessions))
      .catch(() => setSessionLoadError(true));
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
    } catch (err: unknown) {
      const axiosErr = err as { response?: { data?: { detail?: string } } };
      setError(axiosErr.response?.data?.detail || "Failed to get answer. Is the backend running?");
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

          {sessionLoadError && (
            <p style={{ fontSize: "0.75rem", color: "#94a3b8", marginBottom: "0.5rem" }}>
              Could not load session filters.
            </p>
          )}
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
                {result.sources.map((src) => (
                  <div
                    key={src.source_id}
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
                        "{src.text.length > 150 ? src.text.slice(0, 150) + "..." : src.text}"
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
