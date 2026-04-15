import React, { useState, useEffect, useCallback } from "react";
import { Clock, FileText, Loader2, AlertCircle, Database, RefreshCw } from "lucide-react";
import { getSessions, type Session, type NavView } from "../api";
import NavHeader from "./NavHeader";

interface SessionHistoryProps {
  onNavigate: (view: NavView) => void;
  onQuerySession: (sourceId: string) => void;
}

const SessionHistory: React.FC<SessionHistoryProps> = ({ onNavigate, onQuerySession }) => {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSessions = useCallback(() => {
    setLoading(true);
    setError(null);
    getSessions()
      .then((res) => setSessions(res.sessions))
      .catch((err) => setError(err.response?.data?.detail || "Failed to load sessions."))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { fetchSessions(); }, [fetchSessions]);

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
        <p style={{ color: "#64748b", fontSize: "0.875rem", textAlign: "center", marginBottom: "1rem" }}>
          Browse your past recordings. Click a session to ask questions about it.
        </p>
        <div style={{ textAlign: "center", marginBottom: "1.5rem" }}>
          <button
            onClick={fetchSessions}
            disabled={loading}
            style={{
              background: "none",
              border: "1px solid #e2e8f0",
              borderRadius: "0.375rem",
              padding: "0.375rem 0.75rem",
              fontSize: "0.8rem",
              color: "#64748b",
              cursor: loading ? "not-allowed" : "pointer",
              display: "inline-flex",
              alignItems: "center",
              gap: "0.375rem",
            }}
          >
            <RefreshCw size={12} /> Refresh
          </button>
        </div>

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
