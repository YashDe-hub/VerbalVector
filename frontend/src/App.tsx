import { useState, useEffect, type CSSProperties } from "react";
import VerbalVector, { AnalysisResult } from "./components/VerbalVector";
import ResultsDisplay from "./components/ResultsDisplay";
import QueryInterface from "./components/QueryInterface";
import SessionHistory from "./components/SessionHistory";
import { getSessionResult, type NavView } from "./api";
import "./App.css";

type View = NavView | "results";
type PendingStatus = "analyzing" | "ready" | "error";

const POLL_INTERVAL_MS = 2000;
const POLL_MAX_ATTEMPTS = 45; // ~90s; tuned to observed ~61s pipeline

const bannerBase: CSSProperties = {
  padding: "0.75rem 1rem",
  textAlign: "center",
  fontSize: "0.875rem",
  display: "flex",
  gap: "0.75rem",
  alignItems: "center",
  justifyContent: "center",
};

function App() {
  const [currentView, setCurrentView] = useState<View>("analysis");
  const [analysisData, setAnalysisData] = useState<AnalysisResult | null>(null);
  const [scopedSourceId, setScopedSourceId] = useState<string>("");
  const [pending, setPending] = useState<{ sessionId: string; status: PendingStatus } | null>(null);
  const [analysisKey, setAnalysisKey] = useState(0); // bump to remount VerbalVector on retry

  const handleAnalysisComplete = (result: AnalysisResult) => {
    setAnalysisData(result);
    setPending(null); // fast path delivered — stop any poll
    setCurrentView("results");
  };

  const handleAnalyzeAnother = () => {
    setAnalysisData(null);
    setCurrentView("analysis");
  };

  const handleNavigate = (view: NavView) => {
    setScopedSourceId("");
    setCurrentView(view);
  };

  const handleQuerySession = (sourceId: string) => {
    setScopedSourceId(sourceId);
    setCurrentView("query");
  };

  const handleLiveSessionEnding = (sessionId: string) => {
    setAnalysisData(null);
    setPending({ sessionId, status: "analyzing" });
  };

  // Durable poll loop — lives in App so it survives the view-switch unmount of
  // VerbalVector that drops the WebSocket (the original bug).
  useEffect(() => {
    if (pending?.status !== "analyzing") return;
    const sessionId = pending.sessionId;
    let cancelled = false;
    let attempts = 0;
    let timer: ReturnType<typeof setTimeout>;

    const tick = async () => {
      attempts += 1;
      try {
        const res = await getSessionResult(sessionId);
        if (cancelled) return;
        if (res.status === "ready") {
          setAnalysisData(res.data);
          setPending({ sessionId, status: "ready" });
          return;
        }
      } catch {
        // transport/5xx — counts as a failed attempt, keep polling
      }
      if (cancelled) return;
      if (attempts >= POLL_MAX_ATTEMPTS) {
        setPending({ sessionId, status: "error" });
        return;
      }
      timer = setTimeout(tick, POLL_INTERVAL_MS);
    };

    timer = setTimeout(tick, POLL_INTERVAL_MS);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [pending]);

  // Auto-show in place: if the result is ready while the user is still on the
  // analysis view, route to results. If they navigated away, the banner handles it.
  useEffect(() => {
    if (pending?.status === "ready" && currentView === "analysis") {
      setCurrentView("results");
      setPending(null);
    }
  }, [pending, currentView]);

  const showReadyBanner = pending?.status === "ready" && currentView !== "analysis";

  return (
    <div className="App">
      {pending?.status === "analyzing" && currentView !== "analysis" && (
        <div style={{ ...bannerBase, background: "#eef2ff", color: "#4338ca" }}>
          Analyzing your last recording…
        </div>
      )}
      {showReadyBanner && (
        <div style={{ ...bannerBase, background: "#ecfdf5", color: "#065f46" }}>
          <span>✓ Your analysis is ready</span>
          <button onClick={() => { setPending(null); setCurrentView("results"); }}>
            View results
          </button>
        </div>
      )}
      {pending?.status === "error" && (
        <div style={{ ...bannerBase, background: "#fef2f2", color: "#b91c1c" }}>
          <span>Analysis is taking longer than expected.</span>
          <button onClick={() => { setPending(null); handleNavigate("history"); }}>
            Check History
          </button>
          <button onClick={() => { setPending(null); setAnalysisKey((k) => k + 1); setCurrentView("analysis"); }}>
            Retry
          </button>
        </div>
      )}

      {currentView === "analysis" && (
        <VerbalVector
          key={analysisKey}
          onAnalysisComplete={handleAnalysisComplete}
          onNavigate={handleNavigate}
          onLiveSessionEnding={handleLiveSessionEnding}
        />
      )}
      {currentView === "results" && (
        <ResultsDisplay analysisResult={analysisData} onAnalyzeAnother={handleAnalyzeAnother} onNavigate={handleNavigate} />
      )}
      {currentView === "query" && (
        <QueryInterface key={scopedSourceId} onNavigate={handleNavigate} initialSourceId={scopedSourceId} />
      )}
      {currentView === "history" && (
        <SessionHistory onNavigate={handleNavigate} onQuerySession={handleQuerySession} />
      )}
    </div>
  );
}

export default App;
