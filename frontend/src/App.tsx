import { useState } from "react";
import VerbalVector, { AnalysisResult } from "./components/VerbalVector";
import ResultsDisplay from "./components/ResultsDisplay";
import QueryInterface from "./components/QueryInterface";
import SessionHistory from "./components/SessionHistory";
import type { NavView } from "./api";
import "./App.css";

type View = NavView | "results";

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

  const handleNavigate = (view: NavView) => {
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
        <QueryInterface key={scopedSourceId} onNavigate={handleNavigate} initialSourceId={scopedSourceId} />
      )}
      {currentView === "history" && (
        <SessionHistory onNavigate={handleNavigate} onQuerySession={handleQuerySession} />
      )}
    </div>
  );
}

export default App;
