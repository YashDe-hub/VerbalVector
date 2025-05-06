import { useState } from 'react';
import VerbalVector, { AnalysisResult } from './components/VerbalVector'; // Assuming VerbalVector exports AnalysisResult type
import ResultsDisplay from './components/ResultsDisplay';
import './App.css'; // Keep existing App CSS if any

function App() {
  // Lift state up: App manages the current view and the analysis data
  const [currentStage, setCurrentStage] = useState<'input' | 'recording' | 'processing' | 'results'>('input');
  const [analysisData, setAnalysisData] = useState<AnalysisResult | null>(null);

  // Callback for VerbalVector to pass results and signal completion
  const handleAnalysisComplete = (result: AnalysisResult) => {
    setAnalysisData(result);
    setCurrentStage('results');
  };

  // Callback for ResultsDisplay to go back to the input screen
  const handleAnalyzeAnother = () => {
    setAnalysisData(null);
    setCurrentStage('input');
  };

  return (
    <div className="App">
      {currentStage !== 'results' ? (
        <VerbalVector
          // Pass state down only if needed by VerbalVector directly, or manage internally
          // For now, VerbalVector manages its internal stages until completion
          onAnalysisComplete={handleAnalysisComplete}
        />
      ) : (
        <ResultsDisplay
          analysisResult={analysisData} // Pass the fetched/mocked data down
          onAnalyzeAnother={handleAnalyzeAnother} // Pass the reset function down
        />
      )}
    </div>
  );
}

export default App;
