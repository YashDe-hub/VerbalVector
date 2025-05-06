import React from 'react';
import ReactMarkdown from 'react-markdown';
import { AnalysisResult } from './VerbalVector';
import { RefreshCw, BarChart2, Percent, Smile, Mic, Clock, MessageSquare, Hash, FileText } from 'lucide-react';
import '../App.css';

interface ResultsDisplayProps {
  analysisResult: AnalysisResult | null;
  onAnalyzeAnother: () => void;
}

const headerStyle: React.CSSProperties = {
    padding: '1.5rem 2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, backgroundColor: 'white', zIndex: 10, width: '100%', boxSizing: 'border-box'
};
const logoContainerStyle: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: '0.5rem' };
const logoCircleStyle: React.CSSProperties = { width: '2rem', height: '2rem', borderRadius: '9999px', backgroundColor: '#6366f1', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: 'inset 0 1px 2px 0 rgba(0,0,0,0.05)' };
const logoTextStyle: React.CSSProperties = { color: '#fff', fontWeight: 600, fontSize: '1.125rem' };
const titleStyle: React.CSSProperties = { fontSize: '1.5rem', fontWeight: 300, letterSpacing: '0.025em', color: '#334155' };
const titleSpanStyle: React.CSSProperties = { fontWeight: 500 };
const footerStyle: React.CSSProperties = {
    padding: '1rem 2rem', borderTop: '1px solid #e2e8f0', textAlign: 'center',
    fontSize: '0.75rem', color: '#94a3b8', marginTop: '3rem'
};
const mainResultsStyle: React.CSSProperties = { padding: '2rem', maxWidth: '64rem', marginLeft: 'auto', marginRight: 'auto', width: '100%' };
const scoreSectionStyle: React.CSSProperties = { marginBottom: '2.5rem' };
const scoreContainerStyle: React.CSSProperties = { display: 'flex', justifyContent: 'space-around', gap: '1rem', marginBottom: '2rem', padding: '1.5rem', backgroundColor: '#f1f5f9', borderRadius: '0.5rem' };
const scoreItemStyle: React.CSSProperties = { textAlign: 'center', flex: 1 };
const scoreValueStyle: React.CSSProperties = { fontSize: '1.875rem', fontWeight: 600, color: '#4f46e5' };
const scoreLabelStyle: React.CSSProperties = { fontSize: '0.875rem', color: '#64748b', marginTop: '0.25rem' };
const sectionStyle: React.CSSProperties = { marginBottom: '2rem' };
const sectionTitleStyle: React.CSSProperties = { fontSize: '1.25rem', fontWeight: 500, color: '#334155', marginBottom: '1.5rem', borderBottom: '1px solid #e2e8f0', paddingBottom: '0.75rem' };
const transcriptBoxStyle: React.CSSProperties = { maxHeight: '200px', overflowY: 'auto', backgroundColor: '#f8fafc', padding: '1rem', borderRadius: '0.375rem', border: '1px solid #e2e8f0', fontSize: '0.875rem', lineHeight: 1.6, whiteSpace: 'pre-wrap', fontFamily: 'monospace' };
const feedbackBoxStyle: React.CSSProperties = { backgroundColor: '#f8fafc', padding: '1rem', borderRadius: '0.375rem', border: '1px solid #e2e8f0', fontSize: '0.95rem', lineHeight: 1.7 };
const analyzeButtonStyle: React.CSSProperties = { backgroundColor: '#6366f1', color: 'white', fontWeight: 500, padding: '0.75rem 1.5rem', borderRadius: '0.5rem', display: 'inline-flex', alignItems: 'center', gap: '0.5rem', boxShadow: '0 1px 2px 0 rgba(0,0,0,0.05)', transition: 'background-color 0.2s ease-in-out', border: 'none', cursor: 'pointer' };
const analyzeButtonHoverStyle: React.CSSProperties = { backgroundColor: '#4f46e5' };

// NEW: Styles for Detailed Metrics Section
const detailedMetricsContainerStyle: React.CSSProperties = {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', // Responsive grid
    gap: '1.5rem',
    padding: '1.5rem',
    backgroundColor: '#f8fafc', // Slightly different background
    borderRadius: '0.5rem',
    border: '1px solid #e2e8f0'
};
const metricItemStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem'
};
const metricIconStyle: React.CSSProperties = {
    color: '#64748b', // slate-500
    flexShrink: 0
};
const metricInfoStyle: React.CSSProperties = { flexGrow: 1 };
const metricValueStyle: React.CSSProperties = { fontSize: '1.125rem', fontWeight: 500, color: '#334155' };
const metricLabelStyle: React.CSSProperties = { fontSize: '0.8rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' };

// Helper function to format score as percentage
const formatPercent = (value: number | undefined | null): string => {
  if (typeof value !== 'number' || isNaN(value)) return 'N/A';
  return `${Math.round(value)}%`;
};

// NEW: Helper to normalize a value to a 0-100 scale based on min/max range
const normalizeToPercent = (value: number | undefined | null, min: number, max: number): number | null => {
    if (typeof value !== 'number' || isNaN(value)) return null;
    // Clamp value between min and max
    const clampedValue = Math.max(min, Math.min(max, value));
    // Normalize to 0-100
    return ((clampedValue - min) / (max - min)) * 100;
};

// NEW: Helper to safely get score for weighted average (returns 0 if input is not a valid number)
const getScoreValueForWeightedAverage = (value: number | undefined | null): number => {
  return (typeof value === 'number' && !isNaN(value)) ? value : 0;
};

// Format raw metrics for display
const formatMetric = (value: number | undefined | null, decimals: number = 0, unit: string = ''): string => {
    if (typeof value !== 'number' || isNaN(value)) return 'N/A';
    return `${value.toFixed(decimals)}${unit ? ' ' + unit : ''}`;
};

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ analysisResult, onAnalyzeAnother }) => {

  // --- DEBUGGING LOGS --- 
  console.log("[ResultsDisplay] Received analysisResult:", analysisResult);
  // Log features specifically if available
  if (analysisResult?.features) {
      console.log("[ResultsDisplay] Features object received:", analysisResult.features);
  }
  // --- END DEBUGGING LOGS ---

  if (!analysisResult || !analysisResult.features || !analysisResult.feedback) {
    return (
        <div style={{ padding: '2rem', textAlign: 'center' }}>
            <p>Incomplete analysis results received. Please try again.</p>
            <button
                onClick={onAnalyzeAnother}
                style={analyzeButtonStyle}
                onMouseOver={(e) => Object.assign(e.currentTarget.style, analyzeButtonHoverStyle)}
                onMouseOut={(e) => Object.assign(e.currentTarget.style, analyzeButtonStyle)}
            >
                <RefreshCw size={18} /> Try Again
            </button>
        </div>
    );
  }

  // --- Key Scores Calculation (From Features) ---
  // Extract raw values
  const rawClarity = analysisResult.features.speech_clarity;
  const rawVocalExpressiveness = analysisResult.features.vocal_expressiveness;
  const rawPaceWPM = analysisResult.features.words_per_minute;
  const rawUniqueWords = analysisResult.features.unique_word_count;

  // Normalize to percentages (0-100)
  const clarityPercent = rawClarity !== undefined ? rawClarity * 100 : null; // Assuming 0-1 scale
  const vocalVarietyPercent = rawVocalExpressiveness !== undefined ? rawVocalExpressiveness * 100 : null; // Assuming 0-1 scale
  const pacingPercent = normalizeToPercent(rawPaceWPM, 100, 200); // Normalize WPM (100-200 range)
  const engagementPercent = normalizeToPercent(rawUniqueWords, 100, 500); // Normalize unique words (100-500 range)

  // Calculate weighted average for overall score
  const weights = { clarity: 0.3, vocalVariety: 0.3, pacing: 0.2, engagement: 0.2 };
  const weightedSum = 
      getScoreValueForWeightedAverage(clarityPercent) * weights.clarity +
      getScoreValueForWeightedAverage(vocalVarietyPercent) * weights.vocalVariety +
      getScoreValueForWeightedAverage(pacingPercent) * weights.pacing +
      getScoreValueForWeightedAverage(engagementPercent) * weights.engagement;
  const calculatedOverall = Math.round(weightedSum);

  // --- Secondary Metric Extraction (From Features) ---
  const fillerCount = analysisResult.features.total_filler_words;
  const paceWPM = analysisResult.features.words_per_minute;
  const avgSentLength = analysisResult.features.mean_sentence_length;
  const uniqueWords = analysisResult.features.unique_word_count;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        <header style={headerStyle}>
            <div style={logoContainerStyle}>
                <div style={logoCircleStyle}><span style={logoTextStyle}>V</span></div>
                <h1 style={titleStyle}><span style={titleSpanStyle}>Verbal</span> Vector</h1>
            </div>
        </header>

        <main style={mainResultsStyle}>
            <h2 style={{ fontSize: '1.875rem', fontWeight: 300, marginBottom: '2.5rem', textAlign: 'center' }}>
                Analysis Results
            </h2>

            {/* --- Primary Scores Section (Now using features) --- */}
            <div style={scoreSectionStyle}>
                <h3 style={sectionTitleStyle}>Key Scores</h3>
                <div style={scoreContainerStyle}>
                    <div style={scoreItemStyle}>
                        <div style={scoreValueStyle}>{formatPercent(calculatedOverall)}</div>
                        <div style={scoreLabelStyle}>Overall</div>
                    </div>
                    <div style={scoreItemStyle}>
                        <div style={scoreValueStyle}>{formatPercent(clarityPercent)}</div>
                        <div style={scoreLabelStyle}>Clarity</div>
                    </div>
                    <div style={scoreItemStyle}>
                        <div style={scoreValueStyle}>{formatPercent(engagementPercent)}</div>
                        <div style={scoreLabelStyle}>Engagement</div>
                    </div>
                    <div style={scoreItemStyle}>
                        <div style={scoreValueStyle}>{formatPercent(pacingPercent)}</div>
                        <div style={scoreLabelStyle}>Pacing</div>
                    </div>
                    <div style={scoreItemStyle}>
                        <div style={scoreValueStyle}>{formatPercent(vocalVarietyPercent)}</div>
                        <div style={scoreLabelStyle}>Vocal Variety</div>
                    </div>
                </div>
            </div>

            {/* --- Detailed Metrics Section (Reads from features) --- */}
            <div style={sectionStyle}>
                <h3 style={sectionTitleStyle}>Detailed Metrics</h3>
                <div style={detailedMetricsContainerStyle}>
                     {/* Filler Words */}
                     <div style={metricItemStyle}>
                        <Hash size={20} style={metricIconStyle} />
                        <div style={metricInfoStyle}>
                            <div style={metricValueStyle}>{formatMetric(fillerCount)}</div>
                            <div style={metricLabelStyle}>Filler Words</div>
                        </div>
                    </div>
                    {/* Pace WPM */}
                    <div style={metricItemStyle}>
                        <Clock size={20} style={metricIconStyle} />
                        <div style={metricInfoStyle}>
                            <div style={metricValueStyle}>{formatMetric(paceWPM, 0, 'WPM')}</div>
                            <div style={metricLabelStyle}>Pace</div>
                        </div>
                    </div>
                    {/* Avg Sentence Length */}
                    <div style={metricItemStyle}>
                        <MessageSquare size={20} style={metricIconStyle} />
                        <div style={metricInfoStyle}>
                            <div style={metricValueStyle}>{formatMetric(avgSentLength, 1, 'words')}</div>
                            <div style={metricLabelStyle}>Avg. Sentence Length</div>
                        </div>
                    </div>
                    {/* Unique Words */}
                     <div style={metricItemStyle}>
                        <FileText size={20} style={metricIconStyle} />
                        <div style={metricInfoStyle}>
                            <div style={metricValueStyle}>{formatMetric(uniqueWords)}</div>
                            <div style={metricLabelStyle}>Unique Words</div>
                        </div>
                    </div>
                    {/* Add more metrics here if needed */} 
                </div>
            </div>

             {/* --- Feedback Section (Displays feedback string) --- */}
            <div style={sectionStyle}>
                <h3 style={sectionTitleStyle}>Feedback</h3>
                <div style={feedbackBoxStyle} className="prose">
                    <ReactMarkdown>{analysisResult.feedback}</ReactMarkdown>
                </div>
            </div>

            {/* --- Transcript Section (Reads from transcript object) --- */}
            <div style={sectionStyle}>
                <h3 style={sectionTitleStyle}>Transcript</h3>
                <pre style={transcriptBoxStyle}>
                    {(typeof analysisResult.transcript === 'object' && analysisResult.transcript?.text)
                        ? analysisResult.transcript.text
                        : (typeof analysisResult.transcript === 'string' ? analysisResult.transcript : 'Transcript not available.')}
                </pre>
            </div>

            <div style={{ textAlign: 'center', marginTop: '2.5rem' }}>
                <button
                    onClick={onAnalyzeAnother}
                    style={analyzeButtonStyle}
                    onMouseOver={(e) => e.currentTarget.style.backgroundColor = analyzeButtonHoverStyle.backgroundColor || ''}
                    onMouseOut={(e) => e.currentTarget.style.backgroundColor = analyzeButtonStyle.backgroundColor || ''}
                >
                    <RefreshCw size={18} /> Analyze Another
                </button>
            </div>

        </main>

        <footer style={footerStyle}>
            <p>&copy; {new Date().getFullYear()} Verbal Vector – Advanced Speech Analytics</p>
        </footer>
    </div>
  );
};

export default ResultsDisplay;
