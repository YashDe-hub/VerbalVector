import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios'; // Import axios
import { Mic, Upload, Pause, CheckCircle, AlertCircle, Info, Loader2 } from 'lucide-react'; // Removed unused Play, added Loader2
import '../App.css'; // Import the CSS file

// Define expected type for analysis results (adjust as needed based on actual API)
export interface AnalysisResult {
  // Update transcript type to match API response
  transcript: {
    text: string;
    language?: string; // Optional if present
    segments?: any[]; // Optional, type can be refined if needed
  } | string; // Allow string as fallback? Or just the object?
  features: {
    // Primary Score Inputs (some might need normalization)
    clarity?: number;         // Expecting 0-100
    engagement?: number;      // Expecting 0-100
    paceWPM?: number;         // Expecting raw Words Per Minute
    vocalVariety?: number;    // Expecting 0-100

    // Secondary Metrics (Raw values)
    fillerWordCount?: number;
    avgSentenceLength?: number;
    talkTimeRatio?: number;   // Expecting 0-1 ratio
    uniqueWordCount?: number;
    // sentimentScore?: number; // Example: Add if available

    [key: string]: any; // Allow other potential features
  };
  feedback: string;
  overallScore?: number; // Keep API's overall if provided, but we mainly use calculated
  // Removed top-level clarityScore and engagementScore as they should be in features
}

// Define props for VerbalVector
interface VerbalVectorProps {
  onAnalysisComplete: (result: AnalysisResult) => void;
}

const headerStyle: React.CSSProperties = {
    padding: '1.5rem 2rem',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderBottom: '1px solid #e2e8f0',
    position: 'sticky',
    top: 0,
    backgroundColor: 'white',
    zIndex: 10,
    width: '100%',
    boxSizing: 'border-box',
};

const logoContainerStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem', // space-x-2
};

const logoCircleStyle: React.CSSProperties = {
    width: '2rem', // w-8
    height: '2rem', // h-8
    borderRadius: '9999px', // rounded-full
    backgroundColor: '#6366f1', // bg-indigo-500
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    boxShadow: 'inset 0 1px 2px 0 rgba(0,0,0,0.05)', // shadow-inner approx
};

const logoTextStyle: React.CSSProperties = {
    color: '#fff', // text-white
    fontWeight: 600, // font-semibold
    fontSize: '1.125rem', // text-lg
};

const titleStyle: React.CSSProperties = {
    fontSize: '1.5rem', // text-2xl
    fontWeight: 300, // font-light
    letterSpacing: '0.025em', // tracking-wide
    color: '#334155', // Default text color from body
};

const titleSpanStyle: React.CSSProperties = {
    fontWeight: 500, // font-medium
};

const navStyle: React.CSSProperties = {
    display: 'flex',
    gap: '1.5rem',
    fontSize: '0.875rem',
    fontWeight: 500,
    color: '#475569', // text-slate-600 (visible on white)
};

const navLinkStyle: React.CSSProperties = {
     color: '#475569', // text-slate-600
     textDecoration: 'none',
     transition: 'color 0.2s ease-in-out',
};

const navLinkHoverStyle: React.CSSProperties = {
     color: '#6366f1', // hover:text-indigo-500
};

const mainStyle: React.CSSProperties = {
    flexGrow: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '2rem', // p-8
    maxWidth: '64rem', // max-w-5xl
    marginLeft: 'auto',
    marginRight: 'auto',
    width: '100%',
};

const footerStyle: React.CSSProperties = {
    padding: '1rem 2rem', // py-4 px-8
    borderTop: '1px solid #e2e8f0', // border-t border-slate-200
    textAlign: 'center',
    fontSize: '0.75rem', // text-xs
    color: '#94a3b8', // text-slate-400
    marginTop: '3rem', // mt-12
};

const VerbalVector: React.FC<VerbalVectorProps> = ({ onAnalysisComplete }) => {
  const [stage, setStage] = useState<'input' | 'recording' | 'processing'>('input');
  const [isRecording, setIsRecording] = useState(false);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  // Removed progress state
  const [waveformData, setWaveformData] = useState<number[]>([]);
  const [isLoadingApi, setIsLoadingApi] = useState(false); // Loading state for API call
  const [apiError, setApiError] = useState<string | null>(null); // Error state for API call

  // Refs for recording
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null); // To keep track of the stream for cleanup

  // Waveform visualization effect (remains the same)
  useEffect(() => {
    if (stage === 'recording') {
      const interval = setInterval(() => {
        setWaveformData(Array(40).fill(0).map(() => Math.random() * 100));
      }, 100);
      return () => clearInterval(interval);
    } else {
      setWaveformData([]);
    }
  }, [stage]);

  // API call effect for processing stage
  useEffect(() => {
    // Only run when stage becomes 'processing' and we have a file
    if (stage === 'processing' && audioFile) {
      const processFile = async () => {
        setIsLoadingApi(true);
        setApiError(null);
        console.log(`[API Call] Starting upload for: ${audioFile.name}`);

        const formData = new FormData();
        formData.append('file', audioFile);

        try {
          const response = await axios.post<AnalysisResult>('http://localhost:5002/api/upload', formData, {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
            onUploadProgress: (progressEvent) => {
              if (progressEvent.total) {
                // Handle progress if needed
              }
            },
            // Add a timeout? e.g., timeout: 180000 // 3 minutes
          });

          console.log('[API Call] Success, response data:', response.data);

          // Pass the raw response data; defaults/calculations happen in ResultsDisplay
          const rawResult: AnalysisResult = response.data;

          // We might still want a default for overallScore if API doesn't provide it AND
          // we decide to display the API's overall score alongside the calculated one.
          // For now, let's assume ResultsDisplay calculates it.
          // const resultWithDefaults: AnalysisResult = {
          //   ...rawResult,
          //   overallScore: rawResult.overallScore ?? 0, // Default API overall to 0 if missing
          // };

          onAnalysisComplete(rawResult); // Pass raw result up to App
          // App.tsx will change the stage to 'results'

        } catch (err: any) {
          console.error("[API Call] Error caught:", err);
          let errorMessage = 'An unknown error occurred during analysis.';
          if (axios.isAxiosError(err)) {
            if (err.response) {
              // The request was made and the server responded with a status code
              // that falls out of the range of 2xx
              console.error("[API Call] Error response data:", err.response.data);
              console.error("[API Call] Error response status:", err.response.status);
              errorMessage = err.response.data?.error || `Server error: ${err.response.status}`;
            } else if (err.request) {
              // The request was made but no response was received
              console.error("[API Call] No response received:", err.request);
              errorMessage = 'Could not connect to the analysis server. Is it running?';
            } else {
              // Something happened in setting up the request that triggered an Error
              console.error('[API Call] Error setting up request:', err.message);
              errorMessage = `Request setup error: ${err.message}`;
            }
          } else {
             errorMessage = `Error: ${err.message}`;
          }
          setApiError(errorMessage);
          setStage('input'); // Go back to input stage on error
        } finally {
          console.log('[API Call] Finally block.');
          setIsLoadingApi(false);
        }
      };

      processFile();

    } else if (stage === 'processing' && !audioFile) {
        // Handle case where processing started from recording without generating a file yet
        // This part needs the actual recording logic implemented
        console.warn("Entered processing stage without an audio file (likely from recording). Implement recording save first.");
        setApiError("Recording data not processed yet. Please implement recording save.");
        setStage('input'); // Go back for now
    }

  }, [stage, audioFile, onAnalysisComplete]);

  // Cleanup media stream on component unmount
  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach(track => track.stop());
      console.log("Media stream stopped on unmount");
    };
  }, []);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setApiError(null); // Clear previous errors
      setAudioFile(e.target.files[0]);
      setStage('processing');
    }
  };

  const startRecording = async () => {
    setApiError(null);
    setAudioFile(null);
    audioChunksRef.current = []; // Clear previous chunks

    try {
      // Stop any previous stream first
      streamRef.current?.getTracks().forEach(track => track.stop());

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream; // Store the stream
      console.log("Microphone access granted.");

      // Determine preferred MIME type
      const mimeTypes = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/wav', 'audio/mp4', 'audio/aac'];
      let preferredMimeType = '';
      for (const mimeType of mimeTypes) {
          if (MediaRecorder.isTypeSupported(mimeType)) {
              preferredMimeType = mimeType;
              break;
          }
      }
      if (!preferredMimeType) {
          console.error("No supported audio MIME type found for MediaRecorder");
          setApiError("Your browser doesn't support a suitable audio recording format.");
          return;
      }
      console.log("Using MIME type:", preferredMimeType);

      const recorder = new MediaRecorder(stream, { mimeType: preferredMimeType });
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        //   console.log("Audio chunk received, size:", event.data.size);
        }
      };

      recorder.onstop = () => {
        console.log("Recording stopped. Processing chunks.");
        const mimeType = mediaRecorderRef.current?.mimeType || preferredMimeType;
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
        const fileExtension = mimeType.split('/')[1].split(';')[0]; // e.g., 'webm', 'ogg', 'wav'
        const recordedFile = new File([audioBlob], `recording.${fileExtension}`, { type: mimeType });
        console.log("Created File object:", recordedFile.name, "Size:", recordedFile.size, "Type:", recordedFile.type);

        // Stop media stream tracks AFTER processing is done or handed off
        streamRef.current?.getTracks().forEach(track => track.stop());
        console.log("Media stream tracks stopped.");

        if (recordedFile.size === 0) {
             console.error("Recorded file is empty. Not proceeding.");
             setApiError("Recording failed to capture audio. Please check microphone permissions and try again.");
             setStage('input');
        } else {
            setAudioFile(recordedFile); // Set the state
            setStage('processing'); // Trigger the API call useEffect
        }
        audioChunksRef.current = []; // Clear chunks for next recording
      };

      recorder.start(1000); // Start recording, collect data every 1 second (optional)
      setIsRecording(true);
      setStage('recording');
      console.log("MediaRecorder started.");

    } catch (err) {
      console.error("Error accessing microphone or starting recorder:", err);
       if (err instanceof DOMException && (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError')) {
           setApiError("Microphone access denied. Please grant permission in your browser settings.");
       } else if (err instanceof DOMException && err.name === 'NotFoundError') {
           setApiError("No microphone found. Please ensure a microphone is connected and enabled.");
       } else {
            setApiError("Could not start recording. Please ensure microphone is connected and permissions are granted.");
       }
      setIsRecording(false);
      setStage('input');
       streamRef.current?.getTracks().forEach(track => track.stop()); // Clean up stream if error occurred
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      console.log("Stopping MediaRecorder...");
      mediaRecorderRef.current.stop(); // This will trigger the onstop event handler
      setIsRecording(false);
      // Stage change happens in the onstop handler now
    } else {
         console.warn("Stop recording called but no active recorder found.");
         setIsRecording(false);
         setStage('input'); // Fallback to input if something went wrong
          streamRef.current?.getTracks().forEach(track => track.stop()); // Ensure cleanup
    }
  };

  // Removed resetInterface, formatFeedback
  // Removed renderProcessingIndicators

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', width: '100%' }}>
      {/* Header */}
      <header style={headerStyle}>
        {/* Logo and Title (Left) */}
        <div style={logoContainerStyle}>
          <div style={logoCircleStyle}>
            <span style={logoTextStyle}>V</span>
          </div>
          <h1 style={titleStyle}>
            <span style={titleSpanStyle}>Verbal</span> Vector
          </h1>
        </div>
        {/* Navigation Links (Right) */}
        <nav style={navStyle}>
          <a 
            href="#"
            style={navLinkStyle}
            onMouseOver={(e) => e.currentTarget.style.color = navLinkHoverStyle.color || ''}
            onMouseOut={(e) => e.currentTarget.style.color = navLinkStyle.color || ''}
          >
            Dashboard
          </a>
          <a 
            href="#"
            style={navLinkStyle}
            onMouseOver={(e) => e.currentTarget.style.color = navLinkHoverStyle.color || ''}
            onMouseOut={(e) => e.currentTarget.style.color = navLinkStyle.color || ''}
           >
            History
          </a>
          <a 
            href="#"
            style={navLinkStyle}
            onMouseOver={(e) => e.currentTarget.style.color = navLinkHoverStyle.color || ''}
            onMouseOut={(e) => e.currentTarget.style.color = navLinkStyle.color || ''}
          >
            Settings
          </a>
        </nav>
      </header>

      {/* Main Content - Only Input, Recording, Processing Stages */}
      <main style={mainStyle}>
        {/* Display API Error */}
        {apiError && (
          <div style={{
              maxWidth: '42rem', /* max-w-xl */
              marginBottom: '1rem', /* mb-4 */
              padding: '1rem', /* p-4 */
              backgroundColor: '#fee2e2', /* bg-red-100 */
              border: '1px solid #fecaca', /* border-red-300 */
              color: '#b91c1c', /* text-red-700 */
              borderRadius: '0.5rem', /* rounded-lg */
              boxShadow: '0 1px 2px 0 rgba(0,0,0,0.03)', /* shadow-sm */
              display: 'flex',
              alignItems: 'center',
          }}>
              <AlertCircle size={20} style={{ marginRight: '0.5rem', flexShrink: 0 }}/>
              <p style={{ fontSize: '0.875rem', fontWeight: 500 }}>Error: <span style={{ fontWeight: 400 }}>{apiError}</span></p>
          </div>
        )}

        {stage === 'input' && (
          <div style={{ width: '100%', maxWidth: '28rem', display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center' }}>
            <h2 style={{ fontSize: '1.875rem', fontWeight: 300, marginBottom: '2rem' }}>Analyze your speech</h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '1.5rem', width: '100%', marginBottom: '3rem' }}>
              <button
                onClick={startRecording}
                disabled={isRecording || isLoadingApi}
                className="action-button" // Use class from App.css
              >
                <Mic /> {/* Size/color handled by CSS */}
                <span>Record Audio</span>
              </button>
              <label className={`file-input-label ${ (isRecording || isLoadingApi) ? 'disabled' : ''}`}>
                <Upload /> {/* Size/color handled by CSS */}
                <span>Upload File</span>
                <input type="file" accept="audio/*,video/*" onChange={handleFileUpload} className="hidden-input" disabled={isRecording || isLoadingApi}/>
              </label>
            </div>
            <p style={{ color: '#64748b', fontSize: '0.875rem', maxWidth: '20rem' }}>
              Record your speech or upload an audio/video file (.mp3, .wav, .mp4, etc.) to get instant feedback.
            </p>
          </div>
        )}

        {stage === 'recording' && (
          // Recording stage UI remains the same
          <div style={{ width: '100%', maxWidth: '42rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div className="card" style={{ width: '100%', marginBottom: '2rem' }}> {/* Use card class */} 
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                  <span style={{ fontSize: '1.125rem', fontWeight: 500, color: '#4f46e5' }}>Recording...</span>
                  <span style={{ backgroundColor: '#ef4444', padding: '0.125rem 0.5rem', borderRadius: '9999px', color: 'white', fontSize: '0.75rem', fontWeight: 600, animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite' }}>LIVE</span>
              </div>
              <div style={{ height: '6rem', display: 'flex', alignItems: 'flex-end', justifyContent: 'center', gap: '2px', overflow: 'hidden' }}>
                  {waveformData.map((height, i) => (
                  <div
                      key={i}
                      style={{
                          height: `${Math.max(2, height)}%`,
                          width: '6px', /* w-1.5 */
                          background: 'linear-gradient(to bottom, #818cf8, #6366f1)', /* bg-gradient-to-b from-indigo-400 to-indigo-500 */
                          borderRadius: '9999px', /* rounded-full */
                          flexShrink: 0,
                          transition: 'all 100ms ease-out',
                      }}
                  />
                  ))}
              </div>
            </div>
            <button
              onClick={stopRecording}
              style={{
                  backgroundColor: !isRecording ? '#a5b4fc' : '#6366f1', // Disabled color vs active
                  color: 'white',
                  fontWeight: 500,
                  padding: '0.75rem 2rem',
                  borderRadius: '0.5rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -2px rgba(0,0,0,0.1)',
                  transition: 'all 0.2s ease-in-out',
                  transform: 'scale(1)',
                  cursor: !isRecording ? 'not-allowed' : 'pointer', // Disabled cursor
                  opacity: !isRecording ? 0.5 : 1, // Disabled opacity
              }}
              disabled={!isRecording} // Keep the disabled attribute for accessibility
            >
              <Pause size={18} />
              <span>Stop Recording</span>
            </button>
          </div>
        )}

        {stage === 'processing' && (
          // Updated Processing stage UI (simplified, shows loading spinner)
          <div style={{ width: '100%', maxWidth: '42rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div style={{ width: '100%', marginBottom: '2rem' }}>
              <div className="card" style={{ padding: '2.5rem', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}> {/* Use card class */}
                 <Loader2 size={48} className="animate-spin" style={{ color: '#6366f1', marginBottom: '1.5rem' }} />
                 <h2 style={{ fontSize: '1.25rem', fontWeight: 300, textAlign: 'center', color: '#475569', marginBottom: '0.5rem' }}>Analyzing your speech...</h2>
                 <p style={{ fontSize: '0.875rem', color: '#64748b', textAlign: 'center' }}>
                    Processing {audioFile ? `"${audioFile.name}"` : 'Recording'}.
                    This may take a moment.
                 </p>
              </div>
            </div>
          </div>
        )}

      </main>

      {/* Footer */}
      <footer style={footerStyle}>
        <p>&copy; {new Date().getFullYear()} Verbal Vector – Advanced Speech Analytics</p>
      </footer>
    </div>
  );
};

export default VerbalVector; 