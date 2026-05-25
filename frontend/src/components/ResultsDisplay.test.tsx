import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import ResultsDisplay from './ResultsDisplay';

const baseFeatures = {
  speech_clarity: 0.8,
  vocal_expressiveness: 0.7,
  words_per_minute: 150,
  unique_word_count: 200,
  total_filler_words: 5,
  mean_sentence_length: 12,
};

describe('ResultsDisplay transcript rendering', () => {
  it('renders speaker-labeled utterances when transcript has 2+ speakers', () => {
    const analysisResult = {
      message: 'Live session complete.',
      transcript: {
        text: 'Hello. Hi there.',
        utterances: [
          { speaker: 0, text: 'Hello.', start: 0.0, end: 0.5, confidence: 0.98 },
          { speaker: 1, text: 'Hi there.', start: 0.6, end: 1.2, confidence: 0.97 },
        ],
        speakers: [0, 1],
      },
      features: baseFeatures,
      feedback: 'Great job.',
    };
    render(
      <ResultsDisplay
        analysisResult={analysisResult}
        onAnalyzeAnother={() => {}}
        onNavigate={() => {}}
      />,
    );
    expect(screen.getByText(/Speaker 0/)).toBeInTheDocument();
    expect(screen.getByText(/Speaker 1/)).toBeInTheDocument();
    expect(screen.getByText(/Hello\./)).toBeInTheDocument();
    expect(screen.getByText(/Hi there\./)).toBeInTheDocument();
  });

  it('renders flat transcript when only one speaker', () => {
    const analysisResult = {
      message: 'Upload complete.',
      transcript: {
        text: 'Hello. How are you?',
        utterances: [
          { speaker: 0, text: 'Hello.', start: 0.0, end: 0.5, confidence: 0.98 },
          { speaker: 0, text: 'How are you?', start: 0.6, end: 1.5, confidence: 0.97 },
        ],
        speakers: [0],
      },
      features: baseFeatures,
      feedback: 'Good.',
    };
    render(
      <ResultsDisplay
        analysisResult={analysisResult}
        onAnalyzeAnother={() => {}}
        onNavigate={() => {}}
      />,
    );
    expect(screen.queryByText(/Speaker 0:/)).not.toBeInTheDocument();
    expect(screen.getByText('Hello. How are you?')).toBeInTheDocument();
  });

  it('renders flat transcript when utterances field is absent (legacy shape)', () => {
    const analysisResult = {
      message: 'Upload complete.',
      transcript: {
        text: 'Some flat transcript.',
      },
      features: baseFeatures,
      feedback: 'OK.',
    };
    render(
      <ResultsDisplay
        analysisResult={analysisResult}
        onAnalyzeAnother={() => {}}
        onNavigate={() => {}}
      />,
    );
    expect(screen.getByText('Some flat transcript.')).toBeInTheDocument();
    expect(screen.queryByText(/Speaker/)).not.toBeInTheDocument();
  });

  it('renders the existing flat text when transcript is a string (very-old shape)', () => {
    const analysisResult = {
      message: 'Upload complete.',
      transcript: 'String transcript fallback.',
      features: baseFeatures,
      feedback: 'OK.',
    };
    render(
      <ResultsDisplay
        analysisResult={analysisResult}
        onAnalyzeAnother={() => {}}
        onNavigate={() => {}}
      />,
    );
    expect(screen.getByText('String transcript fallback.')).toBeInTheDocument();
  });

  // Defensive: a malformed payload with utterances=[] but speakers=[0,1]
  // should still render via the flat-text fallback (never blow up).
  it('falls back to flat text when utterances is empty even if speakers reports multiple', () => {
    const analysisResult = {
      message: 'Upload complete.',
      transcript: {
        text: 'Some flat transcript.',
        utterances: [],
        speakers: [0, 1],
      },
      features: baseFeatures,
      feedback: 'OK.',
    };
    render(
      <ResultsDisplay
        analysisResult={analysisResult}
        onAnalyzeAnother={() => {}}
        onNavigate={() => {}}
      />,
    );
    expect(screen.getByText('Some flat transcript.')).toBeInTheDocument();
    expect(screen.queryByText(/Speaker/)).not.toBeInTheDocument();
  });

  it('renders null-speaker utterances without a label when mixed with real speakers', () => {
    const analysisResult = {
      message: 'Upload complete.',
      transcript: {
        text: 'Hello. untagged. Hi.',
        utterances: [
          { speaker: 0, text: 'Hello.', start: 0.0, end: 0.5, confidence: 0.98 },
          { speaker: null, text: 'untagged.', start: 0.6, end: 1.0, confidence: 0.95 },
          { speaker: 1, text: 'Hi.', start: 1.1, end: 1.5, confidence: 0.97 },
        ],
        speakers: [0, 1],
      },
      features: baseFeatures,
      feedback: 'OK.',
    };
    render(
      <ResultsDisplay
        analysisResult={analysisResult}
        onAnalyzeAnother={() => {}}
        onNavigate={() => {}}
      />,
    );
    expect(screen.getByText(/Speaker 0/)).toBeInTheDocument();
    expect(screen.getByText(/Speaker 1/)).toBeInTheDocument();
    // The null-speaker utterance must NOT render "Speaker null:" literally
    expect(screen.queryByText(/Speaker null/)).not.toBeInTheDocument();
    expect(screen.getByText(/untagged\./)).toBeInTheDocument();
  });
});
