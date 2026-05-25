import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { LiveTranscript } from './LiveTranscript';

describe('LiveTranscript', () => {
  it('renders a placeholder when there is no transcript yet', () => {
    render(<LiveTranscript interim="" finals={[]} status="recording" />);
    expect(screen.getByText(/listening/i)).toBeInTheDocument();
  });

  it('renders accumulated final transcripts as one block when there is one speaker', () => {
    render(
      <LiveTranscript
        interim=""
        finals={[
          { text: 'Hello there.', speaker: 0 },
          { text: 'How are you today?', speaker: 0 },
        ]}
        status="recording"
      />,
    );
    expect(screen.getByText(/Hello there\./)).toBeInTheDocument();
    expect(screen.getByText(/How are you today\?/)).toBeInTheDocument();
  });

  it('renders interim text in a separate element with the lighter color', () => {
    render(
      <LiveTranscript
        interim="I am"
        finals={[{ text: 'Hello.', speaker: 0 }]}
        status="recording"
      />,
    );
    const interimEl = screen.getByText('I am');
    expect(interimEl).toHaveStyle({ color: '#94a3b8' });
  });

  it('does not show the placeholder once there is any transcript text', () => {
    render(
      <LiveTranscript
        interim="I am"
        finals={[]}
        status="recording"
      />,
    );
    expect(screen.queryByText(/listening/i)).not.toBeInTheDocument();
    expect(screen.getByText('I am')).toBeInTheDocument();
  });

  it('renders Speaker N: prefix when multiple speakers are detected', () => {
    render(
      <LiveTranscript
        interim=""
        finals={[
          { text: 'Hello.', speaker: 0 },
          { text: 'Hi.', speaker: 1 },
        ]}
        status="recording"
      />,
    );
    expect(screen.getByText(/Speaker 0/)).toBeInTheDocument();
    expect(screen.getByText(/Speaker 1/)).toBeInTheDocument();
    expect(screen.getByText(/Hello\./)).toBeInTheDocument();
    expect(screen.getByText(/Hi\./)).toBeInTheDocument();
  });

  it('does NOT render Speaker N: prefix when only one speaker is detected', () => {
    render(
      <LiveTranscript
        interim=""
        finals={[
          { text: 'Hello.', speaker: 0 },
          { text: 'How are you?', speaker: 0 },
        ]}
        status="recording"
      />,
    );
    expect(screen.queryByText(/Speaker 0/)).not.toBeInTheDocument();
    expect(screen.getByText(/Hello\./)).toBeInTheDocument();
  });

  it('does NOT render labels when all speakers are null', () => {
    render(
      <LiveTranscript
        interim=""
        finals={[
          { text: 'Hello.', speaker: null },
          { text: 'World.', speaker: null },
        ]}
        status="recording"
      />,
    );
    expect(screen.queryByText(/Speaker/)).not.toBeInTheDocument();
    expect(screen.getByText(/Hello\./)).toBeInTheDocument();
  });

  it('keeps interim in lighter color even in multi-speaker mode (no speaker tag on interim)', () => {
    render(
      <LiveTranscript
        interim="continuing..."
        finals={[
          { text: 'Hello.', speaker: 0 },
          { text: 'Hi.', speaker: 1 },
        ]}
        status="recording"
      />,
    );
    const interimEl = screen.getByText('continuing...');
    expect(interimEl).toHaveStyle({ color: '#94a3b8' });
  });

  it('renders null-speaker segments without a label when mixed with real speakers', () => {
    render(
      <LiveTranscript
        interim=""
        finals={[
          { text: 'Hello.', speaker: 0 },
          { text: 'untagged.', speaker: null },
          { text: 'Hi.', speaker: 1 },
        ]}
        status="recording"
      />,
    );
    expect(screen.getByText(/Speaker 0/)).toBeInTheDocument();
    expect(screen.getByText(/Speaker 1/)).toBeInTheDocument();
    // The null-speaker segment must NOT render "Speaker null:" literally
    expect(screen.queryByText(/Speaker null/)).not.toBeInTheDocument();
    expect(screen.getByText(/untagged\./)).toBeInTheDocument();
  });
});
