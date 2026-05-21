import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { LiveTranscript } from './LiveTranscript';

describe('LiveTranscript', () => {
  it('renders a placeholder when there is no transcript yet', () => {
    render(<LiveTranscript interim="" finals={[]} status="recording" />);
    expect(screen.getByText(/listening/i)).toBeInTheDocument();
  });

  it('renders accumulated final transcripts as one block', () => {
    render(
      <LiveTranscript
        interim=""
        finals={['Hello there.', 'How are you today?']}
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
        finals={['Hello.']}
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
});
