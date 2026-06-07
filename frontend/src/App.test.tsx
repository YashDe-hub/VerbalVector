import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, act, fireEvent } from '@testing-library/react';

const mockGetSessionResult = vi.fn();
vi.mock('./api', () => ({ getSessionResult: (id: string) => mockGetSessionResult(id) }));

vi.mock('./components/VerbalVector', () => ({
  default: ({ onLiveSessionEnding, onNavigate }: {
    onLiveSessionEnding?: (id: string) => void;
    onNavigate: (v: string) => void;
  }) => (
    <div>
      <button onClick={() => onLiveSessionEnding?.('sess-1')}>fire-ending</button>
      <button onClick={() => onNavigate('query')}>go-query</button>
    </div>
  ),
}));
vi.mock('./components/ResultsDisplay', () => ({
  default: () => <div>RESULTS_VIEW</div>,
}));
vi.mock('./components/QueryInterface', () => ({ default: () => <div>QUERY_VIEW</div> }));
vi.mock('./components/SessionHistory', () => ({ default: () => <div>HISTORY_VIEW</div> }));

import App from './App';

const READY = { message: 'Live session complete.', transcript: { text: 'hi' }, features: {}, feedback: 'ok' };

// NOTE (test-mechanics adaptation): this repo's vitest/jsdom + fake-timers
// combo deadlocks with both `userEvent` and `@testing-library`'s `waitFor`
// (waitFor schedules a real-timer interval that never settles while fake timers
// are installed). So we drive interactions with synchronous `fireEvent` inside
// `act`, advance the poll with `vi.advanceTimersByTimeAsync` inside `act`, and
// assert synchronously afterward (the DOM is already flushed by then). The
// behavioral assertions are unchanged: results shown / ready banner shown /
// timeout error banner shown.
const click = (el: HTMLElement) => act(() => { fireEvent.click(el); });
const tick = (ms: number) => act(async () => { await vi.advanceTimersByTimeAsync(ms); });

describe('App live poll fallback', () => {
  beforeEach(() => { mockGetSessionResult.mockReset(); vi.useFakeTimers(); });
  afterEach(() => { vi.runOnlyPendingTimers(); vi.useRealTimers(); });

  it('auto-shows results when ready while still on the analysis view', async () => {
    mockGetSessionResult.mockResolvedValue({ status: 'ready', data: READY });
    render(<App />);
    await click(screen.getByText('fire-ending'));
    await tick(2000);
    expect(screen.getByText('RESULTS_VIEW')).toBeInTheDocument();
  });

  it('shows a "ready" banner (no forced nav) when the user navigated away', async () => {
    mockGetSessionResult.mockResolvedValue({ status: 'ready', data: READY });
    render(<App />);
    await click(screen.getByText('fire-ending'));
    await click(screen.getByText('go-query'));
    await tick(2000);
    expect(screen.getByText(/analysis is ready/i)).toBeInTheDocument();
    expect(screen.getByText('QUERY_VIEW')).toBeInTheDocument();
    await click(screen.getByRole('button', { name: /view results/i }));
    expect(screen.getByText('RESULTS_VIEW')).toBeInTheDocument();
  });

  it('keeps polling through pending then shows results', async () => {
    mockGetSessionResult
      .mockResolvedValueOnce({ status: 'pending' })
      .mockResolvedValueOnce({ status: 'ready', data: READY });
    render(<App />);
    await click(screen.getByText('fire-ending'));
    await tick(2000);
    await tick(2000);
    expect(screen.getByText('RESULTS_VIEW')).toBeInTheDocument();
  });

  it('shows a timeout error banner after the attempt cap', async () => {
    mockGetSessionResult.mockResolvedValue({ status: 'pending' });
    render(<App />);
    await click(screen.getByText('fire-ending'));
    await tick(45 * 2000);
    expect(screen.getByText(/taking longer than expected/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /check history/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
  });
});
