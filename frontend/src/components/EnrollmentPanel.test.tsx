import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, cleanup } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const mockGetEnrollment = vi.fn();
const mockEnrollVoice = vi.fn();
const mockDeleteEnrollment = vi.fn();
vi.mock('../api', () => ({
  getEnrollment: () => mockGetEnrollment(),
  enrollVoice: (b: Blob) => mockEnrollVoice(b),
  deleteEnrollment: () => mockDeleteEnrollment(),
}));

import { EnrollmentPanel } from './EnrollmentPanel';

class MockMediaRecorder {
  static isTypeSupported() { return true; }
  static instances: MockMediaRecorder[] = [];
  ondataavailable: ((e: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;
  mimeType = 'audio/webm';
  state = 'recording';
  constructor() { MockMediaRecorder.instances.push(this); }
  start() {}
  stop() {
    this.ondataavailable?.({ data: new Blob(['x'], { type: 'audio/webm' }) });
    this.onstop?.();
  }
}

const STATUS = { created_at: '2026-06-12T00:00:00Z', duration_seconds: 31, model: 'ecapa' };

describe('EnrollmentPanel', () => {
  beforeEach(() => {
    mockGetEnrollment.mockReset();
    mockEnrollVoice.mockReset();
    mockDeleteEnrollment.mockReset();
    MockMediaRecorder.instances = [];
    (global as unknown as { MediaRecorder: unknown }).MediaRecorder = MockMediaRecorder;
    Object.defineProperty(global.navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [{ stop: vi.fn() }] }) },
    });
  });
  afterEach(() => {
    cleanup();
    delete (global as unknown as Record<string, unknown>).MediaRecorder;
  });

  it('shows not-enrolled state when GET returns null', async () => {
    mockGetEnrollment.mockResolvedValue(null);
    render(<EnrollmentPanel />);
    await userEvent.click(screen.getByRole('button', { name: /enroll my voice/i }));
    await waitFor(() => expect(screen.getByText(/not enrolled/i)).toBeInTheDocument());
    expect(screen.getByRole('button', { name: /record enrollment/i })).toBeInTheDocument();
  });

  it('shows enrolled state with re-record and delete', async () => {
    mockGetEnrollment.mockResolvedValue(STATUS);
    render(<EnrollmentPanel />);
    await userEvent.click(screen.getByRole('button', { name: /enroll my voice/i }));
    await waitFor(() => expect(screen.getByText(/voice enrolled/i)).toBeInTheDocument());
    expect(screen.getByRole('button', { name: /re-record/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /delete/i })).toBeInTheDocument();
  });

  it('records and uploads an enrollment', async () => {
    mockGetEnrollment.mockResolvedValue(null);
    mockEnrollVoice.mockResolvedValue(STATUS);
    render(<EnrollmentPanel />);
    await userEvent.click(screen.getByRole('button', { name: /enroll my voice/i }));
    await waitFor(() => screen.getByRole('button', { name: /record enrollment/i }));
    await userEvent.click(screen.getByRole('button', { name: /record enrollment/i }));
    await waitFor(() => screen.getByRole('button', { name: /stop & save/i }));
    await userEvent.click(screen.getByRole('button', { name: /stop & save/i }));
    await waitFor(() => expect(mockEnrollVoice).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(screen.getByText(/voice enrolled/i)).toBeInTheDocument());
  });

  it('surfaces upload errors', async () => {
    mockGetEnrollment.mockResolvedValue(null);
    mockEnrollVoice.mockRejectedValue({ response: { data: { detail: 'Enrollment audio must be at least 10 seconds (got 3.0s).' } } });
    render(<EnrollmentPanel />);
    await userEvent.click(screen.getByRole('button', { name: /enroll my voice/i }));
    await waitFor(() => screen.getByRole('button', { name: /record enrollment/i }));
    await userEvent.click(screen.getByRole('button', { name: /record enrollment/i }));
    await waitFor(() => screen.getByRole('button', { name: /stop & save/i }));
    await userEvent.click(screen.getByRole('button', { name: /stop & save/i }));
    await waitFor(() => expect(screen.getByText(/at least 10 seconds/i)).toBeInTheDocument());
  });

  it('deletes the enrollment', async () => {
    mockGetEnrollment.mockResolvedValue(STATUS);
    mockDeleteEnrollment.mockResolvedValue(undefined);
    render(<EnrollmentPanel />);
    await userEvent.click(screen.getByRole('button', { name: /enroll my voice/i }));
    await waitFor(() => screen.getByRole('button', { name: /delete/i }));
    await userEvent.click(screen.getByRole('button', { name: /delete/i }));
    await waitFor(() => expect(mockDeleteEnrollment).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByText(/not enrolled/i)).toBeInTheDocument());
  });
});
