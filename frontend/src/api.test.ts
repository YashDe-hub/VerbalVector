import { describe, it, expect, vi, beforeEach } from 'vitest';

const { mockGet } = vi.hoisted(() => ({ mockGet: vi.fn() }));
vi.mock('axios', () => ({
  default: {
    create: () => ({ get: mockGet, post: vi.fn() }),
    isAxiosError: (e: unknown) => Boolean((e as { isAxiosError?: boolean })?.isAxiosError),
  },
}));

import { getSessionResult } from './api';

describe('getSessionResult', () => {
  beforeEach(() => mockGet.mockReset());

  it('maps 200 to {status:"ready", data}', async () => {
    const payload = { message: 'Live session complete.', transcript: { text: 'hi' }, features: {}, feedback: 'ok' };
    mockGet.mockResolvedValue({ status: 200, data: payload });
    await expect(getSessionResult('a'.repeat(32))).resolves.toEqual({ status: 'ready', data: payload });
  });

  it('maps 202 to {status:"pending"}', async () => {
    mockGet.mockResolvedValue({ status: 202, data: { status: 'pending' } });
    await expect(getSessionResult('b'.repeat(32))).resolves.toEqual({ status: 'pending' });
  });

  it('maps a transient (5xx) error to {status:"failed", permanent:false} with detail', async () => {
    mockGet.mockImplementationOnce(() =>
      Promise.reject({ isAxiosError: true, response: { status: 500, data: { detail: 'boom' } } }),
    );
    await expect(getSessionResult('c'.repeat(32))).resolves.toEqual({
      status: 'failed',
      permanent: false,
      detail: 'boom',
    });
  });

  it('maps a 400 to {status:"failed", permanent:true} (never recovers)', async () => {
    mockGet.mockImplementationOnce(() =>
      Promise.reject({ isAxiosError: true, response: { status: 400, data: { detail: 'Invalid session id.' } } }),
    );
    await expect(getSessionResult('d'.repeat(32))).resolves.toEqual({
      status: 'failed',
      permanent: true,
      detail: 'Invalid session id.',
    });
  });

  it('maps any 4xx (e.g. 403) to permanent:true', async () => {
    mockGet.mockImplementationOnce(() =>
      Promise.reject({ isAxiosError: true, response: { status: 403, data: { detail: 'Forbidden' } } }),
    );
    await expect(getSessionResult('f'.repeat(32))).resolves.toEqual({
      status: 'failed',
      permanent: true,
      detail: 'Forbidden',
    });
  });

  it('maps a non-axios network error to {status:"failed", permanent:false}', async () => {
    mockGet.mockImplementationOnce(() => Promise.reject(new Error('Network Error')));
    await expect(getSessionResult('e'.repeat(32))).resolves.toEqual({
      status: 'failed',
      permanent: false,
      detail: undefined,
    });
  });
});
