import { describe, it, expect, vi, beforeEach } from 'vitest';

const { mockGet } = vi.hoisted(() => ({ mockGet: vi.fn() }));
vi.mock('axios', () => ({
  default: { create: () => ({ get: mockGet, post: vi.fn() }) },
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

  it('throws on transport error (treated as a failed attempt by caller)', async () => {
    mockGet.mockImplementationOnce(() => Promise.reject(new Error('Network Error')));
    await expect(getSessionResult('c'.repeat(32))).rejects.toThrow('Network Error');
  });
});
