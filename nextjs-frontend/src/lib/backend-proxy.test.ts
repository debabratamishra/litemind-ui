import { describe, it, expect, vi, beforeEach } from 'vitest';

const fetchMock = vi.fn();
vi.stubGlobal('fetch', fetchMock);

import { proxyStreamingPost } from '@/lib/backend-proxy';

function fakeRequest(body: string, contentType = 'application/json') {
  return {
    text: async () => body,
    headers: { get: (h: string) => (h.toLowerCase() === 'content-type' ? contentType : null) },
  } as unknown as import('next/server').NextRequest;
}

function upstreamResponse(contentType: string, status = 200): Response {
  const enc = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(enc.encode('upstream-body'));
      controller.close();
    },
  });
  return new Response(stream, { status, headers: { 'content-type': contentType } });
}

beforeEach(() => {
  fetchMock.mockReset();
});

describe('proxyStreamingPost', () => {
  it('forwards the request and returns the upstream response', async () => {
    fetchMock.mockResolvedValue(upstreamResponse('text/event-stream'));
    const resp = await proxyStreamingPost(fakeRequest('{"message":"hi"}'), '/api/chat');
    expect(resp.status).toBe(200);
    expect(resp.headers.get('content-type')).toBe('text/event-stream');

    const [target, init] = fetchMock.mock.calls[0];
    expect(target).toContain('/api/chat');
    expect(init.method).toBe('POST');
    expect(init.body).toBe('{"message":"hi"}');
    expect(init.headers['Content-Type']).toBe('application/json');
  });

  it('passes the content-type from the incoming request', async () => {
    fetchMock.mockResolvedValue(upstreamResponse('application/json'));
    await proxyStreamingPost(fakeRequest('x', 'text/plain'), '/api/rag/query');
    const [, init] = fetchMock.mock.calls[0];
    expect(init.headers['Content-Type']).toBe('text/plain');
  });

  it('uses defaultContentType from opts when upstream has none', async () => {
    const enc = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(enc.encode('body'));
        controller.close();
      },
    });
    fetchMock.mockResolvedValue(new Response(stream, { status: 200 }));
    const resp = await proxyStreamingPost(fakeRequest('{}'), '/api/chat', {
      defaultContentType: 'application/json',
    });
    expect(resp.headers.get('content-type')).toBe('application/json');
  });
});
