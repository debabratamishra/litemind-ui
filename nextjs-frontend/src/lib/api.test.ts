import { describe, it, expect, vi, beforeEach } from 'vitest';

const fetchMock = vi.fn();
vi.stubGlobal('fetch', fetchMock);

import * as api from '@/lib/api';

function makeSSEResponse(chunks: string[]): Response {
  const enc = new TextEncoder();
  let i = 0;
  const stream = new ReadableStream({
    pull(controller) {
      if (i < chunks.length) {
        controller.enqueue(enc.encode(`data: ${JSON.stringify({ chunk: chunks[i] })}\n\n`));
        i += 1;
      } else {
        controller.close();
      }
    },
  });
  return new Response(stream, { status: 200 });
}

function makeTextResponse(text: string): Response {
  const enc = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(enc.encode(text));
      controller.close();
    },
  });
  return new Response(stream, { status: 200 });
}

function makeJsonResponse(body: unknown, ok = true, status = 200): Response {
  return {
    ok,
    status,
    statusText: ok ? 'OK' : 'Error',
    json: async () => body,
  } as unknown as Response;
}

beforeEach(() => {
  fetchMock.mockReset();
});

describe('api client — REST endpoints', () => {
  it('getRagFiles returns parsed json', async () => {
    fetchMock.mockResolvedValue(makeJsonResponse({ files: [{ filename: 'a.txt' }] }));
    const res = await api.getRagFiles();
    expect(res.files).toHaveLength(1);
    expect(res.files[0].filename).toBe('a.txt');
    expect(fetchMock.mock.calls[0][0]).toContain('/api/rag/files');
  });

  it('getRagStatus returns parsed json on success', async () => {
    fetchMock.mockResolvedValue(makeJsonResponse({ status: 'ready', files: 0 }));
    const res = await api.getRagStatus();
    expect(res.status).toBe('ready');
  });

  it('getRagStatus maps a detail error message', async () => {
    fetchMock.mockResolvedValue(makeJsonResponse({ detail: 'boom' }, false, 500));
    await expect(api.getRagStatus()).rejects.toThrow('boom');
  });

  it('getRagStatus maps a message error to the message text', async () => {
    fetchMock.mockResolvedValue(makeJsonResponse({ message: 'nope' }, false, 503));
    await expect(api.getRagStatus()).rejects.toThrow('nope');
  });

  it('deleteRagFile issues a DELETE and resolves', async () => {
    fetchMock.mockResolvedValue(makeJsonResponse({ status: 'deleted' }));
    await api.deleteRagFile('a.txt');
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toContain('/api/rag/files/a.txt');
    expect(init.method).toBe('DELETE');
  });

  it('resetRag posts and returns json', async () => {
    fetchMock.mockResolvedValue(makeJsonResponse({ status: 'ok', files_removed: 3 }));
    const res = await api.resetRag();
    expect(res.files_removed).toBe(3);
    expect(fetchMock.mock.calls[0][1].method).toBe('POST');
  });

  it('getEnhancedModels returns local and cloud models', async () => {
    fetchMock.mockResolvedValue(
      makeJsonResponse({ local_models: [{ name: 'x' }], cloud_models: [{ name: 'y' }] }),
    );
    const res = await api.getEnhancedModels();
    expect(res.local_models).toHaveLength(1);
    expect(res.cloud_models).toHaveLength(1);
  });

  it('checkSerpStatus posts the key and returns json', async () => {
    fetchMock.mockResolvedValue(makeJsonResponse({ status: 'ok', message: 'good' }));
    const res = await api.checkSerpStatus('key123');
    expect(res.status).toBe('ok');
    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(init.body)).toMatchObject({ serp_api_key: 'key123' });
  });

  it('checkRagDuplicate posts the filename and returns json', async () => {
    fetchMock.mockResolvedValue(makeJsonResponse({ is_duplicate: true, reason: 'exists' }));
    const res = await api.checkRagDuplicate({ filename: 'a.txt' });
    expect(res.is_duplicate).toBe(true);
  });

  it('uploadRagFile sends FormData', async () => {
    fetchMock.mockResolvedValue(makeJsonResponse({ status: 'uploaded' }));
    await api.uploadRagFile(new File(['x'], 'a.txt'));
    const [, init] = fetchMock.mock.calls[0];
    expect(init.body).toBeInstanceOf(FormData);
  });
});

describe('api client — streaming', () => {
  it('streamChat yields SSE chunks and surfaces errors', async () => {
    fetchMock.mockResolvedValue(makeSSEResponse(['hello', ' world']));
    const chunks: string[] = [];
    for await (const c of api.streamChat({ message: 'hi' } as api.ChatStreamRequest)) {
      chunks.push(c);
    }
    expect(chunks.join('')).toBe('hello world');
  });

  it('streamChat throws on a backend error frame', async () => {
    const enc = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(enc.encode('data: {"error":"nope"}\n\n'));
        controller.close();
      },
    });
    fetchMock.mockResolvedValue(new Response(stream, { status: 200 }));
    await expect(async () => {
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      for await (const _ of api.streamChat({ message: 'hi' } as api.ChatStreamRequest)) {
        /* drain */
      }
    }).rejects.toThrow('nope');
  });

  it('streamWebSearch yields plain text chunks', async () => {
    fetchMock.mockResolvedValue(makeTextResponse('search result'));
    const chunks: string[] = [];
    for await (const c of api.streamWebSearch({ message: 'hi' } as api.ChatStreamRequest)) {
      chunks.push(c);
    }
    expect(chunks.join('')).toBe('search result');
  });

  it('streamRagQuery yields plain text and maps top_k to n_results', async () => {
    fetchMock.mockResolvedValue(makeTextResponse('rag answer'));
    const chunks: string[] = [];
    for await (const c of api.streamRagQuery({ query: 'q', top_k: 5 } as api.RAGQueryRequest)) {
      chunks.push(c);
    }
    expect(chunks.join('')).toBe('rag answer');
    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(init.body).n_results).toBe(5);
  });
});
