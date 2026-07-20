import type { ChatMessage, Model, RagFile, RagStatusResponse } from '@/lib/types';

/**
 * Frontend API client for the FastAPI backend.
 *
 * Streaming endpoints:
 *   - POST /api/chat/stream    → SSE (`data: {"chunk": "..."}` frames)
 *   - POST /api/chat/web-search → plain text
 *   - POST /api/rag/query      → plain text (with a leading citation frame)
 *
 * The base URL comes from NEXT_PUBLIC_API_URL (default http://localhost:8000),
 * matching the backend default in CLAUDE.md.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

// ─── Request shapes ──────────────────────────────────────────────────────────

export interface ChatStreamRequest {
  message: string;
  model?: string;
  backend?: string;
  api_key?: string | null;
  api_base?: string | null;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  frequency_penalty?: number;
  repetition_penalty?: number;
  top_k?: number;
  min_p?: number;
  seed?: number | null;
  stop?: string[] | null;
  session_id?: string;
  conversation_history?: ChatMessage[] | null;
  use_web_search?: boolean;
  serp_api_key?: string | null;
  is_voice_mode?: boolean;
  enable_generative_ui?: boolean;
  stream?: boolean;
}

export interface RAGQueryRequest {
  query: string;
  model?: string;
  session_id?: string;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  frequency_penalty?: number;
  repetition_penalty?: number;
  min_p?: number;
  seed?: number | null;
  stop?: string[] | null;
  /** Number of results to retrieve; mapped to the backend's `n_results`. */
  top_k?: number;
}

// ─── Low-level streaming helpers ───────────────────────────────────────────────

async function* streamSSE(response: Response): AsyncGenerator<string> {
  const reader = response.body?.getReader();
  if (!reader) throw new Error('Response body is not readable');
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep: number;
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const event = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);

      const dataLine = event
        .split('\n')
        .find((line) => line.startsWith('data:'));
      if (!dataLine) continue;

      const payload = dataLine.slice(5).trim();
      if (!payload) continue;

      let obj: { chunk?: string; error?: string };
      try {
        obj = JSON.parse(payload);
      } catch {
        // Ignore keep-alive / malformed frames.
        continue;
      }
      if (obj.error) throw new Error(obj.error);
      if (obj.chunk !== undefined) yield obj.chunk;
    }
  }
}

async function* streamText(response: Response): AsyncGenerator<string> {
  const reader = response.body?.getReader();
  if (!reader) throw new Error('Response body is not readable');
  const decoder = new TextDecoder();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    yield decoder.decode(value, { stream: true });
  }
}

async function postJSON(path: string, body: unknown, signal?: AbortSignal): Promise<Response> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
    cache: 'no-store',
  });
  return res;
}

// ─── Streaming chat / web search / RAG ──────────────────────────────────────────

/** Stream a chat completion. The backend returns SSE frames; yields text chunks. */
export async function* streamChat(
  request: ChatStreamRequest,
  signal?: AbortSignal,
): AsyncGenerator<string> {
  const res = await postJSON('/api/chat/stream', request, signal);
  if (!res.ok) throw new Error(await errorText(res));
  yield* streamSSE(res);
}

/** Stream a web-search chat completion. The backend returns plain text. */
export async function* streamWebSearch(
  request: ChatStreamRequest,
  signal?: AbortSignal,
): AsyncGenerator<string> {
  const res = await postJSON('/api/chat/web-search', request, signal);
  if (!res.ok) throw new Error(await errorText(res));
  yield* streamText(res);
}

/** Stream a RAG query. The backend returns plain text (with a citation frame). */
export async function* streamRagQuery(
  request: RAGQueryRequest,
  signal?: AbortSignal,
): AsyncGenerator<string> {
  const body = {
    query: request.query,
    model: request.model,
    session_id: request.session_id,
    temperature: request.temperature,
    max_tokens: request.max_tokens,
    top_p: request.top_p,
    frequency_penalty: request.frequency_penalty,
    repetition_penalty: request.repetition_penalty,
    min_p: request.min_p,
    seed: request.seed,
    stop: request.stop,
    n_results: request.top_k,
  };
  const res = await postJSON('/api/rag/query', body, signal);
  if (!res.ok) throw new Error(await errorText(res));
  yield* streamText(res);
}

// ─── REST endpoints ─────────────────────────────────────────────────────────────

export async function getRagFiles(): Promise<{ files: RagFile[] }> {
  const res = await fetch(`${API_BASE}/api/rag/files`, { cache: 'no-store' });
  if (!res.ok) throw new Error(await errorText(res));
  return (await res.json()) as { files: RagFile[] };
}

export async function getRagStatus(): Promise<RagStatusResponse> {
  const res = await fetch(`${API_BASE}/api/rag/status`, { cache: 'no-store' });
  if (!res.ok) throw new Error(await errorText(res));
  return (await res.json()) as RagStatusResponse;
}

export async function deleteRagFile(filename: string): Promise<void> {
  const res = await fetch(
    `${API_BASE}/api/rag/files/${encodeURIComponent(filename)}`,
    { method: 'DELETE', cache: 'no-store' },
  );
  if (!res.ok) throw new Error(await errorText(res));
}

export async function resetRag(): Promise<{ status: string; message: string; files_removed: number }> {
  const res = await fetch(`${API_BASE}/api/rag/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    cache: 'no-store',
  });
  if (!res.ok) throw new Error(await errorText(res));
  return (await res.json()) as { status: string; message: string; files_removed: number };
}

export async function getEnhancedModels(): Promise<{
  local_models: Model[];
  cloud_models: Model[];
}> {
  const res = await fetch(`${API_BASE}/models/enhanced`, { cache: 'no-store' });
  if (!res.ok) throw new Error(await errorText(res));
  return (await res.json()) as { local_models: Model[]; cloud_models: Model[] };
}

export async function checkSerpStatus(apiKey: string | null): Promise<{ status: string; message: string }> {
  const res = await postJSON('/api/chat/serp-status', { serp_api_key: apiKey ?? null });
  if (!res.ok) throw new Error(await errorText(res));
  return (await res.json()) as { status: string; message: string };
}

export async function checkRagDuplicate(body: { filename: string }): Promise<{ is_duplicate: boolean; reason?: string }> {
  const res = await postJSON('/api/rag/check-duplicates', body);
  if (!res.ok) throw new Error(await errorText(res));
  return (await res.json()) as { is_duplicate: boolean; reason?: string };
}

export async function uploadRagFile(file: File): Promise<unknown> {
  const form = new FormData();
  form.append('files', file);
  form.append('chunk_size', '500');

  const res = await fetch(`${API_BASE}/api/rag/upload`, {
    method: 'POST',
    body: form,
    cache: 'no-store',
  });
  if (!res.ok) throw new Error(await errorText(res));
  return (await res.json()) as unknown;
}

// ─── Error extraction ───────────────────────────────────────────────────────────

async function errorText(res: Response): Promise<string> {
  try {
    const data = await res.json();
    if (typeof data?.detail === 'string') return data.detail;
    if (typeof data?.error === 'string') return data.error;
    if (typeof data?.message === 'string') return data.message;
  } catch {
    /* fall through to status text */
  }
  return `${res.status} ${res.statusText}`;
}
