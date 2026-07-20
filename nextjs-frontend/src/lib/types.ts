/**
 * Shared frontend types.
 *
 * These describe the shapes the UI works with. Backend response models live in
 * `app/backend/models/api_models.py`; the field names here mirror those models
 * so the API layer can pass responses through with minimal mapping.
 */

export type BackendType = 'ollama' | 'openrouter' | 'nvidia_nim';

export type ConversationMode = 'chat' | 'rag';

/** A single message as sent to the backend (no client-only fields). */
export interface ChatMessage {
  role: string;
  content: string;
}

/** A message inside a conversation (client side), with an id and stream flag. */
export interface UIMessage {
  id?: string;
  role: 'user' | 'assistant' | string;
  content: string;
  isStreaming?: boolean;
}

/** A single model entry returned by the enhanced models endpoint. */
export interface Model {
  name: string;
  is_local: boolean;
  parameter_size?: string;
  quantization?: string;
  family?: string;
  description?: string;
}

/** A knowledge-base file as listed by the backend. */
export interface RagFile {
  filename: string;
  size?: number;
  indexed?: boolean;
}

/** Response from GET /api/rag/status. */
export interface RagStatusResponse {
  status: string;
  uploaded_files?: number;
  indexed_chunks?: number;
  bm25_corpus_size?: number;
  message?: string;
}

/** A user-configurable generation / connection setting. */
export interface AppSettings {
  backend: BackendType;
  model: string;
  apiKey: string | null;
  apiBase: string | null;
  ollamaUrl: string | null;
  serpApiKey: string | null;
  sessionId: string;
  temperature: number;
  maxTokens: number;
  topP: number;
  topK: number;
  minP: number;
  frequencyPenalty: number;
  repetitionPenalty: number;
  seed: number | null;
  stopSequences: string;
  voiceMode: boolean;
  enableGenerativeUI: boolean;
  genUIDisplayMode: 'rendered' | 'code';
}
