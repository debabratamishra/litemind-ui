'use client';

import { create } from 'zustand';
import type { AppSettings, ConversationMode, RagFile } from '@/lib/types';

/**
 * Global client store (Zustand). Holds conversations, the active conversation
 * id, user settings, and the cached knowledge-base file list.
 */

export interface Message {
  id: string;
  role: 'user' | 'assistant' | string;
  content: string;
  isStreaming?: boolean;
}

export interface Conversation {
  id: string;
  mode: ConversationMode;
  title: string;
  messages: Message[];
  webSearch: boolean;
  updatedAt: string;
}

interface AppState {
  conversations: Conversation[];
  activeId: string | null;
  settings: AppSettings;
  ragFiles: RagFile[];

  addMessage: (
    convId: string,
    message: { role: string; content: string; isStreaming?: boolean },
  ) => void;
  updateLastMessage: (convId: string, content: string, isStreaming: boolean) => void;
  setWebSearch: (convId: string, value: boolean) => void;
  clearConversation: (convId: string) => void;
  createConversation: (mode: ConversationMode) => string;
  selectConversation: (id: string) => void;
  deleteConversation: (id: string) => void;
  setSettings: (partial: Partial<AppSettings>) => void;
  setRagFiles: (files: RagFile[]) => void;
}

function uid(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function newSessionId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `session-${Date.now().toString(36)}`;
}

const DEFAULT_SETTINGS: AppSettings = {
  backend: 'ollama',
  model: '',
  apiKey: null,
  apiBase: null,
  ollamaUrl: null,
  serpApiKey: null,
  sessionId: newSessionId(),
  temperature: 0.7,
  maxTokens: 2048,
  topP: 0.9,
  topK: 40,
  minP: 0.0,
  frequencyPenalty: 0,
  repetitionPenalty: 1.0,
  seed: null,
  stopSequences: '',
  voiceMode: false,
  enableGenerativeUI: false,
  genUIDisplayMode: 'rendered',
};

function patchConversation(
  state: AppState,
  convId: string,
  updater: (conv: Conversation) => Conversation,
): Partial<AppState> {
  return {
    conversations: state.conversations.map((c) =>
      c.id === convId ? { ...updater(c), updatedAt: new Date().toISOString() } : c,
    ),
  };
}

export const useAppStore = create<AppState>((set) => ({
  conversations: [],
  activeId: null,
  settings: DEFAULT_SETTINGS,
  ragFiles: [],

  addMessage: (convId, message) =>
    set((state) =>
      patchConversation(state, convId, (conv) => ({
        ...conv,
        messages: [
          ...conv.messages,
          { id: uid(), role: message.role, content: message.content, isStreaming: message.isStreaming },
        ],
      })),
    ),

  updateLastMessage: (convId, content, isStreaming) =>
    set((state) =>
      patchConversation(state, convId, (conv) => {
        if (conv.messages.length === 0) return conv;
        const messages = conv.messages.slice();
        const last = messages[messages.length - 1];
        messages[messages.length - 1] = { ...last, content, isStreaming };
        return { ...conv, messages };
      }),
    ),

  setWebSearch: (convId, value) =>
    set((state) => patchConversation(state, convId, (conv) => ({ ...conv, webSearch: value }))),

  clearConversation: (convId) =>
    set((state) => patchConversation(state, convId, (conv) => ({ ...conv, messages: [] }))),

  createConversation: (mode) => {
    const id = uid();
    const conv: Conversation = {
      id,
      mode,
      title: mode === 'rag' ? 'New Knowledge Base' : 'New Chat',
      messages: [],
      webSearch: false,
      updatedAt: new Date().toISOString(),
    };
    set((state) => ({
      conversations: [conv, ...state.conversations],
      activeId: id,
    }));
    return id;
  },

  selectConversation: (id) => set({ activeId: id }),

  deleteConversation: (id) =>
    set((state) => {
      const conversations = state.conversations.filter((c) => c.id !== id);
      const activeId = state.activeId === id ? (conversations[0]?.id ?? null) : state.activeId;
      return { conversations, activeId };
    }),

  setSettings: (partial) => set((state) => ({ settings: { ...state.settings, ...partial } })),

  setRagFiles: (files) => set({ ragFiles: files }),
}));

// ─── Selectors ─────────────────────────────────────────────────────────────────

export const selectActiveConversation = (state: AppState): Conversation | undefined =>
  state.conversations.find((c) => c.id === state.activeId);

export const selectActiveId = (state: AppState): string | null => state.activeId;

export const selectSettings = (state: AppState): AppSettings => state.settings;
