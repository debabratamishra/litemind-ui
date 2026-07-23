import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import type { AppSettings } from '@/lib/types';

/** Build a complete AppSettings object, optionally overriding individual fields. */
function makeSettings(overrides: Partial<AppSettings> = {}): AppSettings {
  return {
    backend: 'ollama',
    model: 'gemma3:1b',
    apiKey: null,
    apiBase: null,
    ollamaUrl: null,
    serpApiKey: null,
    providerKeys: { ollama: null, openrouter: null, nvidia_nim: null },
    sessionId: 'test-session',
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
    ...overrides,
  };
}

/** Shared mock state for the Zustand store. */
function makeMockState(settings: AppSettings) {
  return {
    conversations: [
      {
        id: 'conv-1',
        mode: 'rag' as const,
        title: 'Test RAG',
        messages: [],
        webSearch: false,
        updatedAt: new Date().toISOString(),
      },
    ],
    activeId: 'conv-1',
    settings,
    ragFiles: [],
    addMessage: vi.fn(),
    updateLastMessage: vi.fn(),
    setWebSearch: vi.fn(),
    clearConversation: vi.fn(),
    createConversation: vi.fn(),
    selectConversation: vi.fn(),
    deleteConversation: vi.fn(),
    setSettings: vi.fn(),
    setProviderKey: vi.fn(),
    setRagFiles: vi.fn(),
  };
}

// Mutable state that the mock reads from — reassigned in beforeEach.
let mockState: ReturnType<typeof makeMockState>;

// ── Mocks (hoisted by vitest before imports) ──────────────────────────────────

vi.mock('@/lib/store', () => {
  const useAppStore = (selector?: (state: typeof mockState) => unknown) => {
    if (selector) return selector(mockState);
    return mockState;
  };
  useAppStore.getState = () => mockState;
  return {
    useAppStore,
    selectActiveConversation: (state: typeof mockState) =>
      state.conversations.find((c) => c.id === state.activeId),
    selectActiveId: (state: typeof mockState) => state.activeId,
    selectSettings: (state: typeof mockState) => state.settings,
  };
});

vi.mock('@/hooks/use-rag-upload', () => ({
  useRagUpload: () => ({
    upload: vi.fn(),
    uploading: false,
    progress: 0,
    error: null,
  }),
  ACCEPTED_TYPES: '.pdf,.docx,.txt,.md,.csv,.xlsx,.pptx,.html,.htm,.odt,.rtf,.yaml,.json',
}));

vi.mock('@/lib/api', () => ({
  streamRagQuery: vi.fn().mockResolvedValue([]),
}));

// Import after mocks are set up
import RagPage from '@/app/rag/page';

describe('RagPage — provider override integration', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // jsdom lacks scrollIntoView — mock it so the page's useEffect doesn't crash.
    if (typeof HTMLElement !== 'undefined') {
      HTMLElement.prototype.scrollIntoView = vi.fn();
    }
    mockState = makeMockState(makeSettings());
  });

  it('renders the provider override badge when @alias is typed', () => {
    render(<RagPage />);

    const textarea = screen.getByLabelText('RAG query input');
    fireEvent.change(textarea, { target: { value: '@nim/meta/llama-3.3-70b-instruct hello' } });

    // Badge should appear with provider label
    expect(screen.getByText('NIM')).toBeInTheDocument();
    expect(screen.getByText((c) => c.startsWith('·') && c.includes('meta/llama-3.3-70b-instruct'))).toBeInTheDocument();
  });

  it('shows the key prompt when querying with a missing key', async () => {
    render(<RagPage />);

    const textarea = screen.getByLabelText('RAG query input');
    fireEvent.change(textarea, { target: { value: '@nim/meta/llama-3.3-70b-instruct hello' } });

    // Click the Ask button
    const askButton = screen.getByRole('button', { name: 'Submit query' });
    fireEvent.click(askButton);

    // Key prompt should appear
    await waitFor(() => {
      expect(screen.getByText('Provider: NIM')).toBeInTheDocument();
    });
    expect(screen.getByText(/API key required/)).toBeInTheDocument();
  });

  it('does not show the key prompt when the key is configured', async () => {
    mockState = makeMockState(
      makeSettings({
        providerKeys: { ollama: null, openrouter: null, nvidia_nim: 'nvapi-test-key' },
      }),
    );

    render(<RagPage />);

    const textarea = screen.getByLabelText('RAG query input');
    fireEvent.change(textarea, { target: { value: '@nim/meta/llama-3.3-70b-instruct hello' } });

    // Click the Ask button
    const askButton = screen.getByRole('button', { name: 'Submit query' });
    fireEvent.click(askButton);

    // Key prompt should NOT appear
    await waitFor(() => {
      expect(screen.queryByText('Provider: NIM')).not.toBeInTheDocument();
    });
  });

  it('shows the OpenRouter badge when @or alias is used', () => {
    render(<RagPage />);

    const textarea = screen.getByLabelText('RAG query input');
    fireEvent.change(textarea, { target: { value: '@or/openai/gpt-4o hello' } });

    expect(screen.getByText('OpenRouter')).toBeInTheDocument();
    expect(screen.getByText((c) => c.startsWith('·') && c.includes('openai/gpt-4o'))).toBeInTheDocument();
  });

  it('shows the Ollama badge when @ollama alias is used', () => {
    render(<RagPage />);

    const textarea = screen.getByLabelText('RAG query input');
    fireEvent.change(textarea, { target: { value: '@ollama/gemma3:4b hello' } });

    expect(screen.getByText('Ollama')).toBeInTheDocument();
    expect(screen.getByText((c) => c.startsWith('·') && c.includes('gemma3:4b'))).toBeInTheDocument();
  });

  it('removes the override when the X button is clicked', () => {
    render(<RagPage />);

    const textarea = screen.getByLabelText('RAG query input');
    fireEvent.change(textarea, { target: { value: '@nim/meta/llama-3.3-70b-instruct hello' } });

    // Badge should be visible
    expect(screen.getByText('NIM')).toBeInTheDocument();

    // Click the remove button
    const removeButton = screen.getByRole('button', { name: 'Remove provider override' });
    fireEvent.click(removeButton);

    // Badge should be gone
    expect(screen.queryByText('NIM')).not.toBeInTheDocument();
  });

  it('shows warning state on badge when key is missing', () => {
    render(<RagPage />);

    const textarea = screen.getByLabelText('RAG query input');
    fireEvent.change(textarea, { target: { value: '@nim/meta/llama-3.3-70b-instruct hello' } });

    // Badge should show warning icon
    expect(screen.getByText('⚠️')).toBeInTheDocument();
  });

  it('shows checkmark on badge when key is configured', () => {
    mockState = makeMockState(
      makeSettings({
        providerKeys: { ollama: null, openrouter: null, nvidia_nim: 'nvapi-test-key' },
      }),
    );

    render(<RagPage />);

    const textarea = screen.getByLabelText('RAG query input');
    fireEvent.change(textarea, { target: { value: '@nim/meta/llama-3.3-70b-instruct hello' } });

    // Badge should show checkmark
    expect(screen.getByText('✓')).toBeInTheDocument();
  });
});
