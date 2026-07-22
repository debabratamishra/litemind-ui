import { describe, it, expect, beforeEach } from 'vitest';
import {
  useAppStore,
  selectActiveConversation,
  selectActiveId,
  selectSettings,
} from '@/lib/store';

beforeEach(() => {
  useAppStore.setState({ conversations: [], activeId: null, ragFiles: [] });
});

describe('app store — conversations', () => {
  it('createConversation prepends and sets activeId', () => {
    const id = useAppStore.getState().createConversation('chat');
    const state = useAppStore.getState();
    expect(state.conversations).toHaveLength(1);
    expect(state.conversations[0].id).toBe(id);
    expect(state.activeId).toBe(id);
    expect(state.conversations[0].title).toBe('New Chat');
  });

  it('createConversation titles a rag conversation differently', () => {
    const id = useAppStore.getState().createConversation('rag');
    expect(useAppStore.getState().conversations[0].title).toBe('New Knowledge Base');
    expect(id).toBeDefined();
  });

  it('addMessage appends a message to the named conversation', () => {
    const id = useAppStore.getState().createConversation('chat');
    useAppStore.getState().addMessage(id, { role: 'user', content: 'hi' });
    const conv = useAppStore.getState().conversations[0];
    expect(conv.messages).toHaveLength(1);
    expect(conv.messages[0].content).toBe('hi');
    expect(conv.messages[0].role).toBe('user');
  });

  it('updateLastMessage overwrites the final message', () => {
    const id = useAppStore.getState().createConversation('chat');
    useAppStore.getState().addMessage(id, { role: 'assistant', content: 'partial' });
    useAppStore.getState().updateLastMessage(id, 'complete', false);
    const conv = useAppStore.getState().conversations[0];
    expect(conv.messages[0].content).toBe('complete');
    expect(conv.messages[0].isStreaming).toBe(false);
  });

  it('updateLastMessage is a no-op on an empty conversation', () => {
    const id = useAppStore.getState().createConversation('chat');
    useAppStore.getState().updateLastMessage(id, 'x', false);
    expect(useAppStore.getState().conversations[0].messages).toHaveLength(0);
  });

  it('setWebSearch toggles the flag', () => {
    const id = useAppStore.getState().createConversation('chat');
    useAppStore.getState().setWebSearch(id, true);
    expect(useAppStore.getState().conversations[0].webSearch).toBe(true);
  });

  it('clearConversation empties messages', () => {
    const id = useAppStore.getState().createConversation('chat');
    useAppStore.getState().addMessage(id, { role: 'user', content: 'hi' });
    useAppStore.getState().clearConversation(id);
    expect(useAppStore.getState().conversations[0].messages).toHaveLength(0);
  });

  it('selectConversation sets activeId', () => {
    const a = useAppStore.getState().createConversation('chat');
    const b = useAppStore.getState().createConversation('chat');
    useAppStore.getState().selectConversation(a);
    expect(useAppStore.getState().activeId).toBe(a);
    useAppStore.getState().selectConversation(b);
    expect(useAppStore.getState().activeId).toBe(b);
  });

  it('deleteConversation removes it and repairs activeId', () => {
    const a = useAppStore.getState().createConversation('chat');
    const b = useAppStore.getState().createConversation('chat');
    useAppStore.getState().deleteConversation(a);
    const state = useAppStore.getState();
    expect(state.conversations.find((c) => c.id === a)).toBeUndefined();
    expect(state.activeId).toBe(b);
  });
});

describe('app store — settings and files', () => {
  it('setSettings merges partially', () => {
    useAppStore.getState().setSettings({ model: 'gemma3:1b', temperature: 0.3 });
    const s = useAppStore.getState().settings;
    expect(s.model).toBe('gemma3:1b');
    expect(s.temperature).toBe(0.3);
    expect(s.backend).toBe('ollama');
  });

  it('setRagFiles stores the list', () => {
    useAppStore.getState().setRagFiles([{ filename: 'a.txt' } as never]);
    expect(useAppStore.getState().ragFiles).toHaveLength(1);
  });
});

describe('app store — providerKeys', () => {
  it('default settings initialise providerKeys with all null values', () => {
    const { providerKeys } = useAppStore.getState().settings;
    expect(providerKeys).toEqual({ ollama: null, openrouter: null, nvidia_nim: null });
  });

  it('setProviderKey updates a single provider key', () => {
    useAppStore.getState().setProviderKey('openrouter', 'sk-test-123');
    const { providerKeys } = useAppStore.getState().settings;
    expect(providerKeys.openrouter).toBe('sk-test-123');
    expect(providerKeys.ollama).toBeNull();
    expect(providerKeys.nvidia_nim).toBeNull();
  });

  it('setProviderKey can set null to clear a key', () => {
    useAppStore.getState().setProviderKey('nvidia_nim', 'nv-key');
    useAppStore.getState().setProviderKey('nvidia_nim', null);
    expect(useAppStore.getState().settings.providerKeys.nvidia_nim).toBeNull();
  });

  it('setProviderKey preserves other provider keys', () => {
    useAppStore.getState().setProviderKey('openrouter', 'sk-a');
    useAppStore.getState().setProviderKey('nvidia_nim', 'nv-b');
    const { providerKeys } = useAppStore.getState().settings;
    expect(providerKeys.openrouter).toBe('sk-a');
    expect(providerKeys.nvidia_nim).toBe('nv-b');
    expect(providerKeys.ollama).toBeNull();
  });
});

describe('app store — selectors', () => {
  it('selectActiveId and selectSettings read state', () => {
    const id = useAppStore.getState().createConversation('chat');
    const s = useAppStore.getState();
    expect(selectActiveId(s)).toBe(id);
    expect(selectSettings(s).backend).toBe('ollama');
  });

  it('selectActiveConversation finds the active one', () => {
    const id = useAppStore.getState().createConversation('chat');
    const conv = selectActiveConversation(useAppStore.getState());
    expect(conv?.id).toBe(id);
  });
});
