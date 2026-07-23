import { describe, it, expect } from 'vitest';
import type { AppSettings } from '@/lib/types';
import { parseProviderOverride } from './use-provider-override';

/** Build a complete AppSettings object, optionally overriding individual fields. */
function makeSettings(overrides: Partial<AppSettings> = {}): AppSettings {
  return {
    backend: 'ollama',
    model: '',
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

describe('parseProviderOverride — alias mapping', () => {
  it('maps @nim to nvidia_nim', () => {
    const result = parseProviderOverride('@nim/gemma "hello"', makeSettings());
    expect(result).not.toBeNull();
    expect(result!.backend).toBe('nvidia_nim');
    expect(result!.alias).toBe('nim');
  });

  it('maps @openrouter to openrouter', () => {
    const result = parseProviderOverride('@openrouter/gemma "hello"', makeSettings());
    expect(result).not.toBeNull();
    expect(result!.backend).toBe('openrouter');
    expect(result!.alias).toBe('openrouter');
  });

  it('maps @or (shorthand) to openrouter', () => {
    const result = parseProviderOverride('@or/gemma "hello"', makeSettings());
    expect(result).not.toBeNull();
    expect(result!.backend).toBe('openrouter');
    expect(result!.alias).toBe('or');
  });

  it('maps @ollama to ollama', () => {
    const result = parseProviderOverride('@ollama/gemma "hello"', makeSettings());
    expect(result).not.toBeNull();
    expect(result!.backend).toBe('ollama');
    expect(result!.alias).toBe('ollama');
  });
});

describe('parseProviderOverride — model and text extraction', () => {
  it('extracts model and double-quoted text', () => {
    const result = parseProviderOverride('@nim/gemma-3b "explain quantum physics"', makeSettings());
    expect(result).not.toBeNull();
    expect(result!.model).toBe('gemma-3b');
    expect(result!.text).toBe('explain quantum physics');
  });

  it('extracts model and single-quoted text', () => {
    const result = parseProviderOverride("@or/claude 'hello world'", makeSettings());
    expect(result).not.toBeNull();
    expect(result!.model).toBe('claude');
    expect(result!.text).toBe('hello world');
  });

  it('extracts model and bare (unquoted) text', () => {
    const result = parseProviderOverride('@ollama/gemma hello', makeSettings());
    expect(result).not.toBeNull();
    expect(result!.model).toBe('gemma');
    expect(result!.text).toBe('hello');
  });

  it('handles model names containing slashes (e.g. anthropic/claude-3.5-sonnet)', () => {
    const result = parseProviderOverride('@or/anthropic/claude-3.5-sonnet "hi"', makeSettings());
    expect(result).not.toBeNull();
    expect(result!.model).toBe('anthropic/claude-3.5-sonnet');
    expect(result!.text).toBe('hi');
  });

  it('handles model names containing colons (e.g. gemma3:1b)', () => {
    const result = parseProviderOverride('@nim/gemma3:1b "hi"', makeSettings());
    expect(result).not.toBeNull();
    expect(result!.model).toBe('gemma3:1b');
    expect(result!.text).toBe('hi');
  });

  it('handles empty quoted text', () => {
    const result = parseProviderOverride('@nim/gemma ""', makeSettings());
    expect(result).not.toBeNull();
    expect(result!.model).toBe('gemma');
    expect(result!.text).toBe('');
  });
});

describe('parseProviderOverride — no model (provider default)', () => {
  it('returns empty model when no / follows the alias', () => {
    const result = parseProviderOverride('@nim "what is the meaning of life"', makeSettings());
    expect(result).not.toBeNull();
    expect(result!.backend).toBe('nvidia_nim');
    expect(result!.model).toBe('');
    expect(result!.text).toBe('what is the meaning of life');
  });

  it('returns empty model and empty text when only the alias is given', () => {
    const result = parseProviderOverride('@ollama', makeSettings());
    expect(result).not.toBeNull();
    expect(result!.backend).toBe('ollama');
    expect(result!.model).toBe('');
    expect(result!.text).toBe('');
  });

  it('returns empty model with bare text', () => {
    const result = parseProviderOverride('@or hello', makeSettings());
    expect(result).not.toBeNull();
    expect(result!.backend).toBe('openrouter');
    expect(result!.model).toBe('');
    expect(result!.text).toBe('hello');
  });
});

describe('parseProviderOverride — null returns', () => {
  it('returns null when input does not start with @', () => {
    expect(parseProviderOverride('hello world', makeSettings())).toBeNull();
  });

  it('returns null for an empty string', () => {
    expect(parseProviderOverride('', makeSettings())).toBeNull();
  });

  it('returns null for a bare @ with no alias', () => {
    expect(parseProviderOverride('@', makeSettings())).toBeNull();
  });

  it('returns null for an unrecognised alias', () => {
    expect(parseProviderOverride('@claude/model "hi"', makeSettings())).toBeNull();
  });

  it('returns null for an unrecognised shorthand alias', () => {
    expect(parseProviderOverride('@o/model "hi"', makeSettings())).toBeNull();
  });
});

describe('parseProviderOverride — hasKey', () => {
  it('hasKey is false when no key is configured for the backend', () => {
    const settings = makeSettings(); // all providerKeys are null
    const result = parseProviderOverride('@nim/gemma "hi"', settings);
    expect(result).not.toBeNull();
    expect(result!.hasKey).toBe(false);
  });

  it('hasKey is true when a key is configured for nvidia_nim', () => {
    const settings = makeSettings({
      providerKeys: { ollama: null, openrouter: null, nvidia_nim: 'nv-abc-123' },
    });
    const result = parseProviderOverride('@nim/gemma "hi"', settings);
    expect(result).not.toBeNull();
    expect(result!.hasKey).toBe(true);
  });

  it('hasKey is true when a key is configured for openrouter', () => {
    const settings = makeSettings({
      providerKeys: { ollama: null, openrouter: 'sk-or-456', nvidia_nim: null },
    });
    const result = parseProviderOverride('@or/gemma "hi"', settings);
    expect(result).not.toBeNull();
    expect(result!.hasKey).toBe(true);
  });

  it('hasKey is true when a key is configured for ollama', () => {
    const settings = makeSettings({
      providerKeys: { ollama: 'ollama-key', openrouter: null, nvidia_nim: null },
    });
    const result = parseProviderOverride('@ollama/gemma "hi"', settings);
    expect(result).not.toBeNull();
    expect(result!.hasKey).toBe(true);
  });

  it('hasKey is false for an empty-string key', () => {
    const settings = makeSettings({
      providerKeys: { ollama: '', openrouter: null, nvidia_nim: null },
    });
    const result = parseProviderOverride('@ollama/gemma "hi"', settings);
    expect(result).not.toBeNull();
    expect(result!.hasKey).toBe(false);
  });

  it('hasKey only reflects the matched backend, not others', () => {
    const settings = makeSettings({
      providerKeys: { ollama: 'ollama-key', openrouter: null, nvidia_nim: 'nv-key' },
    });
    const orResult = parseProviderOverride('@or/gemma "hi"', settings);
    expect(orResult).not.toBeNull();
    expect(orResult!.hasKey).toBe(false); // openrouter has no key
  });
});

describe('parseProviderOverride — full override object shape', () => {
  it('returns a complete ProviderOverride with all fields populated', () => {
    const settings = makeSettings({
      providerKeys: { ollama: null, openrouter: 'sk-123', nvidia_nim: null },
    });
    const result = parseProviderOverride('@openrouter/claude-3.5 "hello"', settings);
    expect(result).toEqual({
      alias: 'openrouter',
      backend: 'openrouter',
      model: 'claude-3.5',
      text: 'hello',
      hasKey: true,
    });
  });

  it('returns a complete ProviderOverride with empty model', () => {
    const settings = makeSettings({
      providerKeys: { ollama: null, openrouter: null, nvidia_nim: 'nv-789' },
    });
    const result = parseProviderOverride('@nim "explain RAG"', settings);
    expect(result).toEqual({
      alias: 'nim',
      backend: 'nvidia_nim',
      model: '',
      text: 'explain RAG',
      hasKey: true,
    });
  });
});
