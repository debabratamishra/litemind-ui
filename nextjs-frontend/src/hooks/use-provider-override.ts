import type { AppSettings, BackendType, ProviderOverride } from '@/lib/types';

/**
 * Maps provider aliases used in chat input (`@alias`) to their canonical
 * backend type.
 *
 * - `@nim`        → nvidia_nim
 * - `@or`         → openrouter  (shorthand)
 * - `@openrouter` → openrouter
 * - `@ollama`     → ollama
 */
const ALIAS_TO_BACKEND: Record<string, BackendType> = {
  nim: 'nvidia_nim',
  openrouter: 'openrouter',
  or: 'openrouter',
  ollama: 'ollama',
};

/**
 * Parse a provider-override directive from chat input.
 *
 * Recognised forms:
 *   @nim/model "query"          → nvidia_nim backend
 *   @openrouter/model "query"
 *   @or/model "query"           → openrouter (shorthand)
 *   @ollama/model "query"
 *
 * If no `/` follows the alias, `model` is an empty string and the caller
 * should fall back to the provider's default model.
 *
 * Returns `null` when:
 *   - the input does not start with `@`, or
 *   - the alias is not one of the recognised providers.
 *
 * `hasKey` is derived from `settings.providerKeys[backend]` — `true` when a
 * non-empty key is configured for that backend.
 */
export function parseProviderOverride(
  input: string,
  settings: AppSettings,
): ProviderOverride | null {
  if (!input.startsWith('@')) {
    return null;
  }

  // Strip the leading '@' and any leading whitespace.
  const rest = input.slice(1).trimStart();

  // Alias = everything up to the first '/' or whitespace character.
  const aliasEnd = Math.min(
    ...['/', ' ', '\t'].map((sep) => {
      const idx = rest.indexOf(sep);
      return idx === -1 ? rest.length : idx;
    }),
  );
  const alias = rest.slice(0, aliasEnd);

  const backend = ALIAS_TO_BACKEND[alias];
  if (!backend) {
    return null;
  }

  // Everything after the alias.
  let remainder = rest.slice(aliasEnd);

  // Optional /model segment — runs until the next whitespace.
  let model = '';
  if (remainder.startsWith('/')) {
    remainder = remainder.slice(1);
    const modelEnd = remainder.search(/\s/);
    model = modelEnd === -1 ? remainder : remainder.slice(0, modelEnd);
    remainder = modelEnd === -1 ? '' : remainder.slice(modelEnd);
  }

  // Optional text — a double-quoted, single-quoted, or bare token.
  let text = '';
  const trimmed = remainder.trim();
  if (trimmed) {
    const doubleQuoted = trimmed.match(/^"([^"]*)"/);
    if (doubleQuoted) {
      text = doubleQuoted[1];
    } else {
      const singleQuoted = trimmed.match(/^'([^']*)'/);
      if (singleQuoted) {
        text = singleQuoted[1];
      } else {
        text = trimmed;
      }
    }
  }

  return {
    alias,
    backend,
    model,
    text,
    hasKey: !!settings.providerKeys[backend],
  };
}
