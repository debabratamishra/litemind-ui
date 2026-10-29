'use client';

import { X } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import type { ProviderOverride } from '@/lib/types';
import { cn } from '@/lib/utils';

/** Display name and colour for each backend. */
const PROVIDER_INFO: Record<ProviderOverride['backend'], { label: string; color: string }> = {
  nvidia_nim: { label: 'NIM', color: 'bg-cyan-500' },
  openrouter: { label: 'OpenRouter', color: 'bg-violet-500' },
  ollama: { label: 'Ollama', color: 'bg-emerald-600' },
};

export interface ProviderOverrideBadgeProps {
  override: ProviderOverride;
  onRemove: () => void;
  onSetKey: () => void;
}

/**
 * Inline badge that appears above the chat/RAG input when a `@` provider
 * override is active. Shows the provider, model, and key status.
 *
 * - Green ✓ when an API key is configured for the provider.
 * - Amber ⚠️ when the key is missing — clicking the badge opens the key prompt.
 * - The X button removes the override entirely.
 */
export function ProviderOverrideBadge({
  override,
  onRemove,
  onSetKey,
}: ProviderOverrideBadgeProps) {
  const info = PROVIDER_INFO[override.backend];
  const displayModel = override.model || 'default';

  return (
    <div className="mb-2 flex items-center gap-2 text-sm">
      <Badge
        className={cn(
          'cursor-pointer gap-1.5 px-2.5 py-1 text-xs font-medium text-white',
          info.color,
          !override.hasKey && 'bg-amber-500',
        )}
        onClick={onSetKey}
        role="button"
        aria-label={`Provider: ${info.label}, model: ${displayModel}. ${override.hasKey ? 'Key configured' : 'Key missing — click to set'}`}
      >
        <span>{info.label}</span>
        {override.model && <span className="opacity-80">· {displayModel}</span>}
        <span className="ml-1">{override.hasKey ? '✓' : '⚠️'}</span>
      </Badge>
      <Button
        variant="ghost"
        size="sm"
        onClick={onRemove}
        aria-label="Remove provider override"
        className="h-5 w-5 p-0"
      >
        <X className="h-3 w-3" aria-hidden="true" />
      </Button>
    </div>
  );
}
