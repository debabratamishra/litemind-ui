'use client';

import * as React from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import type { BackendType } from '@/lib/types';

const PROVIDER_LABELS: Record<BackendType, string> = {
  nvidia_nim: 'NIM',
  openrouter: 'OpenRouter',
  ollama: 'Ollama',
};

export interface ProviderKeyPromptProps {
  open: boolean;
  provider: BackendType;
  model: string;
  onSetKey: (key: string) => void;
  onCancel: () => void;
  onFallback: () => void;
}

/**
 * Modal that appears when the user tries to send a message with a `@`
 * provider override but no API key is configured for that backend.
 *
 * Lets the user paste a key (sent per-request, never persisted to the backend)
 * or fall back to the default Ollama provider.
 */
export function ProviderKeyPrompt({
  open,
  provider,
  model,
  onSetKey,
  onCancel,
  onFallback,
}: ProviderKeyPromptProps) {
  const [key, setKey] = React.useState('');

  const handleOpenChange = (newOpen: boolean) => {
    if (newOpen) {
      setKey('');
    } else {
      onCancel();
    }
  };

  const handleSetKey = () => {
    if (key.trim()) {
      onSetKey(key.trim());
      setKey('');
    }
  };

  const handleCancel = () => {
    setKey('');
    onCancel();
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent showCloseButton={false}>
        <DialogHeader>
          <DialogTitle>Provider: {PROVIDER_LABELS[provider]}</DialogTitle>
          <DialogDescription>
            {model && `Model: ${model}`}
            <br />
            Status: ⚠️ API key required
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-2">
          <Label htmlFor="provider-api-key">API Key</Label>
          <Input
            id="provider-api-key"
            type="password"
            placeholder="Enter your API key"
            value={key}
            onChange={(e) => setKey(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && key.trim()) {
                handleSetKey();
              }
            }}
            autoComplete="off"
          />
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleCancel}>
            Cancel
          </Button>
          <Button variant="outline" onClick={onFallback}>
            Use Default Ollama Instead
          </Button>
          <Button onClick={handleSetKey} disabled={!key.trim()}>
            Set API Key
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
