'use client';

import * as React from 'react';
import { KeyRound, Loader2, Play, Save, Trash2 } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  deleteApiKey,
  getApiKeys,
  saveApiKey,
  testProvider,
  type ApiKeyRecord,
  type ProviderTestResult,
} from '@/lib/api';

const KIND_OPTIONS = ['llm', 'vlm', 'stt', 'tts', 'embeddings', 'search'] as const;

export default function ProvidersSettingsPage(): React.ReactElement {
  const [keys, setKeys] = React.useState<ApiKeyRecord[]>([]);
  const [kind, setKind] = React.useState<string>('llm');
  const [provider, setProvider] = React.useState('');
  const [label, setLabel] = React.useState('');
  const [value, setValue] = React.useState('');
  const [error, setError] = React.useState<string | null>(null);
  const [saving, setSaving] = React.useState(false);
  const [testing, setTesting] = React.useState(false);
  const [deletingId, setDeletingId] = React.useState<string | null>(null);
  const [testResult, setTestResult] = React.useState<ProviderTestResult | null>(null);

  React.useEffect(() => {
    let cancelled = false;
    getApiKeys()
      .then((rows) => {
        if (!cancelled) setKeys(rows);
      })
      .catch(() => {
        if (!cancelled) setError('Could not load saved keys');
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const onSave = async () => {
    if (!provider.trim()) {
      setError('Provider is required');
      return;
    }
    if (!value) {
      setError('API key is required');
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const saved = await saveApiKey({ kind, provider: provider.trim(), label: label.trim(), value });
      setKeys((rows) => [
        saved,
        ...rows.filter((r) => !(r.kind === saved.kind && r.provider === saved.provider)),
      ]);
      setValue('');
    } catch {
      setError('Could not save key');
    } finally {
      setSaving(false);
    }
  };

  const onDelete = async (id: string) => {
    setDeletingId(id);
    setError(null);
    try {
      await deleteApiKey(id);
      setKeys((rows) => rows.filter((r) => r.id !== id));
    } catch {
      setError('Could not delete key');
    } finally {
      setDeletingId(null);
    }
  };

  const onTest = async () => {
    setTesting(true);
    setError(null);
    setTestResult(null);
    try {
      setTestResult(await testProvider(kind));
    } catch {
      setError('Could not run connection test');
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-6">
      <div>
        <h1 className="text-lg font-semibold">Providers</h1>
        <p className="text-sm text-muted-foreground">
          API keys for LLM, VLM, speech, embedding, and search providers. Keys are encrypted at rest.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Save a key</CardTitle>
          <CardDescription>
            Saving over an existing provider for the same capability replaces it.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-2">
            <Label htmlFor="prov-kind">Capability</Label>
            <Select value={kind} onValueChange={(v) => v && setKind(v)}>
              <SelectTrigger id="prov-kind" className="h-8 text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {KIND_OPTIONS.map((k) => (
                  <SelectItem key={k} value={k} className="text-sm">
                    {k}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="grid gap-2">
            <Label htmlFor="prov-provider">Provider</Label>
            <Input
              id="prov-provider"
              className="h-8 text-sm"
              placeholder="e.g. openrouter, nvidia_nim, ollama"
              value={provider}
              onChange={(e) => setProvider(e.target.value)}
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="prov-label">Label</Label>
            <Input
              id="prov-label"
              className="h-8 text-sm"
              placeholder="e.g. main"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="prov-value">API key</Label>
            <Input
              id="prov-value"
              type="password"
              autoComplete="off"
              className="h-8 text-sm font-mono"
              placeholder="Paste the key"
              value={value}
              onChange={(e) => setValue(e.target.value)}
            />
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button size="sm" onClick={() => void onSave()} disabled={saving}>
              {saving ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
              ) : (
                <Save className="h-3.5 w-3.5" aria-hidden="true" />
              )}
              Save
            </Button>
            <Button variant="outline" size="sm" onClick={() => void onTest()} disabled={testing}>
              {testing ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
              ) : (
                <Play className="h-3.5 w-3.5" aria-hidden="true" />
              )}
              Test
            </Button>
            {testResult && (
              <span aria-live="polite" className="text-xs text-muted-foreground">
                {testResult.ok
                  ? `Connected in ${testResult.latency_ms} ms`
                  : `Test failed: ${testResult.error ?? 'unknown error'}`}
              </span>
            )}
          </div>
        </CardContent>
      </Card>

      <Separator />

      <div className="space-y-3">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Saved keys
        </h2>
        {error && (
          <p className="text-xs text-destructive" role="alert">
            {error}
          </p>
        )}
        {keys.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            No keys saved yet. Keys are stored encrypted and never shown back.
          </p>
        ) : (
          <ul className="space-y-2">
            {keys.map((k) => (
              <li
                key={k.id}
                className="flex items-center justify-between gap-2 rounded border px-3 py-2 text-sm"
              >
                <span className="flex items-center gap-2">
                  <KeyRound className="h-3.5 w-3.5 text-muted-foreground" aria-hidden="true" />
                  <span className="font-medium">{k.provider}</span>
                  <Badge variant="secondary" className="text-[10px]">
                    {k.kind}
                  </Badge>
                  {k.label && <span className="text-xs text-muted-foreground">{k.label}</span>}
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => void onDelete(k.id)}
                  disabled={deletingId === k.id}
                  aria-label={`Delete ${k.provider} key`}
                >
                  {deletingId === k.id ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
                  ) : (
                    <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
                  )}
                  Delete
                </Button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
