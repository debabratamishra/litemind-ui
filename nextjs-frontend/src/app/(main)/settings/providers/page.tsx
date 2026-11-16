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
import { cn } from '@/lib/utils';
import {
  deleteApiKey,
  getApiKeys,
  saveApiKey,
  testProvider,
  type ApiKeyRecord,
  type ProviderTestResult,
} from '@/lib/api';

const KIND_OPTIONS = ['llm', 'voice', 'stt', 'tts', 'embeddings', 'search'] as const;

const PROVIDER_PRESETS: Record<string, string[]> = {
  llm: ['openrouter', 'nvidia_nim'],
  voice: ['elevenlabs', 'sarvam'],
  stt: ['elevenlabs'],
  tts: ['elevenlabs'],
  embeddings: ['openrouter', 'nvidia_nim'],
  search: ['serpapi'],
};

export default function ProvidersSettingsPage(): React.ReactElement {
  const [keys, setKeys] = React.useState<ApiKeyRecord[]>([]);
  const [kind, setKind] = React.useState<string>('llm');
  const [provider, setProvider] = React.useState('');
  const [agentId, setAgentId] = React.useState('');
  const [label, setLabel] = React.useState('');
  const [value, setValue] = React.useState('');
  // Sarvam-specific credential fields
  const [sarvamOrgId, setSarvamOrgId] = React.useState('');
  const [sarvamWorkspaceId, setSarvamWorkspaceId] = React.useState('');
  const [sarvamAppId, setSarvamAppId] = React.useState('');
  const [error, setError] = React.useState<string | null>(null);
  const [saving, setSaving] = React.useState(false);
  const [testing, setTesting] = React.useState(false);
  const [deletingId, setDeletingId] = React.useState<string | null>(null);
  const [testResult, setTestResult] = React.useState<ProviderTestResult | null>(null);

  const isSarvam = provider === 'sarvam';

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

  const handleKindChange = (newKind: string) => {
    setKind(newKind);
    const presets = PROVIDER_PRESETS[newKind] || [];
    if (presets.length > 0 && (!provider.trim() || Object.values(PROVIDER_PRESETS).flat().includes(provider.trim()))) {
      setProvider(presets[0]);
    }
  };

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
      const saved = await saveApiKey({
        kind,
        provider: provider.trim(),
        label: agentId.trim() || label.trim(),
        value,
      });

      let updatedKeys = [
        saved,
        ...keys.filter((r) => !(r.kind === saved.kind && r.provider === saved.provider)),
      ];

      // ElevenLabs Agent ID — stored as a second credential entry
      if (agentId.trim() && provider === 'elevenlabs') {
        try {
          const savedAgent = await saveApiKey({
            kind: 'voice',
            provider: 'elevenlabs_agent_id',
            label: 'Agent ID',
            value: agentId.trim(),
          });
          updatedKeys = [
            savedAgent,
            ...updatedKeys.filter((r) => !(r.kind === savedAgent.kind && r.provider === savedAgent.provider)),
          ];
        } catch {
          /* non-fatal */
        }
      }

      // Sarvam org_id, workspace_id, app_id — each stored as a separate credential
      if (provider === 'sarvam') {
        for (const [field, fieldValue] of [
          ['sarvam_org_id', sarvamOrgId],
          ['sarvam_workspace_id', sarvamWorkspaceId],
          ['sarvam_app_id', sarvamAppId],
        ] as const) {
          if (fieldValue.trim()) {
            try {
              const savedField = await saveApiKey({
                kind: 'voice',
                provider: field,
                label: field.replace('sarvam_', '').replace('_', ' '),
                value: fieldValue.trim(),
              });
              updatedKeys = [
                savedField,
                ...updatedKeys.filter((r) => !(r.kind === savedField.kind && r.provider === savedField.provider)),
              ];
            } catch {
              /* non-fatal */
            }
          }
        }
      }

      setKeys(updatedKeys);
      setValue('');
      setAgentId('');
      setSarvamOrgId('');
      setSarvamWorkspaceId('');
      setSarvamAppId('');
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
          API keys for LLM, voice agent, speech, embedding, and search providers. Keys are encrypted at rest.
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
            <Select value={kind} onValueChange={(v) => v && handleKindChange(v)}>
              <SelectTrigger id="prov-kind" className="h-8 text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {KIND_OPTIONS.map((k) => (
                  <SelectItem key={k} value={k} className="text-sm">
                    {k === 'voice' ? 'voice (speech-to-speech agent)' : k}
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
              placeholder="e.g. elevenlabs, openrouter, nvidia_nim"
              value={provider}
              onChange={(e) => setProvider(e.target.value)}
            />
            {PROVIDER_PRESETS[kind] && (
              <div className="flex flex-wrap items-center gap-1.5 pt-0.5">
                <span className="text-[11px] text-muted-foreground mr-1">Presets:</span>
                {PROVIDER_PRESETS[kind].map((p) => (
                  <button
                    key={p}
                    type="button"
                    onClick={() => setProvider(p)}
                    className={cn(
                      'rounded px-2 py-0.5 text-xs font-mono transition-colors border',
                      provider === p
                        ? 'bg-primary text-primary-foreground border-primary'
                        : 'bg-muted/50 hover:bg-muted text-muted-foreground'
                    )}
                  >
                    {p}
                  </button>
                ))}
              </div>
            )}
          </div>
          {provider === 'elevenlabs' && (
            <div className="grid gap-2 rounded-md border border-primary/20 bg-primary/5 p-3">
              <Label htmlFor="prov-agent" className="text-xs font-semibold">
                {kind === 'voice' ? 'Agent ID (required for voice agent)' : 'Agent ID (Conversational AI — optional for STT / TTS)'}
              </Label>
              <Input
                id="prov-agent"
                className="h-8 text-sm font-mono"
                placeholder="agent_abc123def456 (from ElevenLabs dashboard)"
                value={agentId}
                onChange={(e) => setAgentId(e.target.value)}
              />
              <p className="text-[11px] text-muted-foreground">
                {kind === 'voice'
                  ? 'Get this from elevenlabs.io → Conversational AI. Required for speech-to-speech voice mode — no STT or TTS models needed.'
                  : 'Get this from elevenlabs.io → Conversational AI. Enables real-time speech-to-speech with no local models.'}
              </p>
            </div>
          )}
          {isSarvam && (
            <div className="grid gap-2 rounded-md border border-primary/20 bg-primary/5 p-3">
              <p className="text-xs font-semibold">Sarvam agent identifiers</p>
              <p className="text-[11px] text-muted-foreground">
                Find these in your{' '}
                <a href="https://app.sarvam.ai" target="_blank" rel="noreferrer" className="text-primary hover:underline">
                  Sarvam dashboard
                </a>
                {' '}→ Conversations → your agent → Deploy with Code.
              </p>
              <Label htmlFor="prov-sarvam-org" className="text-xs">Org ID</Label>
              <Input
                id="prov-sarvam-org"
                className="h-8 text-sm font-mono"
                placeholder="org_…"
                value={sarvamOrgId}
                onChange={(e) => setSarvamOrgId(e.target.value)}
              />
              <Label htmlFor="prov-sarvam-ws" className="text-xs">Workspace ID</Label>
              <Input
                id="prov-sarvam-ws"
                className="h-8 text-sm font-mono"
                placeholder="ws_…"
                value={sarvamWorkspaceId}
                onChange={(e) => setSarvamWorkspaceId(e.target.value)}
              />
              <Label htmlFor="prov-sarvam-app" className="text-xs">App ID</Label>
              <Input
                id="prov-sarvam-app"
                className="h-8 text-sm font-mono"
                placeholder="app_…"
                value={sarvamAppId}
                onChange={(e) => setSarvamAppId(e.target.value)}
              />
            </div>
          )}
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
              placeholder={
                provider === 'elevenlabs'
                  ? 'Paste your ElevenLabs xi-api-key'
                  : provider === 'sarvam'
                  ? 'Paste your Sarvam API key'
                  : 'Paste the key'
              }
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
