'use client';

import * as React from 'react';
import { X, RotateCcw, CheckCircle2, XCircle, Loader2, ShieldCheck, Shuffle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Dialog,
  DialogContent,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { KnowledgeBaseSection } from '@/components/knowledge-base';
import { useAppStore } from '@/lib/store';
import { getEnhancedModels, checkSerpStatus } from '@/lib/api';
import { cn } from '@/lib/utils';
import type { Model, BackendType } from '@/lib/types';

function Section({
  title,
  action,
  children,
}: {
  title: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          {title}
        </h3>
        {action}
      </div>
      {children}
    </section>
  );
}

function SliderField({
  id,
  label,
  value,
  display,
  min,
  max,
  step,
  onChange,
  hint,
}: {
  id: string;
  label: string;
  value: number;
  display: string;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  hint?: string;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label htmlFor={id} className="text-xs">
          {label}
        </Label>
        <span className="text-xs font-mono text-muted-foreground">{display}</span>
      </div>
      <Slider
        id={id}
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={(v) => onChange(Array.isArray(v) ? (v[0] as number) : (v as number))}
        aria-label={label}
      />
      {hint && <p className="text-[11px] leading-snug text-muted-foreground">{hint}</p>}
    </div>
  );
}

const GENERATION_DEFAULTS = {
  temperature: 0.7,
  maxTokens: 2048,
  topP: 0.9,
  topK: 40,
  minP: 0.0,
  frequencyPenalty: 0,
  repetitionPenalty: 1.0,
  seed: null,
  stopSequences: '',
} as const;

export function SettingsPanel({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const { settings, setSettings } = useAppStore();
  const [localModels, setLocalModels] = React.useState<Model[]>([]);
  const [cloudModels, setCloudModels] = React.useState<Model[]>([]);
  const [serpStatus, setSerpStatus] = React.useState<{ status: string; message: string } | null>(null);
  const [checking, setChecking] = React.useState(false);

  React.useEffect(() => {
    getEnhancedModels()
      .then((r) => {
        setLocalModels(r.local_models ?? []);
        setCloudModels(r.cloud_models ?? []);
      })
      .catch(() => {
        setLocalModels([]);
        setCloudModels([]);
      });
  }, []);

  const resetGeneration = () => setSettings({ ...GENERATION_DEFAULTS });

  const checkStatus = async () => {
    setChecking(true);
    try {
      const res = await checkSerpStatus(settings.serpApiKey);
      setSerpStatus(res);
    } catch (e) {
      setSerpStatus({ status: 'error', message: e instanceof Error ? e.message : 'Check failed' });
    } finally {
      setChecking(false);
    }
  };

  const serpStatusColor =
    serpStatus?.status === 'valid'
      ? 'text-emerald-600 dark:text-emerald-400'
      : 'text-destructive';
  const SerpStatusIcon =
    serpStatus?.status === 'valid' ? CheckCircle2 : serpStatus ? XCircle : ShieldCheck;

  const isOllama = settings.backend === 'ollama';
  const needsApiKey = settings.backend === 'openrouter' || settings.backend === 'nvidia_nim';
  const allOllama = [
    ...localModels.map((m) => ({ ...m, is_local: true as const })),
    ...cloudModels.map((m) => ({ ...m, is_local: false as const })),
  ];

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent
        showCloseButton={false}
        className="left-0 top-0 translate-x-0 translate-y-0 h-full max-w-sm w-full rounded-none border-r border-border p-0 gap-0 flex flex-col data-[state=closed]:slide-out-to-left data-[state=open]:slide-in-from-left"
      >
        <div className="flex items-center justify-between border-b border-border px-4 py-3 shrink-0">
          <DialogTitle className="text-sm font-semibold">Settings</DialogTitle>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            aria-label="Close settings"
            className="h-7 w-7"
          >
            <X className="h-4 w-4" aria-hidden="true" />
          </Button>
        </div>

        <ScrollArea className="flex-1 min-h-0">
          <div className="space-y-6 p-4">
            <Section title="Models">
              <div className="space-y-2">
                <Label htmlFor="set-backend" className="text-xs">
                  Backend
                </Label>
                <Select
                  value={settings.backend}
                  onValueChange={(v) => v && setSettings({ backend: v as BackendType, model: '' })}
                >
                  <SelectTrigger id="set-backend" className="h-8 text-sm">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="ollama">🖥️ Ollama (local)</SelectItem>
                    <SelectItem value="openrouter">☁️ OpenRouter</SelectItem>
                    <SelectItem value="nvidia_nim">⚡ Nvidia NIM</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {needsApiKey && (
                <div className="space-y-2">
                  <Label htmlFor="set-key" className="text-xs">
                    {settings.backend === 'openrouter' ? 'OpenRouter API key' : 'Nvidia NIM API key'}
                  </Label>
                  <Input
                    id="set-key"
                    type="password"
                    placeholder={settings.backend === 'openrouter' ? 'sk-or-…' : 'nvapi-…'}
                    value={settings.apiKey ?? ''}
                    onChange={(e) => setSettings({ apiKey: e.target.value })}
                    className="h-8 text-sm"
                  />
                </div>
              )}

              <div className="space-y-2">
                <Label htmlFor={isOllama ? 'set-model-sel' : 'set-model-in'} className="text-xs">
                  Model
                </Label>
                {isOllama ? (
                  <Select value={settings.model} onValueChange={(v) => v && setSettings({ model: v })}>
                    <SelectTrigger id="set-model-sel" className="h-8 text-sm">
                      <SelectValue placeholder={allOllama.length ? 'Select model…' : 'No models'} />
                    </SelectTrigger>
                    <SelectContent>
                      {allOllama.length === 0 ? (
                        <SelectItem value="__none__" disabled>
                          Ollama unreachable
                        </SelectItem>
                      ) : (
                        allOllama.map((m) => (
                          <SelectItem key={m.name} value={m.name} className="text-sm">
                            {m.is_local ? '🟢 ' : '☁️ '}
                            {m.name}
                          </SelectItem>
                        ))
                      )}
                    </SelectContent>
                  </Select>
                ) : (
                  <Input
                    id="set-model-in"
                    type="text"
                    placeholder={
                      settings.backend === 'openrouter' ? 'e.g. openai/gpt-4o' : 'e.g. meta/llama-3.3-70b-instruct'
                    }
                    value={settings.model}
                    onChange={(e) => setSettings({ model: e.target.value })}
                    className="h-8 text-sm font-mono"
                  />
                )}
              </div>
            </Section>

            <Separator />

            <Section
              title="Generation"
              action={
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 gap-1 px-2 text-[11px] text-muted-foreground hover:text-foreground"
                  onClick={resetGeneration}
                  aria-label="Reset generation settings to defaults"
                >
                  <RotateCcw className="h-3 w-3" aria-hidden="true" />
                  Reset
                </Button>
              }
            >
              <SliderField
                id="set-temp"
                label="Temperature"
                value={settings.temperature}
                display={settings.temperature.toFixed(1)}
                min={0}
                max={2}
                step={0.1}
                onChange={(v) => setSettings({ temperature: v })}
                hint="Higher values make output more random; lower values more focused and deterministic."
              />
              <div className="space-y-2">
                <Label htmlFor="set-maxtok" className="text-xs">
                  Max tokens
                </Label>
                <Input
                  id="set-maxtok"
                  type="number"
                  min={128}
                  max={32768}
                  step={128}
                  value={settings.maxTokens}
                  onChange={(e) => setSettings({ maxTokens: parseInt(e.target.value) || 2048 })}
                  className="h-8 text-sm"
                />
                <p className="text-[11px] leading-snug text-muted-foreground">
                  Maximum length of the model&apos;s reply.
                </p>
              </div>
              <SliderField
                id="set-topp"
                label="Top-P"
                value={settings.topP}
                display={settings.topP.toFixed(2)}
                min={0}
                max={1}
                step={0.05}
                onChange={(v) => setSettings({ topP: v })}
                hint="Nucleus sampling — only tokens within the top probability mass are considered."
              />
              <SliderField
                id="set-topk"
                label="Top-K"
                value={settings.topK}
                display={String(settings.topK)}
                min={0}
                max={100}
                step={1}
                onChange={(v) => setSettings({ topK: v })}
                hint="Limits sampling to the K most likely next tokens. 0 disables the cutoff."
              />
              <SliderField
                id="set-minp"
                label="Min-P"
                value={settings.minP}
                display={settings.minP.toFixed(2)}
                min={0}
                max={1}
                step={0.01}
                onChange={(v) => setSettings({ minP: v })}
                hint="Minimum probability a token needs relative to the most likely one."
              />
              <SliderField
                id="set-freq"
                label="Frequency penalty"
                value={settings.frequencyPenalty}
                display={settings.frequencyPenalty.toFixed(1)}
                min={-2}
                max={2}
                step={0.1}
                onChange={(v) => setSettings({ frequencyPenalty: v })}
                hint="Penalises tokens by how often they have already appeared."
              />
              <SliderField
                id="set-rep"
                label="Repetition penalty"
                value={settings.repetitionPenalty}
                display={settings.repetitionPenalty.toFixed(2)}
                min={0}
                max={2}
                step={0.05}
                onChange={(v) => setSettings({ repetitionPenalty: v })}
                hint="Penalises repeated tokens (Ollama repeat_penalty). 1.0 turns it off."
              />
              <div className="space-y-2">
                <Label htmlFor="set-seed" className="text-xs">
                  Seed
                </Label>
                <div className="flex items-center gap-2">
                  <Input
                    id="set-seed"
                    type="number"
                    placeholder="random"
                    value={settings.seed ?? ''}
                    onChange={(e) => {
                      const raw = e.target.value.trim();
                      setSettings({ seed: raw === '' ? null : parseInt(raw, 10) || null });
                    }}
                    className="h-8 text-sm font-mono"
                  />
                  <Button
                    variant="outline"
                    size="icon"
                    className="h-8 w-8 shrink-0"
                    onClick={() => setSettings({ seed: null })}
                    aria-label="Use a random seed"
                  >
                    <Shuffle className="h-4 w-4" aria-hidden="true" />
                  </Button>
                </div>
                <p className="text-[11px] leading-snug text-muted-foreground">
                  Fix for reproducible outputs. Empty means a new random seed each run.
                </p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="set-stop" className="text-xs">
                  Stop sequences
                </Label>
                <Input
                  id="set-stop"
                  type="text"
                  placeholder="e.g. &quot;\n\nHuman:&quot;, &quot;###&quot;"
                  value={settings.stopSequences}
                  onChange={(e) => setSettings({ stopSequences: e.target.value })}
                  className="h-8 text-sm font-mono"
                />
                <p className="text-[11px] leading-snug text-muted-foreground">
                  Comma-separated. Generation halts when any of these strings appears.
                </p>
              </div>
            </Section>

            <Separator />

            <Section title="Web search">
              <div className="space-y-2">
                <Label htmlFor="set-serp" className="text-xs">
                  SerpAPI key
                </Label>
                <Input
                  id="set-serp"
                  type="password"
                  autoComplete="off"
                  placeholder="Paste your SerpAPI key"
                  value={settings.serpApiKey ?? ''}
                  onChange={(e) => setSettings({ serpApiKey: e.target.value })}
                  className="h-8 text-sm font-mono"
                />
                <p className="text-[11px] leading-snug text-muted-foreground">
                  Powers web search. Provided keys override the server&apos;s SERP_API_KEY for this session
                  only.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 gap-1.5 text-xs"
                  onClick={checkStatus}
                  disabled={checking}
                >
                  {checking ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
                  ) : (
                    <SerpStatusIcon className="h-3.5 w-3.5" aria-hidden="true" />
                  )}
                  Check status
                </Button>
                {serpStatus && (
                  <span className={cn('flex items-center gap-1 text-[11px]', serpStatusColor)}>
                    <SerpStatusIcon className="h-3.5 w-3.5" aria-hidden="true" />
                    {serpStatus.message}
                  </span>
                )}
              </div>
            </Section>

            <Separator />

            <Section title="Backend keys">
              <div className="space-y-2">
                <Label htmlFor="set-ollama" className="text-xs">
                  Ollama URL
                </Label>
                <Input
                  id="set-ollama"
                  placeholder="http://localhost:11434"
                  value={settings.ollamaUrl ?? ''}
                  onChange={(e) => setSettings({ ollamaUrl: e.target.value })}
                  className="h-8 text-sm"
                />
              </div>
            </Section>

            <Separator />

            <Section title="Knowledge Base">
              <KnowledgeBaseSection />
            </Section>
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}
