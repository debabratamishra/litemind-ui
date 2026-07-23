'use client';

import * as React from 'react';
import { Database, Plus, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import MarkdownRenderer from '@/components/markdown-renderer';
import { SourcesButton, CitationsDialog } from '@/components/citations';
import { RagAttachButton } from '@/components/rag-attach-button';
import { useAppStore, selectActiveConversation, selectActiveId, selectSettings } from '@/lib/store';
import { streamRagQuery, type RAGQueryRequest } from '@/lib/api';
import { useRagUpload } from '@/hooks/use-rag-upload';
import { parseRagContent, convertCitationMarkers, normalizeAnswerWhitespace } from '@/lib/web-search-citations';
import type { UIMessage, ProviderOverride } from '@/lib/types';
import { cn } from '@/lib/utils';
import { parseProviderOverride } from '@/hooks/use-provider-override';
import { ProviderOverrideBadge } from '@/components/provider-override-badge';
import { ProviderKeyPrompt } from '@/components/provider-key-prompt';

/**
 * A single RAG message. Assistant messages that carry a `data: {"citations": …}`
 * frame are rendered with clickable `[n]` citation chips plus a "Sources (N)"
 * button that opens a Dialog listing every cited document chunk — mirroring the
 * web-search citation UX. User messages and plain assistant messages render
 * exactly as before.
 */
function RagMessage({ msg }: { msg: UIMessage }) {
  const [citeOpen, setCiteOpen] = React.useState(false);
  const [citeFocus, setCiteFocus] = React.useState<number | null>(null);

  const openCitations = React.useCallback((focus: number | null) => {
    setCiteFocus(focus);
    setCiteOpen(true);
  }, []);

  if (msg.role === 'user') {
    return <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>;
  }

  const parsed = msg.content ? parseRagContent(msg.content) : null;

  // While streaming, show the (frame-stripped) prose without citation controls.
  if (msg.isStreaming) {
    return <MarkdownRenderer content={parsed ? parsed.answer : msg.content} />;
  }

  if (parsed && parsed.sources.length > 0) {
    const answer = convertCitationMarkers(normalizeAnswerWhitespace(parsed.answer));
    return (
      <>
        <MarkdownRenderer content={answer} onCitationClick={(index) => openCitations(index)} />
        <SourcesButton count={parsed.sources.length} onClick={() => openCitations(null)} />
        <CitationsDialog
          open={citeOpen}
          onOpenChange={setCiteOpen}
          sources={parsed.sources}
          focusIndex={citeFocus}
        />
      </>
    );
  }

  return <MarkdownRenderer content={msg.content} />;
}

export default function RagPage() {
  const conv = useAppStore(selectActiveConversation);
  const activeId = useAppStore(selectActiveId);
  const settings = useAppStore(selectSettings);
  const { addMessage, updateLastMessage, createConversation } = useAppStore();

  const [multiAgent, setMultiAgent] = React.useState(false);
  const [hybridSearch, setHybridSearch] = React.useState(true);
  const [nResults, setNResults] = React.useState(5);
  const [query, setQuery] = React.useState('');
  const [queryLoading, setQueryLoading] = React.useState(false);
  const [keyPromptOpen, setKeyPromptOpen] = React.useState(false);
  const [pendingOverride, setPendingOverride] = React.useState<ProviderOverride | null>(null);
  const abortRef = React.useRef<AbortController | null>(null);
  const bottomRef = React.useRef<HTMLDivElement>(null);
  const { upload, uploading, progress, error } = useRagUpload();
  const dragDepth = React.useRef(0);
  const [dragActive, setDragActive] = React.useState(false);

  // Compute the override live as the user types so the badge can be shown.
  const currentOverride = React.useMemo(
    () => parseProviderOverride(query, settings),
    [query, settings],
  );

  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conv?.messages]);

  const handleQuery = async () => {
    const q = query.trim();
    const convId = activeId;
    if (!q || queryLoading || !convId) return;

    // Parse "@" provider override
    const latestSettings = useAppStore.getState().settings;
    const override = parseProviderOverride(q, latestSettings);

    // If override requires key but none configured, show inline prompt
    if (override && !override.hasKey) {
      setKeyPromptOpen(true);
      setPendingOverride(override);
      return;
    }

    // Determine effective settings from override or defaults
    const effectiveBackend = override?.backend ?? settings.backend;
    const effectiveModel = override?.model || settings.model;
    const effectiveApiKey = override?.hasKey
      ? latestSettings.providerKeys[override.backend]
      : settings.apiKey;
    const effectiveText = override?.text ?? q;

    setQuery('');
    setQueryLoading(true);

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      addMessage(convId, { role: 'user', content: effectiveText });
      addMessage(convId, { role: 'assistant', content: '', isStreaming: true });
      let accumulated = '';
      const stop = settings.stopSequences
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean);

      // The backend selects the RAG strategy from these two flags:
      //   use_multi_agent=false + use_hybrid_search=false -> standard RAG
      //   use_multi_agent=false + use_hybrid_search=true  -> hybrid (BM25 + vector)
      //   use_multi_agent=true                             -> multi-agent (CrewAI)
      // They were previously declared/wired to the UI but never sent, so every
      // query silently fell through to standard RAG.
      const ragRequest: RAGQueryRequest & {
        use_multi_agent?: boolean;
        use_hybrid_search?: boolean;
      } = {
        query: effectiveText,
        model: effectiveModel || undefined,
        backend: effectiveBackend,
        api_key: effectiveApiKey ?? null,
        api_base: settings.apiBase ?? null,
        session_id: settings.sessionId,
        temperature: settings.temperature,
        max_tokens: settings.maxTokens,
        top_p: settings.topP,
        frequency_penalty: settings.frequencyPenalty,
        repetition_penalty: settings.repetitionPenalty,
        min_p: settings.minP,
        seed: settings.seed,
        stop: stop.length ? stop : undefined,
        top_k: nResults,
        use_multi_agent: multiAgent,
        use_hybrid_search: hybridSearch,
      };

      const stream = streamRagQuery(ragRequest, controller.signal);
      for await (const chunk of stream) {
        // The backend prepends a `data: {"citations": {...}}` frame to the
        // plain-text stream. We keep it in the stored content and let the
        // renderer strip it so the citation chips can be built from it.
        accumulated += chunk;
        updateLastMessage(convId, accumulated, true);
      }
      updateLastMessage(convId, accumulated, false);
    } catch (err) {
      if (!(err instanceof Error && err.name === 'AbortError')) {
        updateLastMessage(convId, `⚠️ Error: ${err instanceof Error ? err.message : 'An error occurred.'}`, false);
      }
    } finally {
      setQueryLoading(false);
      abortRef.current = null;
    }
  };

  // ── Provider override key prompt handlers ─────────────────────────────────
  // handleSetKey is a plain function (not useCallback) so it always closes over
  // the latest handleQuery and pendingOverride. After storing the key it
  // re-invokes handleQuery, which re-parses the override — this time hasKey is
  // true because setProviderKey updated the store synchronously.
  const handleSetKey = (key: string) => {
    if (pendingOverride) {
      useAppStore.getState().setProviderKey(pendingOverride.backend, key);
    }
    setKeyPromptOpen(false);
    setPendingOverride(null);
    void handleQuery();
  };

  const handleKeyPromptCancel = React.useCallback(() => {
    setKeyPromptOpen(false);
    setPendingOverride(null);
  }, []);

  const handleKeyPromptFallback = React.useCallback(() => {
    setKeyPromptOpen(false);
    setPendingOverride(null);
    // Fallback to default provider — just clear the input
    setQuery('');
  }, []);

  if (!conv || !activeId) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
        <Database className="h-10 w-10 text-muted-foreground/40" aria-hidden="true" />
        <h2 className="text-xl font-semibold">Knowledge Base</h2>
        <p className="max-w-sm text-sm text-muted-foreground">Attach documents with the paperclip, then start asking questions.</p>
        <Button onClick={() => createConversation('rag')} className="gap-2"><Plus className="h-4 w-4" aria-hidden="true" /> New conversation</Button>
      </div>
    );
  }

  const msgs = conv.messages;
  return (
    <div className="flex h-full flex-col overflow-hidden" aria-label="Knowledge Base">
      <div className="flex flex-wrap items-center gap-3 border-b border-border bg-muted/30 px-4 py-2.5">
        <div className="flex items-center gap-2">
          <Switch id="rag-multi-agent" checked={multiAgent} onCheckedChange={setMultiAgent} className="scale-90" />
          <Label htmlFor="rag-multi-agent" className="text-xs text-muted-foreground cursor-pointer">Multi-agent</Label>
        </div>
        <div className="flex items-center gap-2">
          <Switch id="rag-hybrid" checked={hybridSearch} onCheckedChange={setHybridSearch} className="scale-90" />
          <Label htmlFor="rag-hybrid" className="text-xs text-muted-foreground cursor-pointer">Hybrid search</Label>
        </div>
        <div className="flex items-center gap-2">
          <Label htmlFor="rag-n-results" className="text-xs text-muted-foreground">Results</Label>
          <Input id="rag-n-results" type="number" min={1} max={20} value={nResults} onChange={(e) => setNResults(parseInt(e.target.value) || 5)} className="h-7 w-14 text-xs" />
        </div>
      </div>

      <div className="flex flex-1 flex-col overflow-y-auto min-h-0">
        <div className="flex flex-col gap-4 px-4 py-4 md:px-6">
          {msgs.length === 0 && (
            <div className="flex flex-col items-center justify-center gap-2 py-20 text-center">
              <Database className="h-10 w-10 text-muted-foreground/40" aria-hidden="true" />
              <p className="text-sm font-medium text-muted-foreground">Ask questions about your documents</p>
              <p className="text-xs text-muted-foreground">Attach documents with the paperclip to start querying.</p>
            </div>
          )}
          {msgs.map((m, i) => (
            <div key={m.id ?? i} className={cn('flex', m.role === 'user' ? 'justify-end' : 'justify-start')}>
              <div className={cn('max-w-[75%] rounded-2xl px-4 py-3 text-sm', m.role === 'user' ? 'rounded-br-sm bg-primary text-primary-foreground' : 'rounded-bl-sm border border-border bg-card')}>
                <RagMessage msg={m} />
                {m.isStreaming && <span className="ml-0.5 inline-block h-4 w-0.5 animate-pulse bg-foreground align-middle" aria-hidden="true" />}
              </div>
            </div>
          ))}
          <div ref={bottomRef} aria-hidden="true" />
        </div>
      </div>

      <div
        className={cn(
          'border-t border-border bg-background px-4 py-3 md:px-6',
          dragActive && 'ring-2 ring-inset ring-primary/30',
        )}
        onDragEnter={(e) => {
          if (e.dataTransfer.types.includes('Files')) {
            dragDepth.current += 1;
            setDragActive(true);
          }
        }}
        onDragOver={(e) => {
          if (e.dataTransfer.types.includes('Files')) e.preventDefault();
        }}
        onDragLeave={() => {
          dragDepth.current = Math.max(0, dragDepth.current - 1);
          if (dragDepth.current === 0) setDragActive(false);
        }}
        onDrop={(e) => {
          e.preventDefault();
          dragDepth.current = 0;
          setDragActive(false);
          if (e.dataTransfer.files?.length) void upload(e.dataTransfer.files);
        }}
      >
        {currentOverride && (
          <ProviderOverrideBadge
            override={currentOverride}
            onRemove={() => setQuery('')}
            onSetKey={() => {
              if (!currentOverride.hasKey) {
                setKeyPromptOpen(true);
                setPendingOverride(currentOverride);
              }
            }}
          />
        )}
        <div className="flex items-end gap-2">
          <RagAttachButton
            onFiles={(files) => void upload(files)}
            uploading={uploading}
            disabled={queryLoading}
          />
          <Textarea value={query} onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void handleQuery(); } }}
            placeholder="Ask about your documents…" disabled={queryLoading} rows={2} className="flex-1 resize-none text-sm" aria-label="RAG query input" />
          <Button size="sm" onClick={() => void handleQuery()} disabled={!query.trim() || queryLoading} className="h-[60px] px-4" aria-label={queryLoading ? 'Searching…' : 'Submit query'}>
            {queryLoading ? '…' : 'Ask'}
          </Button>
        </div>
        {(uploading || error) && (
          <div className="mt-2 flex items-center gap-2 text-xs" aria-live="polite">
            {uploading && (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" aria-hidden="true" />
                <span className="text-muted-foreground">Uploading… {progress}%</span>
              </>
            )}
            {error && (
              <span className="text-destructive" role="alert">{error}</span>
            )}
          </div>
        )}
      </div>

      <ProviderKeyPrompt
        open={keyPromptOpen}
        provider={pendingOverride?.backend ?? 'ollama'}
        model={pendingOverride?.model ?? ''}
        onSetKey={handleSetKey}
        onCancel={handleKeyPromptCancel}
        onFallback={handleKeyPromptFallback}
      />
    </div>
  );
}
