'use client';

import * as React from 'react';
import { Globe, Database, Plus, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { Separator } from '@/components/ui/separator';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import MarkdownRenderer from '@/components/markdown-renderer';
import { RagAttachButton } from '@/components/rag-attach-button';
import { useAppStore, selectActiveConversation, selectActiveId, selectSettings } from '@/lib/store';
import { streamRagQuery } from '@/lib/api';
import { useRagUpload } from '@/hooks/use-rag-upload';
import { cn } from '@/lib/utils';

export default function RagPage() {
  const conv = useAppStore(selectActiveConversation);
  const activeId = useAppStore(selectActiveId);
  const settings = useAppStore(selectSettings);
  const { addMessage, updateLastMessage, setWebSearch, createConversation } = useAppStore();

  const [multiAgent, setMultiAgent] = React.useState(false);
  const [hybridSearch, setHybridSearch] = React.useState(true);
  const [nResults, setNResults] = React.useState(5);
  const [query, setQuery] = React.useState('');
  const [queryLoading, setQueryLoading] = React.useState(false);
  const abortRef = React.useRef<AbortController | null>(null);
  const bottomRef = React.useRef<HTMLDivElement>(null);
  const { upload, uploading, progress, error } = useRagUpload();
  const dragDepth = React.useRef(0);
  const [dragActive, setDragActive] = React.useState(false);

  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conv?.messages]);

  const handleQuery = async () => {
    const q = query.trim();
    const convId = activeId;
    if (!q || queryLoading || !convId) return;
    const webSearch = useAppStore.getState().conversations.find((c) => c.id === convId)?.webSearch ?? false;
    setQuery('');
    setQueryLoading(true);

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      addMessage(convId, { role: 'user', content: q });
      addMessage(convId, { role: 'assistant', content: '', isStreaming: true });
      let accumulated = '';
      const stop = settings.stopSequences
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean);

      const stream = streamRagQuery(
        {
          query: q,
          model: settings.model,
          session_id: settings.sessionId,
          temperature: settings.temperature,
          max_tokens: settings.maxTokens,
          top_p: settings.topP,
          frequency_penalty: settings.frequencyPenalty,
          repetition_penalty: settings.repetitionPenalty,
          min_p: settings.minP,
          seed: settings.seed,
          stop: stop.length ? stop : undefined,
          serp_api_key: settings.serpApiKey ?? null,
          top_k: nResults,
          use_web_search: webSearch,
        },
        controller.signal,
      );
      for await (const chunk of stream) { accumulated += chunk; updateLastMessage(convId, accumulated, true); }
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
        <Separator orientation="vertical" className="h-5" />
        <Tooltip>
          <TooltipTrigger render={<Button variant={conv.webSearch ? 'default' : 'outline'} size="sm" className="gap-1.5" onClick={() => setWebSearch(activeId, !conv.webSearch)} aria-label="Toggle web search" aria-pressed={conv.webSearch}><Globe className="h-4 w-4" aria-hidden="true" />Web</Button>} />
          <TooltipContent>Combine docs with web search</TooltipContent>
        </Tooltip>
      </div>

      <div className="flex flex-1 flex-col overflow-y-auto min-h-0">
        <div className="flex flex-col gap-4 px-4 py-4 md:px-6">
          {msgs.length === 0 && (
            <div className="flex flex-col items-center justify-center gap-2 py-20 text-center">
              <Database className="h-10 w-10 text-muted-foreground/40" aria-hidden="true" />
              <p className="text-sm font-medium text-muted-foreground">Ask questions about your documents</p>
              <p className="text-xs text-muted-foreground">{conv.webSearch ? 'Web search is on — answers may also use the web.' : 'Attach documents with the paperclip to start querying.'}</p>
            </div>
          )}
          {msgs.map((m, i) => (
            <div key={m.id ?? i} className={cn('flex', m.role === 'user' ? 'justify-end' : 'justify-start')}>
              <div className={cn('max-w-[75%] rounded-2xl px-4 py-3 text-sm', m.role === 'user' ? 'rounded-br-sm bg-primary text-primary-foreground' : 'rounded-bl-sm border border-border bg-card')}>
                {m.role === 'user' ? <p className="whitespace-pre-wrap leading-relaxed">{m.content}</p>
                  : <MarkdownRenderer content={m.content} />}
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
    </div>
  );
}
