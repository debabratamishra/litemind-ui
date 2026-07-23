'use client';

import * as React from 'react';
import {
  Send, Globe, Mic, MicOff, Bot, Code, Sparkles, Plus, Phone, PhoneOff, MoreVertical,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem } from '@/components/ui/dropdown-menu';
import MessageBubble from '@/components/message-bubble';
import VoiceActivityIndicator from '@/components/voice-activity';
import { useAppStore, selectActiveConversation, selectActiveId, selectSettings } from '@/lib/store';
import { streamChat, streamWebSearch } from '@/lib/api';
import type { ChatMessage } from '@/lib/types';
import { cn } from '@/lib/utils';
import { useVoiceInput } from '@/hooks/use-voice-input';
import { useRealtimeVoice } from '@/hooks/use-realtime-voice';
import { parseProviderOverride } from '@/hooks/use-provider-override';
import { ProviderOverrideBadge } from '@/components/provider-override-badge';
import { ProviderKeyPrompt } from '@/components/provider-key-prompt';
import type { ProviderOverride } from '@/lib/types';

const DISPLAY_MODES = [
  { mode: 'rendered', label: 'Rendered', Icon: Sparkles },
  { mode: 'code', label: 'Code', Icon: Code },
] as const;

function ModeSelector({ mode, onChange }: { mode: 'rendered' | 'code'; onChange: (m: 'rendered' | 'code') => void }) {
  return (
    <div className="flex items-center gap-0.5 rounded-lg border border-border bg-muted/40 p-0.5" role="group" aria-label="Generative UI display mode">
      {DISPLAY_MODES.map(({ mode: m, label, Icon }) => (
        <Button key={m} type="button" variant={mode === m ? 'default' : 'ghost'} size="sm" className="h-7 gap-1.5 px-2.5 text-xs" aria-pressed={mode === m} onClick={() => onChange(m)}>
          <Icon className="h-3.5 w-3.5" aria-hidden="true" />{label}
        </Button>
      ))}
    </div>
  );
}

export default function ChatPage() {
  const conv = useAppStore(selectActiveConversation);
  const activeId = useAppStore(selectActiveId);
  const settings = useAppStore(selectSettings);
  const { addMessage, updateLastMessage, setWebSearch, clearConversation, createConversation } = useAppStore();

  const [input, setInput] = React.useState('');
  const [isStreaming, setIsStreaming] = React.useState(false);
  const [voiceOn, setVoiceOn] = React.useState(false);
  const [keyPromptOpen, setKeyPromptOpen] = React.useState(false);
  const [pendingOverride, setPendingOverride] = React.useState<ProviderOverride | null>(null);
  const abortRef = React.useRef<AbortController | null>(null);
  const bottomRef = React.useRef<HTMLDivElement>(null);
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);

  // Compute the override live as the user types so the badge can be shown.
  const currentOverride = React.useMemo(
    () => parseProviderOverride(input, settings),
    [input, settings],
  );

  const { state: voiceState, isSupported: voiceSupported, start: startVoice, stop: stopVoice } = useVoiceInput(
    (transcript) => { setInput(transcript); setVoiceOn(false); setTimeout(() => handleSend(transcript), 50); }
  );

  // ── Realtime voice mode (independent of the browser-dictation Mic above) ──
  const assistantActiveRef = React.useRef(false);
  const convIdRef = React.useRef(activeId);
  convIdRef.current = activeId;

  const voiceCallbacks = React.useMemo(() => ({
    onUserTranscript: (text: string) => {
      const id = convIdRef.current;
      if (id) addMessage(id, { role: 'user', content: text });
    },
    onAssistantText: (delta: string) => {
      const id = convIdRef.current;
      if (!id) return;
      const cv = useAppStore.getState().conversations.find((c) => c.id === id);
      const last = cv?.messages.slice(-1)[0];
      if (!assistantActiveRef.current) {
        addMessage(id, { role: 'assistant', content: delta, isStreaming: true });
        assistantActiveRef.current = true;
      } else if (last && last.role === 'assistant') {
        updateLastMessage(id, last.content + delta, true);
      }
    },
    onAssistantEnd: () => {
      const id = convIdRef.current;
      if (id) {
        const cv = useAppStore.getState().conversations.find((c) => c.id === id);
        const last = cv?.messages.slice(-1)[0];
        if (last && last.role === 'assistant') updateLastMessage(id, last.content, false);
      }
      assistantActiveRef.current = false;
    },
    onError: (message: string) => {
      const id = convIdRef.current;
      if (id) addMessage(id, { role: 'assistant', content: `⚠️ Voice error: ${message}` });
    },
  }), [addMessage, updateLastMessage]);

  const { state: realtimeState, start: startRealtime, stop: stopRealtime, isConnected: realtimeOn } = useRealtimeVoice(settings, voiceCallbacks);

  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conv?.messages]);

  React.useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape' && isStreaming) abortRef.current?.abort(); };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [isStreaming]);

  const handleSend = React.useCallback(async (overrideText?: string) => {
    const text = (overrideText ?? input).trim();
    const convId = activeId;
    if (!text || isStreaming || !convId) return;

    // Parse "@" provider override
    const override = parseProviderOverride(text, settings);

    // If override requires key but none configured, show inline prompt
    if (override && !override.hasKey) {
      setKeyPromptOpen(true);
      setPendingOverride(override);
      return;
    }

    const webSearch = useAppStore.getState().conversations.find((c) => c.id === convId)?.webSearch ?? false;

    // Determine effective settings from override or defaults
    const effectiveBackend = override?.backend ?? settings.backend;
    const effectiveModel = override?.model || settings.model;
    const effectiveApiKey = override?.hasKey
      ? settings.providerKeys[override.backend]
      : settings.apiKey;
    const effectiveText = override?.text ?? text;

    setInput('');
    setIsStreaming(true);
    addMessage(convId, { role: 'user', content: effectiveText });
    addMessage(convId, { role: 'assistant', content: '', isStreaming: true });

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      let accumulated = '';
      const history: ChatMessage[] = useAppStore.getState().conversations
        .find((c) => c.id === convId)?.messages.slice(-20)
        .filter((m) => !m.isStreaming).map((m) => ({ role: m.role, content: m.content })) ?? [];

      const stop = settings.stopSequences
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean);

      const stream = webSearch
        ? streamWebSearch({
            message: effectiveText, model: effectiveModel || undefined, backend: effectiveBackend,
            api_key: effectiveApiKey ?? null, api_base: settings.apiBase ?? null,
            session_id: settings.sessionId, temperature: settings.temperature, max_tokens: settings.maxTokens,
            top_p: settings.topP, frequency_penalty: settings.frequencyPenalty,
            repetition_penalty: settings.repetitionPenalty, top_k: settings.topK, min_p: settings.minP,
            seed: settings.seed, stop: stop.length ? stop : undefined,
            serp_api_key: settings.serpApiKey ?? null,
            conversation_history: history.length ? history : null, use_web_search: true,
          }, controller.signal)
        : streamChat({
            message: effectiveText, model: effectiveModel || undefined, backend: effectiveBackend,
            api_key: effectiveApiKey ?? null, api_base: settings.apiBase ?? null,
            temperature: settings.temperature, max_tokens: settings.maxTokens, top_p: settings.topP,
            frequency_penalty: settings.frequencyPenalty, repetition_penalty: settings.repetitionPenalty,
            top_k: settings.topK, min_p: settings.minP, seed: settings.seed,
            stop: stop.length ? stop : undefined,
            session_id: settings.sessionId, conversation_history: history.length ? history : null,
            is_voice_mode: settings.voiceMode, enable_generative_ui: settings.enableGenerativeUI, stream: true,
          }, controller.signal);

      for await (const chunk of stream) { accumulated += chunk; updateLastMessage(convId, accumulated, true); }
      updateLastMessage(convId, accumulated, false);
    } catch (err) {
      if (!(err instanceof Error && err.name === 'AbortError')) {
        updateLastMessage(convId, `⚠️ Error: ${err instanceof Error ? err.message : 'An error occurred.'}`, false);
      }
    } finally {
      setIsStreaming(false);
      abortRef.current = null;
      setTimeout(() => textareaRef.current?.focus(), 50);
    }
  }, [input, isStreaming, activeId, settings, addMessage, updateLastMessage]);

  // ── Provider override key prompt handlers ─────────────────────────────────
  const handleSetKey = React.useCallback((key: string) => {
    if (pendingOverride) {
      useAppStore.getState().setProviderKey(pendingOverride.backend, key);
    }
    setKeyPromptOpen(false);
    setPendingOverride(null);
  }, [pendingOverride]);

  const handleKeyPromptCancel = React.useCallback(() => {
    setKeyPromptOpen(false);
    setPendingOverride(null);
  }, []);

  const handleKeyPromptFallback = React.useCallback(() => {
    setKeyPromptOpen(false);
    setPendingOverride(null);
    // Fallback to default provider — just clear the input
    setInput('');
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void handleSend(); }
  };

  const handleAction = React.useCallback((action: string, payload?: string) => {
    if (action === 'send_message' && payload) void handleSend(payload);
  }, [handleSend]);

  if (!conv || !activeId) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
        <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10">
          <Bot className="h-7 w-7 text-primary" aria-hidden="true" />
        </div>
        <h2 className="text-xl font-semibold">How can I help you today?</h2>
        <Button onClick={() => createConversation('chat')} className="gap-2">
          <Plus className="h-4 w-4" aria-hidden="true" /> New conversation
        </Button>
      </div>
    );
  }

  const msgs = conv.messages;
  return (
    <div className="flex h-full flex-col overflow-hidden" aria-label="Chat interface">
      <div className="flex h-9 shrink-0 items-center justify-end border-b border-border bg-background px-4 md:px-6">
        <DropdownMenu>
          <DropdownMenuTrigger
            render={
              <Button variant="ghost" size="icon" className="h-8 w-8" aria-label="Conversation options">
                <MoreVertical className="h-4 w-4" aria-hidden="true" />
              </Button>
            }
          />
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={() => activeId && clearConversation(activeId)}>
              Clear conversation
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {settings.enableGenerativeUI && (
        <div className="flex h-11 shrink-0 items-center justify-between border-b border-border bg-background px-4 md:px-6">
          <span className="text-xs font-medium text-muted-foreground">Generative UI</span>
          <ModeSelector mode={settings.genUIDisplayMode} onChange={(m) => useAppStore.getState().setSettings({ genUIDisplayMode: m })} />
        </div>
      )}

      <div className="flex flex-1 flex-col overflow-y-auto min-h-0">
        <div className="flex flex-col gap-4 px-4 py-6 md:px-8">
          {msgs.length === 0 ? (
            <div className="flex flex-1 flex-col items-center justify-center gap-3 py-24 text-center">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10">
                <Bot className="h-7 w-7 text-primary" aria-hidden="true" />
              </div>
              <h2 className="text-xl font-semibold">How can I help you today?</h2>
              <p className="max-w-sm text-sm text-muted-foreground">Ask me anything — I can answer, help you write, analyse documents, or search the web.</p>
            </div>
          ) : (
            msgs.map((msg) => (
              <MessageBubble
                key={msg.id}
                msg={msg}
                settings={settings}
                onAction={handleAction}
              />
            ))
          )}
          <div ref={bottomRef} aria-hidden="true" />
        </div>
      </div>

      <div className="border-t border-border bg-background px-4 py-3 md:px-8">
        {conv.webSearch && (
          <div className="mb-2 flex items-center gap-2">
            <span className="rounded-md bg-secondary px-2 py-0.5 text-xs">🌐 Web search active</span>
            <span className="text-xs text-muted-foreground">(Escape to cancel)</span>
          </div>
        )}
        {realtimeOn && (
          <div className="mb-2 flex items-center gap-2 text-xs text-muted-foreground">
            <VoiceActivityIndicator state={realtimeState} />
            <span>{realtimeState === 'speaking' ? 'Assistant is speaking' : 'Listening…'}</span>
          </div>
        )}
        {currentOverride && (
          <ProviderOverrideBadge
            override={currentOverride}
            onRemove={() => setInput('')}
            onSetKey={() => {
              if (!currentOverride.hasKey) {
                setKeyPromptOpen(true);
                setPendingOverride(currentOverride);
              }
            }}
          />
        )}
        <div className="flex items-end gap-2">
          <Textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={conv.webSearch ? 'Search the web…' : 'Message LiteMindUI…'}
            disabled={isStreaming}
            rows={1}
            className="min-h-[44px] max-h-[200px] flex-1 resize-none overflow-y-auto pr-2 text-sm leading-relaxed"
            aria-label="Chat message input"
          />
          <div className="flex items-center gap-1.5 pb-0.5">
            <Tooltip>
              <TooltipTrigger render={<Button variant={conv.webSearch ? 'default' : 'outline'} size="icon" className="h-9 w-9" onClick={() => setWebSearch(activeId, !conv.webSearch)} aria-label={conv.webSearch ? 'Disable web search' : 'Enable web search'} aria-pressed={conv.webSearch}><Globe className="h-4 w-4" aria-hidden="true" /></Button>} />
              <TooltipContent>Web search</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger
                render={
                  <Button
                    variant={realtimeOn ? 'default' : 'outline'}
                    size="icon"
                    className={cn('h-9 w-9', realtimeOn && 'bg-primary text-primary-foreground')}
                    onClick={() => (realtimeOn ? stopRealtime() : startRealtime())}
                    aria-label={realtimeOn ? 'Stop voice mode' : 'Start voice mode'}
                    aria-pressed={realtimeOn}
                  >
                    {realtimeOn ? <PhoneOff className="h-4 w-4" aria-hidden="true" /> : <Phone className="h-4 w-4" aria-hidden="true" />}
                  </Button>
                }
              />
              <TooltipContent>Voice mode</TooltipContent>
            </Tooltip>
            {voiceSupported && (
              <Tooltip>
                <TooltipTrigger render={<Button variant={voiceOn ? 'default' : 'outline'} size="icon" className={cn('h-9 w-9', voiceOn && 'animate-pulse bg-red-500 text-white')} onClick={() => { if (voiceOn) { stopVoice(); setVoiceOn(false); } else { setVoiceOn(true); startVoice(); } }} aria-label={voiceOn ? 'Stop voice input' : 'Start voice input'} aria-pressed={voiceOn} disabled={voiceState === 'processing'}>{voiceOn ? <MicOff className="h-4 w-4" aria-hidden="true" /> : <Mic className="h-4 w-4" aria-hidden="true" />}</Button>} />
                <TooltipContent>{voiceOn ? 'Stop recording' : 'Voice input'}</TooltipContent>
              </Tooltip>
            )}
            <Button size="icon" className="h-9 w-9" onClick={() => void handleSend()} disabled={!input.trim() || isStreaming} aria-label={isStreaming ? 'Sending…' : 'Send message'}>
              <Send className="h-4 w-4" aria-hidden="true" />
            </Button>
          </div>
        </div>
        <p className="mt-1.5 text-center text-[11px] text-muted-foreground">Enter to send · Shift+Enter for newline · Escape to cancel</p>
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
