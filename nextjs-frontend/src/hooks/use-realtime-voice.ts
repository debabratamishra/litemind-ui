'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { getVoiceSession, getSarvamSession } from '@/lib/api';

// Type-only import — both SDKs are loaded dynamically to keep them out of the SSR bundle.
import type { VoiceConversation } from '@elevenlabs/client';
import type { ConversationAgent } from 'sarvam-conv-ai-sdk/browser';

export type VoiceState = 'idle' | 'connecting' | 'listening' | 'speaking' | 'error';

export interface VoiceCallbacks {
  onReady?: () => void;
  onUserTranscript?: (text: string) => void;
  onAssistantText?: (text: string) => void;
  onAssistantEnd?: () => void;
  onError?: (message: string) => void;
  onEnded?: () => void;
}

export interface VoiceRequestSettings {
  voiceProvider?: 'elevenlabs' | 'sarvam' | null;
  // ElevenLabs
  agentId?: string | null;
  apiKey?: string | null;
  // unused legacy fields kept for interface stability
  model?: string | null;
  backend?: string | null;
  temperature?: number;
  maxTokens?: number;
  voice?: string | null;
}

export function useRealtimeVoice(settings: VoiceRequestSettings, callbacks: VoiceCallbacks) {
  const [state, setState] = useState<VoiceState>('idle');

  // One ref per provider — only one is ever populated at a time.
  const elevenRef = useRef<VoiceConversation | null>(null);
  const sarvamRef = useRef<ConversationAgent | null>(null);

  const settingsRef = useRef(settings);
  const callbacksRef = useRef(callbacks);
  useEffect(() => {
    settingsRef.current = settings;
    callbacksRef.current = callbacks;
  }, [settings, callbacks]);

  const cleanup = useCallback(() => {
    if (elevenRef.current) {
      try { elevenRef.current.endSession(); } catch { /* ignore */ }
      elevenRef.current = null;
    }
    if (sarvamRef.current) {
      try { sarvamRef.current.stop(); } catch { /* ignore */ }
      sarvamRef.current = null;
    }
    setState('idle');
  }, []);

  const stop = useCallback(() => {
    cleanup();
    callbacksRef.current.onEnded?.();
  }, [cleanup]);

  // ── ElevenLabs session ────────────────────────────────────────────────────
  const startElevenLabs = useCallback(async () => {
    const { agentId, apiKey } = settingsRef.current;
    const session = await getVoiceSession(agentId ?? undefined, apiKey ?? undefined);

    const { Conversation } = await import('@elevenlabs/client');

    const conversation = await Conversation.startSession({
      signedUrl: session.signed_url,
      connectionType: 'websocket',

      onConnect: () => {
        setState('listening');
        callbacksRef.current.onReady?.();
      },

      onDisconnect: () => {
        elevenRef.current = null;
        setState('idle');
        callbacksRef.current.onEnded?.();
      },

      onError: (message: string) => {
        setState('error');
        callbacksRef.current.onError?.(message);
      },

      onModeChange: ({ mode }: { mode: 'speaking' | 'listening' }) => {
        setState(mode === 'speaking' ? 'speaking' : 'listening');
      },

      onMessage: ({ role, message }: { role: 'user' | 'agent'; message: string }) => {
        if (role === 'user') {
          callbacksRef.current.onUserTranscript?.(message);
        } else {
          callbacksRef.current.onAssistantText?.(message);
          callbacksRef.current.onAssistantEnd?.();
        }
      },
    }) as VoiceConversation;

    elevenRef.current = conversation;
  }, []);

  // ── Sarvam session ────────────────────────────────────────────────────────
  const startSarvam = useCallback(async () => {
    // Fetch resolved credentials from our backend — API key never lives in the store.
    const creds = await getSarvamSession();

    const {
      ConversationAgent,
      BrowserAudioInterface,
      InteractionType,
      AgentState,
    } = await import('sarvam-conv-ai-sdk/browser');

    const audioInterface = new BrowserAudioInterface(16000);

    const agent = new ConversationAgent({
      apiKey: creds.api_key,
      config: {
        org_id: creds.org_id,
        workspace_id: creds.workspace_id,
        app_id: creds.app_id,
        user_identifier: `litemind-${Date.now()}`,
        user_identifier_type: 'custom',
        interaction_type: InteractionType.CALL,
        input_sample_rate: 16000,
        output_sample_rate: 16000,
      },
      audioInterface,

      stateCallback: (newState: typeof AgentState[keyof typeof AgentState]) => {
        switch (newState) {
          case AgentState.LISTENING:
            setState('listening');
            break;
          case AgentState.SPEAKING:
            setState('speaking');
            break;
          case AgentState.CONNECTING:
          case AgentState.CONNECTED:
            setState('connecting');
            break;
          case AgentState.ERROR:
            setState('error');
            break;
          default:
            setState('idle');
        }
      },

      startCallback: async () => {
        setState('listening');
        callbacksRef.current.onReady?.();
      },

      // Role enum values: Role.USER = 'user', Role.BOT = 'bot'.
      // Compare against raw string literals to avoid dynamic-import closure issues
      // where the captured Role reference can be undefined when the callback fires.
      transcriptCallback: async (msg: { role: string; content: string }) => {
        if (msg.role === 'user') {
          callbacksRef.current.onUserTranscript?.(msg.content);
        } else {
          callbacksRef.current.onAssistantText?.(msg.content);
          callbacksRef.current.onAssistantEnd?.();
        }
      },

      endCallback: async () => {
        sarvamRef.current = null;
        setState('idle');
        callbacksRef.current.onEnded?.();
      },
    });

    sarvamRef.current = agent;
    await agent.start();
  }, []);

  // ── Unified start ─────────────────────────────────────────────────────────
  const start = useCallback(async () => {
    if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
      setState('error');
      callbacksRef.current.onError?.('Voice mode is not supported in this browser.');
      return;
    }

    setState('connecting');

    try {
      const provider = settingsRef.current.voiceProvider ?? 'elevenlabs';
      if (provider === 'sarvam') {
        await startSarvam();
      } else {
        await startElevenLabs();
      }
    } catch (err) {
      setState('error');
      callbacksRef.current.onError?.(
        err instanceof Error ? err.message : 'Failed to start voice session.'
      );
      cleanup();
    }
  }, [startElevenLabs, startSarvam, cleanup]);

  useEffect(() => () => cleanup(), [cleanup]);

  return { state, start, stop, isConnected: state !== 'idle' && state !== 'error' };
}
