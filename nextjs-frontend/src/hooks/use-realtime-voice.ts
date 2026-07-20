'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

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
  model?: string | null;
  backend?: string | null;
  apiKey?: string | null;
  apiBase?: string | null;
  temperature?: number;
  maxTokens?: number;
  voice?: string | null;
}

export function useRealtimeVoice(settings: VoiceRequestSettings, callbacks: VoiceCallbacks) {
  const [state, setState] = useState<VoiceState>('idle');
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const dcRef = useRef<RTCDataChannel | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const settingsRef = useRef(settings);
  const callbacksRef = useRef(callbacks);
  useEffect(() => {
    settingsRef.current = settings;
    callbacksRef.current = callbacks;
  }, [settings, callbacks]);

  const cleanup = useCallback(() => {
    try { dcRef.current?.close(); } catch { /* ignore */ }
    try { pcRef.current?.close(); } catch { /* ignore */ }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    if (audioRef.current) audioRef.current.srcObject = null;
    pcRef.current = null;
    dcRef.current = null;
    streamRef.current = null;
    setState('idle');
  }, []);

  const stop = useCallback(() => {
    try { dcRef.current?.send(JSON.stringify({ type: 'end' })); } catch { /* ignore */ }
    cleanup();
  }, [cleanup]);

  const start = useCallback(async () => {
    if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
      setState('error');
      callbacksRef.current.onError?.('Voice mode is not supported in this browser.');
      return;
    }
    setState('connecting');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const pc = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
      });
      pcRef.current = pc;

      // MUST be created before the offer, or Pipecat disables the channel.
      const dc = pc.createDataChannel('events');
      dcRef.current = dc;
      dc.onmessage = (e: MessageEvent) => {
        let msg: { type: string; text?: string; message?: string };
        try { msg = JSON.parse(e.data); } catch { return; }
        const cb = callbacksRef.current;
        switch (msg.type) {
          case 'ready': cb.onReady?.(); break;
          case 'user_transcript': setState('listening'); cb.onUserTranscript?.(msg.text ?? ''); break;
          case 'assistant_text': setState('speaking'); cb.onAssistantText?.(msg.text ?? ''); break;
          case 'assistant_end': setState('listening'); cb.onAssistantEnd?.(); break;
          case 'error': setState('error'); cb.onError?.(msg.message ?? ''); break;
          case 'ended': cb.onEnded?.(); cleanup(); break;
        }
      };
      pc.ondatachannel = (e: RTCDataChannelEvent) => { e.channel.onmessage = dc.onmessage; };
      stream.getTracks().forEach((t) => pc.addTrack(t, stream));

      const audio = new Audio();
      audio.autoplay = true;
      audioRef.current = audio;
      pc.ontrack = (e: RTCTrackEvent) => { audio.srcObject = e.streams[0]; };
      pc.onconnectionstatechange = () => {
        if (pc.connectionState === 'failed' || pc.connectionState === 'closed') cleanup();
      };

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      const s = settingsRef.current;
      const resp = await fetch('/api/voice/offer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pc_id: crypto.randomUUID(),
          sdp: offer.sdp,
          type: offer.type,
          model: s.model ?? null,
          backend: s.backend ?? null,
          api_key: s.apiKey ?? null,
          api_base: s.apiBase ?? null,
          temperature: s.temperature ?? 0.7,
          max_tokens: s.maxTokens ?? 512,
          voice: s.voice ?? null,
        }),
      });
      if (!resp.ok) {
        setState('error');
        callbacksRef.current.onError?.('Failed to start voice session.');
        cleanup();
        return;
      }
      const answer = await resp.json();
      await pc.setRemoteDescription(new RTCSessionDescription({ type: answer.type, sdp: answer.sdp }));
      setState('listening');
    } catch (err) {
      setState('error');
      callbacksRef.current.onError?.(err instanceof Error ? err.message : 'Voice start failed.');
      cleanup();
    }
  }, [cleanup]);

  useEffect(() => () => cleanup(), [cleanup]);

  return { state, start, stop, isConnected: state !== 'idle' && state !== 'error' };
}
