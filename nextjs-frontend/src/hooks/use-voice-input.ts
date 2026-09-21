'use client';

import {
  useState,
  useCallback,
  useRef,
  useEffect,
  useSyncExternalStore,
} from 'react';

// ── Minimal type declarations for the Web Speech API ─────────────────────────
// TypeScript's dom lib may not ship full SpeechRecognition types on all
// versions, so we define exactly what we need here.

interface ISpeechRecognitionResult {
  readonly isFinal: boolean;
  [index: number]: { transcript: string; confidence: number };
}

interface ISpeechRecognitionResultList {
  readonly length: number;
  [index: number]: ISpeechRecognitionResult;
}

interface ISpeechRecognitionEvent extends Event {
  readonly resultIndex: number;
  readonly results: ISpeechRecognitionResultList;
}

interface ISpeechRecognitionErrorEvent extends Event {
  readonly error: string;
  readonly message: string;
}

interface ISpeechRecognition extends EventTarget {
  lang: string;
  interimResults: boolean;
  maxAlternatives: number;
  continuous: boolean;
  onstart: (() => void) | null;
  onresult: ((event: ISpeechRecognitionEvent) => void) | null;
  onerror: ((event: ISpeechRecognitionErrorEvent) => void) | null;
  onend: (() => void) | null;
  start(): void;
  stop(): void;
  abort(): void;
}

interface ISpeechRecognitionConstructor {
  new (): ISpeechRecognition;
}

// Augment window to include prefixed variants
declare global {
  interface Window {
    SpeechRecognition?: ISpeechRecognitionConstructor;
    webkitSpeechRecognition?: ISpeechRecognitionConstructor;
  }
}

// ── Hook ─────────────────────────────────────────────────────────────────────

// No-op subscription used by useSyncExternalStore for values that never change
// after hydration (the client snapshot is already final).
const emptySubscribe = () => () => {};

export type VoiceInputState = 'idle' | 'listening' | 'processing' | 'error';

export interface UseVoiceInputReturn {
  state: VoiceInputState;
  transcript: string;
  error: string | null;
  isSupported: boolean;
  start: () => void;
  stop: () => void;
  reset: () => void;
}

export function useVoiceInput(
  onResult?: (transcript: string) => void,
): UseVoiceInputReturn {
  const [state, setState] = useState<VoiceInputState>('idle');
  const [transcript, setTranscript] = useState('');
  const [error, setError] = useState<string | null>(null);
  const recognitionRef = useRef<ISpeechRecognition | null>(null);

  // Detect support client-side only. useSyncExternalStore returns the server
  // snapshot (false) during SSR and the first client render, then the real
  // client value after hydration — avoiding a setState-in-effect while keeping
  // server/client HTML consistent.
  const isSupported = useSyncExternalStore(
    emptySubscribe,
    () => !!(window.SpeechRecognition || window.webkitSpeechRecognition),
    () => false,
  );

  const stop = useCallback(() => {
    recognitionRef.current?.stop();
  }, []);

  const reset = useCallback(() => {
    // Abort (not stop) so the onend handler fires without side-effects,
    // then immediately null-out the ref so the onend handler below is a no-op.
    if (recognitionRef.current) {
      recognitionRef.current.onend = null;
      recognitionRef.current.abort();
      recognitionRef.current = null;
    }
    setTranscript('');
    setError(null);
    setState('idle');
  }, []);

  const start = useCallback(() => {
    if (typeof window === 'undefined') return;

    const SpeechRecognitionClass =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognitionClass) {
      setError('Voice input is not supported in this browser.');
      setState('error');
      return;
    }

    // Safari (webkitSpeechRecognition) throws InvalidStateError if you call
    // start() while a previous instance is still alive. Always abort the old
    // one and detach its handlers before creating a new session.
    if (recognitionRef.current) {
      recognitionRef.current.onend = null;
      recognitionRef.current.onerror = null;
      recognitionRef.current.onresult = null;
      recognitionRef.current.onstart = null;
      recognitionRef.current.abort();
      recognitionRef.current = null;
    }

    // Reset to a clean slate for this turn
    setTranscript('');
    setError(null);
    setState('idle');

    const recognition = new SpeechRecognitionClass();
    recognition.lang = 'en-US';
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;
    recognition.continuous = false;

    recognition.onstart = () => {
      setState('listening');
    };

    recognition.onresult = (event: ISpeechRecognitionEvent) => {
      let interim = '';
      let final = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        const text = result[0].transcript;
        if (result.isFinal) {
          final += text;
        } else {
          interim += text;
        }
      }

      const combined = (final || interim).trim();
      setTranscript(combined);

      if (final) {
        setState('processing');
        onResult?.(final.trim());
      }
    };

    recognition.onerror = (event: ISpeechRecognitionErrorEvent) => {
      // Safari fires an 'aborted' error when the recognition session ends
      // naturally after receiving a final result (it internally calls abort()
      // after stop()). This is not a real error — ignore it so that the state
      // set by onresult ('processing') is preserved and the turn completes
      // cleanly. The same applies to the generic 'abort' error code.
      if (event.error === 'aborted' || event.error === 'abort') return;

      const msg =
        event.error === 'no-speech'
          ? 'No speech detected. Try again.'
          : event.error === 'not-allowed'
            ? 'Microphone access denied.'
            : `Voice error: ${event.error}`;
      setError(msg);
      setState('error');
    };

    recognition.onend = () => {
      // Transition back to idle only if we are still in listening state
      // (i.e. the session ended without a final result — timeout, silence, etc.).
      // If state is 'processing' the turn already succeeded; leave it alone so
      // the caller can observe the completed transcript.
      // If state is 'error' an onerror handler already ran; leave that too.
      setState((prev) => (prev === 'listening' ? 'idle' : prev));
    };

    recognitionRef.current = recognition;
    recognition.start();
  }, [isSupported, onResult]); // eslint-disable-line react-hooks/exhaustive-deps

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.onend = null;
        recognitionRef.current.onerror = null;
        recognitionRef.current.abort();
        recognitionRef.current = null;
      }
    };
  }, []);

  return { state, transcript, error, isSupported, start, stop, reset };
}
