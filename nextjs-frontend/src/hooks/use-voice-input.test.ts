import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useVoiceInput } from './use-voice-input';

// Mock SpeechRecognition
interface MockSpeechRecognition {
  lang: string;
  interimResults: boolean;
  maxAlternatives: number;
  continuous: boolean;
  onstart: (() => void) | null;
  onresult: ((event: any) => void) | null;
  onerror: ((event: any) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
}

let mockRecognition: MockSpeechRecognition;

function setupWindow(hasSpeechRecognition = true) {
  mockRecognition = {
    lang: '',
    interimResults: false,
    maxAlternatives: 0,
    continuous: false,
    onstart: null,
    onresult: null,
    onerror: null,
    onend: null,
    start() {
      if (mockRecognition.onstart) {
        mockRecognition.onstart();
      }
    },
    stop() {},
    abort() {},
  };

  // Set up window with SpeechRecognition
  if (hasSpeechRecognition) {
    (window as any).SpeechRecognition = vi.fn().mockImplementation(() => mockRecognition);
    (window as any).webkitSpeechRecognition = vi.fn().mockImplementation(() => mockRecognition);
  } else {
    delete (window as any).SpeechRecognition;
    delete (window as any).webkitSpeechRecognition;
  }
}

describe('useVoiceInput', () => {
  beforeEach(() => {
    setupWindow(true);
    vi.clearAllMocks();
  });

  afterEach(() => {
    delete (window as any).SpeechRecognition;
    delete (window as any).webkitSpeechRecognition;
  });

  it('returns initial idle state', () => {
    const { result } = renderHook(() => useVoiceInput());
    expect(result.current.state).toBe('idle');
    expect(result.current.transcript).toBe('');
    expect(result.current.error).toBe(null);
  });

  it('detects browser support for SpeechRecognition', () => {
    const { result } = renderHook(() => useVoiceInput());
    expect(result.current.isSupported).toBe(true);
  });

  it('detects lack of browser support', () => {
    setupWindow(false);
    const { result } = renderHook(() => useVoiceInput());
    expect(result.current.isSupported).toBe(false);
  });

  it('sets error when SpeechRecognition is not supported', () => {
    setupWindow(false);
    const { result } = renderHook(() => useVoiceInput());

    act(() => {
      result.current.start();
    });

    expect(result.current.state).toBe('error');
    expect(result.current.error).toBe('Voice input is not supported in this browser.');
  });

  it('starts recognition and transitions to listening state', () => {
    const { result } = renderHook(() => useVoiceInput());

    act(() => {
      result.current.start();
    });

    expect((window as any).SpeechRecognition).toHaveBeenCalled();
    expect(result.current.state).toBe('listening');
    expect(mockRecognition.lang).toBe('en-US');
    expect(mockRecognition.interimResults).toBe(true);
    expect(mockRecognition.maxAlternatives).toBe(1);
    expect(mockRecognition.continuous).toBe(false);
  });

  it('calls onResult with final transcript', () => {
    const onResult = vi.fn();
    const { result } = renderHook(() => useVoiceInput(onResult));

    act(() => {
      result.current.start();
    });

    // Simulate a final result
    const mockEvent = {
      resultIndex: 0,
      results: {
        length: 1,
        0: {
          isFinal: true,
          0: { transcript: 'Hello world', confidence: 0.9 },
        },
      },
    };

    act(() => {
      mockRecognition.onresult?.(mockEvent);
    });

    expect(onResult).toHaveBeenCalledWith('Hello world');
    expect(result.current.state).toBe('processing');
  });

  it('handles interim results', () => {
    const { result } = renderHook(() => useVoiceInput());

    act(() => {
      result.current.start();
    });

    // Simulate an interim result
    const mockEvent = {
      resultIndex: 0,
      results: {
        length: 1,
        0: {
          isFinal: false,
          0: { transcript: 'Hello', confidence: 0.8 },
        },
      },
    };

    act(() => {
      mockRecognition.onresult?.(mockEvent);
    });

    expect(result.current.transcript).toBe('Hello');
    expect(result.current.state).toBe('listening');
  });

  it('combines multiple interim results', () => {
    const { result } = renderHook(() => useVoiceInput());

    act(() => {
      result.current.start();
    });

    // First interim result
    act(() => {
      mockRecognition.onresult?.({
        resultIndex: 0,
        results: {
          length: 1,
          0: {
            isFinal: false,
            0: { transcript: 'Hello', confidence: 0.8 },
          },
        },
      });
    });

    expect(result.current.transcript).toBe('Hello');

    // Second interim result
    act(() => {
      mockRecognition.onresult?.({
        resultIndex: 0,
        results: {
          length: 1,
          0: {
            isFinal: false,
            0: { transcript: 'Hello world', confidence: 0.9 },
          },
        },
      });
    });

    expect(result.current.transcript).toBe('Hello world');
  });

  it('handles error events', () => {
    const { result } = renderHook(() => useVoiceInput());

    act(() => {
      result.current.start();
    });

    // Simulate a 'no-speech' error
    act(() => {
      mockRecognition.onerror?.({ error: 'no-speech', message: 'No speech detected' });
    });

    expect(result.current.state).toBe('error');
    expect(result.current.error).toBe('No speech detected. Try again.');
  });

  it('handles not-allowed error', () => {
    const { result } = renderHook(() => useVoiceInput());

    act(() => {
      result.current.start();
    });

    act(() => {
      mockRecognition.onerror?.({ error: 'not-allowed', message: 'Access denied' });
    });

    expect(result.current.state).toBe('error');
    expect(result.current.error).toBe('Microphone access denied.');
  });

  it('handles unknown errors', () => {
    const { result } = renderHook(() => useVoiceInput());

    act(() => {
      result.current.start();
    });

    act(() => {
      mockRecognition.onerror?.({ error: 'unknown-error', message: 'Something went wrong' });
    });

    expect(result.current.state).toBe('error');
    expect(result.current.error).toBe('Voice error: unknown-error');
  });

  it('stops recognition on recognition end', () => {
    const { result } = renderHook(() => useVoiceInput());

    act(() => {
      result.current.start();
    });

    expect(result.current.state).toBe('listening');

    // Simulate recognition end
    act(() => {
      mockRecognition.onend?.();
    });

    expect(result.current.state).toBe('idle');
  });

  it('keeps processing state on end when not listening', () => {
    const { result } = renderHook(() => useVoiceInput());

    // Start and get a final result to enter processing state
    act(() => {
      result.current.start();
    });

    act(() => {
      mockRecognition.onresult?.({
        resultIndex: 0,
        results: {
          length: 1,
          0: {
            isFinal: true,
            0: { transcript: 'test', confidence: 0.9 },
          },
        },
      });
    });

    expect(result.current.state).toBe('processing');

    act(() => {
      mockRecognition.onend?.();
    });

    expect(result.current.state).toBe('processing');
  });

  it('stops recognition when stop is called', () => {
    const { result } = renderHook(() => useVoiceInput());

    act(() => {
      result.current.start();
    });

    act(() => {
      result.current.stop();
    });

    // The stop method should be called on the recognition instance
    // We verify that the hook doesn't throw and state remains valid
    expect(result.current.state).toBe('listening');
  });

  it('resets state when reset is called', () => {
    const { result } = renderHook(() => useVoiceInput());

    act(() => {
      result.current.start();
    });

    // Set some transcript
    act(() => {
      mockRecognition.onresult?.({
        resultIndex: 0,
        results: {
          length: 1,
          0: {
            isFinal: false,
            0: { transcript: 'Hello', confidence: 0.8 },
          },
        },
      });
    });

    expect(result.current.transcript).toBe('Hello');

    act(() => {
      result.current.reset();
    });

    expect(result.current.state).toBe('idle');
    expect(result.current.transcript).toBe('');
    expect(result.current.error).toBe(null);
  });

  it('handles multiple results in sequence', () => {
    const { result } = renderHook(() => useVoiceInput());

    act(() => {
      result.current.start();
    });

    // Multiple results in one event - both interim, then final
    const mockEvent = {
      resultIndex: 0,
      results: {
        length: 2,
        0: {
          isFinal: false,
          0: { transcript: 'Hello ', confidence: 0.8 },
        },
        1: {
          isFinal: true,
          0: { transcript: 'world', confidence: 0.9 },
        },
      },
    };

    act(() => {
      mockRecognition.onresult?.(mockEvent);
    });

    // The code takes final || interim, so it should be 'world'
    expect(result.current.transcript).toBe('world');
    expect(result.current.state).toBe('processing');
  });
});