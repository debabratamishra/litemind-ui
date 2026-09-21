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

  // ── Safari multi-turn and abort-error scenarios ───────────────────────────

  it('can be started again after a completed turn (multi-turn)', () => {
    const onResult = vi.fn();
    const { result } = renderHook(() => useVoiceInput(onResult));

    // ── Turn 1 ──
    act(() => { result.current.start(); });
    expect(result.current.state).toBe('listening');

    // Final result arrives
    act(() => {
      mockRecognition.onresult?.({
        resultIndex: 0,
        results: { length: 1, 0: { isFinal: true, 0: { transcript: 'First turn', confidence: 0.9 } } },
      });
    });
    expect(result.current.state).toBe('processing');
    expect(onResult).toHaveBeenCalledWith('First turn');

    // Recognition ends
    act(() => { mockRecognition.onend?.(); });
    // state stays 'processing' (not listening at end)
    expect(result.current.state).toBe('processing');

    // ── Turn 2 — start() should succeed without throwing ──
    // A fresh mockRecognition is needed because start() creates a new instance.
    const firstInstance = mockRecognition;
    // Re-install mock so the constructor returns a fresh object
    mockRecognition = {
      ...mockRecognition,
      onstart: null, onresult: null, onerror: null, onend: null,
      start: vi.fn().mockImplementation(() => { mockRecognition.onstart?.(); }),
      stop: vi.fn(),
      abort: vi.fn(),
    };
    (window as any).SpeechRecognition = vi.fn().mockImplementation(() => mockRecognition);
    (window as any).webkitSpeechRecognition = vi.fn().mockImplementation(() => mockRecognition);

    act(() => { result.current.start(); });
    expect(result.current.state).toBe('listening');
    expect(result.current.transcript).toBe('');

    act(() => {
      mockRecognition.onresult?.({
        resultIndex: 0,
        results: { length: 1, 0: { isFinal: true, 0: { transcript: 'Second turn', confidence: 0.9 } } },
      });
    });
    expect(onResult).toHaveBeenCalledWith('Second turn');
    expect(onResult).toHaveBeenCalledTimes(2);

    // The first instance's handlers were detached before abort() was called
    expect(firstInstance.onend).toBeNull();
    expect(firstInstance.onerror).toBeNull();
  });

  it('ignores Safari abort error fired after a natural recognition end', () => {
    const { result } = renderHook(() => useVoiceInput());

    act(() => { result.current.start(); });

    // Final result → state becomes 'processing'
    act(() => {
      mockRecognition.onresult?.({
        resultIndex: 0,
        results: { length: 1, 0: { isFinal: true, 0: { transcript: 'Hello Safari', confidence: 0.9 } } },
      });
    });
    expect(result.current.state).toBe('processing');

    // Safari fires onerror with 'aborted' after a natural session end
    act(() => { mockRecognition.onerror?.({ error: 'aborted', message: '' }); });
    // Must stay 'processing', not flip to 'error'
    expect(result.current.state).toBe('processing');
    expect(result.current.error).toBeNull();

    // Same for the 'abort' variant
    act(() => { mockRecognition.onerror?.({ error: 'abort', message: '' }); });
    expect(result.current.state).toBe('processing');
    expect(result.current.error).toBeNull();
  });

  it('abort error during listening (not after final result) is also ignored', () => {
    // Some Safari versions fire abort when you call stop() manually.
    // We should not show an error in that case either.
    const { result } = renderHook(() => useVoiceInput());

    act(() => { result.current.start(); });
    expect(result.current.state).toBe('listening');

    act(() => { mockRecognition.onerror?.({ error: 'aborted', message: '' }); });
    // Should not transition to 'error'
    expect(result.current.state).toBe('listening');
    expect(result.current.error).toBeNull();
  });

  it('start() aborts and detaches the previous instance before creating a new one', () => {
    const { result } = renderHook(() => useVoiceInput());

    act(() => { result.current.start(); });
    const firstInstance = mockRecognition;
    const abortSpy = vi.spyOn(firstInstance, 'abort');

    // Wire up a second mock instance
    mockRecognition = {
      ...firstInstance,
      onstart: null, onresult: null, onerror: null, onend: null,
      start: vi.fn().mockImplementation(() => { mockRecognition.onstart?.(); }),
      stop: vi.fn(),
      abort: vi.fn(),
    };
    (window as any).SpeechRecognition = vi.fn().mockImplementation(() => mockRecognition);
    (window as any).webkitSpeechRecognition = vi.fn().mockImplementation(() => mockRecognition);

    act(() => { result.current.start(); });

    // Old instance must have been aborted and its handlers nulled
    expect(abortSpy).toHaveBeenCalledTimes(1);
    expect(firstInstance.onend).toBeNull();
    expect(firstInstance.onerror).toBeNull();
    expect(firstInstance.onresult).toBeNull();
    expect(firstInstance.onstart).toBeNull();
    // New session is listening
    expect(result.current.state).toBe('listening');
  });

  it('reset() nulls the ref so orphaned onend does not fire', () => {
    const { result } = renderHook(() => useVoiceInput());

    act(() => { result.current.start(); });
    const instance = mockRecognition;

    act(() => { result.current.reset(); });
    expect(result.current.state).toBe('idle');

    // Simulate the browser firing onend after the abort — should be a no-op
    // because reset() detached the handler before calling abort()
    act(() => { instance.onend?.(); });
    expect(result.current.state).toBe('idle');
  });
});