import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useRealtimeVoice } from "./use-realtime-voice";

function mockBrowser() {
  const wsHandlers: Record<string, ((e: any) => void)> = {};
  class FakeWebSocket {
    static OPEN = 1;
    readyState = 1;
    send = vi.fn();
    close = vi.fn().mockImplementation(() => {
      this.readyState = 3;
      wsHandlers["close"]?.({});
    });
    set onopen(fn: any) {
      wsHandlers["open"] = fn;
      queueMicrotask(() => {
        if (this.readyState === 1 && wsHandlers["open"] === fn) {
          fn({});
        }
      });
    }
    set onmessage(fn: any) {
      wsHandlers["message"] = fn;
    }
    set onerror(fn: any) {
      wsHandlers["error"] = fn;
    }
    set onclose(fn: any) {
      wsHandlers["close"] = fn;
    }
    constructor(public url: string) {}
  }
  (globalThis as any).WebSocket = FakeWebSocket;

  const mockAudioContext = {
    currentTime: 0,
    createMediaStreamSource: vi.fn().mockReturnValue({ connect: vi.fn() }),
    createScriptProcessor: vi.fn().mockReturnValue({
      connect: vi.fn(),
      disconnect: vi.fn(),
      onaudioprocess: null,
    }),
    createGain: vi.fn().mockReturnValue({
      gain: { value: 1 },
      connect: vi.fn(),
    }),
    createBuffer: vi.fn().mockReturnValue({
      duration: 1,
      getChannelData: vi.fn().mockReturnValue(new Float32Array(100)),
    }),
    createBufferSource: vi.fn().mockReturnValue({
      connect: vi.fn(),
      start: vi.fn(),
      stop: vi.fn(),
      onended: null,
    }),
    destination: {},
    close: vi.fn().mockResolvedValue(undefined),
  };
  (globalThis as any).AudioContext = vi.fn().mockImplementation(() => mockAudioContext);

  (globalThis as any).navigator = {
    mediaDevices: { getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [] }) },
  };

  (globalThis as any).fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ signed_url: "wss://api.elevenlabs.io/v1/convai/conversation?signed=true", agent_id: "agent-123" }),
  });

  return { wsHandlers, mockAudioContext };
}

describe("useRealtimeVoice", () => {
  beforeEach(() => mockBrowser());

  it("transitions to listening on successful ElevenLabs ConvAI connection", async () => {
    const { result } = renderHook(() => useRealtimeVoice({}, {}));
    await act(async () => {
      await result.current.start();
    });
    expect(result.current.state).toBe("listening");
    expect((globalThis as any).fetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/voice/session"),
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("fires onAssistantText when an agent_response event arrives", async () => {
    const env = mockBrowser();
    const onAssistantText = vi.fn();
    const { result } = renderHook(() =>
      useRealtimeVoice({}, { onAssistantText }),
    );
    await act(async () => {
      await result.current.start();
    });
    await act(async () => {
      env.wsHandlers["message"]?.({
        data: JSON.stringify({
          type: "agent_response",
          agent_response_event: { agent_response: "Hi there from ElevenLabs" },
        }),
      });
    });
    expect(onAssistantText).toHaveBeenCalledWith("Hi there from ElevenLabs");
  });

  it("fires onUserTranscript when a user_transcript event arrives", async () => {
    const env = mockBrowser();
    const onUserTranscript = vi.fn();
    const { result } = renderHook(() =>
      useRealtimeVoice({}, { onUserTranscript }),
    );
    await act(async () => {
      await result.current.start();
    });
    await act(async () => {
      env.wsHandlers["message"]?.({
        data: JSON.stringify({
          type: "user_transcript",
          user_transcription_event: { user_transcript: "Hello assistant" },
        }),
      });
    });
    expect(onUserTranscript).toHaveBeenCalledWith("Hello assistant");
  });
});
