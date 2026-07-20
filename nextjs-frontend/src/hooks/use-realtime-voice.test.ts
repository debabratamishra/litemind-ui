import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useRealtimeVoice } from "./use-realtime-voice";

function mockBrowser() {
  const dataChannelHandlers: Record<string, ((e: { data: string }) => void)> = {};
  const dc = {
    send: vi.fn(),
    close: vi.fn(),
    set onmessage(fn: any) {
      dataChannelHandlers["message"] = fn;
    },
    get onmessage() {
      return dataChannelHandlers["message"];
    },
  };
  const pc = {
    createDataChannel: () => dc,
    createOffer: vi.fn().mockResolvedValue({ sdp: "offer-sdp", type: "offer" }),
    setLocalDescription: vi.fn().mockResolvedValue(undefined),
    setRemoteDescription: vi.fn().mockResolvedValue(undefined),
    addTrack: vi.fn(),
    close: vi.fn(),
    ondatachannel: null as any,
    ontrack: null as any,
    onconnectionstatechange: null as any,
    connectionState: "connected",
  };
  (globalThis as any).RTCPeerConnection = vi.fn().mockReturnValue(pc);
  (globalThis as any).RTCSessionDescription = class {
    constructor(public init: any) {}
  };
  (globalThis as any).navigator = {
    mediaDevices: { getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [] }) },
  };
  if (!(globalThis as any).crypto?.randomUUID) {
    Object.defineProperty(globalThis, "crypto", {
      value: { randomUUID: () => "pc-123" },
      configurable: true,
      writable: true,
    });
  }
  (globalThis as any).fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ sdp: "answer-sdp", type: "answer", pc_id: "pc-123" }),
  });
  (globalThis as any).Audio = class {
    autoplay = false;
    srcObject: any = null;
  };
  return { pc, dc, dataChannelHandlers };
}

describe("useRealtimeVoice", () => {
  beforeEach(() => mockBrowser());

  it("transitions to listening on a successful offer", async () => {
    const { result } = renderHook(() => useRealtimeVoice({}, {}));
    await act(async () => {
      await result.current.start();
    });
    expect(result.current.state).toBe("listening");
    expect((globalThis as any).fetch).toHaveBeenCalledWith(
      "/api/voice/offer",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("fires onAssistantText when a data-channel message arrives", async () => {
    const env = mockBrowser();
    const onAssistantText = vi.fn();
    const { result } = renderHook(() =>
      useRealtimeVoice({}, { onAssistantText }),
    );
    await act(async () => {
      await result.current.start();
    });
    await act(async () => {
      env.dataChannelHandlers["message"]({
        data: JSON.stringify({ type: "assistant_text", text: "Hi there" }),
      });
    });
    expect(onAssistantText).toHaveBeenCalledWith("Hi there");
    expect(result.current.state).toBe("speaking");
  });
});
