import { describe, it, expect } from "vitest";
import type { BackendType, AppSettings, UIMessage, ChatMessage } from "@/lib/types";

describe("types — BackendType union", () => {
  it("accepts exactly the three known provider values", () => {
    const backends: BackendType[] = ["ollama", "openrouter", "nvidia_nim"];
    expect(backends).toEqual(["ollama", "openrouter", "nvidia_nim"]);
  });
});

describe("types — AppSettings shape", () => {
  it("documents the expected keys with correct value domains", () => {
    const sample: AppSettings = {
      backend: "ollama",
      model: "gemma3:1b",
      apiKey: null,
      apiBase: null,
      ollamaUrl: null,
      serpApiKey: null,
      sessionId: "sess-1",
      temperature: 0.7,
      maxTokens: 2048,
      topP: 0.9,
      topK: 40,
      minP: 0.0,
      frequencyPenalty: 0,
      repetitionPenalty: 1.0,
      seed: null,
      stopSequences: "",
      voiceMode: false,
      enableGenerativeUI: false,
      genUIDisplayMode: "rendered",
    };

    const requiredKeys: (keyof AppSettings)[] = [
      "backend",
      "model",
      "apiKey",
      "apiBase",
      "ollamaUrl",
      "serpApiKey",
      "sessionId",
      "temperature",
      "maxTokens",
      "topP",
      "topK",
      "minP",
      "frequencyPenalty",
      "repetitionPenalty",
      "seed",
      "stopSequences",
      "voiceMode",
      "enableGenerativeUI",
      "genUIDisplayMode",
    ];
    for (const key of requiredKeys) {
      expect(key in sample).toBe(true);
    }
    expect(["rendered", "code"]).toContain(sample.genUIDisplayMode);
    expect(["ollama", "openrouter", "nvidia_nim"]).toContain(sample.backend);
  });
});

describe("types — message shapes", () => {
  it("ChatMessage requires role + content", () => {
    const m: ChatMessage = { role: "user", content: "hi" };
    expect(m.role).toBe("user");
    expect(m.content).toBe("hi");
  });

  it("UIMessage allows optional id and isStreaming", () => {
    const m: UIMessage = { role: "assistant", content: "resp", isStreaming: true };
    expect(m.isStreaming).toBe(true);
  });
});
