import { describe, it, expect } from "vitest";
import type { BackendType, AppSettings, ProviderOverride, UIMessage, ChatMessage } from "@/lib/types";

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
      providerKeys: { ollama: null, openrouter: null, nvidia_nim: null },
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
      "providerKeys",
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

  it("providerKeys is a Record of BackendType to string|null", () => {
    const sample: AppSettings = {
      backend: "openrouter",
      model: "llama-3.3-70b",
      apiKey: "sk-123",
      apiBase: null,
      ollamaUrl: null,
      serpApiKey: null,
      providerKeys: { ollama: null, openrouter: "sk-123", nvidia_nim: null },
      sessionId: "sess-2",
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
    expect(sample.providerKeys.openrouter).toBe("sk-123");
    expect(sample.providerKeys.ollama).toBeNull();
    expect(sample.providerKeys.nvidia_nim).toBeNull();
  });
});

describe("types — ProviderOverride", () => {
  it("has all required fields with correct value domains", () => {
    const override: ProviderOverride = {
      alias: "claude",
      backend: "openrouter",
      model: "anthropic/claude-3.5-sonnet",
      text: "Claude via OpenRouter",
      hasKey: true,
    };
    expect(override.alias).toBe("claude");
    expect(override.backend).toBe("openrouter");
    expect(override.model).toBe("anthropic/claude-3.5-sonnet");
    expect(override.text).toBe("Claude via OpenRouter");
    expect(override.hasKey).toBe(true);
  });

  it("hasKey can be false when no key is configured", () => {
    const override: ProviderOverride = {
      alias: "local",
      backend: "ollama",
      model: "gemma3:1b",
      text: "Local Ollama",
      hasKey: false,
    };
    expect(override.hasKey).toBe(false);
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
