import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import MessageBubble from "@/components/message-bubble";
import type { AppSettings, UIMessage } from "@/lib/types";

const settings: AppSettings = {
  model: "gemma3:1b",
  backend: "ollama",
  apiKey: null,
  apiBase: null,
  ollamaUrl: null,
  serpApiKey: null,
  providerKeys: { ollama: null, openrouter: null, nvidia_nim: null },
  temperature: 0.7,
  maxTokens: 2048,
  topP: 1,
  topK: 40,
  minP: 0.05,
  frequencyPenalty: 0,
  repetitionPenalty: 1,
  seed: null,
  stopSequences: "",
  sessionId: "",
  enableGenerativeUI: false,
  genUIDisplayMode: "rendered",
  voiceMode: false,
};

const WEB_SEARCH_MSG: UIMessage = {
  id: "m1",
  role: "assistant",
  content: `Spain defeated Argentina 1- 0 in extra time to win the 2 026 FIFA World Cup [1][3][4][7].

Sources:

[1] **BBC Sport** (bbc.com) - [Link](https://www.bbc.com/sport/football)
    *Full match report from the final in Miami.*

[3] **CBS News** (cbsnews.com) - [Link](https://www.cbsnews.com/world-cup)
    *Spain wins 2026 FIFA World Cup with a 1-0 win over Argentina.*`,
  isStreaming: false,
};

describe("MessageBubble — web-search citations", () => {
  it("renders inline [n] markers as clickable citation chips", () => {
    render(
      <MessageBubble msg={WEB_SEARCH_MSG} settings={settings} onAction={vi.fn()} />,
    );

    expect(screen.getByLabelText("Citation 1")).toBeTruthy();
    expect(screen.getByLabelText("Citation 3")).toBeTruthy();
    expect(screen.getByLabelText("Citation 4")).toBeTruthy();
    expect(screen.getByLabelText("Citation 7")).toBeTruthy();
  });

  it("shows a Sources (N) button", () => {
    render(
      <MessageBubble msg={WEB_SEARCH_MSG} settings={settings} onAction={vi.fn()} />,
    );
    expect(
      screen.getByRole("button", { name: /Sources \(2\)/ }),
    ).toBeTruthy();
  });

  it("does not trap the answer in whitespace-pre-wrap", () => {
    const { container } = render(
      <MessageBubble msg={WEB_SEARCH_MSG} settings={settings} onAction={vi.fn()} />,
    );
    expect(container.querySelector(".whitespace-pre-wrap")).toBeNull();
  });

  it("opens the citations dialog with working links when a chip is clicked", async () => {
    render(
      <MessageBubble msg={WEB_SEARCH_MSG} settings={settings} onAction={vi.fn()} />,
    );

    fireEvent.click(screen.getByLabelText("Citation 1"));

    const bbcLink = await screen.findByRole("link", { name: "BBC Sport" });
    expect(bbcLink.getAttribute("href")).toBe(
      "https://www.bbc.com/sport/football",
    );
    // Both parsed sources are present and clickable.
    expect(
      screen.getByRole("link", { name: "CBS News" }).getAttribute("href"),
    ).toBe("https://www.cbsnews.com/world-cup");
  });

  it("opens the citations dialog from the Sources button", async () => {
    render(
      <MessageBubble msg={WEB_SEARCH_MSG} settings={settings} onAction={vi.fn()} />,
    );

    fireEvent.click(screen.getByRole("button", { name: /Sources \(2\)/ }));

    const bbcLink = await screen.findByRole("link", { name: "BBC Sport" });
    expect(bbcLink.getAttribute("href")).toBe(
      "https://www.bbc.com/sport/football",
    );
  });

  it("renders a normal (non-web-search) answer without a Sources button", () => {
    const plain: UIMessage = {
      id: "m2",
      role: "assistant",
      content: "The capital of France is Paris.",
      isStreaming: false,
    };
    render(<MessageBubble msg={plain} settings={settings} onAction={vi.fn()} />);
    expect(screen.queryByRole("button", { name: /Sources/ })).toBeNull();
  });
});
