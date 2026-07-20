import { describe, it, expect } from "vitest";
import {
  parseRagContent,
  convertCitationMarkers,
  normalizeAnswerWhitespace,
  parseWebSearchContent,
} from "@/lib/web-search-citations";

describe("parseRagContent", () => {
  it("strips the leading SSE citation frame and extracts sources", () => {
    const raw = [
      'data: {"citations": {"1": {"id": "doc1", "content": "snippet", "metadata": {"filename": "report.pdf", "url": "http://example.com/r"}}}}',
      "",
      "The answer mentions a fact [1].",
    ].join("\n");

    const { answer, sources } = parseRagContent(raw);
    expect(answer).toBe("The answer mentions a fact [1].");
    expect(sources).toHaveLength(1);
    expect(sources[0].index).toBe(1);
    expect(sources[0].title).toBe("report.pdf");
    expect(sources[0].url).toBe("http://example.com/r");
    expect(sources[0].domain).toBe("example.com");
    expect(sources[0].snippet).toBe("snippet");
  });

  it("returns the raw text with empty sources when no frame is present", () => {
    const raw = "answer text\nSources:\n[1] A (a.com)";
    const { answer, sources } = parseRagContent(raw);
    expect(answer).toContain("answer");
    expect(sources).toEqual([]);
  });

  it("falls back gracefully on a malformed citation frame", () => {
    const raw = 'data: {not valid json}\n\nSome answer';
    const { answer, sources } = parseRagContent(raw);
    expect(answer).toBe(raw);
    expect(sources).toEqual([]);
  });

  it("uses metadata title/source/id as fallback title", () => {
    const raw = [
      'data: {"citations": {"2": {"id": "id-2", "metadata": {"source": "paper.md"}}}}',
      "",
      "body",
    ].join("\n");
    const { sources } = parseRagContent(raw);
    expect(sources[0].title).toBe("paper.md");
  });
});

describe("convertCitationMarkers", () => {
  it("turns bare [n] markers into #cite-n markdown links", () => {
    expect(convertCitationMarkers("see [1] ref")).toBe("see [1](#cite-1) ref");
  });

  it("converts multiple distinct markers", () => {
    expect(convertCitationMarkers("a [1] b [3]")).toBe("a [1](#cite-1) b [3](#cite-3)");
  });

  it("does not touch already-linked markers [n](...)", () => {
    const input = "see [1](http://x.com) now";
    expect(convertCitationMarkers(input)).toBe(input);
  });
});

describe("normalizeAnswerWhitespace", () => {
  it("collapses 3+ blank lines into a single blank line and trims ends", () => {
    expect(normalizeAnswerWhitespace("a\n\n\n\nb")).toBe("a\n\nb");
  });

  it("normalizes CRLF to LF and strips trailing whitespace on lines", () => {
    expect(normalizeAnswerWhitespace("a  \n\r\nb  ")).toBe("a\n\nb");
  });

  it("trims leading and trailing whitespace", () => {
    expect(normalizeAnswerWhitespace("  hello  ")).toBe("hello");
  });
});

describe("parseWebSearchContent", () => {
  const raw = [
    "Spain won [1].",
    "",
    "Sources:",
    "[1] **BBC Sport** (bbc.com) - [Link](https://www.bbc.com/sport/football)",
    "    *Full match report from the final.*",
    "",
    "[2] **CBS News** (cbsnews.com) - [Link](https://www.cbsnews.com/world-cup)",
  ].join("\n");

  it("separates the answer prose from the Sources block", () => {
    const { answer, sources } = parseWebSearchContent(raw);
    expect(answer).toBe("Spain won [1].");
    expect(sources).toHaveLength(2);
  });

  it("extracts indexed citations with title, domain, url and snippet", () => {
    const { sources } = parseWebSearchContent(raw);
    expect(sources[0]).toMatchObject({
      index: 1,
      title: "BBC Sport",
      domain: "bbc.com",
      url: "https://www.bbc.com/sport/football",
      snippet: "Full match report from the final.",
    });
    expect(sources[1].title).toBe("CBS News");
    expect(sources[1].url).toBe("https://www.cbsnews.com/world-cup");
  });

  it("returns the original text with empty sources when no Sources block exists", () => {
    const { answer, sources } = parseWebSearchContent("Just a plain answer.");
    expect(answer).toBe("Just a plain answer.");
    expect(sources).toEqual([]);
  });
});
