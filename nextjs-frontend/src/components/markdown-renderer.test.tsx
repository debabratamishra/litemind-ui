import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import MarkdownRenderer from "@/components/markdown-renderer";

/**
 * Regression coverage for the web-search answer layout bug.
 *
 * Web-search answers are plain text (synthesised prose) followed by a
 * newline-separated `Sources:` block whose citations use markdown links:
 *   [n] **Title** (domain) - [Link](https://...)
 *
 * These used to render through a `<p className="whitespace-pre-wrap">`, which
 * (a) preserved every newline so each fragment sat on its own line and
 * (b) let the flex chat bubble shrink to the narrowest fragment, leaving the
 * viewport empty. The fix routes assistant messages through MarkdownRenderer.
 *
 * This test pins the resolved behaviour:
 *  - the answer reads as connected prose (no per-fragment line breaks),
 *  - source citations become real, clickable links,
 *  - nothing stays trapped in `whitespace-pre-wrap`.
 */
const WEB_SEARCH_ANSWER = `Spain defeated Argentina 1-0 in extra time to win the 2026 FIFA World Cup [1].

Sources:

[1] **BBC Sport** (bbc.com) - [Link](https://www.bbc.com/sport/football)
    *Full match report from the final in Miami.*

[2] **Reuters** (reuters.com) - [Link](https://www.reuters.com/sports/soccer)
    *Tournament recap and goalscorer details.*`;

describe("MarkdownRenderer — web-search answer layout", () => {
  it("renders the answer as connected prose, not one fragment per line", () => {
    render(<MarkdownRenderer content={WEB_SEARCH_ANSWER} />);

    // The full sentence must appear intact in a single paragraph — the bug
    // split it into short lines via whitespace-pre-wrap.
    const answerText = screen.getByText(
      /Spain defeated Argentina 1-0 in extra time to win the 2026 FIFA World Cup/,
    );
    expect(answerText.tagName).toBe("P");
    expect(answerText.textContent).toContain("win the 2026 FIFA World Cup [1].");
  });

  it("turns source citations into functional links", () => {
    const { container } = render(<MarkdownRenderer content={WEB_SEARCH_ANSWER} />);

    const bbc = container.querySelector(
      'a[href="https://www.bbc.com/sport/football"]',
    );
    expect(bbc).not.toBeNull();
    expect(bbc?.textContent).toBe("Link");

    const reuters = container.querySelector(
      'a[href="https://www.reuters.com/sports/soccer"]',
    );
    expect(reuters).not.toBeNull();

    // Snippets are preserved (not truncated or dropped).
    expect(screen.getByText(/Full match report from the final in Miami/)).toBeTruthy();
    expect(
      screen.getByText(/Tournament recap and goalscorer details/),
    ).toBeTruthy();
  });

  it("does not trap content in whitespace-pre-wrap (so it uses the bubble width)", () => {
    const { container } = render(<MarkdownRenderer content={WEB_SEARCH_ANSWER} />);

    const preWrapped = container.querySelector(".whitespace-pre-wrap");
    expect(preWrapped).toBeNull();
  });

  it("keeps the answer text fully intact (no SerpAPI content altered or removed)", () => {
    const { container } = render(<MarkdownRenderer content={WEB_SEARCH_ANSWER} />);
    expect(container.textContent).toContain("Spain defeated Argentina 1-0");
    expect(container.textContent).toContain("BBC Sport");
    expect(container.textContent).toContain("Reuters");
  });
});
