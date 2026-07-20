import { describe, it, expect } from "vitest";
import { parseGenerativeUI } from "@/lib/generative-ui";

describe("parseGenerativeUI", () => {
  it("returns a single text segment when there is no fenced block", () => {
    const segs = parseGenerativeUI("just text");
    expect(segs).toHaveLength(1);
    expect(segs[0].type).toBe("text");
    if (segs[0].type === "text") expect(segs[0].content).toBe("just text");
  });

  it("extracts a ui: block as a `ui` segment with parsed JSON data", () => {
    const content = '```ui:chart\n{"x":1}\n```';
    const segs = parseGenerativeUI(content);
    const ui = segs.find((s) => s.type === "ui");
    expect(ui).toBeDefined();
    if (ui && ui.type === "ui") {
      expect(ui.component).toBe("chart");
      expect(ui.data).toEqual({ x: 1 });
      expect(ui.content).toContain('{"x":1}');
    }
  });

  it("keeps HTML components (webapp) as verbatim content with no parsed data", () => {
    const content = '```ui:webapp\n<div>hi</div>\n```';
    const segs = parseGenerativeUI(content);
    const ui = segs.find((s) => s.type === "ui");
    expect(ui).toBeDefined();
    if (ui && ui.type === "ui") {
      expect(ui.component).toBe("webapp");
      expect(ui.content).toContain("<div>hi</div>");
      expect(ui.data).toBeUndefined();
    }
  });

  it("treats ordinary (non-ui) fenced blocks as text so markdown can render them", () => {
    const content = "```js\nconsole.log(1)\n```";
    const segs = parseGenerativeUI(content);
    expect(segs).toHaveLength(1);
    expect(segs[0].type).toBe("text");
    if (segs[0].type === "text") expect(segs[0].content).toContain("console.log");
  });

  it("falls back to a text segment for unparseable JSON ui blocks", () => {
    const content = "```ui:chart\nnot json\n```";
    const segs = parseGenerativeUI(content);
    expect(segs).toHaveLength(1);
    expect(segs[0].type).toBe("text");
    if (segs[0].type === "text") expect(segs[0].content).toContain("ui:chart");
  });

  it("interleaves text and ui segments in order", () => {
    const content = "before\n```ui:chart\n{\"a\":1}\n```\nafter";
    const segs = parseGenerativeUI(content);
    expect(segs.map((s) => s.type)).toEqual(["text", "ui", "text"]);
    if (segs[0].type === "text") expect(segs[0].content).toContain("before");
    if (segs[2].type === "text") expect(segs[2].content).toContain("after");
  });
});
