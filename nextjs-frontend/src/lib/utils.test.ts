import { describe, it, expect } from "vitest";
import { cn } from "@/lib/utils";

describe("cn", () => {
  it("joins class names with spaces", () => {
    expect(cn("a", "b")).toBe("a b");
  });

  it("merges tailwind conflicts (tailwind-merge: later wins)", () => {
    expect(cn("px-2", "px-4")).toBe("px-4");
  });

  it("resolves conflicting boolean/color utilities", () => {
    expect(cn("text-red-500", "text-blue-500")).toBe("text-blue-500");
  });

  it("handles conditional and array inputs via clsx", () => {
    expect(cn("a", false && "b", ["c", "d"], { e: true, f: false })).toBe("a c d e");
  });

  it("returns an empty string for no args", () => {
    expect(cn()).toBe("");
  });
});
