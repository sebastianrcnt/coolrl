import { describe, it, expect } from "vitest";
import { formatSignedValue, formatDuration, formatBytes, escapeHtml } from "./format";

describe("formatSignedValue", () => {
  it("prefixes positive with +", () => expect(formatSignedValue(0.5)).toBe("+0.50"));
  it("keeps - for negative", () => expect(formatSignedValue(-1.2)).toBe("-1.20"));
  it("returns null for null input", () => expect(formatSignedValue(null)).toBeNull());
  it("returns null for undefined input", () => expect(formatSignedValue(undefined)).toBeNull());
  it("formats zero as +0.00", () => expect(formatSignedValue(0)).toBe("+0.00"));
});

describe("formatDuration", () => {
  it("shows one decimal for < 10s", () => expect(formatDuration(1200)).toBe("1.2초"));
  it("shows zero decimals for >= 10s", () => expect(formatDuration(12000)).toBe("12초"));
  it("returns empty string for null", () => expect(formatDuration(null)).toBe(""));
  it("returns empty string for undefined", () => expect(formatDuration(undefined)).toBe(""));
});

describe("formatBytes", () => {
  it("formats bytes", () => expect(formatBytes(512)).toBe("512B"));
  it("formats kilobytes", () => expect(formatBytes(1024)).toBe("1.0KB"));
  it("formats megabytes with one decimal when < 10", () => {
    expect(formatBytes(5 * 1024 * 1024)).toBe("5.0MB");
  });
  it("formats megabytes with no decimal when >= 10", () => {
    expect(formatBytes(20 * 1024 * 1024)).toBe("20MB");
  });
  it("returns n/a for 0", () => expect(formatBytes(0)).toBe("n/a"));
  it("returns n/a for negative", () => expect(formatBytes(-1)).toBe("n/a"));
});

describe("escapeHtml", () => {
  it("escapes all five special chars", () => {
    expect(escapeHtml(`<div class="a">&'test</div>`)).toBe(
      "&lt;div class=&quot;a&quot;&gt;&amp;&#39;test&lt;/div&gt;"
    );
  });
  it("passes through plain strings unchanged", () => {
    expect(escapeHtml("hello world")).toBe("hello world");
  });
  it("coerces non-strings", () => {
    expect(escapeHtml(42)).toBe("42");
  });
});
