import { describe, expect, it } from "vitest";
import { regionHue } from "./regionHue";

describe("regionHue", () => {
  it("maps known region ids to their CSS hue token", () => {
    expect(regionHue("sensory")).toBe("var(--c-sensory)");
    expect(regionHue("hippocampus")).toBe("var(--c-hippocampus)");
    expect(regionHue("motor")).toBe("var(--c-motor)");
  });

  it("falls back to a neutral token for unknown ids", () => {
    expect(regionHue("mystery")).toBe("var(--text-dim)");
  });
});
