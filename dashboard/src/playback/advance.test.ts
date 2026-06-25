import { describe, expect, it } from "vitest";
import { advancePlayback } from "./advance";

describe("advancePlayback", () => {
  it("advances the window playhead within T", () => {
    const r = advancePlayback({ winTi: 3, envStep: 0, T: 32, frameCount: 10 });
    expect(r).toEqual({ winTi: 4, envStep: 0 });
  });

  it("wraps the window and advances the episode at T boundary", () => {
    const r = advancePlayback({ winTi: 31, envStep: 0, T: 32, frameCount: 10 });
    expect(r).toEqual({ winTi: 0, envStep: 1 });
  });

  it("wraps the episode back to 0 at the last frame", () => {
    const r = advancePlayback({ winTi: 31, envStep: 9, T: 32, frameCount: 10 });
    expect(r).toEqual({ winTi: 0, envStep: 0 });
  });

  it("is safe with zero frames", () => {
    const r = advancePlayback({ winTi: 31, envStep: 0, T: 32, frameCount: 0 });
    expect(r).toEqual({ winTi: 0, envStep: 0 });
  });
});
