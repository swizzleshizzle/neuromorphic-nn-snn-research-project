import { describe, expect, it } from "vitest";
import type { Frame } from "../contract";
import { damp, lerpVec3, neuronGlow } from "./interp";

const frame = {
  field: { sensory: { spikes: [[1, 0], [0, 0], [0, 1], [0, 0]] } },
} as unknown as Frame;

describe("neuronGlow", () => {
  it("0.06 baseline, +0.5 on spike at ti, +0.18 on spike at ti-1 (wrapped)", () => {
    expect(neuronGlow(frame, "sensory", 0, 0, 4)).toMatchObject({ sp: 1 });
    expect(neuronGlow(frame, "sensory", 0, 0, 4).act).toBeCloseTo(0.56); // sp at 0, prev = ti-1 = 3 -> 0
    expect(neuronGlow(frame, "sensory", 0, 1, 4).act).toBeCloseTo(0.24); // ti=1 no spike, prev ti=0 spikes -> 0.06+0.18
    expect(neuronGlow(frame, "missing", 0, 0, 4)).toEqual({ sp: 0, act: 0.06 });
  });
});

describe("lerpVec3", () => {
  it("interpolates componentwise", () => {
    expect(lerpVec3([0, 0, 0], [2, 4, 6], 0.5)).toEqual([1, 2, 3]);
    expect(lerpVec3([1, 1, 1], [3, 3, 3], 0)).toEqual([1, 1, 1]);
  });
});

describe("damp", () => {
  it("stays at current when dt=0 and approaches target as dt grows", () => {
    expect(damp(0, 1, 6, 0)).toBe(0);
    expect(damp(0, 1, 6, 10)).toBeCloseTo(1, 5);
    const mid = damp(0, 1, 6, 0.1);
    expect(mid).toBeGreaterThan(0);
    expect(mid).toBeLessThan(1);
  });
});
