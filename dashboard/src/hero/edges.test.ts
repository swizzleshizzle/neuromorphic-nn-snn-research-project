import { describe, expect, it } from "vitest";
import type { PathwayState, TraceHeader } from "../contract";
import { buildEdges, edgeState, quadPoint } from "./edges";
import { pulseCount, pulsePhase } from "./edges";

describe("edgeState", () => {
  it("no pathway frame: open iff not gated", () => {
    expect(edgeState(undefined, false)).toEqual({ inten: 0, open: true, quiescent: false });
    expect(edgeState(undefined, true)).toEqual({ inten: 0, open: false, quiescent: true });
  });
  it("ungated pathway frame is always open", () => {
    expect(edgeState({ intensity: 0.4 } as PathwayState, false)).toEqual({ inten: 0.4, open: true, quiescent: false });
  });
  it("gate_open > 0.5 opens; array takes the max; <= 0.5 is quiescent", () => {
    expect(edgeState({ intensity: 0.2, gate_open: 0.8 } as PathwayState, true).open).toBe(true);
    expect(edgeState({ intensity: 0.2, gate_open: [0.1, 0.9] } as PathwayState, true).open).toBe(true);
    expect(edgeState({ intensity: 0.2, gate_open: 0.3 } as PathwayState, true)).toMatchObject({ open: false, quiescent: true });
  });
});

describe("buildEdges", () => {
  it("maps header pathways to edges", () => {
    const header = { pathways: [{ id: "a_b", src: "a", dst: "b", gated: true }] } as unknown as TraceHeader;
    expect(buildEdges(header)).toEqual([{ id: "a_b", src: "a", dst: "b", gated: true }]);
  });
});

describe("quadPoint", () => {
  it("endpoints at t=0 and t=1; bow=0 gives the straight midpoint", () => {
    expect(quadPoint([0, 0, 0], [2, 0, 0], 0, 0)).toEqual([0, 0, 0]);
    expect(quadPoint([0, 0, 0], [2, 0, 0], 0, 1)).toEqual([2, 0, 0]);
    expect(quadPoint([0, 0, 0], [2, 0, 0], 0, 0.5)).toEqual([1, 0, 0]);
  });
  it("bow offsets the midpoint perpendicular in XY", () => {
    const mid = quadPoint([0, 0, 0], [2, 0, 0], 1, 0.5);
    expect(mid[0]).toBeCloseTo(1);
    expect(Math.abs(mid[1])).toBeGreaterThan(0); // bowed off the straight line
  });
});

describe("pulseCount", () => {
  it("returns 0 below threshold", () => {
    expect(pulseCount(0.04, 0.05, 3)).toBe(0);
    expect(pulseCount(0, 0.05, 3)).toBe(0);
  });

  it("returns 1 exactly at threshold", () => {
    expect(pulseCount(0.05, 0.05, 3)).toBe(1);
  });

  it("returns maxPulses at full intensity", () => {
    expect(pulseCount(1, 0.05, 3)).toBe(3);
  });

  it("is monotonically non-decreasing in intensity", () => {
    let prev = -1;
    for (let x = 0; x <= 1.0001; x += 0.05) {
      const c = pulseCount(x, 0.05, 3);
      expect(c).toBeGreaterThanOrEqual(prev);
      prev = c;
    }
  });
});

describe("pulsePhase", () => {
  it("is 0 at winTi=0 for pulse 0", () => {
    expect(pulsePhase(0, 32, 0, 3, 0)).toBe(0);
  });

  it("freezes: identical winTi gives identical phase", () => {
    const a = pulsePhase(7, 32, 1, 3, 0.21);
    const b = pulsePhase(7, 32, 1, 3, 0.21);
    expect(a).toBe(b);
  });

  it("always returns a value in [0, 1)", () => {
    for (let winTi = 0; winTi < 32; winTi++) {
      for (let k = 0; k < 3; k++) {
        const p = pulsePhase(winTi, 32, k, 3, 0.6);
        expect(p).toBeGreaterThanOrEqual(0);
        expect(p).toBeLessThan(1);
      }
    }
  });

  it("advances with the playhead", () => {
    expect(pulsePhase(16, 32, 0, 3, 0)).toBeCloseTo(0.5);
  });

  it("guards T=0 (no divide by zero)", () => {
    expect(pulsePhase(5, 0, 0, 1, 0)).toBe(0);
  });
});
