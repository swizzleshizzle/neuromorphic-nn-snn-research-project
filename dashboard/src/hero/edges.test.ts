import { describe, expect, it } from "vitest";
import type { PathwayState, TraceHeader } from "../contract";
import { buildEdges, edgeState, quadPoint } from "./edges";

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
