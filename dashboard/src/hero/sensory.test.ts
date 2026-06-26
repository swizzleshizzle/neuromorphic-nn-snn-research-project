import { describe, expect, it } from "vitest";
import type { Frame } from "../contract";
import { aggregateSensoryGrid } from "./sensory";

describe("aggregateSensoryGrid", () => {
  it("returns null when encoding is absent", () => {
    expect(aggregateSensoryGrid(undefined)).toBeNull();
  });

  it("argmax of the agent plane (first g*g) and goal plane (next g*g) over the window", () => {
    // grid_n = 2 -> 4 cells per plane, 8 columns total. agent fires cell 1, goal fires cell 3.
    const enc = {
      sensory_input: {
        grid_n: 2,
        planes: ["agent", "goal"],
        index: "row-major",
        spikes: [
          [0, 1, 0, 0, /*goal*/ 0, 0, 0, 1],
          [0, 1, 0, 0, /*goal*/ 0, 0, 0, 1],
        ],
      },
    } as unknown as Frame["encoding"];
    expect(aggregateSensoryGrid(enc)).toEqual({ agentCell: 1, goalCell: 3 });
  });

  it("reports -1 for a plane that never fires", () => {
    const enc = {
      sensory_input: { grid_n: 2, planes: ["agent", "goal"], index: "row-major", spikes: [[0, 0, 0, 0, 0, 0, 0, 0]] },
    } as unknown as Frame["encoding"];
    expect(aggregateSensoryGrid(enc)).toEqual({ agentCell: -1, goalCell: -1 });
  });
});
