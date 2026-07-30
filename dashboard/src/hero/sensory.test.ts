import { describe, expect, it } from "vitest";
import type { Frame } from "../contract";
import { aggregateCubeFacelets, aggregateSensoryGrid } from "./sensory";

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

  it("returns null for a cube encoding", () => {
    const enc = { sensory_input: { spikes: [new Array(144).fill(0)], cube_n: 2, n_colors: 6, index: "" } };
    expect(aggregateSensoryGrid(enc as never)).toBeNull();
  });
});

it("takes the argmax color per facelet", () => {
  // 24 facelets x 6 colors = 144. Facelet f is color f % 6, spiking twice.
  const row = new Array(144).fill(0);
  for (let f = 0; f < 24; f++) row[f * 6 + (f % 6)] = 1;
  const enc = { sensory_input: { spikes: [row, row], cube_n: 2, n_colors: 6, index: "" } };
  const got = aggregateCubeFacelets(enc as never);
  expect(got).toHaveLength(24);
  for (let f = 0; f < 24; f++) expect(got![f]).toBe(f % 6);
});

it("returns null for a gridworld encoding", () => {
  const enc = { sensory_input: { spikes: [[0, 1]], grid_n: 1, planes: [], index: "" } };
  expect(aggregateCubeFacelets(enc as never)).toBeNull();
});
