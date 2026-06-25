import { describe, expect, it } from "vitest";
import type { Frame, TraceHeader } from "../contract";
import { buildNeurons, isSpiking } from "./layout";

const header = {
  schema_version: "1.0",
  brain: { id: "five-region", config_hash: "x", seed: 0, T: 4 },
  task: { type: "gridworld", grid_n: 5, action_labels: [] },
  regions: [
    { id: "sensory", label: "Sensory", n_neurons: 2, role: "input", render: "dots" },
    { id: "motor", label: "Motor", n_neurons: 3, role: "output", render: "dots" },
  ],
  pathways: [],
} as TraceHeader;

describe("buildNeurons", () => {
  it("emits one entry per neuron across all regions, in flow order", () => {
    const ns = buildNeurons(header);
    expect(ns).toHaveLength(5); // 2 + 3
    expect(ns.filter((n) => n.region === "sensory")).toHaveLength(2);
    expect(ns[0]).toMatchObject({ region: "sensory", idx: 0 });
    // x increases left-to-right by region order
    const sx = ns.find((n) => n.region === "sensory")!.x;
    const mx = ns.find((n) => n.region === "motor")!.x;
    expect(mx).toBeGreaterThan(sx);
  });
});

describe("isSpiking", () => {
  const frame = {
    field: { sensory: { spikes: [[1, 0], [0, 0], [0, 1], [0, 0]] } },
  } as unknown as Frame;

  it("reads field[region].spikes[ti][idx]", () => {
    expect(isSpiking(frame, "sensory", 0, 0)).toBe(true);
    expect(isSpiking(frame, "sensory", 1, 0)).toBe(false);
    // spikes[2] = [0, 1] → neuron idx 1 fires at window step ti=2.
    // (plan had transposed args (1,2); idx 2 is out of range for a 2-neuron region)
    expect(isSpiking(frame, "sensory", 2, 1)).toBe(true);
  });

  it("returns false when the region or index is absent", () => {
    expect(isSpiking(frame, "missing", 0, 0)).toBe(false);
    expect(isSpiking(frame, "sensory", 0, 9)).toBe(false);
  });
});
