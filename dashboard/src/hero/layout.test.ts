import { describe, expect, it } from "vitest";
import type { Frame, TraceHeader } from "../contract";
import { buildHeroNeurons, clusterCentroids, isSpiking, shapeOf } from "./layout";

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

const heroHeader = {
  schema_version: "1.0",
  brain: { id: "b", config_hash: "x", seed: 0, T: 4 },
  task: { type: "gridworld", grid_n: 5, action_labels: [] },
  regions: [
    { id: "sensory", label: "Sensory Cortex", n_neurons: 4, role: "input", render: "dots" },
    { id: "router", label: "Thalamic Router", n_neurons: 9, role: "gating", render: "dots" },
    { id: "motor", label: "Motor", n_neurons: 5, role: "output", render: "dots" },
  ],
  pathways: [],
} as unknown as TraceHeader;

describe("shapeOf", () => {
  it("columns for small regions, grid for perfect squares, disc otherwise", () => {
    expect(shapeOf(5)).toBe("col");   // <= 8
    expect(shapeOf(8)).toBe("col");
    expect(shapeOf(9)).toBe("grid");  // perfect square > 8
    expect(shapeOf(12)).toBe("disc"); // non-square > 8
  });
});

describe("buildHeroNeurons", () => {
  const ns = buildHeroNeurons(heroHeader);

  it("emits one entry per neuron across all regions in header order", () => {
    expect(ns).toHaveLength(18); // 4 + 9 + 5
    expect(ns.filter((n) => n.region === "router")).toHaveLength(9);
    expect(ns[0]).toMatchObject({ region: "sensory", idx: 0 });
  });

  it("gives every neuron finite cloud and flow coordinates", () => {
    for (const n of ns) {
      for (const v of [...n.cloudPos, ...n.flowPos]) expect(Number.isFinite(v)).toBe(true);
      expect(n.flowPos[2]).toBe(0); // flow is a z=0 plane
    }
  });

  it("mean-centers the cloud near the origin", () => {
    const sum = ns.reduce((s, n) => [s[0] + n.cloudPos[0], s[1] + n.cloudPos[1], s[2] + n.cloudPos[2]], [0, 0, 0]);
    for (const axis of sum) expect(Math.abs(axis / ns.length)).toBeLessThan(1e-6);
  });

  it("spreads regions left-to-right by header order on the flow X axis", () => {
    const cx = (id: string) => {
      const pts = ns.filter((n) => n.region === id);
      return pts.reduce((s, n) => s + n.flowPos[0], 0) / pts.length;
    };
    expect(cx("sensory")).toBeLessThan(cx("router"));
    expect(cx("router")).toBeLessThan(cx("motor"));
  });
});

describe("clusterCentroids", () => {
  it("returns one centroid per region", () => {
    const c = clusterCentroids(buildHeroNeurons(heroHeader), "cloud");
    expect(c.size).toBe(3);
    expect(c.get("router")).toHaveLength(3);
  });
});
