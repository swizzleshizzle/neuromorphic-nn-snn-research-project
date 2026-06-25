import { describe, expect, it } from "vitest";
import { parseTrace } from "./parseTrace";

const HEADER = {
  schema_version: "1.0",
  brain: { id: "five-region", config_hash: "ab12cd34", seed: 0, T: 32 },
  task: { type: "gridworld", grid_n: 5, action_labels: ["up", "right", "down", "left"] },
  regions: [{ id: "sensory", label: "Sensory Cortex", n_neurons: 64, role: "input", render: "dots" }],
  pathways: [{ id: "sens_pfc", src: "sensory", dst: "prefrontal", gated: false }],
};
const FRAME = { episode: 0, step: 0, t: 0, task: {}, regions: {}, pathways: {}, router: {}, field: {} };

describe("parseTrace", () => {
  it("splits header (line 0) from frames", () => {
    const text = JSON.stringify(HEADER) + "\n" + JSON.stringify(FRAME) + "\n" + JSON.stringify(FRAME) + "\n";
    const trace = parseTrace(text);
    expect(trace.header.schema_version).toBe("1.0");
    expect(trace.header.regions[0].id).toBe("sensory");
    expect(trace.frames).toHaveLength(2);
  });

  it("ignores blank trailing lines", () => {
    const text = JSON.stringify(HEADER) + "\n" + JSON.stringify(FRAME) + "\n\n";
    expect(parseTrace(text).frames).toHaveLength(1);
  });

  it("throws on empty input", () => {
    expect(() => parseTrace("")).toThrow();
  });
});
