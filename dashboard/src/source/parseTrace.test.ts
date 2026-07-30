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

const gridHeader = {
  schema_version: "1.1",
  brain: { id: "five-region", config_hash: "abc12345", seed: 0, T: 8 },
  task: { type: "gridworld", grid_n: 5, action_labels: ["up", "right", "down", "left"] },
  regions: [], pathways: [], policy_regions: [],
};
const cubeHeader = {
  ...gridHeader,
  task: { type: "cube", cube_n: 2, action_labels: ["U", "U'", "R", "R'", "F", "F'"] },
};
const jsonl = (h: unknown, fs: unknown[]) =>
  [JSON.stringify(h), ...fs.map((f) => JSON.stringify(f))].join("\n");

describe("parseTrace task stamping", () => {
  it("stamps gridworld frames from the header", () => {
    const t = parseTrace(jsonl(gridHeader, [{ step: 0, task: { agent: [1, 2], goal: [3, 4] } }]));
    expect(t.frames[0].task.type).toBe("gridworld");
  });

  it("stamps cube frames from the header", () => {
    const t = parseTrace(jsonl(cubeHeader, [{ step: 0, task: { facelets: new Array(24).fill(0) } }]));
    const task = t.frames[0].task;
    expect(task.type).toBe("cube");
    if (task.type === "cube") expect(task.facelets).toHaveLength(24);
  });

  it("throws when a cube header carries gridworld frames", () => {
    expect(() =>
      parseTrace(jsonl(cubeHeader, [{ step: 0, task: { agent: [1, 2], goal: [3, 4] } }])),
    ).toThrow(/cube.*agent|agent.*cube/i);
  });

  it("throws when a gridworld header carries cube frames", () => {
    expect(() =>
      parseTrace(jsonl(gridHeader, [{ step: 0, task: { facelets: new Array(24).fill(0) } }])),
    ).toThrow(/gridworld.*facelets|facelets.*gridworld/i);
  });
});
