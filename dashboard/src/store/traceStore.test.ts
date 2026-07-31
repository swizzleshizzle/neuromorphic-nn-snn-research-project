import { beforeEach, describe, expect, it } from "vitest";
import type { Frame, TraceHeader } from "../contract";
import { useTraceStore } from "./traceStore";

const header = {
  schema_version: "1.0",
  brain: { id: "five-region", config_hash: "ab12cd34", seed: 0, T: 32 },
  task: { type: "gridworld", grid_n: 5, action_labels: ["up", "right", "down", "left"] },
  regions: [],
  pathways: [],
} as TraceHeader;

const frame = (step: number) =>
  ({ episode: 0, step, t: step, task: {}, regions: {}, pathways: {}, router: {}, field: {} }) as unknown as Frame;

describe("traceStore", () => {
  beforeEach(() => useTraceStore.getState().reset());

  it("load sets header, frames, T, and resets playhead", () => {
    useTraceStore.getState().load(header, [frame(0), frame(1)]);
    const s = useTraceStore.getState();
    expect(s.header?.brain.T).toBe(32);
    expect(s.frames).toHaveLength(2);
    expect(s.T).toBe(32);
    expect(s.envStep).toBe(0);
    expect(s.winTi).toBe(0);
  });

  it("setEnvStep clamps to frame range", () => {
    useTraceStore.getState().load(header, [frame(0), frame(1)]);
    useTraceStore.getState().setEnvStep(5);
    expect(useTraceStore.getState().envStep).toBe(1);
    useTraceStore.getState().setEnvStep(-3);
    expect(useTraceStore.getState().envStep).toBe(0);
  });

  it("play/pause toggle the flag", () => {
    useTraceStore.getState().play();
    expect(useTraceStore.getState().playing).toBe(true);
    useTraceStore.getState().pause();
    expect(useTraceStore.getState().playing).toBe(false);
  });

  it("toggles hero layout", () => {
    expect(useTraceStore.getState().heroLayout).toBe("cloud");
    useTraceStore.getState().setHeroLayout("flow");
    expect(useTraceStore.getState().heroLayout).toBe("flow");
  });

  it("stamps loaded frames with the header task type", () => {
    const header = {
      schema_version: "1.1",
      brain: { id: "b", config_hash: "x", seed: 0, T: 1 },
      task: { type: "cube", cube_n: 2, action_labels: ["U"] },
      regions: [], pathways: [],
    } as unknown as TraceHeader;
    const frame = { episode: 0, step: 0, t: 0, task: { facelets: [] } } as unknown as Frame;
    useTraceStore.getState().load(header, [frame]);
    expect(useTraceStore.getState().frames[0].task.type).toBe("cube");
  });

  it("stamps appended live frames with the header task type", () => {
    const header = {
      schema_version: "1.1",
      brain: { id: "b", config_hash: "x", seed: 0, T: 1 },
      task: { type: "cube", cube_n: 2, action_labels: ["U"] },
      regions: [], pathways: [],
    } as unknown as TraceHeader;
    useTraceStore.getState().load(header, []);
    useTraceStore.getState().appendFrame(
      { episode: 0, step: 0, t: 0, task: { facelets: [] } } as unknown as Frame,
    );
    expect(useTraceStore.getState().frames[0].task.type).toBe("cube");
  });

  const cubeHeader = {
    schema_version: "1.1",
    brain: { id: "b", config_hash: "x", seed: 0, T: 1 },
    task: { type: "cube", cube_n: 2, action_labels: ["U"] },
    regions: [], pathways: [],
  } as unknown as TraceHeader;

  const gridHeader = {
    schema_version: "1.1",
    brain: { id: "b", config_hash: "x", seed: 0, T: 1 },
    task: { type: "gridworld", grid_n: 5, action_labels: ["up", "right", "down", "left"] },
    regions: [], pathways: [],
  } as unknown as TraceHeader;

  it("load() throws when a cube header carries a gridworld frame", () => {
    const badFrame = { episode: 0, step: 0, t: 0, task: { agent: [1, 2], goal: [3, 4] } } as unknown as Frame;
    expect(() => useTraceStore.getState().load(cubeHeader, [badFrame])).toThrow(/cube.*agent|agent.*cube/i);
  });

  it("appendFrame() throws when a cube header carries a gridworld frame", () => {
    useTraceStore.getState().load(cubeHeader, []);
    const badFrame = { episode: 0, step: 0, t: 0, task: { agent: [1, 2], goal: [3, 4] } } as unknown as Frame;
    expect(() => useTraceStore.getState().appendFrame(badFrame)).toThrow(/cube.*agent|agent.*cube/i);
  });

  it("load() throws when a gridworld header carries a cube frame", () => {
    const badFrame = { episode: 0, step: 0, t: 0, task: { facelets: new Array(24).fill(0) } } as unknown as Frame;
    expect(() => useTraceStore.getState().load(gridHeader, [badFrame])).toThrow(/gridworld.*facelets|facelets.*gridworld/i);
  });
});
