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
});
