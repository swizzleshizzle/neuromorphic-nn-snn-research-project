import { beforeEach, describe, expect, it } from "vitest";
import type { Frame, TraceHeader } from "../contract";
import { useTraceStore } from "./traceStore";

const header = { schema_version: "1.0", brain: { id: "b", config_hash: "h", seed: 0, T: 1 } } as unknown as TraceHeader;
const frame = (step: number) => ({ episode: 0, step, t: step } as unknown as Frame);

describe("traceStore live", () => {
  beforeEach(() => useTraceStore.getState().reset());

  it("appendFrame appends and follows the tail", () => {
    const s = useTraceStore.getState();
    s.load(header, []);
    s.appendFrame(frame(0));
    expect(useTraceStore.getState().frames.length).toBe(1);
    expect(useTraceStore.getState().envStep).toBe(0);
    s.appendFrame(frame(1));
    expect(useTraceStore.getState().frames.length).toBe(2);
    expect(useTraceStore.getState().envStep).toBe(1);
  });

  it("connectionState defaults to idle and is settable", () => {
    expect(useTraceStore.getState().connectionState).toBe("idle");
    useTraceStore.getState().setConnectionState("live");
    expect(useTraceStore.getState().connectionState).toBe("live");
  });

  it("reset restores idle connectionState", () => {
    useTraceStore.getState().setConnectionState("ended");
    useTraceStore.getState().reset();
    expect(useTraceStore.getState().connectionState).toBe("idle");
  });
});
