import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { WebSocketTraceSource } from "./WebSocketTraceSource";

class FakeWS {
  static instances: FakeWS[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;
  constructor(public url: string) {
    FakeWS.instances.push(this);
  }
  send(obj: unknown) {
    this.onmessage?.({ data: JSON.stringify(obj) });
  }
  close() {
    this.closed = true;
    this.onclose?.();
  }
}

const factory = (url: string) => new FakeWS(url) as never;

beforeEach(() => {
  FakeWS.instances = [];
  vi.useFakeTimers();
});
afterEach(() => vi.useRealTimers());

describe("WebSocketTraceSource", () => {
  it("resolves open() on the header and delivers frames", async () => {
    const states: string[] = [];
    const src = new WebSocketTraceSource("ws://x/stream", { wsFactory: factory, onState: (s) => states.push(s) });
    const p = src.open();
    const frames: number[] = [];
    src.subscribe((f) => frames.push((f as { step: number }).step));
    FakeWS.instances[0].send({ type: "header", data: { schema_version: "1.0" } });
    await expect(p).resolves.toEqual({ schema_version: "1.0" });
    FakeWS.instances[0].send({ type: "frame", data: { step: 0 } });
    FakeWS.instances[0].send({ type: "frame", data: { step: 1 } });
    expect(frames).toEqual([0, 1]);
    expect(states).toContain("connecting");
    expect(states).toContain("live");
  });

  it("reconnects with backoff on a drop before end, firing onReconnect on the new header", async () => {
    const states: string[] = [];
    const reconns: unknown[] = [];
    const src = new WebSocketTraceSource("ws://x/stream", {
      wsFactory: factory,
      onState: (s) => states.push(s),
      onReconnect: (h) => reconns.push(h),
      baseDelayMs: 100,
    });
    const p = src.open();
    src.subscribe(() => {});
    FakeWS.instances[0].send({ type: "header", data: { schema_version: "1.0" } });
    await p;
    FakeWS.instances[0].close(); // drop before end
    expect(states).toContain("reconnecting");
    vi.advanceTimersByTime(100);
    expect(FakeWS.instances.length).toBe(2); // reconnected
    FakeWS.instances[1].send({ type: "header", data: { schema_version: "1.0" } });
    expect(reconns.length).toBe(1); // second header => reset signal
  });

  it("goes ended on end and does not reconnect", async () => {
    const states: string[] = [];
    const src = new WebSocketTraceSource("ws://x/stream", { wsFactory: factory, onState: (s) => states.push(s) });
    const p = src.open();
    src.subscribe(() => {});
    FakeWS.instances[0].send({ type: "header", data: {} });
    await p;
    FakeWS.instances[0].send({ type: "end" });
    expect(states[states.length - 1]).toBe("ended");
    vi.advanceTimersByTime(10000);
    expect(FakeWS.instances.length).toBe(1); // no reconnect after end
  });

  it("rejects open() and goes error on a pre-open error", async () => {
    const src = new WebSocketTraceSource("ws://x/stream", { wsFactory: factory });
    const p = src.open();
    FakeWS.instances[0].send({ type: "error", data: { reason: "trace not found" } });
    await expect(p).rejects.toThrow("trace not found");
    vi.advanceTimersByTime(10000);
    expect(FakeWS.instances.length).toBe(1); // no reconnect after error
  });
});
