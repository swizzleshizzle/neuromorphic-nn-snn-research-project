import type { Frame, TraceHeader } from "../contract";

/** Frontend twin of the Python TraceSink. The render path is agnostic to origin. */
export interface TraceSource {
  /** Resolve the run header. */
  open(): Promise<TraceHeader>;
  /** Emit frames: all-at-once for a file, live for a websocket. Call after open(). */
  subscribe(onFrame: (frame: Frame) => void): void;
  /** Release resources. */
  close(): void;
}
