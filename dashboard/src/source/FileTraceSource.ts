import type { Frame, Trace, TraceHeader } from "../contract";
import { parseTrace } from "./parseTrace";
import type { TraceSource } from "./TraceSource";

/** Loads a JSONL trace by URL and replays its frames synchronously on subscribe. */
export class FileTraceSource implements TraceSource {
  private trace?: Trace;

  constructor(private readonly url: string) {}

  async open(): Promise<TraceHeader> {
    const res = await fetch(this.url);
    if (!res.ok) {
      throw new Error(`FileTraceSource: fetch ${this.url} failed (${res.status})`);
    }
    this.trace = parseTrace(await res.text());
    return this.trace.header;
  }

  subscribe(onFrame: (frame: Frame) => void): void {
    if (!this.trace) {
      throw new Error("FileTraceSource: call open() before subscribe()");
    }
    for (const frame of this.trace.frames) {
      onFrame(frame);
    }
  }

  close(): void {
    this.trace = undefined;
  }
}
