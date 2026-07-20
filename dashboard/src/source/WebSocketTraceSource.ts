import type { Frame, TraceHeader } from "../contract";
import type { ConnectionState } from "../store/traceStore";
import type { TraceSource } from "./TraceSource";

interface WSLike {
  onopen: (() => void) | null;
  onmessage: ((ev: { data: string }) => void) | null;
  onclose: (() => void) | null;
  onerror: (() => void) | null;
  close(): void;
}

export interface WebSocketTraceSourceOpts {
  onState?: (s: ConnectionState) => void;
  /** Fires when a header arrives after the first one (post-reconnect). Consumer should reset the store. */
  onReconnect?: (header: TraceHeader) => void;
  wsFactory?: (url: string) => WSLike;
  baseDelayMs?: number;
  maxDelayMs?: number;
}

/** Live TraceSource: connects to the file-tail WebSocket server, with auto-reconnect. */
export class WebSocketTraceSource implements TraceSource {
  private ws?: WSLike;
  private onFrame?: (f: Frame) => void;
  private opened = false;
  private ended = false;
  private errored = false;
  private closed = false;
  private retry = 0;
  private resolveHeader?: (h: TraceHeader) => void;
  private rejectHeader?: (e: Error) => void;

  constructor(private readonly url: string, private readonly opts: WebSocketTraceSourceOpts = {}) {}

  open(): Promise<TraceHeader> {
    return new Promise<TraceHeader>((resolve, reject) => {
      this.resolveHeader = resolve;
      this.rejectHeader = reject;
      this.connect();
    });
  }

  subscribe(onFrame: (f: Frame) => void): void {
    this.onFrame = onFrame;
  }

  close(): void {
    this.closed = true;
    this.ws?.close();
  }

  private setState(s: ConnectionState): void {
    this.opts.onState?.(s);
  }

  private connect(): void {
    this.setState(this.opened ? "reconnecting" : "connecting");
    const make = this.opts.wsFactory ?? ((u: string) => new WebSocket(u) as unknown as WSLike);
    const ws = make(this.url);
    this.ws = ws;
    ws.onmessage = (ev) => this.handle(ev.data);
    ws.onclose = () => this.handleClose();
    ws.onerror = () => {}; // onclose drives reconnect
  }

  private handle(data: string): void {
    const msg = JSON.parse(data) as { type: string; data?: unknown };
    if (msg.type === "header") {
      this.retry = 0;
      this.setState("live");
      if (!this.opened) {
        this.opened = true;
        this.resolveHeader?.(msg.data as TraceHeader);
      } else {
        this.opts.onReconnect?.(msg.data as TraceHeader);
      }
    } else if (msg.type === "frame") {
      this.onFrame?.(msg.data as Frame);
    } else if (msg.type === "end") {
      this.ended = true;
      this.setState("ended");
      this.ws?.close();
    } else if (msg.type === "error") {
      this.errored = true;
      this.setState("error");
      if (!this.opened) this.rejectHeader?.(new Error(String((msg.data as { reason?: string })?.reason ?? "stream error")));
      this.ws?.close();
    }
  }

  private handleClose(): void {
    if (this.closed || this.ended || this.errored) return;
    const base = this.opts.baseDelayMs ?? 250;
    const max = this.opts.maxDelayMs ?? 5000;
    const delay = Math.min(max, base * 2 ** this.retry);
    this.retry += 1;
    this.setState("reconnecting");
    setTimeout(() => {
      if (!this.closed && !this.ended && !this.errored) this.connect();
    }, delay);
  }
}
