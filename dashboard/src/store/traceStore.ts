import { create } from "zustand";
import type { Frame, TraceHeader } from "../contract";
import { advancePlayback } from "../playback/advance";

export type ConnectionState = "idle" | "connecting" | "live" | "reconnecting" | "ended" | "error";

interface TraceStore {
  header?: TraceHeader;
  frames: Frame[];
  T: number;
  envStep: number;
  winTi: number;
  playing: boolean;
  heroLayout: "cloud" | "flow";
  connectionState: ConnectionState;

  load: (header: TraceHeader, frames: Frame[]) => void;
  appendFrame: (frame: Frame) => void;
  setConnectionState: (s: ConnectionState) => void;
  setEnvStep: (i: number) => void;
  setWinTi: (ti: number) => void;
  play: () => void;
  pause: () => void;
  tickWindow: () => void;
  reset: () => void;
  setHeroLayout: (v: "cloud" | "flow") => void;
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

/** Stamp a frame's task with the run's task type from the header.
 *
 * The wire format carries the type once, in the header. parseTrace stamps file
 * traces, but WebSocketTraceSource hands raw frames straight to appendFrame, so
 * the store is the only point both paths share. Idempotent.
 */
const stamp = (frame: Frame, header?: TraceHeader): Frame => {
  const type = header?.task?.type;
  if (!type || !frame?.task) return frame;
  return { ...frame, task: { ...frame.task, type } as Frame["task"] };
};

export const useTraceStore = create<TraceStore>((set, get) => ({
  frames: [],
  T: 1,
  envStep: 0,
  winTi: 0,
  playing: false,
  heroLayout: "cloud",
  connectionState: "idle",

  load: (header, frames) =>
    set({
      header,
      frames: frames.map((f) => stamp(f, header)),
      T: header.brain.T, envStep: 0, winTi: 0, playing: false,
    }),

  appendFrame: (frame) =>
    set((s) => {
      const frames = [...s.frames, stamp(frame, s.header)];
      return { frames, envStep: frames.length - 1 }; // follow-live (unconditional for MVP)
    }),

  setConnectionState: (connectionState) => set({ connectionState }),

  setEnvStep: (i) => {
    const max = Math.max(0, get().frames.length - 1);
    set({ envStep: clamp(i, 0, max) });
  },

  setWinTi: (ti) => set({ winTi: clamp(ti, 0, Math.max(0, get().T - 1)) }),

  play: () => set({ playing: true }),
  pause: () => set({ playing: false }),

  tickWindow: () =>
    set((s) => advancePlayback({ winTi: s.winTi, envStep: s.envStep, T: s.T, frameCount: s.frames.length })),

  reset: () => set({ header: undefined, frames: [], T: 1, envStep: 0, winTi: 0, playing: false, connectionState: "idle" }),

  setHeroLayout: (v) => set({ heroLayout: v }),
}));
