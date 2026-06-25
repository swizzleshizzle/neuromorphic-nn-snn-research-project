import { create } from "zustand";
import type { Frame, TraceHeader } from "../contract";
import { advancePlayback } from "../playback/advance";

interface TraceStore {
  header?: TraceHeader;
  frames: Frame[];
  T: number;
  envStep: number;
  winTi: number;
  playing: boolean;

  load: (header: TraceHeader, frames: Frame[]) => void;
  setEnvStep: (i: number) => void;
  setWinTi: (ti: number) => void;
  play: () => void;
  pause: () => void;
  tickWindow: () => void;
  reset: () => void;
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

export const useTraceStore = create<TraceStore>((set, get) => ({
  frames: [],
  T: 1,
  envStep: 0,
  winTi: 0,
  playing: false,

  load: (header, frames) =>
    set({ header, frames, T: header.brain.T, envStep: 0, winTi: 0, playing: false }),

  setEnvStep: (i) => {
    const max = Math.max(0, get().frames.length - 1);
    set({ envStep: clamp(i, 0, max) });
  },

  setWinTi: (ti) => set({ winTi: clamp(ti, 0, Math.max(0, get().T - 1)) }),

  play: () => set({ playing: true }),
  pause: () => set({ playing: false }),

  tickWindow: () =>
    set((s) => advancePlayback({ winTi: s.winTi, envStep: s.envStep, T: s.T, frameCount: s.frames.length })),

  reset: () => set({ header: undefined, frames: [], T: 1, envStep: 0, winTi: 0, playing: false }),
}));
