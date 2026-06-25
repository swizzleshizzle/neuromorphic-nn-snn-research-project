import { useEffect } from "react";
import { useTraceStore } from "../store/traceStore";

const STEP_HZ = 7; // ~7 window steps / second

/** Drives the store's window playhead with a single rAF loop while `playing`. */
export function usePlayback(): void {
  const playing = useTraceStore((s) => s.playing);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let last = performance.now();
    let acc = 0;
    const stepDur = 1 / STEP_HZ;

    const tick = (now: number) => {
      acc += (now - last) / 1000;
      last = now;
      while (acc >= stepDur) {
        acc -= stepDur;
        useTraceStore.getState().tickWindow();
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);
}
