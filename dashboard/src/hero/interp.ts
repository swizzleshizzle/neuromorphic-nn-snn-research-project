import type { Frame } from "../contract";
import type { Vec3 } from "./layout";

/** Spike state + afterglow for one neuron at window step ti (ports the comp's _neuronGlow). */
const FLASH_WINDOW = 3; // timesteps over which a fire highlight decays out

export function neuronGlow(
  frame: Frame,
  region: string,
  idx: number,
  ti: number,
  T: number,
): { sp: number; act: number; flash: number } {
  const arr = frame.field?.[region]?.spikes;
  if (!arr) return { sp: 0, act: 0.06, flash: 0 };
  const sp = arr[ti]?.[idx] ?? 0;
  const prev = (ti - 1 + T) % T;
  const spPrev = arr[prev]?.[idx] ?? 0;
  const act = 0.06 + (sp ? 0.5 : 0) + (spPrev ? 0.18 : 0);
  // flash: continuous "just fired" highlight that decays over FLASH_WINDOW steps.
  // 1.0 the step it fires, then 1 - age/W, reaching 0 once the spike is W steps old.
  let flash = 0;
  for (let a = 0; a < FLASH_WINDOW; a++) {
    const spiked = arr[(ti - a + T) % T]?.[idx] ?? 0;
    if (spiked) {
      flash = Math.max(flash, 1 - a / FLASH_WINDOW);
    }
  }
  return { sp, act, flash };
}

export function lerpVec3(a: Vec3, b: Vec3, t: number): Vec3 {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

/** Frame-rate-independent exponential approach toward target. */
export function damp(current: number, target: number, lambda: number, dt: number): number {
  return target + (current - target) * Math.exp(-lambda * dt);
}
