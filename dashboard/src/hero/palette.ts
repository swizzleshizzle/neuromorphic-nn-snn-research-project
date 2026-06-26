export const REGION_HUE: Record<string, string> = {
  sensory: "#3fd2ff",
  hippocampus: "#ad8bff",
  prefrontal: "#ffd24a",
  router: "#ff5a8a",
  motor: "#46f0a0",
};
/** Region hex color with a deterministic hue fallback for unknown ids. */
export function hueFor(id: string, i: number): string {
  return REGION_HUE[id] ?? `hsl(${(i * 67) % 360} 88% 66%)`;
}
