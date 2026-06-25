const HUE_VAR: Record<string, string> = {
  sensory: "var(--c-sensory)",
  hippocampus: "var(--c-hippocampus)",
  prefrontal: "var(--c-prefrontal)",
  router: "var(--c-router)",
  motor: "var(--c-motor)",
};

/** CSS color token for a region/pathway-source id; neutral for unknown ids. */
export function regionHue(regionId: string): string {
  return HUE_VAR[regionId] ?? "var(--text-dim)";
}
