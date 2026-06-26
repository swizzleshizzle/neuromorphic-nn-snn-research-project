import { useTraceStore } from "../../store/traceStore";

export function HeroCaption() {
  const header = useTraceStore((s) => s.header);
  const layout = useTraceStore((s) => s.heroLayout);
  if (!header) return null;
  const caption =
    layout === "cloud"
      ? "distributed neuron cloud · field spikes · auto-orbit"
      : header.regions.map((r) => r.label.split(" ")[0].toLowerCase()).join(" → ");
  return (
    <div
      data-hero-caption
      style={{
        position: "absolute",
        top: 14,
        left: 18,
        display: "flex",
        alignItems: "center",
        gap: 9,
        font: "500 9px/1 'IBM Plex Mono', monospace",
        color: "var(--text-faint)",
        letterSpacing: ".1em",
        pointerEvents: "none",
        zIndex: 10,
      }}
    >
      <span style={{ color: "var(--text-dim)" }}>NEURON FIELD</span>
      <span>{caption}</span>
    </div>
  );
}
