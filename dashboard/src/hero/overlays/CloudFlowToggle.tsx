import type React from "react";
import { useTraceStore } from "../../store/traceStore";

export function CloudFlowToggle() {
  const layout = useTraceStore((s) => s.heroLayout);
  const set = useTraceStore((s) => s.setHeroLayout);
  const seg = (active: boolean): React.CSSProperties => ({
    padding: "6px 11px",
    borderRadius: 6,
    border: "none",
    cursor: "pointer",
    font: "600 10px/1 'Space Grotesk', sans-serif",
    background: active ? "var(--c-router)" : "transparent",
    color: active ? "var(--text-bright)" : "var(--text-dim)",
  });
  return (
    <div
      data-hero-toggle
      style={{
        position: "absolute",
        top: 14,
        right: 18,
        display: "flex",
        gap: 3,
        padding: 3,
        borderRadius: 8,
        background: "var(--panel)",
        border: "1px solid var(--edge)",
        backdropFilter: "var(--blur)",
        zIndex: 10,
      }}
    >
      <button style={seg(layout === "cloud")} onClick={() => set("cloud")}>
        3D Cloud
      </button>
      <button style={seg(layout === "flow")} onClick={() => set("flow")}>
        Flow Map
      </button>
    </div>
  );
}
